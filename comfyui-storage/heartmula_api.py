#!/usr/bin/env python3
"""
HeartMuLa REST API Server for OpenFork DGN Client
Provides a simple HTTP API for music generation using HeartMuLa model.
"""

import os
import sys
import uuid
import logging
import tempfile
import traceback
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import torch
import torchaudio

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    logger.warning("bitsandbytes not available - quantization will be disabled")
    BITSANDBYTES_AVAILABLE = False
    BitsAndBytesConfig = None

# Import HeartMuLa pipeline
try:
    from heartlib.inference import HeartMuLaGenPipeline
except ImportError as e:
    logger.error(f"Failed to import HeartMuLa pipeline: {e}")
    logger.error("Make sure heartlib is installed: pip install -e /app/heartlib_repo")
    sys.exit(1)

app = FastAPI(
    title="HeartMuLa API",
    version="1.0.0",
    description="Music generation API using HeartMuLa foundation models"
)

# Job storage with timestamps for cleanup
jobs = {}
JOB_RETENTION_HOURS = 24  # Clean up jobs older than 24 hours

# Working directory
WORK_DIR = Path("/app")
OUTPUT_DIR = WORK_DIR / "output"
INPUT_DIR = WORK_DIR / "input"
MODEL_DIR = WORK_DIR / "ckpt"

# Global pipeline variable
pipe = None

class GenerateRequest(BaseModel):
    """Request model for music generation."""
    lyrics: str = Field(default="", description="Song lyrics with optional structure tags like [Verse], [Chorus]")
    style_prompt: str = Field(default="", description="Comma-separated style tags (e.g., 'piano,happy,romantic')")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    max_audio_length_ms: int = Field(default=95000, ge=10000, le=300000, description="Maximum audio length in milliseconds (10s-300s)")
    topk: int = Field(default=250, ge=1, le=1000, description="Top-k sampling parameter")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Sampling temperature")
    cfg_scale: float = Field(default=3.0, ge=1.0, le=10.0, description="Classifier-free guidance scale")
    model_version: str = Field(default="3B", description="Model version (3B or 7B)")


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str  # pending, processing, completed, failed
    output_path: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    estimated_duration_seconds: Optional[float] = None


def cleanup_old_jobs():
    """Remove jobs older than retention period."""
    cutoff_time = datetime.now() - timedelta(hours=JOB_RETENTION_HOURS)
    jobs_to_delete = []
    
    for job_id, job_data in jobs.items():
        created_at = datetime.fromisoformat(job_data.get('created_at', datetime.now().isoformat()))
        if created_at < cutoff_time:
            jobs_to_delete.append(job_id)
            # Delete output file
            if job_data.get('output_path'):
                try:
                    Path(job_data['output_path']).unlink(missing_ok=True)
                except Exception as e:
                    logger.error(f"Failed to delete file for job {job_id}: {e}")
    
    for job_id in jobs_to_delete:
        del jobs[job_id]
    
    if jobs_to_delete:
        logger.info(f"Cleaned up {len(jobs_to_delete)} old jobs")


def generate_music_sync(job_id: str, request: GenerateRequest):
    """Synchronous music generation using HeartMuLa pipeline."""
    global pipe
    start_time = datetime.now()
    
    try:
        jobs[job_id]["status"] = "processing"
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{job_id}.wav"
        
        logger.info(f"[{job_id}] Starting generation...")
        logger.info(f"[{job_id}] Style: '{request.style_prompt}', Lyrics: {len(request.lyrics)} chars")
        logger.info(f"[{job_id}] Duration: {request.max_audio_length_ms}ms, Seed: {request.seed}")
        
        if pipe is None:
            raise RuntimeError("HeartMuLa pipeline is not loaded.")

        # Set seed for reproducibility
        torch.manual_seed(request.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request.seed)

        # Prepare inputs
        inputs = {
            "lyrics": request.lyrics,
            "tags": request.style_prompt,
        }
        
        # Generation - HeartMuLa saves directly to save_path
        logger.info(f"[{job_id}] Running inference (this may take several minutes)...")
        wav = pipe(
            inputs,
            max_audio_length_ms=request.max_audio_length_ms,
            save_path=str(output_path),
            topk=request.topk,
            temperature=request.temperature,
            cfg_scale=request.cfg_scale,
        )
        
        # Verify output
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            duration = datetime.now() - start_time
            
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["output_path"] = str(output_path)
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
            
            logger.info(f"[{job_id}] ✓ Completed in {duration.total_seconds():.1f}s")
            logger.info(f"[{job_id}] Output: {output_path.name} ({file_size:.2f} MB)")
        else:
            # Fallback: check if wav tensor was returned
            if isinstance(wav, torch.Tensor):
                logger.warning(f"[{job_id}] save_path not used, saving tensor manually...")
                # HeartMuLa uses 32kHz sample rate (verify in actual code)
                torchaudio.save(str(output_path), wav.cpu(), 32000)
                
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["output_path"] = str(output_path)
                jobs[job_id]["completed_at"] = datetime.now().isoformat()
                logger.info(f"[{job_id}] ✓ Saved manually")
            else:
                raise RuntimeError(f"Output file not created and no tensor returned")

    except Exception as e:
        duration = datetime.now() - start_time
        logger.error(f"[{job_id}] ✗ Failed after {duration.total_seconds():.1f}s: {e}")
        traceback.print_exc()
        
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()


@app.on_event("startup")
async def startup_event():
    """Load the model on startup."""
    global pipe
    logger.info("=" * 60)
    logger.info("HeartMuLa API Server Starting...")
    logger.info("=" * 60)
    
    # System info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU memory: {memory_gb:.1f} GB")
    
    # Load pipeline
    try:
        logger.info("Loading HeartMuLa pipeline...")
        model_path = str(MODEL_DIR)
        
        # Check if model files exist
        required_paths = [
            MODEL_DIR / "HeartMuLa-oss-3B",
            MODEL_DIR / "HeartCodec-oss",
            MODEL_DIR / "gen_config.json",
            MODEL_DIR / "tokenizer.json"
        ]
        
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Required model file not found: {path}")
        
        logger.info(f"Model path: {model_path}")
        
        # Quantization setup
        quantization_type = os.environ.get("HEARTMULA_QUANTIZATION", "none").lower()
        bnb_config = None
        
        if quantization_type == "4bit" and BITSANDBYTES_AVAILABLE:
            logger.info("Enabling 4-bit quantization (nf4)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif quantization_type == "4bit":
            logger.warning("4-bit quantization requested but bitsandbytes not available")
        
        # Load pipeline
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        logger.info(f"Device: {device}, dtype: {dtype}")
        
        pipe = HeartMuLaGenPipeline.from_pretrained(
            model_path,
            device=device,
            dtype=dtype,
            version="3B",
            bnb_config=bnb_config
        )
        
        logger.info("✓ HeartMuLa pipeline loaded successfully!")
        logger.info("=" * 60)
        logger.info("Server ready to accept requests")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("✗ FAILED TO LOAD MODEL")
        logger.error("=" * 60)
        logger.error(f"Error: {e}")
        traceback.print_exc()
        logger.error("=" * 60)
        logger.error("Server will start but generation will fail")
        logger.error("=" * 60)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "HeartMuLa Music Generation API",
        "version": "1.0.0",
        "model_loaded": pipe is not None,
        "endpoints": {
            "health": "/health",
            "generate": "/generate (POST)",
            "status": "/status/{job_id}",
            "download": "/download/{job_id}",
            "cleanup": "/cleanup (POST)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if pipe is not None else "model_not_loaded",
        "cuda_available": torch.cuda.is_available(),
        "active_jobs": len([j for j in jobs.values() if j["status"] in ["pending", "processing"]]),
        "total_jobs": len(jobs)
    }


@app.post("/generate", response_model=JobStatus)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Start music generation job.
    Returns job_id that can be used to poll for status.
    
    Estimated generation time: ~1-3 minutes for 95s of audio (RTF ≈ 1.0)
    """
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check server logs.")

    job_id = str(uuid.uuid4())
    
    # Estimate duration based on audio length (RTF ≈ 1.0 baseline)
    estimated_seconds = request.max_audio_length_ms / 1000.0
    
    jobs[job_id] = {
        "status": "pending",
        "output_path": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": None
    }
    
    # Clean up old jobs periodically
    if len(jobs) % 10 == 0:  # Every 10 jobs
        background_tasks.add_task(cleanup_old_jobs)
    
    # Run generation in background
    background_tasks.add_task(generate_music_sync, job_id, request)
    
    return JobStatus(
        job_id=job_id,
        status="pending",
        created_at=jobs[job_id]["created_at"],
        estimated_duration_seconds=estimated_seconds
    )


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Get the status of a generation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        output_path=job.get("output_path"),
        error=job.get("error"),
        created_at=job.get("created_at"),
        completed_at=job.get("completed_at")
    )


@app.get("/download/{job_id}")
async def download(job_id: str):
    """Download the generated audio file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] == "pending":
        raise HTTPException(status_code=202, detail="Job is pending")
    elif job["status"] == "processing":
        raise HTTPException(status_code=202, detail="Job is still processing")
    elif job["status"] == "failed":
        raise HTTPException(status_code=400, detail=f"Job failed: {job.get('error', 'Unknown error')}")
    
    output_path = job.get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename=f"{job_id}.wav",
        headers={"Content-Disposition": f"attachment; filename={job_id}.wav"}
    )


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its output file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs.pop(job_id)
    
    # Delete output file if it exists
    if job.get("output_path"):
        try:
            Path(job["output_path"]).unlink(missing_ok=True)
            logger.info(f"Deleted job {job_id} and its output file")
        except Exception as e:
            logger.error(f"Failed to delete output file: {e}")
    
    return {"message": f"Job {job_id} deleted"}


@app.post("/cleanup")
async def manual_cleanup():
    """Manually trigger cleanup of old jobs."""
    initial_count = len(jobs)
    cleanup_old_jobs()
    cleaned_count = initial_count - len(jobs)
    
    return {
        "message": "Cleanup completed",
        "jobs_removed": cleaned_count,
        "jobs_remaining": len(jobs)
    }


if __name__ == "__main__":
    import uvicorn
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting uvicorn server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )