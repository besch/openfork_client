#!/usr/bin/env python3
"""
HeartMuLa REST API Server for OpenFork DGN Client
Provides a simple HTTP API for music generation using HeartMuLa model.

IMPROVEMENTS:
- Better error handling and logging during model loading
- Progress indicators for each loading stage
- Memory monitoring
- Graceful degradation on errors
- Health endpoint reports loading progress
"""

import os
import sys
import uuid
import logging
import tempfile
import traceback
import gc
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import torch
import torchaudio

# Setup logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
    logger.info("✓ bitsandbytes available")
except ImportError:
    logger.warning("⚠️  bitsandbytes not available - quantization will be disabled")
    BITSANDBYTES_AVAILABLE = False
    BitsAndBytesConfig = None

# Import HeartMuLa pipeline
try:
    from heartlib import HeartMuLaGenPipeline
    logger.info("✓ heartlib imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import HeartMuLa pipeline: {e}")
    logger.error("Make sure heartlib is installed: pip install -e /app/heartlib_repo")
    sys.exit(1)

app = FastAPI(
    title="HeartMuLa API",
    version="1.0.0",
    description="Music generation API using HeartMuLa foundation models"
)

# Job storage with timestamps for cleanup
jobs = {}
JOB_RETENTION_HOURS = 24

# Working directory
WORK_DIR = Path("/app")
OUTPUT_DIR = WORK_DIR / "output"
INPUT_DIR = WORK_DIR / "input"
MODEL_DIR = WORK_DIR / "ckpt"

# Global pipeline variable and loading state
pipe = None
load_error = None
loading_progress = {
    "stage": "not_started",
    "progress": 0,
    "message": "Model not loaded yet"
}


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
    status: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    estimated_duration_seconds: Optional[float] = None


def update_loading_progress(stage: str, progress: int, message: str):
    """Update the loading progress for health endpoint."""
    global loading_progress
    loading_progress = {
        "stage": stage,
        "progress": progress,
        "message": message
    }
    logger.info(f"[{progress}%] {stage}: {message}")


def log_memory_usage():
    """Log current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")


def cleanup_old_jobs():
    """Remove jobs older than retention period."""
    cutoff_time = datetime.now() - timedelta(hours=JOB_RETENTION_HOURS)
    jobs_to_delete = []
    
    for job_id, job_data in jobs.items():
        created_at = datetime.fromisoformat(job_data.get('created_at', datetime.now().isoformat()))
        if created_at < cutoff_time:
            jobs_to_delete.append(job_id)
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
        
        # Log memory before generation
        log_memory_usage()
        
        # Generation
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
            file_size = output_path.stat().st_size / (1024 * 1024)
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
                torchaudio.save(str(output_path), wav.cpu(), 32000)
                
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["output_path"] = str(output_path)
                jobs[job_id]["completed_at"] = datetime.now().isoformat()
                logger.info(f"[{job_id}] ✓ Saved manually")
            else:
                raise RuntimeError(f"Output file not created and no tensor returned")
        
        # Log memory after generation
        log_memory_usage()

    except Exception as e:
        duration = datetime.now() - start_time
        logger.error(f"[{job_id}] ✗ Failed after {duration.total_seconds():.1f}s: {e}")
        traceback.print_exc()
        
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()


@app.on_event("startup")
async def startup_event():
    """Load the model on startup with detailed progress tracking."""
    global pipe, load_error
    logger.info("=" * 80)
    logger.info("HeartMuLa API Server Starting...")
    logger.info("=" * 80)
    
    try:
        # Stage 0: Clear GPU memory before starting
        update_loading_progress("memory_cleanup", 2, "Clearing GPU memory")
        if torch.cuda.is_available():
            logger.info("Clearing GPU cache before model loading...")
            torch.cuda.empty_cache()
            gc.collect()
            
            # Log initial GPU state
            initial_allocated = torch.cuda.memory_allocated() / 1024**3
            initial_reserved = torch.cuda.memory_reserved() / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            logger.info(f"GPU Memory Before Loading:")
            logger.info(f"  Allocated: {initial_allocated:.2f}GB")
            logger.info(f"  Reserved: {initial_reserved:.2f}GB")
            logger.info(f"  Total: {total_memory:.2f}GB")
            logger.info(f"  Free: {total_memory - initial_allocated:.2f}GB")
            
            # Critical check: if >90% memory already used, fail early
            if initial_allocated > total_memory * 0.9:
                error_msg = (
                    f"GPU memory already {initial_allocated:.2f}GB / {total_memory:.2f}GB used. "
                    f"Cannot load HeartMuLa model. Other services are consuming GPU memory. "
                    f"Stop ComfyUI, YUME, or other GPU services first."
                )
                logger.error(error_msg)
                update_loading_progress("failed", 0, error_msg)
                load_error = error_msg
                return
        
        # Stage 1: System info
        update_loading_progress("system_check", 5, "Checking system configuration")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            
            # Get GPU compute capability
            major, minor = torch.cuda.get_device_capability(0)
            compute_capability = f"{major}.{minor}"
            logger.info(f"GPU Compute Capability: {compute_capability}")
            
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU memory: {memory_gb:.1f} GB")
            
            # Validate compute capability
            # PyTorch 2.4 with CUDA 12.4 requires compute capability >= 5.0
            # Common issues:
            # - Kepler GPUs (GTX 700 series): compute capability 3.x (NOT SUPPORTED)
            # - Maxwell GPUs (GTX 900 series): compute capability 5.x (SUPPORTED)
            # - Pascal and newer: compute capability >= 6.0 (FULLY SUPPORTED)
            if major < 5:
                error_msg = (
                    f"GPU compute capability {compute_capability} is too old for PyTorch 2.4 + CUDA 12.4. "
                    f"Minimum required: 5.0 (Maxwell/GTX 900 series or newer). "
                    f"Your GPU ({torch.cuda.get_device_name(0)}) is not supported by this container. "
                    f"Consider using an older PyTorch build or a newer GPU."
                )
                logger.error(error_msg)
                update_loading_progress("failed", 0, error_msg)
                load_error = error_msg
                return
            elif major == 5:
                logger.warning(f"⚠️  GPU has compute capability {compute_capability} (Maxwell architecture)")
                logger.warning("   Some optimizations may not be available. Pascal (6.0+) or newer recommended.")
            
            # Warn if GPU has less than 12GB
            if memory_gb < 12:
                logger.warning(f"⚠️  GPU has only {memory_gb:.1f}GB VRAM")
                logger.warning("   HeartMuLa 3B model requires 12-16GB VRAM")
                logger.warning("   Consider enabling 4-bit quantization: HEARTMULA_QUANTIZATION=4bit")

        
        # Stage 2: Verify model files
        update_loading_progress("file_check", 10, "Verifying model files")
        model_path = str(MODEL_DIR)
        
        required_paths = [
            (MODEL_DIR / "HeartMuLa-oss-3B", "HeartMuLa model"),
            (MODEL_DIR / "HeartCodec-oss", "HeartCodec model"),
            (MODEL_DIR / "gen_config.json", "Generation config"),
            (MODEL_DIR / "tokenizer.json", "Tokenizer"),
        ]
        
        missing_files = []
        for path, name in required_paths:
            if not path.exists():
                missing_files.append(f"{name} ({path})")
                logger.error(f"✗ Missing: {name} at {path}")
            else:
                logger.info(f"✓ Found: {name}")
        
        if missing_files:
            error_msg = f"Missing required files: {', '.join(missing_files)}"
            update_loading_progress("failed", 0, error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"✓ All model files present")
        
        # Stage 3: Configure quantization
        update_loading_progress("config", 20, "Configuring model parameters")
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
            logger.info("✓ Quantization config created (will use ~6GB VRAM)")
        elif quantization_type == "8bit" and BITSANDBYTES_AVAILABLE:
            logger.info("Enabling 8-bit quantization (int8)...")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            logger.info("✓ Quantization config created (will use ~8-10GB VRAM)")
        elif quantization_type in ["4bit", "8bit"]:
            logger.warning(f"⚠️  {quantization_type} quantization requested but bitsandbytes not available")
            logger.warning("   Model will load in full precision (requires 16GB+ VRAM)")
        else:
            logger.info("Loading model in full precision (requires 16GB+ VRAM)")
        
        # Stage 4: Prepare device and dtype
        update_loading_progress("device_setup", 25, "Configuring device and dtype")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        logger.info(f"Device: {device}, dtype: {dtype}")
        log_memory_usage()
        
        # Stage 5: Load pipeline (this is the long part)
        update_loading_progress("loading_pipeline", 30, "Loading HeartMuLa pipeline (this may take 10-15 minutes)...")
        
        logger.info("=" * 80)
        logger.info("LOADING PIPELINE - This will take 10-15 minutes")
        logger.info("=" * 80)
        
        pipe = HeartMuLaGenPipeline.from_pretrained(
            model_path,
            device=device,
            dtype=dtype,
            version="3B",
            bnb_config=bnb_config
        )
        
        update_loading_progress("pipeline_loaded", 80, "Pipeline loaded, initializing...")
        
        # Stage 6: Warmup (optional but helpful)
        update_loading_progress("warmup", 90, "Running warmup generation")
        try:
            logger.info("Running warmup generation to compile kernels...")
            log_memory_usage()
            
            # Quick 5-second warmup
            warmup_output = OUTPUT_DIR / "warmup.wav"
            pipe(
                {"lyrics": "test warmup", "tags": "test"},
                max_audio_length_ms=5000,
                save_path=str(warmup_output),
                topk=50,
                temperature=1.0,
                cfg_scale=1.5,
            )
            
            if warmup_output.exists():
                warmup_output.unlink()
            
            logger.info("✓ Warmup completed successfully")
            log_memory_usage()
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")
        
        # Stage 7: Complete
        update_loading_progress("ready", 100, "Model ready")
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        log_memory_usage()
        
        logger.info("=" * 80)
        logger.info("✓ HeartMuLa pipeline loaded successfully!")
        logger.info("=" * 80)
        logger.info("Server ready to accept requests")
        logger.info("=" * 80)
        
    except Exception as e:
        load_error = str(e)
        update_loading_progress("failed", 0, f"Loading failed: {str(e)}")
        logger.error("=" * 80)
        logger.error("✗ FAILED TO LOAD MODEL")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        traceback.print_exc()
        logger.error("=" * 80)
        
        # Provide specific guidance for common errors
        error_str = str(e).lower()
        
        # CUDA kernel compatibility error
        if "no kernel image is available" in error_str or "cuda error" in error_str:
            logger.error("CUDA KERNEL COMPATIBILITY ERROR DETECTED")
            logger.error("=" * 80)
            logger.error("This error typically means your GPU architecture is not supported")
            logger.error("by the PyTorch binaries in this container.")
            logger.error("")
            logger.error("DIAGNOSTIC STEPS:")
            logger.error("")
            if torch.cuda.is_available():
                try:
                    major, minor = torch.cuda.get_device_capability(0)
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.error(f"  GPU: {gpu_name}")
                    logger.error(f"  Compute Capability: {major}.{minor}")
                    logger.error(f"  PyTorch version: {torch.__version__}")
                    logger.error(f"  CUDA version: {torch.version.cuda}")
                    logger.error("")
                    
                    if major < 5:
                        logger.error("  ❌ Your GPU is TOO OLD (compute capability < 5.0)")
                        logger.error("     PyTorch 2.4 + CUDA 12.4 requires at least compute capability 5.0")
                        logger.error("")
                        logger.error("  SOLUTIONS:")
                        logger.error("     1. Use a newer GPU (GTX 900 series or newer)")
                        logger.error("     2. Use an older Docker image with PyTorch 1.x + CUDA 11.x")
                    elif major >= 5:
                        logger.error(f"  ⚠️  GPU compute capability {major}.{minor} should be supported")
                        logger.error("      This might be a driver/library mismatch issue")
                        logger.error("")
                        logger.error("  SOLUTIONS:")
                        logger.error("     1. Update NVIDIA driver to latest version")
                        logger.error("     2. Verify CUDA toolkit version matches container (12.4)")
                        logger.error("     3. Rebuild container with --no-cache")
                except Exception as diag_e:
                    logger.error(f"  Could not get GPU diagnostics: {diag_e}")
            logger.error("=" * 80)
        
        # CUDA OOM error
        elif "cuda out of memory" in error_str or "out of memory" in error_str:
            logger.error("CUDA OUT OF MEMORY ERROR DETECTED")
            logger.error("=" * 80)
            logger.error("SOLUTIONS:")
            logger.error("  1. Enable 4-bit quantization:")
            logger.error("     export HEARTMULA_QUANTIZATION=4bit")
            logger.error("  2. Stop other GPU services:")
            logger.error("     pkill -f comfyui")
            logger.error("     pkill -f yume_api")
            logger.error("  3. Use a GPU with more VRAM:")
            logger.error("     - Minimum: 12GB (with 4-bit quantization)")
            logger.error("     - Recommended: 16GB+ (full precision)")
            logger.error("=" * 80)
        
        logger.error("Server will start but generation will fail")
        logger.error("=" * 80)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "HeartMuLa Music Generation API",
        "version": "1.0.0",
        "model_loaded": pipe is not None,
        "loading_progress": loading_progress,
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
    """Health check endpoint with detailed loading status."""
    status = "healthy"
    if pipe is None:
        if load_error:
            status = "error"
        elif loading_progress["stage"] == "not_started":
            status = "initializing"
        elif loading_progress["stage"] == "failed":
            status = "error"
        else:
            status = "model_loading"
    
    response = {
        "status": status,
        "loading_progress": loading_progress,
        "load_error": load_error,
        "cuda_available": torch.cuda.is_available(),
        "active_jobs": len([j for j in jobs.values() if j["status"] in ["pending", "processing"]]),
        "total_jobs": len(jobs)
    }
    
    # Add GPU memory info
    if torch.cuda.is_available():
        try:
            response["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
            response["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 2)
            response["gpu_memory_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        except:
            pass
    
    return response


@app.post("/generate", response_model=JobStatus)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Start music generation job."""
    if pipe is None:
        error_detail = "Model is not loaded. "
        if load_error:
            error_detail += f"Error: {load_error}"
        elif loading_progress["stage"] != "ready":
            error_detail += f"Still loading: {loading_progress['message']} ({loading_progress['progress']}%)"
        raise HTTPException(status_code=503, detail=error_detail)

    job_id = str(uuid.uuid4())
    estimated_seconds = request.max_audio_length_ms / 1000.0
    
    jobs[job_id] = {
        "status": "pending",
        "output_path": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": None
    }
    
    if len(jobs) % 10 == 0:
        background_tasks.add_task(cleanup_old_jobs)
    
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
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting uvicorn server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )