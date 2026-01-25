#!/usr/bin/env python3
"""
HeartMuLa REST API Server - AGGRESSIVE MEMORY MANAGEMENT VERSION
This version completely reloads the model between jobs to ensure memory is released.

CRITICAL CHANGE: After each generation, we DELETE the entire pipeline and reload it.
This is slower (adds ~30s overhead) but guarantees memory cleanup.
"""

import os
import sys

# CRITICAL: Set CUDA memory allocation config BEFORE importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.9"

import uuid
import logging
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

# Setup logging
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
    sys.exit(1)

app = FastAPI(
    title="HeartMuLa API (Aggressive Reload)",
    version="2.0.0",
    description="Music generation API with aggressive memory management"
)

# Job storage
jobs = {}
JOB_RETENTION_HOURS = 24

# Working directory
WORK_DIR = Path("/app")
OUTPUT_DIR = WORK_DIR / "output"
MODEL_DIR = WORK_DIR / "ckpt"

# Global pipeline variable - NOW RELOADED AFTER EACH JOB
pipe = None
load_error = None
model_config = None  # Store config for reloading


class GenerateRequest(BaseModel):
    """Request model for music generation."""
    lyrics: str = Field(default="", description="Song lyrics")
    style_prompt: str = Field(default="", description="Style tags")
    seed: Optional[int] = Field(default=None, description="Random seed")
    max_audio_length_ms: int = Field(default=95000, ge=10000, le=300000)
    topk: int = Field(default=250, ge=1, le=1000)
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    cfg_scale: float = Field(default=3.0, ge=1.0, le=10.0)
    model_version: str = Field(default="3B")


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


def log_memory_usage():
    """Log current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")


def aggressive_cleanup():
    """
    NUCLEAR OPTION: Delete everything and force cleanup.
    This is called after EVERY job to ensure memory is completely freed.
    """
    global pipe
    
    logger.info("=" * 80)
    logger.info("🧹 AGGRESSIVE CLEANUP: Deleting pipeline and freeing all memory")
    logger.info("=" * 80)
    
    # Log memory before cleanup
    log_memory_usage()
    
    # Delete the pipeline
    if pipe is not None:
        try:
            # Try to move models to CPU first (helps with cleanup)
            if hasattr(pipe, '_mula') and pipe._mula is not None:
                try:
                    pipe._mula.cpu()
                except:
                    pass
            if hasattr(pipe, '_codec') and pipe._codec is not None:
                try:
                    pipe._codec.cpu()
                except:
                    pass
        except:
            pass
        
        del pipe
        pipe = None
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache multiple times (sometimes one isn't enough)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # Yes, twice
    
    # Log memory after cleanup
    log_memory_usage()
    
    logger.info("✓ Aggressive cleanup complete")
    logger.info("=" * 80)


def load_model():
    """
    Load the HeartMuLa model.
    This is called:
    1. On startup
    2. After EVERY job (to reload after deletion)
    """
    global pipe, model_config
    
    logger.info("🔄 Loading HeartMuLa model...")
    log_memory_usage()
    
    if model_config is None:
        raise RuntimeError("Model config not initialized")
    
    # Clear cache before loading
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load the pipeline with stored config
    pipe = HeartMuLaGenPipeline.from_pretrained(
        str(MODEL_DIR),
        **model_config
    )
    
    logger.info("✓ Model loaded successfully")
    log_memory_usage()


def generate_music_sync(job_id: str, request: GenerateRequest):
    """
    Synchronous music generation.
    After generation, pipeline is DELETED and RELOADED.
    """
    global pipe
    start_time = datetime.now()
    
    try:
        jobs[job_id]["status"] = "processing"
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{job_id}.wav"
        
        logger.info(f"[{job_id}] Starting generation...")
        logger.info(f"[{job_id}] Style: '{request.style_prompt}', Lyrics: {len(request.lyrics)} chars")
        
        # Ensure model is loaded
        if pipe is None:
            logger.info(f"[{job_id}] Model not loaded, loading now...")
            load_model()
        
        # Generate seed
        import random
        actual_seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(actual_seed)
        
        # Prepare inputs
        inputs = {
            "lyrics": request.lyrics,
            "tags": request.style_prompt,
        }
        
        # Memory cleanup BEFORE generation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        log_memory_usage()
        
        # Generation
        logger.info(f"[{job_id}] Running inference...")
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
            # Fallback for tensor output
            if isinstance(wav, torch.Tensor):
                logger.warning(f"[{job_id}] Saving tensor manually...")
                torchaudio.save(str(output_path), wav.cpu(), 32000)
                
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["output_path"] = str(output_path)
                jobs[job_id]["completed_at"] = datetime.now().isoformat()
            else:
                raise RuntimeError("Output file not created")
        
        # CRITICAL: AGGRESSIVE CLEANUP AFTER EVERY JOB
        aggressive_cleanup()
        
        # Reload model for next job
        logger.info(f"[{job_id}] Reloading model for next job...")
        load_model()

    except Exception as e:
        duration = datetime.now() - start_time
        logger.error(f"[{job_id}] ✗ Failed after {duration.total_seconds():.1f}s: {e}")
        traceback.print_exc()
        
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        # Cleanup even on error
        aggressive_cleanup()
        
        # Try to reload for next job
        try:
            load_model()
        except Exception as reload_error:
            logger.error(f"Failed to reload model after error: {reload_error}")


@app.on_event("startup")
async def startup_event():
    """Initialize model config and load model once on startup."""
    global pipe, model_config, load_error
    
    logger.info("=" * 80)
    logger.info("HeartMuLa API Server Starting (Aggressive Reload Mode)")
    logger.info("=" * 80)
    
    try:
        # Configure model loading parameters
        quantization_type = os.environ.get("HEARTMULA_QUANTIZATION", "none").lower()
        bnb_config = None
        
        # Auto-detect for <20GB GPUs
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram_gb < 20 and quantization_type == "none":
                logger.info(f"Auto-enabling 4-bit for {vram_gb:.1f}GB VRAM")
                quantization_type = "4bit"
        
        # Setup quantization if needed
        if quantization_type == "4bit" and BITSANDBYTES_AVAILABLE:
            quant_type = "nf4"
            if torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability()
                if major >= 10:
                    quant_type = "fp4"
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            logger.info(f"✓ Using {quant_type} quantization")
        
        # Determine device/dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # Determine offloading strategy
        total_vram_gb = 0
        if torch.cuda.is_available():
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        use_lazy_load = total_vram_gb < 32
        use_codec_cpu_offload = total_vram_gb < 14
        
        if use_codec_cpu_offload:
            logger.warning(f"⚠️  Very limited VRAM ({total_vram_gb:.1f}GB) - offloading codec to CPU")
        
        # Store config for reloading
        model_config = {
            "device": device,
            "dtype": dtype,
            "version": "3B",
            "lazy_load": use_lazy_load,
        }
        
        if use_codec_cpu_offload:
            model_config["device"] = {
                "mula": torch.device("cuda"),
                "codec": torch.device("cpu"),
            }
            model_config["dtype"] = {
                "mula": dtype,
                "codec": torch.float32,
            }
        
        if bnb_config:
            model_config["bnb_config"] = bnb_config
        
        # Load model once
        load_model()
        
        logger.info("=" * 80)
        logger.info("✓ API Ready - Aggressive reload mode active")
        logger.info("✓ Model will be reloaded after EVERY job to ensure memory cleanup")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        traceback.print_exc()
        load_error = str(e)


@app.get("/")
async def root():
    return {
        "service": "HeartMuLa API (Aggressive Reload)",
        "version": "2.0.0",
        "model_loaded": pipe is not None,
        "mode": "aggressive_reload"
    }


@app.get("/health")
async def health_check():
    status = "healthy" if pipe is not None else "error"
    response = {
        "status": status,
        "model_loaded": pipe is not None,
        "load_error": load_error,
        "cuda_available": torch.cuda.is_available(),
        "active_jobs": len([j for j in jobs.values() if j["status"] in ["pending", "processing"]]),
    }
    
    if torch.cuda.is_available():
        try:
            response["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
            response["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 2)
        except:
            pass
    
    return response


@app.post("/generate", response_model=JobStatus)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "status": "pending",
        "output_path": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": None
    }
    
    background_tasks.add_task(generate_music_sync, job_id, request)
    
    return JobStatus(
        job_id=job_id,
        status="pending",
        created_at=jobs[job_id]["created_at"]
    )


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
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
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job {job['status']}")
    
    output_path = job.get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename=f"{job_id}.wav"
    )


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs.pop(job_id)
    
    if job.get("output_path"):
        try:
            Path(job["output_path"]).unlink(missing_ok=True)
        except:
            pass
    
    return {"message": f"Job {job_id} deleted"}


if __name__ == "__main__":
    import uvicorn
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting uvicorn server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )