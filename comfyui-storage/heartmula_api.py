#!/usr/bin/env python3
"""
HeartMuLa REST API Server - ULTRA MEMORY OPTIMIZED VERSION
For 16GB VRAM GPUs - Forces codec to CPU and uses aggressive memory management.

Key optimizations:
1. Force codec CPU offloading for all GPUs <20GB
2. Use lazy_load=True (official recommendation)
3. Clear CUDA cache before/after every operation
4. Aggressive garbage collection
"""

import os
import sys

# CRITICAL: Set CUDA memory allocation config BEFORE importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:128"

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

# Import HeartMuLa pipeline
try:
    from heartlib import HeartMuLaGenPipeline
    logger.info("✓ heartlib imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import HeartMuLa pipeline: {e}")
    sys.exit(1)

app = FastAPI(
    title="HeartMuLa API (Ultra Memory Optimized)",
    version="2.1.0",
    description="Music generation API optimized for 16GB VRAM"
)

# Job storage
jobs = {}
JOB_RETENTION_HOURS = 24

# Working directory
WORK_DIR = Path("/app")
OUTPUT_DIR = WORK_DIR / "output"
MODEL_DIR = WORK_DIR / "ckpt"

# Global pipeline variable
pipe = None
load_error = None
model_config = None


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


def log_memory_usage(context=""):
    """Log current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        logger.info(f"[{context}] GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free of {total:.2f}GB total")


def aggressive_cleanup():
    """
    Ultra-aggressive memory cleanup.
    Called before model loading and after generation.
    """
    logger.info("🧹 Aggressive cleanup starting...")
    log_memory_usage("Before cleanup")
    
    # Force garbage collection multiple times
    for i in range(3):
        gc.collect()
    
    # Clear CUDA cache aggressively
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # Yes, twice
        torch.cuda.reset_peak_memory_stats()
    
    log_memory_usage("After cleanup")
    logger.info("✓ Cleanup complete")


def load_model():
    """
    Load the HeartMuLa model with ultra memory optimization.
    """
    global pipe, model_config
    
    logger.info("=" * 80)
    logger.info("🔄 Loading HeartMuLa model with memory optimization...")
    logger.info("=" * 80)
    
    if model_config is None:
        raise RuntimeError("Model config not initialized")
    
    # Aggressive cleanup before loading
    aggressive_cleanup()
    
    # Load the pipeline
    logger.info(f"Loading with config: {model_config}")
    pipe = HeartMuLaGenPipeline.from_pretrained(
        str(MODEL_DIR),
        **model_config
    )
    
    logger.info("✓ Model loaded successfully")
    log_memory_usage("After model load")
    logger.info("=" * 80)


def unload_model():
    """
    Completely unload the model to free memory.
    """
    global pipe
    
    logger.info("=" * 80)
    logger.info("🗑️  Unloading model...")
    logger.info("=" * 80)
    
    if pipe is not None:
        try:
            # Move components to CPU first
            if hasattr(pipe, '_mula') and pipe._mula is not None:
                try:
                    pipe._mula.cpu()
                except Exception as e:
                    logger.warning(f"Could not move mula to CPU: {e}")
            
            if hasattr(pipe, '_codec') and pipe._codec is not None:
                try:
                    pipe._codec.cpu()
                except Exception as e:
                    logger.warning(f"Could not move codec to CPU: {e}")
        except Exception as e:
            logger.warning(f"Error during component cleanup: {e}")
        
        # Delete the pipeline
        del pipe
        pipe = None
    
    # Aggressive cleanup
    aggressive_cleanup()
    
    logger.info("✓ Model unloaded")
    logger.info("=" * 80)


def generate_music_sync(job_id: str, request: GenerateRequest):
    """
    Synchronous music generation with reload cycle.
    Unloads model after each job to guarantee memory cleanup.
    """
    global pipe
    start_time = datetime.now()
    
    try:
        jobs[job_id]["status"] = "processing"
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{job_id}.wav"
        
        logger.info(f"[{job_id}] Starting generation...")
        logger.info(f"[{job_id}] Style: '{request.style_prompt}', Lyrics: {len(request.lyrics)} chars")
        log_memory_usage(f"Job {job_id} start")
        
        # Ensure model is loaded
        if pipe is None:
            logger.info(f"[{job_id}] Model not loaded, loading now...")
            load_model()
        
        # Additional cleanup before generation
        aggressive_cleanup()
        log_memory_usage(f"Job {job_id} pre-gen")
        
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
        
        # Generation
        logger.info(f"[{job_id}] Running inference...")
        with torch.inference_mode():  # Use inference mode for memory efficiency
            wav = pipe(
                inputs,
                max_audio_length_ms=request.max_audio_length_ms,
                save_path=str(output_path),
                topk=request.topk,
                temperature=request.temperature,
                cfg_scale=request.cfg_scale,
            )
        
        # Immediate cleanup after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
        
        # CRITICAL: Unload model after EVERY job
        unload_model()
        
        # Reload for next job
        logger.info(f"[{job_id}] Reloading model for next job...")
        load_model()

    except Exception as e:
        duration = datetime.now() - start_time
        logger.error(f"[{job_id}] ✗ Failed after {duration.total_seconds():.1f}s: {e}")
        traceback.print_exc()
        
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        # Cleanup on error
        unload_model()
        
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
    logger.info("HeartMuLa API Server Starting (Ultra Memory Optimized)")
    logger.info("=" * 80)
    
    try:
        # Get VRAM info
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Detected GPU with {vram_gb:.2f}GB VRAM")
        else:
            vram_gb = 0
            logger.warning("No CUDA device detected")
        
        # CRITICAL: For 16GB GPUs, FORCE codec to CPU
        # This is the key to making it work on 16GB
        force_codec_cpu = vram_gb < 20  # Any GPU under 20GB
        
        if force_codec_cpu:
            logger.warning(f"⚠️  GPU has {vram_gb:.1f}GB VRAM - FORCING codec to CPU")
            logger.warning("⚠️  This will slow down generation but prevent OOM")
            
            model_config = {
                "device": {
                    "mula": torch.device("cuda"),
                    "codec": torch.device("cpu"),  # Force codec to CPU
                },
                "dtype": {
                    "mula": torch.bfloat16,
                    "codec": torch.float32,  # CPU uses float32
                },
                "version": "3B",
                "lazy_load": True,  # Official recommendation
            }
        else:
            # Large GPU - can keep everything on CUDA
            model_config = {
                "device": torch.device("cuda"),
                "dtype": torch.bfloat16,
                "version": "3B",
                "lazy_load": True,
            }
        
        logger.info(f"Model config: {model_config}")
        
        # Load model
        load_model()
        
        logger.info("=" * 80)
        logger.info("✓ API Ready - Ultra memory optimized mode")
        logger.info("✓ Codec offloaded to CPU for memory efficiency")
        logger.info("✓ Model will reload after EVERY job")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        traceback.print_exc()
        load_error = str(e)


@app.get("/")
async def root():
    return {
        "service": "HeartMuLa API (Ultra Memory Optimized)",
        "version": "2.1.0",
        "model_loaded": pipe is not None,
        "mode": "codec_cpu_offload"
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
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            response["gpu_memory_allocated_gb"] = round(allocated, 2)
            response["gpu_memory_reserved_gb"] = round(reserved, 2)
            response["gpu_memory_total_gb"] = round(total, 2)
            response["gpu_memory_free_gb"] = round(total - allocated, 2)
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