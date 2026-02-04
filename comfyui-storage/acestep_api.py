#!/usr/bin/env python3
"""
ACE-Step-1.5 REST API Server for OpenFork.
standardized to match HeartMuLa and DiffRhythm patterns.
"""

import os
import sys

# CRITICAL: Set CUDA memory allocation config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.7,max_split_size_mb:128"

import uuid
import logging
import traceback
import gc
import torch
import torchaudio
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AceStepAPI")

# Import ACE-Step pipeline
# In the Dockerfile, this script is placed in /app/acestep_repo/
try:
    from acestep.acestep_v15_pipeline import AceStepV15Pipeline
    logger.info("✓ ACE-Step pipeline imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import ACE-Step pipeline: {e}")
    sys.exit(1)

app = FastAPI(
    title="ACE-Step 1.5 API (OpenFork)",
    version="1.5.0",
    description="Music generation API for ACE-Step 1.5"
)

# Job storage
jobs = {}

# Working directory
WORK_DIR = Path("/app/acestep_repo")
OUTPUT_DIR = Path("/app/output")
CHECKPOINT_DIR = WORK_DIR / "checkpoints"

# Environment variable for CPU offloading (set in 8GB Dockerfile)
CPU_OFFLOAD = os.getenv("ACESTEP_CPU_OFFLOAD", "false").lower() == "true"

# Global pipeline variable
pipe = None
load_error = None

class GenerateRequest(BaseModel):
    """Request model for music generation."""
    lyrics: str = Field(default="", description="Song lyrics")
    style_prompt: str = Field(default="", description="Style tags")
    seed: Optional[int] = Field(default=None, description="Random seed")
    max_audio_length_ms: int = Field(default=60000, ge=5000, le=300000)
    topk: int = Field(default=250, ge=1, le=1000)
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    cfg_scale: float = Field(default=3.0, ge=1.0, le=10.0)
    model_version: str = Field(default="turbo")

class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

def cleanup():
    """Manual memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_model():
    """Load the ACE-Step model."""
    global pipe, load_error
    try:
        logger.info(f"🔄 Loading ACE-Step model (CPU Offload: {CPU_OFFLOAD})...")
        cleanup()
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize pipeline
        # ACE-Step 1.5.0+ supports offload_to_cpu in pipeline init or generate
        pipe = AceStepV15Pipeline(
            config_path="acestep-v15-turbo",
            lm_model_path="acestep-5Hz-lm-1.7B",
            device=device,
            offload_to_cpu=CPU_OFFLOAD
        )
        logger.info(f"✓ ACE-Step model loaded successfully on {device}")
        load_error = None
    except Exception as e:
        logger.error(f"✗ Failed to load ACE-Step model: {e}")
        traceback.print_exc()
        load_error = str(e)
        raise e

def generate_music_sync(job_id: str, request: GenerateRequest):
    """Synchronous generation task."""
    start_time = datetime.now()
    try:
        jobs[job_id]["status"] = "processing"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{job_id}.wav"
        
        logger.info(f"[{job_id}] Starting generation: Style='{request.style_prompt}', Lyrics={len(request.lyrics)} chars")
        
        if pipe is None:
            load_model()
            
        import random
        actual_seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        
        # ACE-Step pipeline call
        # Based on their Gradio interface / CLI logic
        wav = pipe.generate(
            lyrics=request.lyrics,
            tags=request.style_prompt,
            negative_tags="",
            seed=actual_seed,
            duration_ms=request.max_audio_length_ms,
            temperature=request.temperature,
            top_k=request.topk,
            top_p=0.95,
            cfg_scale=request.cfg_scale,
        )
        
        # Save output
        # wav is usually a torch.Tensor (1, length) or similar
        torchaudio.save(str(output_path), wav.cpu(), 32000)
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_path"] = str(output_path)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        duration = datetime.now() - start_time
        logger.info(f"[{job_id}] ✓ Completed in {duration.total_seconds():.1f}s")
        
    except Exception as e:
        logger.error(f"[{job_id}] ✗ Failed: {e}")
        traceback.print_exc()
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
    finally:
        cleanup()

@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except:
        logger.warning("Pipeline will be loaded on first request if startup load failed")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if pipe is not None else "initializing",
        "model_loaded": pipe is not None,
        "load_error": load_error,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/generate", response_model=JobStatus)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }
    background_tasks.add_task(generate_music_sync, job_id, request)
    return JobStatus(job_id=job_id, status="pending", created_at=jobs[job_id]["created_at"])

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return JobStatus(job_id=job_id, **job)

@app.get("/download/{job_id}")
async def download(job_id: str):
    if job_id not in jobs or jobs[job_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail="Result not ready or not found")
    return FileResponse(jobs[job_id]["output_path"], media_type="audio/wav", filename=f"{job_id}.wav")

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    if job_id in jobs:
        path = jobs[job_id].get("output_path")
        if path and os.path.exists(path):
            os.remove(path)
        jobs.pop(job_id)
    return {"status": "deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
