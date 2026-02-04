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

# Import ACE-Step handler
try:
    from acestep.handler import AceStepHandler
    logger.info("ACE-Step handler imported successfully")
except ImportError as e:
    logger.error(f"Failed to import ACE-Step handler: {e}")
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

# Global handler variable
handler = None
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

# Patch handler to slice latents for VRAM efficiency if needed
def patch_handler(handler_instance):
    import types
    # Use the existing logger instead of importing loguru
    global logger
    
    # Patch service_generate to slice latents if CFG returned 2x batch size
    # We use the class method to avoid double self passing if the instance method is already bound
    original_service_generate = handler_instance.__class__.service_generate
    
    def patched_service_generate(self, *args, **kwargs):
        # Call the original class method with self
        outputs = original_service_generate(self, *args, **kwargs)
        if "target_latents" in outputs:
            lats = outputs["target_latents"]
            # Detect if it's CFG overhead (batch 2 for 1 requested)
            # Find captions in args or kwargs
            captions = kwargs.get("captions")
            if captions is None and len(args) > 0:
                captions = args[0]
            
            if captions is not None and isinstance(captions, list) and len(captions) == 1 and lats.shape[0] == 2:
                logger.info(f"Monkeypatch: Slicing CFG latents from {lats.shape} to [1, ...] to save VRAM")
                # Take the second half (conditioned)
                outputs["target_latents"] = lats[1:] 
        return outputs
    
    handler_instance.service_generate = types.MethodType(patched_service_generate, handler_instance)
    logger.info("ACE-Step handler monkeypatched for VRAM optimization")

def load_model():
    """Load the ACE-Step model using the handler."""
    global handler, load_error
    try:
        if handler is not None:
            return
            
        cleanup() # Keep cleanup before loading
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading ACE-Step model on {device} (Offload={CPU_OFFLOAD})")
        
        handler = AceStepHandler()
        
        # Initialize service
        status, success = handler.initialize_service(
            project_root=str(WORK_DIR),
            config_path="acestep-v15-turbo",
            device=device,
            offload_to_cpu=CPU_OFFLOAD,
            offload_dit_to_cpu=CPU_OFFLOAD
        )
        
        # Apply VRAM optimization patch
        patch_handler(handler)
        
        logger.info(f"Model initialization: {status}")
        if not success:
            raise Exception(f"Model failed to initialize: {status}")
            
        load_error = None
    except Exception as e:
        logger.error(f"Failed to load ACE-Step model: {e}")
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
        
        if handler is None:
            load_model()
            
        import random
        actual_seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        
        # ACE-Step handler call
        # audio_duration should be length of the audio in seconds
        # We enforce batch_size=1 to avoid OOM on 8GB cards
        result = handler.generate_music(
            captions=request.style_prompt,
            lyrics=request.lyrics,
            seed=actual_seed,
            audio_duration=request.max_audio_length_ms / 1000.0,
            inference_steps=8, # Default for turbo
            guidance_scale=request.cfg_scale,
            batch_size=1,
            use_tiled_decode=True
        )
        
        if result.get("is_error"):
            raise Exception(result.get("error", "Unknown error in generation"))
            
        audios = result.get("audios", [])
        if not audios:
            raise Exception("No audio generated")
            
        wav = audios[0].get("tensor")
        sample_rate = audios[0].get("sample_rate", 32000)
        
        if wav is None:
            raise Exception("No audio tensor found in result")
            
        # Save output
        if isinstance(wav, torch.Tensor):
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            torchaudio.save(str(output_path), wav.cpu(), sample_rate)
        else:
            # Fallback if it's already a path or list
            logger.warning(f"Unexpected audio type: {type(wav)}")
            raise Exception("Generated audio is not a tensor")
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_path"] = str(output_path)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        duration = datetime.now() - start_time
        logger.info(f"[{job_id}] Completed in {duration.total_seconds():.1f}s")
        
    except Exception as e:
        logger.error(f"[{job_id}] Failed: {e}")
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
        "status": "healthy" if handler is not None else "initializing",
        "model_loaded": handler is not None,
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
