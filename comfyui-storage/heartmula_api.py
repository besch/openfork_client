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
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import torchaudio

# Import HeartMuLa pipeline
# Assuming code is cloned into /app/HeartMuLa or accessible via path
# Check if we need to adjust sys.path
sys.path.append("/app")
from heartlib.inference import HeartMuLaGenPipeline
from transformers import BitsAndBytesConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="HeartMuLa API", version="1.0.0")

# Job storage
jobs = {}

# Working directory
WORK_DIR = Path("/app")
OUTPUT_DIR = WORK_DIR / "output"
INPUT_DIR = WORK_DIR / "input"
MODEL_DIR = WORK_DIR / "ckpt"

# Global pipeline variable
pipe = None

class GenerateRequest(BaseModel):
    """Request model for music generation."""
    lyrics: str = ""
    style_prompt: str = ""  # Mapped to 'tags'
    seed: int = 42
    max_audio_length_ms: int = 95000 # Default to 95 seconds to match DiffRhythm
    topk: int = 250
    temperature: float = 1.0
    cfg_scale: float = 3.0
    model_version: str = "3B" # "3B" or "7B"


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str  # pending, processing, completed, failed
    output_path: Optional[str] = None
    error: Optional[str] = None


def generate_music_sync(job_id: str, request: GenerateRequest):
    """Synchronous music generation using HeartMuLa pipeline."""
    global pipe
    try:
        jobs[job_id]["status"] = "processing"
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # INPUT_DIR.mkdir(parents=True, exist_ok=True) # Not strictly needed if passing strings
        
        output_path = OUTPUT_DIR / f"{job_id}.wav"
        
        logger.info(f"Generating music for job {job_id}...")
        logger.info(f"Tags: {request.style_prompt}, Lyrics length: {len(request.lyrics)} chars")
        
        if pipe is None:
            raise RuntimeError("HeartMuLa pipeline is not loaded.")

        # Set seed if possible (HeartMuLa might use torch.manual_seed globally or pass generator)
        torch.manual_seed(request.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request.seed)

        # Call the pipeline
        # pipe(inputs, max_audio_length_ms, save_path, topk, temperature, cfg_scale)
        # inputs is a dict with 'lyrics' and 'tags'
        
        # Note: mapping 'style_prompt' to 'tags'
        inputs = {
            "lyrics": request.lyrics,
            "tags": request.style_prompt,
        }
        
        # Generation
        logger.info("Starting inference...")
        wav = pipe(
            inputs,
            max_audio_length_ms=request.max_audio_length_ms,
            save_path=str(output_path),
            topk=request.topk,
            temperature=request.temperature,
            cfg_scale=request.cfg_scale,
        )
        
        if output_path.exists():
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["output_path"] = str(output_path)
            logger.info(f"Job {job_id} completed successfully: {output_path}")
        else:
             # Just in case save_path wasn't respected or something else happened, 
             # check if 'wav' is returned and save it manually if needed.
             # Based on example run_music_generation.py, it takes save_path.
             # If wav is a tensor, we could save it manually.
             if isinstance(wav, torch.Tensor):
                 logger.info("Saving output tensor manually...")
                 torchaudio.save(str(output_path), wav.cpu(), 32000) # Assuming 32kHz, verifying in docs/code usage later if needed
                 jobs[job_id]["status"] = "completed"
                 jobs[job_id]["output_path"] = str(output_path)
             else:
                logger.error(f"Output file was not created at {output_path}")
                raise RuntimeError("Output file was not created")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        traceback.print_exc()
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.on_event("startup")
async def startup_event():
    """Load the model on startup."""
    global pipe
    logger.info("HeartMuLa API starting...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU: {device_name}")
        
    try:
        logger.info("Loading HeartMuLa pipeline...")
        # Define model paths - assuming they are in /app/ckpt/
        # Adjust these paths based on actual Docker download location
        # The example used args.model_path (./ckpt) and args.version (3B)
        # So pipeline construction does:
        # self.v2_folder = os.path.join(model_path, f"HeartMuLa-oss-{version}")
        # self.vae_folder = os.path.join(model_path, "HeartCodec-oss")
        
        model_path = "/app/ckpt" 
        quantization_type = os.environ.get("HEARTMULA_QUANTIZATION", "none")
        
        bnb_config = None
        if quantization_type == "4bit":
            logger.info("Enabling 4-bit quantization (nf4)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        pipe = HeartMuLaGenPipeline.from_pretrained(
            model_path,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=torch.bfloat16,
            version="3B", # Default to 3B for now
            bnb_config=bnb_config
        )
        logger.info("HeartMuLa pipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load HeartMuLa pipeline: {e}")
        # We don't crash the server, but generation will fail
        pass


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if pipe is not None else "model_not_loaded",
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/generate", response_model=JobStatus)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Start music generation job.
    Returns job_id that can be used to poll for status.
    """
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "status": "pending",
        "output_path": None,
        "error": None
    }
    
    # Run generation in background
    background_tasks.add_task(
        generate_music_sync,
        job_id,
        request
    )
    
    return JobStatus(job_id=job_id, status="pending")


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
        error=job.get("error")
    )


@app.get("/download/{job_id}")
async def download(job_id: str):
    """Download the generated audio file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job['status']}")
    
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
    """Delete a job and its output file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs.pop(job_id)
    
    # Delete output file if it exists
    if job.get("output_path"):
        Path(job["output_path"]).unlink(missing_ok=True)
    
    return {"message": "Job deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
