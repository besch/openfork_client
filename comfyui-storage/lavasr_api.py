#!/usr/bin/env python3
"""
LavaSR REST API Server for OpenFork DGN Client
Provides an HTTP API for speech restoration and enhancement.
Accepts audio file upload, returns enhanced audio.
"""

import os
import sys
import uuid
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Try to import LavaSR - must be installed via pip install .
try:
    from LavaSR.model import LavaEnhance2
except ImportError:
    logging.error("LavaSR library not found. Ensure it is installed in the environment.")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Job storage
jobs = {}

# Directories
WORK_DIR = Path("/app")
OUTPUT_DIR = WORK_DIR / "output"
INPUT_DIR = WORK_DIR / "input"
CHECKPOINT_DIR = WORK_DIR / "checkpoints"

# Global model instance
lava_model = None
model_loading = False
model_error = None

def load_model():
    """Load the LavaSR model to GPU."""
    global lava_model, model_loading, model_error
    model_loading = True
    try:
        logger.info("Loading LavaSR model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # The library uses Hugging Face for model management
        lava_model = LavaEnhance2("YatharthS/LavaSR", device=device)
        logger.info(f"LavaSR model loaded on {device}")
    except Exception as e:
        logger.error(f"Failed to load LavaSR model: {e}", exc_info=True)
        model_error = str(e)
    finally:
        model_loading = False

def run_lavasr_inference(job_id: str, input_path: str):
    """Run LavaSR inference using the Python API."""
    try:
        jobs[job_id]["status"] = "processing"
        
        output_path = OUTPUT_DIR / f"{job_id}_restored.wav"
        
        if lava_model is None:
            raise RuntimeError("LavaSR model not loaded")

        logger.info(f"Starting enhancement for job {job_id}")
        start_time = time.time()
        
        # Load and enhance audio
        input_audio, input_sr = lava_model.load_audio(input_path)
        output_audio_tensor = lava_model.enhance(input_audio)
        
        # Convert to numpy and save
        output_audio = output_audio_tensor.cpu().numpy().squeeze()
        sf.write(str(output_path), output_audio, 48000) # Output is 48kHz by default
        
        elapsed = time.time() - start_time
        logger.info(f"Job {job_id} completed in {elapsed:.2f}s: {output_path}")

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_path"] = str(output_path)

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        # Clean up input audio
        try:
            Path(input_path).unlink(missing_ok=True)
        except Exception:
            pass

from contextlib import asynccontextmanager
import threading

@asynccontextmanager
async def lifespan(app: FastAPI):

    """Ensure directories exist and start model loading."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model in a background thread so the API is reachable immediately
    threading.Thread(target=load_model, daemon=True).start()
    
    logger.info("LavaSR API starting (model loading in background)...")
    yield

app = FastAPI(title="LavaSR API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model_error is None else "error",
        "model_loaded": lava_model is not None,
        "model_loading": model_loading,
        "model_error": model_error,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/enhance")
async def enhance_endpoint(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
):
    """
    Start speech restoration.
    Accepts audio file upload.
    """
    if lava_model is None:
        if model_error:
            raise HTTPException(status_code=503, detail=f"Model failed to load: {model_error}")
        raise HTTPException(status_code=503, detail="Model is still loading")
    job_id = str(uuid.uuid4())

    # Save uploaded audio
    input_path = INPUT_DIR / f"{job_id}_{audio.filename}"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    jobs[job_id] = {
        "status": "pending",
        "output_path": None,
        "error": None,
    }

    background_tasks.add_task(
        run_lavasr_inference,
        job_id,
        str(input_path)
    )

    return {"job_id": job_id, "status": "pending"}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get the status of a restoration job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "output_path": job.get("output_path"),
        "error": job.get("error"),
    }


@app.get("/download/{job_id}")
async def download(job_id: str):
    """Download the restored audio file."""
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
        filename=f"{job_id}_restored.wav",
    )


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its output file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs.pop(job_id)

    if job.get("output_path"):
        Path(job["output_path"]).unlink(missing_ok=True)

    return {"message": "Job deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
