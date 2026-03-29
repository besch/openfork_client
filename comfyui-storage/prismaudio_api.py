#!/usr/bin/env python3
"""
Prism Audio (PRiSM) REST API Server for OpenFork DGN Client
Provides an HTTP API for video-to-audio synthesis using PRiSM.
Accepts video file upload + prompt, returns generated audio.
"""

import os
import sys
import uuid
import logging
import shutil
import time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import torch
import torchaudio
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Job storage
jobs = {}

# Directories
WORK_DIR = Path("/app")
OUTPUT_DIR = WORK_DIR / "output"
INPUT_DIR = WORK_DIR / "input"
CKPT_DIR = WORK_DIR / "ckpts"

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Globals for loaded model
_model = None
_tokenizer = None
_processor = None


def wrap_cot(prompt: str) -> str:
    """Wrap simple prompt into PRiSM Chain-of-Thought format if tags are missing."""
    tags = ["<Semantic>", "<Temporal>", "<Aesthetic>", "<Spatial>"]
    if any(tag in prompt for tag in tags):
        return prompt
    
    # Basic wrapping for single-line prompts
    return (
        f"<Semantic>{prompt}</Semantic>"
        f"<Temporal>Natural flow synchronous with video video events.</Temporal>"
        f"<Aesthetic>Professional quality, clear audio, natural ambiance.</Aesthetic>"
        f"<Spatial>Centered spatial placement.</Spatial>"
    )


def load_model():
    """Load PRiSM model at startup."""
    global _model, _tokenizer, _processor

    logger.info("Loading PRiSM (Prism Audio) model...")
    
    # Note: This logic depends on the specific PRiSM implementation.
    # We follow the common pattern found in ThinkSound/PrismAudio.
    try:
        # Assuming the repo structure provides a high-level API or we use the underlying classes
        # In a real environment, we'd import from the cloned repo.
        # Here we prepare the structure for the Docker environment.
        
        # Add app to path if needed (if predict logic is nested)
        sys.path.append(str(WORK_DIR))
        
        # Placeholder for actual model loading logic from PRiSM repo
        # Example:
        # from model.PrismAudio import load_prismaudio
        # _model, _processor = load_prismaudio(CKPT_DIR, device=DEVICE)
        
        logger.info("PRiSM model loaded successfully (Placeholder logic for now)")
        
    except Exception as e:
        logger.error(f"Failed to load PRiSM model: {e}")
        # In Docker, we might want to exit if the model fails to load
        # sys.exit(1)


@torch.inference_mode()
def generate_audio_sync(
    job_id: str,
    video_path: str,
    prompt: str,
    duration: float,
    seed: int,
    cfg_strength: float,
    num_steps: int,
):
    """Run PRiSM generation synchronously."""
    try:
        jobs[job_id]["status"] = "processing"
        
        cot_prompt = wrap_cot(prompt)
        logger.info(f"Generating audio for job {job_id}: prompt='{prompt}', duration={duration}s")
        logger.info(f"CoT Prompt: {cot_prompt}")

        # PRiSM Inference logic implementation
        # 1. Prepare video features
        # 2. Run model inference
        # 3. Save output to job folder
        
        # Simulated delay for now
        time.sleep(2)
        
        # Save output (Simulated result for now - actual PRiSM logic goes here)
        output_path = OUTPUT_DIR / f"{job_id}.wav"
        
        # Dummy wav for structure verification
        # import numpy as np
        # audio_data = np.zeros((1, 48000 * 5))
        # torchaudio.save(str(output_path), torch.from_numpy(audio_data).float(), 48000)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_path"] = str(output_path)
        logger.info(f"Job {job_id} completed: {output_path}")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        # Clean up input video
        try:
            Path(video_path).unlink(missing_ok=True)
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and clean up on shutdown."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Prism Audio API starting...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GB")

    load_model()
    yield


app = FastAPI(title="Prism Audio API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "prismaudio",
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/generate")
async def generate_endpoint(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    prompt: str = Form(""),
    duration: str = Form("8"),
    seed: str = Form("42"),
    cfg_strength: str = Form("4.5"),
    num_steps: str = Form("25"),
):
    """
    Start video-to-audio generation via PRiSM.
    """
    job_id = str(uuid.uuid4())

    # Save uploaded video
    video_path = INPUT_DIR / f"{job_id}_{video.filename}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    jobs[job_id] = {
        "status": "pending",
        "output_path": None,
        "error": None,
    }

    background_tasks.add_task(
        generate_audio_sync,
        job_id,
        str(video_path),
        prompt,
        float(duration),
        int(seed),
        float(cfg_strength),
        int(num_steps),
    )

    return {"job_id": job_id, "status": "pending"}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get the status of a generation job."""
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
        filename=f"{job_id}.wav",
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
