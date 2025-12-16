#!/usr/bin/env python3
"""
DiffRhythm REST API Server for OpenFork DGN Client
Provides a simple HTTP API for music generation from lyrics and style prompts.
Uses subprocess to call the official DiffRhythm inference scripts.
"""

import os
import sys
import uuid
import subprocess
import logging
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="DiffRhythm API", version="1.0.0")

# Job storage
jobs = {}

# Working directory
WORK_DIR = Path("/app")
OUTPUT_DIR = WORK_DIR / "output"
INPUT_DIR = WORK_DIR / "input"


class GenerateRequest(BaseModel):
    """Request model for music generation."""
    lyrics: str = ""
    style_prompt: str = "Pop"
    seed: int = 0
    chunked: bool = True  # Enable chunked decoding for 8GB VRAM


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str  # pending, processing, completed, failed
    output_path: Optional[str] = None
    error: Optional[str] = None


def generate_music_sync(job_id: str, lyrics: str, style_prompt: str, seed: int, chunked: bool):
    """Synchronous music generation using subprocess to call DiffRhythm inference."""
    try:
        jobs[job_id]["status"] = "processing"
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create temp file for lyrics in LRC format
        lyrics_path = INPUT_DIR / f"{job_id}_lyrics.lrc"
        output_path = OUTPUT_DIR / f"{job_id}.wav"
        
        # Write lyrics to file (empty or with content)
        with open(lyrics_path, 'w', encoding='utf-8') as f:
            if lyrics.strip():
                f.write(lyrics)
            else:
                # Empty file for instrumental
                f.write("")
        
        logger.info(f"Generating music for job {job_id}...")
        logger.info(f"Style: {style_prompt}, Lyrics length: {len(lyrics)} chars")
        
        # Build command to call DiffRhythm inference
        cmd = [
            sys.executable, "-u", "infer/infer.py",
            "--lrc-path", str(lyrics_path),
            "--ref-prompt", style_prompt,
            "--output-dir", str(OUTPUT_DIR),
            "--audio-length", "95",  # Standard length
        ]
        
        if chunked:
            cmd.append("--chunked")
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the inference
        result = subprocess.run(
            cmd,
            cwd=str(WORK_DIR),
            capture_output=True,
            text=True,
            timeout=900,  # 15 minute timeout
            env={**os.environ, "PYTHONPATH": str(WORK_DIR)}
        )
        
        logger.info(f"Inference stdout: {result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout}")
        
        if result.returncode != 0:
            logger.error(f"Inference failed with code {result.returncode}: {result.stderr}")
            raise RuntimeError(f"Inference failed: {result.stderr[-500:] if result.stderr else 'Unknown error'}")
        
        # Find the output file (DiffRhythm may use different naming)
        output_files = list(OUTPUT_DIR.glob("*.wav"))
        
        # Try to find the most recent wav file
        if output_files:
            # Sort by modification time, get newest
            output_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            actual_output = output_files[0]
            
            # Rename to our expected path if needed
            if actual_output != output_path:
                actual_output.rename(output_path)
        
        # Clean up lyrics file
        lyrics_path.unlink(missing_ok=True)
        
        if output_path.exists():
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["output_path"] = str(output_path)
            logger.info(f"Job {job_id} completed successfully: {output_path}")
        else:
            # List what's in output dir for debugging
            logger.error(f"Output files in {OUTPUT_DIR}: {list(OUTPUT_DIR.glob('*'))}")
            raise RuntimeError("Output file was not created")
            
    except subprocess.TimeoutExpired:
        logger.error(f"Job {job_id} timed out")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = "Generation timed out after 15 minutes"
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.on_event("startup")
async def startup_event():
    """Log startup info."""
    logger.info("DiffRhythm API starting...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/generate", response_model=JobStatus)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Start music generation job.
    Returns job_id that can be used to poll for status.
    """
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
        request.lyrics,
        request.style_prompt,
        request.seed,
        request.chunked
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
