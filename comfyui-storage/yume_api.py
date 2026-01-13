#!/usr/bin/env python3
"""
YUME REST API Server for OpenFork DGN Client

Provides a simple HTTP API for video generation from text or images.
Uses subprocess to call the official YUME inference scripts.
Implements a Request Queue to prevent OOM / internal concurrency clashes.

Reference: https://github.com/stdstu12/YUME
Model: https://huggingface.co/stdstu123/Yume-5B-720P
"""

import os
import sys
import uuid
import subprocess
import logging
import shutil
import asyncio
from pathlib import Path
from typing import Optional, Dict
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="YUME API", version="1.1.0")

# Job storage
jobs = {}

# Queue for processing jobs sequentially
job_queue = asyncio.Queue()
processing_lock = asyncio.Lock()

# Directories
YUME_DIR = Path("/opt/YUME")
MODEL_DIR = Path("/opt/models/Yume")
OUTPUT_DIR = Path("/opt/output")
INPUT_DIR = Path("/opt/input")


class GenerateRequest(BaseModel):
    """Request model for video generation."""
    prompt: str = "A beautiful landscape with rolling hills and a sunset sky"
    negative_prompt: str = "low quality, distorted, bad animation, blurry, watermark"
    num_frames: int = 49
    width: int = 1280
    height: int = 720
    steps: int = 30
    cfg: float = 7.0
    seed: int = 0


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str  # pending, queued, processing, completed, failed
    output_path: Optional[str] = None
    error: Optional[str] = None
    progress: Optional[int] = None
    queue_position: Optional[int] = None


async def worker_loop():
    """Background worker that processes jobs from the queue one at a time."""
    logger.info("Worker loop started")
    while True:
        job_id = await job_queue.get()
        try:
            logger.info(f"Worker picked up job {job_id}")
            async with processing_lock:
                 # Check if job was cancelled while in queue
                if job_id not in jobs:
                    logger.info(f"Job {job_id} was removed/cancelled before processing")
                    continue
                
                # Update status
                jobs[job_id]["status"] = "processing"
                
                # Extract parameters
                job_data = jobs[job_id]
                params = job_data["params"]
                
                # Execute generation (blocking subprocess call wrapped in executor)
                await asyncio.to_thread(
                    generate_video_sync,
                    job_id,
                    params.get("prompt"),
                    params.get("negative_prompt"),
                    params.get("num_frames"),
                    params.get("width"),
                    params.get("height"),
                    params.get("steps"),
                    params.get("cfg"),
                    params.get("seed"),
                    params.get("image_path")
                )
                
        except Exception as e:
            logger.error(f"Worker loop error processing job {job_id}: {e}")
            if job_id in jobs:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = str(e)
        finally:
            job_queue.task_done()


def generate_video_sync(
    job_id: str,
    prompt: str,
    negative_prompt: str,
    num_frames: int,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    image_path: Optional[str] = None
):
    """Synchronous video generation using subprocess to call YUME inference."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create temp directories for this job
        job_output_dir = OUTPUT_DIR / job_id
        job_input_dir = INPUT_DIR / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)
        job_input_dir.mkdir(parents=True, exist_ok=True)
        
        # Write caption to file
        caption_path = job_input_dir / "caption.txt"
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        logger.info(f"Generating video for job {job_id}...")
        logger.info(f"Prompt: {prompt[:100]}..., Frames: {num_frames}, Resolution: {width}x{height}")
        
        # Use random seed if 0
        if seed == 0:
            import random
            seed = random.randint(1, 2**32 - 1)
        
        # Build command to call YUME inference
        cmd = [
            sys.executable, "-m", "fastvideo.sample.sample_5b",
            "--seed", str(seed),
            "--gradient_checkpointing",
            "--train_batch_size", "1",
            "--max_sample_steps", "600000",
            "--mixed_precision", "bf16",
            "--allow_tf32",
            "--video_output_dir", str(job_output_dir),
            "--num_euler_timesteps", str(steps),
            "--rand_num_img", "0.6",
            "--caption_path", str(caption_path),
        ]
        
        # Image-to-Video or Text-to-Video mode
        if image_path and Path(image_path).exists():
            logger.info(f"Mode: Image-to-Video with {image_path}")
            jpg_dir = job_input_dir / "jpg"
            jpg_dir.mkdir(exist_ok=True)
            
            # Copy and possibly resize image
            from PIL import Image
            img = Image.open(image_path)
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            img.save(jpg_dir / "input_0.jpg", quality=95)
            
            cmd.extend(["--jpg_dir", str(jpg_dir)])
        else:
            logger.info("Mode: Text-to-Video")
            cmd.extend(["--T2V"])
            cmd.extend(["--prompt", prompt])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Set environment for single-GPU inference
        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["CUDA_VISIBLE_DEVICES"] = "0"
        # Suppress noisy transformers/hub warnings that hide real errors
        env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        env["TRANSFORMERS_VERBOSITY"] = "error"
        env["HF_HOME"] = "/tmp/huggingface_cache"
        # Required for YUME's distributed training code (single-GPU mode)
        env["LOCAL_RANK"] = "0"
        env["RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = "12355"
        
        # Run the inference
        result = subprocess.run(
            cmd,
            cwd=str(YUME_DIR),
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
            env=env
        )
        
        # Log output
        if result.stdout:
            # Only log last few lines to avoid bloating logs
            stdout_tail = result.stdout[-2000:]
            logger.info(f"Inference stdout tail: {stdout_tail}")
            
        if result.stderr:
            # Log more stderr to help debugging
            stderr_tail = result.stderr[-4000:]
            logger.warning(f"Inference stderr tail: {stderr_tail}")
        
        if result.returncode != 0:
            logger.error(f"Inference failed with code {result.returncode}")
            # Try to find the actual error in stderr instead of just taking the very end
            err_msg = "Unknown error"
            if result.stderr:
                # Look for common error markers
                lines = result.stderr.splitlines()
                # Find the last line that looks like a real error
                important_lines = [l for l in lines if any(x in l for x in ["Error", "Exception", "Traceback", "Out of memory", "RuntimeError"])]
                if important_lines:
                    err_msg = important_lines[-1]
                else:
                    # Fallback to the last 1000 chars if no markers found
                    err_msg = result.stderr[-1000:].strip()
            
            raise RuntimeError(f"Inference failed: {err_msg}")
        
        # Find the output video
        output_files = list(job_output_dir.glob("**/*.mp4"))
        if not output_files:
            output_files = list(job_output_dir.glob("**/*.avi"))
        
        if output_files:
            # Get the most recent one
            output_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            actual_output = output_files[0]
            
            # Move to standardized location
            final_output = OUTPUT_DIR / f"{job_id}.mp4"
            shutil.copy2(actual_output, final_output)
            
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["output_path"] = str(final_output)
            logger.info(f"Job {job_id} completed successfully: {final_output}")
        else:
            # List what's in output dir for debugging
            all_files = list(job_output_dir.rglob("*"))
            logger.error(f"No output video found. Files in {job_output_dir}: {all_files}")
            raise RuntimeError(f"Output video was not created. Found files: {[f.name for f in all_files]}")
            
    except subprocess.TimeoutExpired:
        logger.error(f"Job {job_id} timed out")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = "Generation timed out after 30 minutes"
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        # Final cleanup attempt
        try:
             # Cleanup temp input/output dirs
            if 'job_input_dir' in locals():
                shutil.rmtree(job_input_dir, ignore_errors=True)
            if 'job_output_dir' in locals():
                shutil.rmtree(job_output_dir, ignore_errors=True)
        except Exception:
            pass


@app.on_event("startup")
async def startup_event():
    """Log startup info and verify YUME is available."""
    logger.info("YUME API starting...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Check YUME installation
    if YUME_DIR.exists():
        logger.info(f"YUME directory found: {YUME_DIR}")
    else:
        logger.warning(f"YUME directory not found: {YUME_DIR}")
    
    if MODEL_DIR.exists():
        logger.info(f"Model directory found: {MODEL_DIR}")
    else:
        logger.warning(f"Model directory not found: {MODEL_DIR}")

    # Start the background worker loop
    asyncio.create_task(worker_loop())


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "yume_available": YUME_DIR.exists(),
        "model_available": MODEL_DIR.exists(),
        "queue_size": job_queue.qsize(),
        "processing_active": processing_lock.locked()
    }


@app.post("/generate", response_model=JobStatus)
async def generate_text_to_video(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Queue text-to-video generation job.
    Returns job_id immediately. Job tracks in "pending"/"queued" state until picked up.
    """
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "status": "queued",
        "output_path": None,
        "error": None,
        "progress": 0,
        "params": {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "num_frames": request.num_frames,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "cfg": request.cfg,
            "seed": request.seed,
            "image_path": None
        }
    }
    
    # Add to asyncio queue
    await job_queue.put(job_id)
    
    return JobStatus(job_id=job_id, status="queued", queue_position=job_queue.qsize())


@app.post("/generate-i2v", response_model=JobStatus)
async def generate_image_to_video(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    prompt: str = Form("The camera slowly pans across the scene"),
    negative_prompt: str = Form("low quality, distorted, bad animation, blurry, watermark"),
    num_frames: int = Form(49),
    steps: int = Form(30),
    cfg: float = Form(7.0),
    seed: int = Form(0),
):
    """
    Queue image-to-video generation job.
    Returns job_id immediately. Job tracks in "pending"/"queued" state until picked up.
    """
    job_id = str(uuid.uuid4())
    
    # Save uploaded image immediately so we have it for the worker
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_path = INPUT_DIR / f"{job_id}_input.jpg"
    
    with open(image_path, "wb") as f:
        content = await image.read()
        f.write(content)
    
    # Get image dimensions to clamp/validate
    from PIL import Image
    try:
        img = Image.open(image_path)
        width, height = img.size
        # Clamp to valid YUME dimensions
        width = min(max(width, 512), 1920)
        height = min(max(height, 288), 1080)
        # Make divisible by 64
        width = (width // 64) * 64
        height = (height // 64) * 64
    except Exception as e:
        logger.error(f"Error processing image {job_id}: {e}")
        # Use defaults if image fails
        width, height = 1280, 720
    
    jobs[job_id] = {
        "status": "queued",
        "output_path": None,
        "error": None,
        "progress": 0,
        "params": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_frames": num_frames,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "image_path": str(image_path)
        }
    }
    
    # Add to asyncio queue
    await job_queue.put(job_id)
    
    return JobStatus(job_id=job_id, status="queued", queue_position=job_queue.qsize())


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Get the status of a generation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Calculate rough queue position if queued
    q_pos = None
    if job["status"] == "queued":
        # Note: This is an approximation as we can't easily peek into asyncio.Queue
        # reliable enough for exact position, but strictly speaking it's in the queue
        pass 

    return JobStatus(
        job_id=job_id,
        status=job["status"],
        output_path=job.get("output_path"),
        error=job.get("error"),
        progress=job.get("progress"),
        queue_position=None # Not easily trackable in basic queue
    )


@app.get("/download/{job_id}")
async def download(job_id: str):
    """Download the generated video file."""
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
        media_type="video/mp4",
        filename=f"{job_id}.mp4"
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
    
    # Delete any input file
    input_path = INPUT_DIR / f"{job_id}_input.jpg"
    input_path.unlink(missing_ok=True)
    
    return {"message": "Job deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
