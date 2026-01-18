#!/usr/bin/env python3
"""
YUME REST API Server for OpenFork DGN Client (Direct Model Loading)

Provides a simple HTTP API for video generation from text or images.
Uses wan23.Yume directly for memory-efficient single-GPU inference,
instead of subprocess with FSDP which causes OOM issues.

Reference: https://github.com/stdstu12/YUME
Model: https://huggingface.co/stdstu123/Yume-5B-720P
"""

import os
import sys
import uuid
import logging
import shutil
import asyncio
import random
from pathlib import Path
from typing import Optional, Dict
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
import fastapi
import json
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="YUME API", version="2.0.0")

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
DEVICE_ID = 0

# Global model storage (loaded once at startup)
class Models:
    device = None
    wan_model = None
    vae = None
    transformer = None
    text_encoder = None
    loaded = False

MODELS = Models()


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


def load_yume_model():
    """Load the YUME model using wan23.Yume for single-GPU inference."""
    global MODELS
    
    if MODELS.loaded:
        logger.info("YUME model already loaded")
        return True
        
    try:
        logger.info(f"Loading YUME model from {MODEL_DIR}...")
        
        # Add YUME to path
        if str(YUME_DIR) not in sys.path:
            sys.path.insert(0, str(YUME_DIR))
        
        # Enable TF32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Import YUME modules
        import importlib
        from wan23.configs import WAN_CONFIGS
        
        _wan23 = importlib.import_module("wan23")
        
        # Set device
        MODELS.device = torch.device(f"cuda:{DEVICE_ID}")
        
        # Load model using ti2v-5B config (text/image to video 5B)
        cfg = WAN_CONFIGS["ti2v-5B"]
        logger.info(f"Loading Yume with config: ti2v-5B")
        
        # Create symlink if needed
        ckpt_dir = YUME_DIR / "Yume-5B-720P"
        if not ckpt_dir.exists():
            ckpt_dir.symlink_to(MODEL_DIR)
        
        MODELS.wan_model = _wan23.Yume(
            config=cfg, 
            checkpoint_dir=str(MODEL_DIR),
            device_id=DEVICE_ID
        )
        
        # Store references with correct attribute names
        MODELS.vae = MODELS.wan_model.vae
        MODELS.transformer = MODELS.wan_model.model  # .model not .dit
        MODELS.text_encoder = MODELS.wan_model.text_encoder
        
        MODELS.loaded = True
        logger.info("YUME model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load YUME model: {e}", exc_info=True)
        return False


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
                
                # Execute generation
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
    """Synchronous video generation using wan23.Yume with internal loop."""
    # Local imports and helpers to keep this function self-contained
    import numpy as np
    from PIL import Image
    from diffusers.utils import export_to_video

    def get_sampling_sigmas(steps: int, shift: float):
        sigma = np.linspace(1, 0, steps + 1)[:steps]
        return (shift * sigma / (1 + (shift - 1) * sigma))

    def _to_bf16(x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.bfloat16)
        if isinstance(x, (list, tuple)):
            return type(x)(_to_bf16(t, device) for t in x)
        return x

    try:
        if not MODELS.loaded:
            raise RuntimeError("YUME model not loaded")
            
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating video for job {job_id}...")
        logger.info(f"Prompt: {prompt[:100]}..., Frames: {num_frames}, Resolution: {width}x{height}, Steps: {steps}")
        
        # Use random seed if 0
        if seed == 0:
            seed = random.randint(1, 2**32 - 1)
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        wan = MODELS.wan_model
        device = MODELS.device
        
        # Prepare input image for I2V mode (Placeholder for future I2V support)
        if image_path:
             logger.warning("I2V inputs received but currently using T2V loop logic. I2V specific adaptation pending.")
        
        # Calculate max_area based on resolution
        max_area = width * height
        
        logger.info(f"Starting generation prep with seed {seed}...")
        
        # 1. Prepare Inputs (Noise, Context) using wan.generate (which acts as prep function)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            gen_ret = wan.generate(
                input_prompt=prompt,
                img=None, # Force T2V prep
                size=(width, height),
                n_prompt=negative_prompt if negative_prompt else "",
                frame_num=num_frames,
                max_area=max_area,
                sampling_steps=steps, # Not used for prep but passed
                guide_scale=cfg, # Not used for prep
                shift=8.0, 
                seed=seed
            )
            
        # Handle return values (T2V)
        # gen_ret is (arg_c, arg_null, noise)
        arg_c, arg_null, noise = gen_ret
        
        # Move to bf16
        noise = _to_bf16(noise, device)
        # arg_c context is already on device from generate()
        
        # 2. Diffusion Loop
        logger.info(f"Starting diffusion sampling loop for {steps} steps...")
        
        # Calculate latent frame zero
        latent_frame_zero = (num_frames - 1) // 4 + 1
        
        shift = 8.0
        sampling_sigmas = get_sampling_sigmas(steps, shift)
        latent = noise.clone()
        transformer = MODELS.transformer
        
        for i in range(steps):
             # Update progress periodically
             if i % 5 == 0:
                 jobs[job_id]["progress"] = int((i / steps) * 100)
             
             # Calculate timestep vector
             ts_scalar = [sampling_sigmas[i]*1000]
             timestep = torch.tensor(ts_scalar).to(device)
             tvec = timestep
             
             latent_model_input = [_to_bf16(latent, device)]
             
             with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                  # Forward pass
                  # webapp uses flag=False for simple T2V
                  noise_pred = transformer(latent_model_input, t=tvec, latent_frame_zero=latent_frame_zero, **arg_c, flag=False)[0]
             
             # Scheduler step (Euler/Flow)
             tail = latent
             pred_tail = noise_pred
             if i+1 == steps:
                 new_tail = tail + (0.0 - sampling_sigmas[i]) * pred_tail
             else:
                 new_tail = tail + (sampling_sigmas[i+1] - sampling_sigmas[i]) * pred_tail
             latent = new_tail
             
        # 3. Decode
        logger.info("Decoding latent video...")
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
             video_frames = MODELS.vae.decode([latent])[0]
             
        # 4. Save
        final_output = OUTPUT_DIR / f"{job_id}.mp4"
        logger.info(f"Saving video to {final_output}...")
        
        # Post-process: [-1, 1] -> [0, 255]
        # video_frames is (C, F, H, W)
        v = (video_frames.clamp(-1,1).add(1).div(2))
        v = (v * 255).byte().cpu().numpy()
        v = v.transpose(1, 2, 3, 0) # (F, H, W, C)
        
        frames = [Image.fromarray(f) for f in v]
        export_to_video(frames, str(final_output), fps=16) # webapp default
        
        if final_output.exists():
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["output_path"] = str(final_output)
            jobs[job_id]["progress"] = 100
            logger.info(f"Job {job_id} completed successfully: {final_output}")
        else:
            raise RuntimeError(f"Output video was not created at {final_output}")
            
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        # Cleanup input files
        if image_path:
             try:
                 Path(image_path).unlink(missing_ok=True)
             except Exception:
                 pass
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.on_event("startup")
async def startup_event():
    """Log startup info and load YUME model."""
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

    # Load the model in background to not block startup
    logger.info("Scheduling background model load...")
    asyncio.create_task(background_load_model())

    # Start the background worker loop
    asyncio.create_task(worker_loop())

async def background_load_model():
    """Run blocking model load in thread pool."""
    logger.info("Starting background model load...")
    success = await asyncio.to_thread(load_yume_model)
    if not success:
        logger.error("Failed to load YUME model - API will return errors for generation requests")
    else:
        logger.info("Model load completed successfully")


@app.get("/health")
async def health_check():
    """Health check endpoint. Returns 503 if model is not loaded (to keep start_cloud.sh waiting)."""
    is_loaded = MODELS.loaded
    
    if not is_loaded:
        # Check if loading is still in progress (by checking if task is running)
        # For now, just return 503 so the startup script waits
        logger.info("Health check: Model not loaded yet")
        return fastapi.Response(
            content=json.dumps({
                "status": "loading",
                "model_loaded": False
            }),
            status_code=503,
            media_type="application/json"
        )
        
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "yume_available": YUME_DIR.exists(),
        "model_available": MODEL_DIR.exists(),
        "model_loaded": True,
        "queue_size": job_queue.qsize(),
        "processing_active": processing_lock.locked()
    }


@app.post("/generate", response_model=JobStatus)
async def generate_text_to_video(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Queue text-to-video generation job.
    Returns job_id immediately. Job tracks in "pending"/"queued" state until picked up.
    """
    if not MODELS.loaded:
        raise HTTPException(status_code=503, detail="YUME model not loaded")
    
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
    if not MODELS.loaded:
        raise HTTPException(status_code=503, detail="YUME model not loaded")
    
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
    
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        output_path=job.get("output_path"),
        error=job.get("error"),
        progress=job.get("progress"),
        queue_position=None
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
