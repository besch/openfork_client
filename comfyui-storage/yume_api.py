#!/usr/bin/env python3
"""
YUME REST API Server for OpenFork DGN Client (Direct Model Loading)

Provides a simple HTTP API for video generation from text or images.
Uses wan.Yume directly for memory-efficient single-GPU inference.

Reference: https://github.com/stdstu12/YUME
Model: https://huggingface.co/stdstu123/Yume-5B-720P
"""

import os
import sys
import uuid
import logging
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
    num_frames: int = 13
    width: int = 848
    height: int = 480
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
        # Check if key exists, if not fallback to hardcoded equivalent
        if "ti2v-5B" in WAN_CONFIGS:
             cfg = WAN_CONFIGS["ti2v-5B"]
             logger.info(f"Loading Yume with config: ti2v-5B")
        else:
             logger.warning("ti2v-5B config not found in WAN_CONFIGS, checking available keys...")
             logger.info(f"Keys: {WAN_CONFIGS.keys()}")
             raise ValueError("ti2v-5B config missing")
        
        # Create symlink if needed
        ckpt_dir = YUME_DIR / "Yume-5B-720P"
        if not ckpt_dir.exists() and MODEL_DIR.exists():
            try:
                ckpt_dir.symlink_to(MODEL_DIR)
                logger.info(f"Created symlink: {ckpt_dir} -> {MODEL_DIR}")
            except Exception as e:
                logger.warning(f"Could not create symlink: {e}")
        
        logger.info(f"Instantiating _wan23.Yume with config={cfg}, checkpoint_dir={MODEL_DIR}, device_id={DEVICE_ID}")
        try:
            MODELS.wan_model = _wan23.Yume(
                config=cfg, 
                checkpoint_dir=str(MODEL_DIR),
                device_id=DEVICE_ID
            )
            logger.info("_wan23.Yume instantiated successfully.")
        except Exception as e:
            logger.error(f"Error instantiating _wan23.Yume: {e}", exc_info=True)
            raise

        logger.info("Accessing VAE...")
        # Store references with correct attribute names for wan23
        MODELS.vae = MODELS.wan_model.vae
        logger.info("Accessing Transformer...")
        MODELS.transformer = MODELS.wan_model.model  # .model not .dit in wan23
        logger.info("Accessing Text Encoder...")
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
    """
    Synchronous video generation using wan.Yume.
    
    This uses the high-level generate() API from YUME which handles
    the entire pipeline internally.
    """
    try:
        if not MODELS.loaded:
            raise RuntimeError("YUME model not loaded")
            
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating video for job {job_id}...")
        logger.info(f"Prompt: {prompt[:100]}...")
        logger.info(f"Frames: {num_frames}, Resolution: {width}x{height}, Steps: {steps}")
        
        # Use random seed if 0
        if seed == 0:
            seed = random.randint(1, 2**32 - 1)
            logger.info(f"Using random seed: {seed}")
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        wan = MODELS.wan_model
        device = MODELS.device
        
        # Prepare input image for I2V mode
        img_input = None
        if image_path and os.path.exists(image_path):
            from PIL import Image
            img_input = Image.open(image_path).convert("RGB")
            logger.info(f"Loaded input image: {image_path}")
        
        # Update progress
        jobs[job_id]["progress"] = 10
        
        # Calculate max_area based on resolution
        max_area = width * height
        
        logger.info(f"Starting YUME generation with seed {seed}...")
        
        # Use the high-level generate() method to get initialization tensors
        # This returns a tuple of (arg_c, arg_null, noise, mask2, img_lat)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                gen_ret = wan.generate(
                    input_prompt=prompt,
                    img=img_input,  # None for T2V, PIL Image for I2V
                    size=(width, height),
                    n_prompt=negative_prompt if negative_prompt else "",
                    frame_num=num_frames,
                    max_area=max_area,
                    sampling_steps=steps,
                    guide_scale=cfg,
                    shift=8.0,  # Flow matching shift parameter
                    seed=seed
                )

        # Unpack return values
        # The generate method returns 7 values:
        # latent_model_input, timestep, arg_c, noise, model_input, clip_context, arg_null
        
        # Robust check: If wan.generate returns video frames (list or tensor), return them directly
        if not isinstance(gen_ret, tuple):
             logger.info("wan.generate returned direct video output, skipping manual loop.")
             video_output = gen_ret
        else:
            # Manual diffusion loop required
            # yume_api_check says: arg_c, arg_null, noise = gen_ret 
            # But here we stick to tuple unpacking or check length
            # Let's trust it returns a tuple that we can unpack or use
            if len(gen_ret) == 3:
                 arg_c, arg_null, noise = gen_ret
                 # Fix for missing variables needed below
                 latent_model_input_init = None 
                 timestep_init = None
                 model_input_init = None
                 clip_context = None
            else:
                 latent_model_input_init, timestep_init, arg_c, noise, model_input_init, clip_context, arg_null = gen_ret
            
            is_i2v = img_input is not None
            
            # Setup scheduler
            from wan23.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
            
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000, 
                shift=1,
                use_dynamic_shifting=False
            )
            sample_scheduler.set_timesteps(steps, device=device, shift=8.0)
            timesteps = sample_scheduler.timesteps
    
            # Main diffusion loop
            logger.info(f"Starting diffusion loop for {steps} steps...")
            
            # Prepare latent model input
            latents = noise
            
            current_step = 0
            
            with torch.no_grad():
                for _, t in enumerate(timesteps):
                    latent_model_input = latents
                    timestep = [t]
                    timestep = torch.stack(timestep).to(device)
                    
                    # Model forward pass
                    noise_pred_cond = wan.model(
                        latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = wan.model(
                        latent_model_input, t=timestep, **arg_null)[0]
                    
                    noise_pred = noise_pred_uncond + cfg * (
                        noise_pred_cond - noise_pred_uncond)
                    
                    # Scheduler step
                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False
                    )[0]
                    latents = [temp_x0.squeeze(0)]
                    
                    # Update progress
                    current_step += 1
                    progress = 10 + int((current_step / steps) * 80)
                    jobs[job_id]["progress"] = progress
    
            # VAE Decoding
            logger.info("Decoding latents with VAE...")
            x0 = latents
            
            with torch.no_grad():
                videos = wan.vae.decode(x0)
                
            video_output = videos[0] # (C, F, H, W)
        
        jobs[job_id]["progress"] = 90
        
        
        # Save the output
        final_output = OUTPUT_DIR / f"{job_id}.mp4"
        logger.info(f"Saving video to {final_output}...")
        
        # Convert to frames
        # Normalize to [0, 255]
        if isinstance(video_output, torch.Tensor):
            video_output = video_output.cpu()
            
            if video_output.dim() == 4:
                # (C, F, H, W) -> (F, H, W, C)
                video_output = video_output.permute(1, 2, 3, 0)
            
            video_output = (video_output.clamp(-1, 1).add(1).div(2) * 255).byte().numpy()
            
            from PIL import Image
            frames = [Image.fromarray(frame) for frame in video_output]
        elif isinstance(video_output, list):
             # Assuming list of PIL Images
             frames = video_output
        else:
             # Try to convert numpy
             import numpy as np
             if isinstance(video_output, np.ndarray):
                 from PIL import Image
                 frames = [Image.fromarray(frame) for frame in video_output]
             else:
                 raise ValueError(f"Unknown video output type: {type(video_output)}")
        
        # Export
        from diffusers.utils import export_to_video
        export_to_video(frames, str(final_output), fps=16)
        
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
    """Health check endpoint. Returns 503 if model is not loaded."""
    is_loaded = MODELS.loaded
    
    if not is_loaded:
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