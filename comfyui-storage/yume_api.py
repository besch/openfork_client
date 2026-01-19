#!/usr/bin/env python3
"""
YUME REST API Server - Version 2.0.4
FIXED: Proper manual diffusion loop implementation

The wan.generate() method returns a tuple that requires manual diffusion loop processing.
This version implements the loop correctly based on YUME's expected workflow.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="YUME API", version="2.0.4")

jobs = {}
job_queue = asyncio.Queue()
processing_lock = asyncio.Lock()

YUME_DIR = Path("/opt/YUME")
MODEL_DIR = Path("/opt/models/Yume")
OUTPUT_DIR = Path("/opt/output")
INPUT_DIR = Path("/opt/input")
DEVICE_ID = 0

class Models:
    device = None
    wan_model = None
    vae = None
    transformer = None
    text_encoder = None
    loaded = False

MODELS = Models()


class GenerateRequest(BaseModel):
    prompt: str = "A beautiful landscape with rolling hills and a sunset sky"
    negative_prompt: str = "low quality, distorted, bad animation, blurry, watermark"
    num_frames: int = 13
    width: int = 848
    height: int = 480
    steps: int = 30
    cfg: float = 7.0
    seed: int = 0


class JobStatus(BaseModel):
    job_id: str
    status: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    progress: Optional[int] = None
    queue_position: Optional[int] = None


def load_yume_model():
    global MODELS
    
    if MODELS.loaded:
        logger.info("YUME model already loaded")
        return True
        
    try:
        logger.info(f"Loading YUME model from {MODEL_DIR}...")
        
        if str(YUME_DIR) not in sys.path:
            sys.path.insert(0, str(YUME_DIR))
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        import importlib
        from wan23.configs import WAN_CONFIGS
        
        _wan23 = importlib.import_module("wan23")
        
        MODELS.device = torch.device(f"cuda:{DEVICE_ID}")
        
        if "ti2v-5B" in WAN_CONFIGS:
             cfg = WAN_CONFIGS["ti2v-5B"]
             logger.info(f"Loading Yume with config: ti2v-5B")
        else:
             logger.warning("ti2v-5B config not found")
             logger.info(f"Available configs: {list(WAN_CONFIGS.keys())}")
             raise ValueError("ti2v-5B config missing")
        
        ckpt_dir = YUME_DIR / "Yume-5B-720P"
        if not ckpt_dir.exists() and MODEL_DIR.exists():
            try:
                ckpt_dir.symlink_to(MODEL_DIR)
                logger.info(f"Created symlink: {ckpt_dir} -> {MODEL_DIR}")
            except Exception as e:
                logger.warning(f"Could not create symlink: {e}")
        
        MODELS.wan_model = _wan23.Yume(
            config=cfg, 
            checkpoint_dir=str(MODEL_DIR),
            device_id=DEVICE_ID
        )
        
        MODELS.vae = MODELS.wan_model.vae
        MODELS.transformer = MODELS.wan_model.model
        MODELS.text_encoder = MODELS.wan_model.text_encoder
        
        MODELS.loaded = True
        logger.info("YUME model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load YUME model: {e}", exc_info=True)
        return False


async def worker_loop():
    logger.info("Worker loop started")
    while True:
        job_id = await job_queue.get()
        try:
            logger.info(f"Worker picked up job {job_id}")
            async with processing_lock:
                if job_id not in jobs:
                    logger.info(f"Job {job_id} was removed before processing")
                    continue
                
                jobs[job_id]["status"] = "processing"
                
                job_data = jobs[job_id]
                params = job_data["params"]
                
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
            logger.error(f"Worker error processing job {job_id}: {e}", exc_info=True)
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
    Video generation with proper manual diffusion loop.
    """
    try:
        if not MODELS.loaded:
            raise RuntimeError("YUME model not loaded")
            
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"=== JOB {job_id} START ===")
        logger.info(f"Prompt: {prompt[:100]}...")
        logger.info(f"Params: {num_frames}f {width}x{height} {steps}steps cfg={cfg}")
        
        if seed == 0:
            seed = random.randint(1, 2**32 - 1)
            logger.info(f"Using random seed: {seed}")
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        wan = MODELS.wan_model
        device = MODELS.device
        
        # Load input image for I2V
        img_input = None
        if image_path and os.path.exists(image_path):
            from PIL import Image
            img_input = Image.open(image_path).convert("RGB")
            logger.info(f"Loaded input image: {image_path}")
        
        jobs[job_id]["progress"] = 10
        
        max_area = width * height
        
        logger.info("Calling wan.generate()...")
        
        # Call generate - it returns intermediate values for manual loop
        with torch.no_grad():
            gen_ret = wan.generate(
                input_prompt=prompt,
                img=img_input,
                size=(width, height),
                n_prompt=negative_prompt if negative_prompt else "",
                frame_num=num_frames,
                max_area=max_area,
                sampling_steps=steps,
                guide_scale=cfg,
                shift=8.0,
                seed=seed,
                offload_model=True  # Return tuple for manual processing
            )
        
        logger.info(f"wan.generate() returned tuple of length {len(gen_ret)}")
        
        # Unpack the tuple
        # Based on YUME source, this should be: (arg_c, arg_null, noise)
        # or: (latent_model_input, timestep, arg_c, noise, model_input, clip_context, arg_null)
        
        if len(gen_ret) == 3:
            arg_c, arg_null, noise = gen_ret
            logger.info("Got 3-element tuple: (arg_c, arg_null, noise)")
        elif len(gen_ret) == 7:
            latent_model_input_init, timestep_init, arg_c, noise, model_input_init, clip_context, arg_null = gen_ret
            logger.info("Got 7-element tuple: full diffusion setup")
        else:
            raise ValueError(f"Unexpected tuple length: {len(gen_ret)}")
        
        # Log initial noise shape
        if isinstance(noise, torch.Tensor):
            logger.info(f"Initial noise shape: {noise.shape}")
        elif isinstance(noise, list):
            logger.info(f"Initial noise is list of {len(noise)} tensors")
            if len(noise) > 0 and isinstance(noise[0], torch.Tensor):
                logger.info(f"  noise[0] shape: {noise[0].shape}")
        
        jobs[job_id]["progress"] = 15
        
        # Setup scheduler
        from wan23.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, 
            shift=1,
            use_dynamic_shifting=False
        )
        scheduler.set_timesteps(steps, device=device, shift=8.0)
        timesteps = scheduler.timesteps
        
        logger.info(f"Scheduler initialized with {len(timesteps)} timesteps")
        
        # Prepare initial latents
        if isinstance(noise, torch.Tensor):
            latent = noise.to(device)
        elif isinstance(noise, list):
            latent = noise[0].to(device) if isinstance(noise[0], torch.Tensor) else noise[0]
        else:
            raise ValueError(f"Unexpected noise type: {type(noise)}")
        
        logger.info(f"Initial latent shape: {latent.shape}, device: {latent.device}")
        
        # Manual diffusion loop
        logger.info("Starting manual diffusion loop...")
        
        # Move model to GPU
        wan.model.to(device)
        
        try:
            with torch.no_grad():
                for step_idx, t in enumerate(timesteps):
                    # Prepare inputs
                    # The model expects latent as a list
                    latent_input = [latent]
                    
                    # Timestep should be a 1D tensor
                    if isinstance(t, torch.Tensor):
                        if t.dim() == 0:
                            timestep = t.unsqueeze(0).to(device)
                        else:
                            timestep = t.to(device)
                    else:
                        timestep = torch.tensor([t], dtype=torch.float32, device=device)
                    
                    # Model forward passes
                    noise_pred_cond = wan.model(latent_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = wan.model(latent_input, t=timestep, **arg_null)[0]
                    
                    # CFG
                    noise_pred = noise_pred_uncond + cfg * (noise_pred_cond - noise_pred_uncond)
                    
                    # Scheduler step
                    # The scheduler expects [B, C, F, H, W] tensors
                    # But our tensors might be [C, F, H, W]
                    
                    sample_for_scheduler = latent
                    noise_for_scheduler = noise_pred
                    
                    # Add batch dimension if needed
                    if sample_for_scheduler.dim() == 4:
                        sample_for_scheduler = sample_for_scheduler.unsqueeze(0)
                    if noise_for_scheduler.dim() == 4:
                        noise_for_scheduler = noise_for_scheduler.unsqueeze(0)
                    
                    # Call scheduler
                    latent_next = scheduler.step(
                        model_output=noise_for_scheduler,
                        timestep=t,
                        sample=sample_for_scheduler,
                        return_dict=False
                    )[0]
                    
                    # Remove batch dimension if we added it
                    if latent.dim() == 4 and latent_next.dim() == 5:
                        latent = latent_next.squeeze(0)
                    else:
                        latent = latent_next
                    
                    if (step_idx + 1) % 5 == 0 or step_idx == 0:
                        logger.info(f"Step {step_idx + 1}/{len(timesteps)}: latent shape={latent.shape}")
                    
                    # Update progress
                    progress = 15 + int((step_idx + 1) / len(timesteps) * 70)
                    jobs[job_id]["progress"] = progress
                    
        finally:
            # Move model back to CPU to free VRAM
            wan.model.cpu()
            torch.cuda.empty_cache()
        
        logger.info(f"Diffusion complete. Final latent shape: {latent.shape}")
        jobs[job_id]["progress"] = 85
        
        # VAE Decode
        logger.info("Decoding with VAE...")
        
        # Move VAE to GPU
        if hasattr(wan.vae, 'model'):
            wan.vae.model.to(device)
        else:
            wan.vae.to(device)
        
        try:
            with torch.no_grad():
                # VAE expects list of latents
                if not isinstance(latent, list):
                    latent = [latent]
                
                videos = wan.vae.decode(latent)
                video_output = videos[0]  # (C, F, H, W)
                
            logger.info(f"Decoded video shape: {video_output.shape}")
            
        finally:
            # Move VAE back to CPU
            if hasattr(wan.vae, 'model'):
                wan.vae.model.cpu()
            else:
                wan.vae.cpu()
            torch.cuda.empty_cache()
        
        jobs[job_id]["progress"] = 90
        
        # Save video
        final_output = OUTPUT_DIR / f"{job_id}.mp4"
        logger.info(f"Saving video to {final_output}...")
        
        video_output = video_output.cpu()
        
        # Convert (C, F, H, W) -> (F, H, W, C)
        if video_output.dim() == 4:
            video_output = video_output.permute(1, 2, 3, 0)
        
        # Normalize to [0, 255]
        video_output = (video_output.clamp(-1, 1).add(1).div(2) * 255).byte().numpy()
        
        from PIL import Image
        frames = [Image.fromarray(frame) for frame in video_output]
        
        logger.info(f"Exporting {len(frames)} frames...")
        from diffusers.utils import export_to_video
        export_to_video(frames, str(final_output), fps=16)
        
        if final_output.exists():
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["output_path"] = str(final_output)
            jobs[job_id]["progress"] = 100
            logger.info(f"=== JOB {job_id} COMPLETED ===")
        else:
            raise RuntimeError("Output video was not created")
            
    except Exception as e:
        logger.error(f"=== JOB {job_id} FAILED ===", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        if image_path:
            try:
                Path(image_path).unlink(missing_ok=True)
            except:
                pass
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.on_event("startup")
async def startup_event():
    logger.info("YUME API v2.0.4 starting...")
    logger.info(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    asyncio.create_task(background_load_model())
    asyncio.create_task(worker_loop())


async def background_load_model():
    logger.info("Loading model in background...")
    success = await asyncio.to_thread(load_yume_model)
    if not success:
        logger.error("Failed to load YUME model")
    else:
        logger.info("Model loaded successfully")


@app.get("/health")
async def health_check():
    is_loaded = MODELS.loaded
    
    if not is_loaded:
        return fastapi.Response(
            content=json.dumps({"status": "loading", "model_loaded": False}),
            status_code=503,
            media_type="application/json"
        )
        
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": True,
        "queue_size": job_queue.qsize()
    }


@app.post("/generate", response_model=JobStatus)
async def generate_text_to_video(request: GenerateRequest, background_tasks: BackgroundTasks):
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
    if not MODELS.loaded:
        raise HTTPException(status_code=503, detail="YUME model not loaded")
    
    job_id = str(uuid.uuid4())
    
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_path = INPUT_DIR / f"{job_id}_input.jpg"
    
    with open(image_path, "wb") as f:
        content = await image.read()
        f.write(content)
    
    from PIL import Image
    try:
        img = Image.open(image_path)
        width, height = img.size
        width = min(max(width, 512), 1920)
        height = min(max(height, 288), 1080)
        width = (width // 64) * 64
        height = (height // 64) * 64
    except Exception as e:
        logger.error(f"Error processing image: {e}")
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
    
    await job_queue.put(job_id)
    return JobStatus(job_id=job_id, status="queued", queue_position=job_queue.qsize())


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
        progress=job.get("progress")
    )


@app.get("/download/{job_id}")
async def download(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed: {job['status']}")
    
    output_path = job.get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(output_path, media_type="video/mp4", filename=f"{job_id}.mp4")


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs.pop(job_id)
    
    if job.get("output_path"):
        Path(job["output_path"]).unlink(missing_ok=True)
    
    input_path = INPUT_DIR / f"{job_id}_input.jpg"
    input_path.unlink(missing_ok=True)
    
    return {"message": "Job deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)