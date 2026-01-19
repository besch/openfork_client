#!/usr/bin/env python3
"""
YUME API - Version 2.0.3
MINIMAL VERSION: Simplified to use ONLY the high-level wan.generate() API

This version removes the manual diffusion loop entirely and relies on
wan.generate() to handle everything internally. This should be the
correct approach based on the YUME library design.
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="YUME API", version="2.0.3")

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

# Global model storage
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
    status: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    progress: Optional[int] = None
    queue_position: Optional[int] = None


def load_yume_model():
    """Load the YUME model."""
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
             logger.warning("ti2v-5B config not found in WAN_CONFIGS")
             logger.info(f"Available configs: {list(WAN_CONFIGS.keys())}")
             raise ValueError("ti2v-5B config missing")
        
        ckpt_dir = YUME_DIR / "Yume-5B-720P"
        if not ckpt_dir.exists() and MODEL_DIR.exists():
            try:
                ckpt_dir.symlink_to(MODEL_DIR)
                logger.info(f"Created symlink: {ckpt_dir} -> {MODEL_DIR}")
            except Exception as e:
                logger.warning(f"Could not create symlink: {e}")
        
        logger.info(f"Instantiating wan23.Yume...")
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
    """Background worker."""
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
    Synchronous video generation using wan.generate().
    SIMPLIFIED VERSION: Let wan.generate() handle everything.
    """
    try:
        if not MODELS.loaded:
            raise RuntimeError("YUME model not loaded")
            
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"=== JOB {job_id} START ===")
        logger.info(f"Prompt: {prompt[:100]}...")
        logger.info(f"Params: {num_frames} frames, {width}x{height}, {steps} steps, cfg={cfg}")
        
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
        logger.info(f"  offload_model=False (keep models on GPU)")
        
        # KEY CHANGE: Set offload_model=False to prevent dimension issues
        # The offload logic may be causing tensor dimension problems
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                try:
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
                        offload_model=False  # CRITICAL: Keep on GPU
                    )
                    logger.info(f"wan.generate() completed. Return type: {type(gen_ret)}")
                    
                    # Debug the return value
                    if isinstance(gen_ret, tuple):
                        logger.info(f"  Tuple length: {len(gen_ret)}")
                        for i, item in enumerate(gen_ret):
                            if isinstance(item, torch.Tensor):
                                logger.info(f"  gen_ret[{i}]: Tensor shape={item.shape}")
                            else:
                                logger.info(f"  gen_ret[{i}]: {type(item)}")
                    elif isinstance(gen_ret, torch.Tensor):
                        logger.info(f"  Tensor shape: {gen_ret.shape}")
                    elif isinstance(gen_ret, list):
                        logger.info(f"  List length: {len(gen_ret)}")
                        for i, item in enumerate(gen_ret):
                            if isinstance(item, torch.Tensor):
                                logger.info(f"  gen_ret[{i}]: Tensor shape={item.shape}")
                    
                except Exception as e:
                    logger.error(f"wan.generate() failed: {e}", exc_info=True)
                    raise
        
        jobs[job_id]["progress"] = 85
        
        # Now we need to decode the output
        # The exact format depends on what wan.generate() returns
        logger.info("Processing generation output...")
        
        video_output = None
        
        # Handle different return types
        if isinstance(gen_ret, torch.Tensor):
            # Direct tensor - assume it's latents
            logger.info("Got direct tensor, decoding with VAE...")
            latents = gen_ret
            
            if latents.dim() == 4:
                # Add batch dimension if needed
                latents = latents.unsqueeze(0)
            
            logger.info(f"Latent shape: {latents.shape}")
            
            with torch.no_grad():
                videos = wan.vae.decode([latents])
            video_output = videos[0]
            logger.info(f"Decoded video shape: {video_output.shape}")
            
        elif isinstance(gen_ret, list):
            # List of tensors
            logger.info("Got list, processing first element...")
            if len(gen_ret) > 0 and isinstance(gen_ret[0], torch.Tensor):
                latents = gen_ret[0]
                logger.info(f"Latent shape: {latents.shape}")
                
                with torch.no_grad():
                    videos = wan.vae.decode([latents])
                video_output = videos[0]
                logger.info(f"Decoded video shape: {video_output.shape}")
            else:
                raise ValueError(f"Unexpected list content: {type(gen_ret[0])}")
                
        elif isinstance(gen_ret, tuple):
            # This shouldn't happen with offload_model=False, but handle it
            logger.warning("Got tuple return - this indicates wan.generate() didn't complete")
            logger.warning("This usually means offload_model=True caused issues")
            raise RuntimeError("wan.generate() returned incomplete result (tuple). Try reducing resolution or frame count.")
        
        else:
            raise ValueError(f"Unexpected return type from wan.generate(): {type(gen_ret)}")
        
        jobs[job_id]["progress"] = 90
        
        # Save the output
        final_output = OUTPUT_DIR / f"{job_id}.mp4"
        logger.info(f"Saving video to {final_output}...")
        
        if isinstance(video_output, torch.Tensor):
            video_output = video_output.cpu()
            
            logger.info(f"Video tensor shape: {video_output.shape}, dims: {video_output.dim()}")
            
            if video_output.dim() == 4:
                # (C, F, H, W) -> (F, H, W, C)
                video_output = video_output.permute(1, 2, 3, 0)
            elif video_output.dim() == 5:
                # (B, C, F, H, W) -> (F, H, W, C)
                video_output = video_output[0].permute(1, 2, 3, 0)
            
            video_output = (video_output.clamp(-1, 1).add(1).div(2) * 255).byte().numpy()
            
            from PIL import Image
            frames = [Image.fromarray(frame) for frame in video_output]
        elif isinstance(video_output, list):
            frames = video_output
        else:
            import numpy as np
            if isinstance(video_output, np.ndarray):
                from PIL import Image
                frames = [Image.fromarray(frame) for frame in video_output]
            else:
                raise ValueError(f"Unknown video output type: {type(video_output)}")
        
        logger.info(f"Exporting {len(frames)} frames to video...")
        from diffusers.utils import export_to_video
        export_to_video(frames, str(final_output), fps=16)
        
        if final_output.exists():
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["output_path"] = str(final_output)
            jobs[job_id]["progress"] = 100
            logger.info(f"=== JOB {job_id} COMPLETED ===")
        else:
            raise RuntimeError(f"Output video was not created")
            
    except Exception as e:
        logger.error(f"=== JOB {job_id} FAILED ===", exc_info=True)
        logger.error(f"Error: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        if image_path:
             try:
                 Path(image_path).unlink(missing_ok=True)
             except Exception:
                 pass
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.on_event("startup")
async def startup_event():
    """Startup."""
    logger.info("YUME API starting...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    if YUME_DIR.exists():
        logger.info(f"YUME directory found: {YUME_DIR}")
    else:
        logger.warning(f"YUME directory not found: {YUME_DIR}")
    
    if MODEL_DIR.exists():
        logger.info(f"Model directory found: {MODEL_DIR}")
    else:
        logger.warning(f"Model directory not found: {MODEL_DIR}")

    asyncio.create_task(background_load_model())
    asyncio.create_task(worker_loop())


async def background_load_model():
    """Load model in background."""
    logger.info("Starting background model load...")
    success = await asyncio.to_thread(load_yume_model)
    if not success:
        logger.error("Failed to load YUME model")
    else:
        logger.info("Model load completed successfully")


@app.get("/health")
async def health_check():
    """Health check."""
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
    """Queue T2V job."""
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
    """Queue I2V job."""
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
        logger.error(f"Error processing image {job_id}: {e}")
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
    """Get job status."""
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
    """Download video."""
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
    """Delete job."""
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