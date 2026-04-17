#!/usr/bin/env python3
"""
ERNIE-Image REST API Server for OpenFork DGN Client
Baidu ERNIE-Image text-to-image generation via diffusers pipeline.

Supports two model variants via ERNIE_MODEL_ID env var:
  - baidu/ERNIE-Image      (standard, 50 steps)
  - baidu/ERNIE-Image-Turbo (turbo, 8 steps)
"""

import os
import uuid
import logging
import time
import base64
import io
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

WORK_DIR = Path("/app")
OUTPUT_DIR = WORK_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

jobs = {}

pipe = None
model_loading = False
model_error = None

MODEL_ID = os.environ.get("ERNIE_MODEL_ID", "baidu/ERNIE-Image")
DEFAULT_STEPS = int(os.environ.get("ERNIE_DEFAULT_STEPS", "50"))
MODEL_DTYPE = os.environ.get("ERNIE_DTYPE", "auto")


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: Optional[int] = None
    guidance_scale: float = 5.0
    seed: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    model_id: str
    model_loaded: bool


def load_model():
    global pipe, model_loading, model_error
    model_loading = True
    try:
        from diffusers import DiffusionPipeline

        logger.info(f"Loading ERNIE-Image model: {MODEL_ID} (dtype={MODEL_DTYPE})")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if MODEL_DTYPE == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif MODEL_DTYPE == "bf16":
            dtype = torch.bfloat16
        elif MODEL_DTYPE == "fp16":
            dtype = torch.float16
        elif MODEL_DTYPE == "fp8":
            dtype = torch.float8_e4m3fn
        else:
            dtype = torch.bfloat16

        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
        ).to(device)

        logger.info(f"ERNIE-Image model loaded on {device} with dtype {dtype}")
    except Exception as e:
        logger.error(f"Failed to load ERNIE-Image model: {e}", exc_info=True)
        model_error = str(e)
    finally:
        model_loading = False


def run_generation(job_id: str, request: GenerateRequest):
    try:
        jobs[job_id]["status"] = "processing"
        logger.info(f"Starting generation for job {job_id}: {request.prompt[:80]}...")

        steps = request.num_inference_steps or DEFAULT_STEPS
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(request.seed)

        start_time = time.time()
        result = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            width=request.width,
            height=request.height,
            num_inference_steps=steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        )
        elapsed = time.time() - start_time
        logger.info(
            f"Generation completed for job {job_id} in {elapsed:.1f}s ({steps} steps)"
        )

        image = result.images[0]
        output_path = OUTPUT_DIR / f"{job_id}.png"
        image.save(str(output_path), format="PNG")

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_path"] = str(output_path)
        jobs[job_id]["elapsed_seconds"] = elapsed
        jobs[job_id]["steps"] = steps
    except Exception as e:
        logger.error(f"Generation failed for job {job_id}: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@asynccontextmanager
async def lifespan(app_instance):
    import threading

    thread = threading.Thread(target=load_model, daemon=True)
    thread.start()
    yield


app = FastAPI(title="ERNIE-Image API", lifespan=lifespan)


@app.get("/health")
async def health():
    return HealthResponse(
        status="ok" if pipe is not None else ("loading" if model_loading else "error"),
        model_id=MODEL_ID,
        model_loaded=pipe is not None,
    )


@app.post("/generate")
async def generate(request: GenerateRequest):
    global model_error
    if model_loading:
        raise HTTPException(status_code=503, detail="Model is still loading")
    if pipe is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {model_error}")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "prompt": request.prompt,
        "created_at": time.time(),
    }

    import threading

    thread = threading.Thread(
        target=run_generation, args=(job_id, request), daemon=True
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/output/{job_id}")
async def get_output(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status is {job['status']}")
    output_path = job.get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(output_path, media_type="image/png", filename=f"{job_id}.png")
