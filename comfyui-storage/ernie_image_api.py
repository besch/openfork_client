#!/usr/bin/env python3
"""
ERNIE-Image REST API Server for OpenFork DGN Client
Baidu ERNIE-Image text-to-image generation via diffusers pipeline.

Supports two model variants via ERNIE_MODEL_ID env var:
  - baidu/ERNIE-Image      (standard, 50 steps)
  - baidu/ERNIE-Image-Turbo (turbo, 8 steps)

Model weights are expected to be pre-baked into the Docker image at /app/models.
The server starts immediately and loads the model in a background thread so
that /health is reachable right away and the client can start polling.
"""

import os
import uuid
import logging
import time
from pathlib import Path
from typing import Optional

import torch
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

jobs: dict = {}

pipe = None
model_loading = False
model_error: Optional[str] = None

MODEL_ID = os.environ.get("ERNIE_MODEL_ID", "baidu/ERNIE-Image-Turbo")
# local_files_only=True because weights are baked into the image.
# Falls back to downloading if the cache is somehow missing (e.g. dev builds).
HF_HOME = os.environ.get("HF_HOME", "/app/models")
DEFAULT_STEPS = int(os.environ.get("ERNIE_DEFAULT_STEPS", "8"))
MODEL_DTYPE = os.environ.get("ERNIE_DTYPE", "fp16")
# Turbo needs guidance_scale=1.0; standard ERNIE-Image uses 4.0-5.0
_IS_TURBO = "turbo" in MODEL_ID.lower()
DEFAULT_CFG = float(os.environ.get("ERNIE_DEFAULT_CFG", "1.0" if _IS_TURBO else "4.0"))


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: Optional[int] = None
    # -1 sentinel → use model default (1.0 for Turbo, 4.0 for standard)
    guidance_scale: float = -1.0
    seed: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    model_id: str
    model_loaded: bool
    error: Optional[str] = None


def _resolve_dtype() -> torch.dtype:
    if MODEL_DTYPE == "bf16":
        return torch.bfloat16
    if MODEL_DTYPE == "fp16":
        return torch.float16
    if MODEL_DTYPE == "fp8":
        return torch.float8_e4m3fn
    # auto
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_model() -> None:
    global pipe, model_loading, model_error
    model_loading = True
    try:
        from diffusers import ErnieImagePipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = _resolve_dtype()

        logger.info(
            f"Loading ERNIE-Image model: {MODEL_ID} | device={device} | dtype={dtype}"
        )

        # Try local cache first (baked into image), fall back to HF download.
        try:
            pipe = ErnieImagePipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                local_files_only=True,
                cache_dir=HF_HOME,
            ).to(device)
            logger.info("Model loaded from local cache.")
        except Exception as local_err:
            logger.warning(
                f"Local cache load failed ({local_err}). Falling back to HF download..."
            )
            pipe = ErnieImagePipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                cache_dir=HF_HOME,
            ).to(device)
            logger.info("Model downloaded and loaded from HuggingFace.")

        logger.info(f"ERNIE-Image model ready on {device} with dtype {dtype}")
    except Exception as e:
        logger.error(f"Failed to load ERNIE-Image model: {e}", exc_info=True)
        model_error = str(e)
    finally:
        model_loading = False


def run_generation(job_id: str, request: GenerateRequest) -> None:
    try:
        jobs[job_id]["status"] = "processing"
        logger.info(f"Starting generation for job {job_id}: {request.prompt[:80]}...")

        steps = request.num_inference_steps or DEFAULT_STEPS
        cfg = request.guidance_scale if request.guidance_scale >= 0 else DEFAULT_CFG
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
            guidance_scale=cfg,
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
async def lifespan(app_instance: FastAPI):
    import threading

    thread = threading.Thread(target=load_model, daemon=True)
    thread.start()
    yield


app = FastAPI(title="ERNIE-Image API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if pipe is not None:
        status = "ok"
    elif model_loading:
        status = "loading"
    else:
        status = "error"

    return HealthResponse(
        status=status,
        model_id=MODEL_ID,
        model_loaded=pipe is not None,
        error=model_error,
    )


@app.post("/generate")
async def generate(request: GenerateRequest):
    if model_loading:
        raise HTTPException(status_code=503, detail="Model is still loading")
    if pipe is None:
        raise HTTPException(
            status_code=503, detail=f"Model not loaded: {model_error}"
        )

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)