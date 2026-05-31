#!/usr/bin/env python3
"""
NVIDIA PiD REST API for OpenFork.

This server keeps the PiD decoder resident in memory and exposes an async
still-image upscale API. The baked image contains PiD plus the Flux/Z-Image
compatible 2k decoder checkpoint and Flux VAE weights.
"""

import gc
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

from pid._src.inference._demo_common import save_image
from pid._src.inference._demo_from_clean_common import (
    _add_noise,
    _load_input_image,
    _vae_decode,
)
from pid._src.inference.checkpoint_registry import get_pid_checkpoint
from pid._src.utils.model_loader import load_model_from_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

WORK_DIR = Path("/app")
PID_DIR = Path(os.environ.get("PID_REPO_DIR", "/app/PiD"))
INPUT_DIR = WORK_DIR / "input"
OUTPUT_DIR = WORK_DIR / "output"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PID_BACKBONE = os.environ.get("PID_BACKBONE", "zimage")
PID_CKPT_TYPE = os.environ.get("PID_CKPT_TYPE", "2k")
PID_CONFIG_FILE = os.environ.get("PID_CONFIG_FILE", "pid/_src/configs/pid/config.py")
DEFAULT_PROMPT = os.environ.get("PID_DEFAULT_PROMPT", "high quality detailed image")

jobs: dict = {}
model = None
model_id = f"{PID_BACKBONE}:{PID_CKPT_TYPE}"
model_loading = False
model_error: Optional[str] = None
_inference_semaphore = threading.Semaphore(1)


class HealthResponse(BaseModel):
    status: str
    model_id: str
    model_loaded: bool
    error: Optional[str] = None


def _is_oom_error(exc: Exception) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _cleanup_cuda(context: str, synchronize: bool = False) -> None:
    for _ in range(2):
        gc.collect()

    if not torch.cuda.is_available():
        return

    if synchronize:
        try:
            torch.cuda.synchronize()
        except Exception as exc:
            logger.warning("CUDA synchronize failed during %s: %s", context, exc)

    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning("torch.cuda.empty_cache failed during %s: %s", context, exc)

    if hasattr(torch.cuda, "ipc_collect"):
        try:
            torch.cuda.ipc_collect()
        except Exception as exc:
            logger.warning("torch.cuda.ipc_collect failed during %s: %s", context, exc)


def load_pid_model() -> None:
    global model, model_loading, model_error
    model_loading = True
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("PiD requires CUDA; torch.cuda.is_available() is false")

        os.chdir(PID_DIR)
        torch.enable_grad(False)
        torch.backends.cuda.matmul.allow_tf32 = True

        checkpoint = get_pid_checkpoint(PID_BACKBONE, PID_CKPT_TYPE)
        logger.info(
            "Loading PiD model: backbone=%s ckpt_type=%s experiment=%s checkpoint=%s",
            PID_BACKBONE,
            PID_CKPT_TYPE,
            checkpoint.experiment,
            checkpoint.checkpoint_path,
        )
        loaded_model, _config = load_model_from_checkpoint(
            experiment_name=checkpoint.experiment,
            checkpoint_path=checkpoint.checkpoint_path,
            config_file=PID_CONFIG_FILE,
            enable_fsdp=False,
            experiment_opts=[],
            strict=False,
            load_ema_to_reg=False,
        )
        loaded_model.eval()
        model = loaded_model
        logger.info("PiD model loaded successfully")
    except Exception as exc:
        logger.error("Failed to load PiD model: %s", exc, exc_info=True)
        model_error = str(exc)
    finally:
        model_loading = False


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _round_down_to_multiple(value: int, multiple: int = 16) -> int:
    return max(multiple, (value // multiple) * multiple)


def _prepare_input_image(
    source_path: Path,
    job_id: str,
    max_long_edge: int,
    preserve_aspect: bool,
) -> Path:
    if not preserve_aspect:
        return source_path

    with Image.open(source_path) as raw_image:
        image = raw_image.convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Input image has invalid dimensions")

        scale = min(1.0, max_long_edge / max(width, height))
        target_width = _round_down_to_multiple(int(width * scale))
        target_height = _round_down_to_multiple(int(height * scale))

        left = max(0, (width - _round_down_to_multiple(width)) // 2)
        top = max(0, (height - _round_down_to_multiple(height)) // 2)
        crop_width = _round_down_to_multiple(width)
        crop_height = _round_down_to_multiple(height)
        image = image.crop((left, top, left + crop_width, top + crop_height))

        if (target_width, target_height) != image.size:
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        prepared_path = INPUT_DIR / f"{job_id}_prepared.png"
        image.save(prepared_path, "PNG", optimize=True)
        return prepared_path


def run_upscale(
    job_id: str,
    input_path: Path,
    prompt: str,
    input_resolution: int,
    degrade_sigma: float,
    cfg_scale: float,
    pid_inference_steps: int,
    scale: int,
    seed: int,
    preserve_aspect: bool,
) -> None:
    logger.info("PiD job %s waiting for inference slot", job_id)
    _inference_semaphore.acquire()
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress_pct"] = 5
        _cleanup_cuda(f"job {job_id} preflight")

        if model is None:
            raise RuntimeError(f"PiD model is not loaded: {model_error}")

        input_resolution = max(256, min(int(input_resolution), 512))
        degrade_sigma = max(0.0, min(float(degrade_sigma), 1.0))
        cfg_scale = float(cfg_scale)
        pid_inference_steps = max(1, min(int(pid_inference_steps), 8))
        scale = 4
        seed = int(seed)
        caption = (prompt or DEFAULT_PROMPT).strip() or DEFAULT_PROMPT
        prepared_path = _prepare_input_image(
            input_path,
            job_id,
            input_resolution,
            preserve_aspect,
        )

        start_time = time.time()
        logger.info(
            "PiD job %s config: input=%s input_resolution=%s sigma=%s cfg=%s steps=%s scale=%s seed=%s preserve_aspect=%s",
            job_id,
            prepared_path,
            input_resolution,
            degrade_sigma,
            cfg_scale,
            pid_inference_steps,
            scale,
            seed,
            preserve_aspect,
        )

        input_tensor = _load_input_image(
            str(prepared_path),
            input_resolution,
            keep_input_size=preserve_aspect,
        ).to(dtype=torch.bfloat16, device="cuda")
        jobs[job_id]["progress_pct"] = 20

        with torch.no_grad():
            clean_latent = model.encode_lq_latent(input_tensor)
            vae_compression = int(model.vae_encoder.spatial_compression_factor)
            vae_h = int(clean_latent.shape[-2]) * vae_compression
            vae_w = int(clean_latent.shape[-1]) * vae_compression
            target_hw = (vae_h * scale, vae_w * scale)

            generator = torch.Generator(device="cuda").manual_seed(seed)
            latent = _add_noise(clean_latent.float(), degrade_sigma, generator).to(
                dtype=torch.bfloat16,
                device="cuda",
            )
            jobs[job_id]["progress_pct"] = 40

            vae_img = _vae_decode(model, latent)
            lq_placeholder = torch.zeros_like(
                vae_img,
                dtype=torch.bfloat16,
                device="cuda",
            )
            data_batch = {
                model.config.input_caption_key: [caption],
                "LQ_video_or_image": lq_placeholder,
                "LQ_latent": latent.to(dtype=torch.bfloat16, device="cuda"),
                "degrade_sigma": torch.tensor(
                    [degrade_sigma],
                    device="cuda",
                    dtype=torch.float32,
                ),
            }
            jobs[job_id]["progress_pct"] = 65
            samples_out = model.generate_samples_from_batch(
                data_batch,
                cfg_scale=cfg_scale,
                num_steps=pid_inference_steps,
                seed=seed,
                shift=None,
                image_size=target_hw,
            )

        output_path = OUTPUT_DIR / f"{job_id}.png"
        ours_img = samples_out[0].float().cpu().clamp(-1, 1)
        save_image(ours_img, str(output_path))

        elapsed = time.time() - start_time
        jobs[job_id].update(
            {
                "status": "completed",
                "output_path": str(output_path),
                "elapsed_seconds": elapsed,
                "input_width": input_tensor.shape[-1],
                "input_height": input_tensor.shape[-2],
                "output_width": target_hw[1],
                "output_height": target_hw[0],
                "progress_pct": 100,
            }
        )
        logger.info("PiD job %s completed in %.1fs -> %s", job_id, elapsed, output_path)
    except Exception as exc:
        logger.error("PiD job %s failed: %s", job_id, exc, exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)
        if _is_oom_error(exc):
            jobs[job_id]["error_code"] = "cuda_oom"
    finally:
        _cleanup_cuda(f"job {job_id} teardown", synchronize=True)
        _inference_semaphore.release()


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    thread = threading.Thread(target=load_pid_model, daemon=True)
    thread.start()
    yield


app = FastAPI(title="OpenFork PiD Image Upscale API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if model is not None:
        status = "ok"
    elif model_loading:
        status = "loading"
    else:
        status = "error"

    return HealthResponse(
        status=status,
        model_id=model_id,
        model_loaded=model is not None,
        error=model_error,
    )


@app.post("/upscale")
async def upscale(
    image: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
    input_resolution: int = Form(512),
    degrade_sigma: float = Form(0.0),
    cfg_scale: float = Form(1.0),
    pid_inference_steps: int = Form(4),
    scale: int = Form(4),
    seed: int = Form(5),
    preserve_aspect: str = Form("true"),
):
    if model_loading:
        raise HTTPException(status_code=503, detail="PiD model is still loading")
    if model is None:
        raise HTTPException(status_code=503, detail=f"PiD model not loaded: {model_error}")

    job_id = str(uuid.uuid4())
    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"
    input_path = INPUT_DIR / f"{job_id}{suffix}"
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    input_path.write_bytes(content)

    jobs[job_id] = {
        "status": "queued",
        "prompt": prompt,
        "created_at": time.time(),
        "progress_pct": 0,
    }

    thread = threading.Thread(
        target=run_upscale,
        args=(
            job_id,
            input_path,
            prompt,
            input_resolution,
            degrade_sigma,
            cfg_scale,
            pid_inference_steps,
            scale,
            seed,
            _as_bool(preserve_aspect),
        ),
        daemon=True,
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
