"""
FastAPI REST API wrapper for daVinci-MagiHuman

daVinci-MagiHuman (GAIR-NLP / Sand.ai) is a 15B single-stream Transformer that
jointly generates synchronized video + audio in one pass. It is purpose-built for
human-centric avatar video: expressive faces, speech-lip sync, body motion.

This server is copied to /app/ in the Docker container and started by start_cloud.sh.
It wraps the Python inference library and exposes it as a REST API on port 8000.

Architecture:
  - Singleton model loaded on first request (lazy to avoid startup OOM)
  - One inference at a time (GPU is fully occupied during generation)
  - Background task pattern: POST /generate/t2v or /generate/i2v → job_id
  - GET /status/{job_id} → poll for completion
  - GET /download/{job_id} → retrieve MP4
  - DELETE /job/{job_id} → cleanup

Endpoints:
  GET  /health               — liveness check
  POST /generate/t2v         — text-to-video (JSON body)
  POST /generate/i2v         — image-to-video (multipart form)
  GET  /status/{job_id}      — job status + progress
  GET  /download/{job_id}    — download completed video
  DELETE /job/{job_id}       — cleanup job files
"""

import gc
import logging
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DaVinci API] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

DAVINCI_ROOT = os.environ.get("DAVINCI_ROOT", "/opt/davinci-magihuman")
DAVINCI_MODELS = os.environ.get("DAVINCI_MODELS", "/models/davinci-magihuman")
DAVINCI_OUTPUT = os.environ.get("DAVINCI_OUTPUT", "/tmp/davinci_outputs")
WAN22_TI2V_PATH = os.environ.get("WAN22_TI2V_PATH", f"{DAVINCI_MODELS}/Wan2.2-TI2V-5B")

# Fixed inference parameters for the distilled model
_DEFAULT_STEPS = 8          # distilled — only valid value
_DEFAULT_FPS = 16           # daVinci-MagiHuman default
_DEFAULT_HEIGHT = 256       # distilled baseline resolution
_DEFAULT_WIDTH = 256
_DEFAULT_DURATION_S = 5     # seconds

os.makedirs(DAVINCI_OUTPUT, exist_ok=True)

# ─── App + State ─────────────────────────────────────────────────────────────

app = FastAPI(title="daVinci-MagiHuman API", version="1.0.0")

# In-memory job registry: job_id → {status, output_path, error, ...}
_jobs: dict = {}
_jobs_lock = threading.Lock()

# Serialize inference (one job at a time — GPU is fully saturated)
_inference_semaphore = threading.Semaphore(1)
_pipeline = None
_pipeline_lock = threading.Lock()


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class T2VRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    duration_seconds: float = _DEFAULT_DURATION_S
    height: int = _DEFAULT_HEIGHT
    width: int = _DEFAULT_WIDTH
    num_steps: int = _DEFAULT_STEPS
    seed: int = 0
    use_distilled: bool = True


# ─── Model Loading ────────────────────────────────────────────────────────────

def _ensure_davinci_root_in_path():
    if DAVINCI_ROOT not in sys.path:
        sys.path.insert(0, DAVINCI_ROOT)


def _load_pipeline():
    """Load daVinci-MagiHuman inference pipeline (once, singleton)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        logger.info("Loading daVinci-MagiHuman pipeline…")
        _ensure_davinci_root_in_path()

        try:
            # daVinci-MagiHuman exposes its pipeline through the inference package.
            # The exact import path depends on the repo structure.
            # We try the most likely paths in order.
            from inference.pipeline.pipeline import MagiHumanPipeline  # type: ignore
        except ImportError:
            try:
                from inference.pipeline import MagiHumanPipeline  # type: ignore
            except ImportError:
                logger.error(
                    "Could not import MagiHumanPipeline from daVinci-MagiHuman. "
                    "Ensure the repo is at DAVINCI_ROOT=%s and installed correctly.",
                    DAVINCI_ROOT,
                )
                raise

        # Locate config file — daVinci ships example configs under example/distill/
        config_dir = Path(DAVINCI_ROOT) / "example" / "distill"
        config_file = config_dir / "config.json"

        if not config_file.exists():
            # Fall back to base model config
            config_file = Path(DAVINCI_ROOT) / "example" / "base" / "config.json"

        if not config_file.exists():
            raise FileNotFoundError(
                f"No daVinci config found under {DAVINCI_ROOT}/example/. "
                "Ensure the container was built correctly."
            )

        logger.info("Loading pipeline from config: %s", config_file)
        _pipeline = MagiHumanPipeline.from_config(str(config_file), model_dir=DAVINCI_MODELS)
        logger.info("daVinci-MagiHuman pipeline loaded ✓")
        return _pipeline


# ─── Inference Helpers ────────────────────────────────────────────────────────

def _set_job(job_id: str, **kwargs):
    with _jobs_lock:
        if job_id not in _jobs:
            _jobs[job_id] = {}
        _jobs[job_id].update(kwargs)


def _run_t2v(job_id: str, prompt: str, negative_prompt: str,
             duration_seconds: float, height: int, width: int,
             num_steps: int, seed: int, use_distilled: bool):
    """Run T2V inference in background thread."""
    _inference_semaphore.acquire()
    try:
        _set_job(job_id, status="processing")
        pipeline = _load_pipeline()

        num_frames = max(1, int(duration_seconds * _DEFAULT_FPS))
        output_path = os.path.join(DAVINCI_OUTPUT, f"{job_id}.mp4")
        generator = torch.Generator(device="cuda").manual_seed(seed) if seed else None

        logger.info(
            "T2V [%s]: prompt=%r | frames=%d | steps=%d | %dx%d",
            job_id, prompt[:80], num_frames, num_steps, width, height,
        )

        pipeline.generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            output_path=output_path,
            generator=generator,
            use_distilled=use_distilled,
        )

        if os.path.exists(output_path):
            _set_job(job_id, status="completed", output_path=output_path)
            logger.info("T2V [%s] completed → %s", job_id, output_path)
        else:
            _set_job(job_id, status="failed", error="Output file not produced")

    except Exception as exc:
        logger.error("T2V [%s] failed: %s", job_id, exc, exc_info=True)
        _set_job(job_id, status="failed", error=str(exc))
    finally:
        _inference_semaphore.release()
        # Release GPU cache between jobs
        torch.cuda.empty_cache()
        gc.collect()


def _run_i2v(job_id: str, image_path: str, prompt: str, negative_prompt: str,
             duration_seconds: float, height: int, width: int,
             num_steps: int, seed: int, use_distilled: bool):
    """Run I2V inference in background thread."""
    _inference_semaphore.acquire()
    try:
        _set_job(job_id, status="processing")
        pipeline = _load_pipeline()

        num_frames = max(1, int(duration_seconds * _DEFAULT_FPS))
        output_path = os.path.join(DAVINCI_OUTPUT, f"{job_id}.mp4")
        generator = torch.Generator(device="cuda").manual_seed(seed) if seed else None

        from PIL import Image  # noqa: PLC0415
        ref_image = Image.open(image_path).convert("RGB")

        logger.info(
            "I2V [%s]: prompt=%r | frames=%d | steps=%d | %dx%d",
            job_id, prompt[:80], num_frames, num_steps, width, height,
        )

        pipeline.generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            reference_image=ref_image,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            output_path=output_path,
            generator=generator,
            use_distilled=use_distilled,
        )

        if os.path.exists(output_path):
            _set_job(job_id, status="completed", output_path=output_path)
            logger.info("I2V [%s] completed → %s", job_id, output_path)
        else:
            _set_job(job_id, status="failed", error="Output file not produced")

    except Exception as exc:
        logger.error("I2V [%s] failed: %s", job_id, exc, exc_info=True)
        _set_job(job_id, status="failed", error=str(exc))
    finally:
        _inference_semaphore.release()
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                pass
        torch.cuda.empty_cache()
        gc.collect()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe — also returns GPU info."""
    try:
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "None"
        vram_free_mb = (
            torch.cuda.mem_get_info(0)[0] // (1024 * 1024) if cuda_ok else 0
        )
    except Exception as exc:
        logger.warning("Health check GPU query failed: %s", exc)
        cuda_ok, gpu_name, vram_free_mb = False, "Error", 0

    return {
        "status": "ok",
        "cuda": cuda_ok,
        "gpu_name": gpu_name,
        "vram_free_mb": vram_free_mb,
        "model_loaded": _pipeline is not None,
    }


@app.post("/generate/t2v")
async def generate_t2v(req: T2VRequest, background_tasks: BackgroundTasks):
    """Submit a text-to-video generation job."""
    job_id = str(uuid.uuid4())
    _set_job(job_id, status="pending", job_id=job_id)

    background_tasks.add_task(
        _run_t2v,
        job_id=job_id,
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        duration_seconds=req.duration_seconds,
        height=req.height,
        width=req.width,
        num_steps=req.num_steps,
        seed=req.seed,
        use_distilled=req.use_distilled,
    )
    logger.info("T2V job queued: %s", job_id)
    return {"job_id": job_id}


@app.post("/generate/i2v")
async def generate_i2v(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    duration_seconds: float = Form(_DEFAULT_DURATION_S),
    height: int = Form(_DEFAULT_HEIGHT),
    width: int = Form(_DEFAULT_WIDTH),
    num_steps: int = Form(_DEFAULT_STEPS),
    seed: int = Form(0),
    use_distilled: bool = Form(True),
):
    """Submit an image-to-video (avatar) generation job."""
    job_id = str(uuid.uuid4())

    # Persist uploaded image for background thread
    image_path = f"/tmp/{job_id}_input_image"
    contents = await image.read()
    with open(image_path, "wb") as f:
        f.write(contents)

    _set_job(job_id, status="pending", job_id=job_id)
    background_tasks.add_task(
        _run_i2v,
        job_id=job_id,
        image_path=image_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        duration_seconds=duration_seconds,
        height=height,
        width=width,
        num_steps=num_steps,
        seed=seed,
        use_distilled=use_distilled,
    )
    logger.info("I2V job queued: %s", job_id)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    """Poll job status."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return job


@app.get("/download/{job_id}")
def download(job_id: str):
    """Download completed video."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job or job.get("status") != "completed":
        return JSONResponse(status_code=404, content={"error": "Output not ready"})
    output_path = job.get("output_path")
    if not output_path or not os.path.exists(output_path):
        return JSONResponse(status_code=404, content={"error": "Output file missing"})
    return FileResponse(output_path, media_type="video/mp4", filename=f"{job_id}.mp4")


@app.delete("/job/{job_id}")
def delete_job(job_id: str):
    """Clean up job files and registry entry."""
    with _jobs_lock:
        job = _jobs.pop(job_id, None)
    if job:
        out_path = job.get("output_path")
        if out_path and os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
    return {"status": "ok"}


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting daVinci-MagiHuman API server on port 8000")
    logger.info("  DAVINCI_ROOT   = %s", DAVINCI_ROOT)
    logger.info("  DAVINCI_MODELS = %s", DAVINCI_MODELS)
    logger.info("  DAVINCI_OUTPUT = %s", DAVINCI_OUTPUT)
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1, log_level="info")
