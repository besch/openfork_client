"""
Stable Audio 3 Small-SFX REST API wrapper.

This exposes Stability AI's Stable Audio 3 sound-effects checkpoint through the
same small REST contract used by OpenFork's other non-ComfyUI audio services.
"""

import gc
import logging
import os
import random
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torchaudio
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from stable_audio_3 import StableAudioModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenFork Stable Audio 3 SFX API")

OUTPUT_DIR = Path(os.environ.get("STABLE_AUDIO3_OUTPUT_DIR", "/app/output"))
MODEL_ID = os.environ.get("STABLE_AUDIO3_MODEL_ID", "small-sfx")
DEVICE = os.environ.get(
    "STABLE_AUDIO3_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)
MAX_DURATION_SECONDS = int(os.environ.get("STABLE_AUDIO3_MAX_DURATION_SECONDS", "120"))
DEFAULT_DURATION_SECONDS = float(os.environ.get("STABLE_AUDIO3_DEFAULT_DURATION", "8"))
DEFAULT_STEPS = int(os.environ.get("STABLE_AUDIO3_DEFAULT_STEPS", "8"))
DEFAULT_CFG_SCALE = float(os.environ.get("STABLE_AUDIO3_DEFAULT_CFG_SCALE", "1.0"))
MAX_STEPS = int(os.environ.get("STABLE_AUDIO3_MAX_STEPS", "50"))
USE_MODEL_HALF = os.environ.get("STABLE_AUDIO3_MODEL_HALF", "true").lower() in {
    "1",
    "true",
    "yes",
} and DEVICE.startswith("cuda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    duration: float = DEFAULT_DURATION_SECONDS
    seed: int = -1
    steps: int = DEFAULT_STEPS
    cfg_scale: float = DEFAULT_CFG_SCALE
    chunked_decode: Optional[bool] = None


jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()
model_lock = threading.Lock()

logger.info(
    "Loading Stable Audio 3 model %s on %s (half=%s)",
    MODEL_ID,
    DEVICE,
    USE_MODEL_HALF,
)
model = StableAudioModel.from_pretrained(
    MODEL_ID,
    device=DEVICE,
    model_half=USE_MODEL_HALF,
)
sample_rate = int(model.model.sample_rate)
logger.info("Stable Audio 3 ready: model=%s sample_rate=%s", MODEL_ID, sample_rate)


def _set_job(job_id: str, **updates) -> None:
    with jobs_lock:
        jobs.setdefault(job_id, {}).update(updates)


def _clamp_float(value, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _clamp_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _normalize_seed(seed) -> int:
    try:
        parsed = int(seed)
    except (TypeError, ValueError):
        parsed = -1
    if parsed < 0:
        return random.randint(0, 99999)
    return parsed


def _generate_stable_audio(
    job_id: str,
    prompt: str,
    negative_prompt: Optional[str],
    duration: float,
    seed: int,
    steps: int,
    cfg_scale: float,
    chunked_decode: Optional[bool],
) -> None:
    duration = _clamp_float(
        duration, DEFAULT_DURATION_SECONDS, 1.0, MAX_DURATION_SECONDS
    )
    steps = _clamp_int(steps, DEFAULT_STEPS, 1, MAX_STEPS)
    cfg_scale = _clamp_float(cfg_scale, DEFAULT_CFG_SCALE, 0.0, 25.0)
    seed = _normalize_seed(seed)
    output_path = OUTPUT_DIR / f"{job_id}.wav"

    _set_job(
        job_id,
        status="processing",
        error=None,
        model=MODEL_ID,
        duration=duration,
        seed=seed,
        steps=steps,
        cfg_scale=cfg_scale,
    )

    try:
        with model_lock:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            logger.info(
                "Stable Audio 3 job %s: duration=%.2fs steps=%s cfg=%.2f seed=%s",
                job_id,
                duration,
                steps,
                cfg_scale,
                seed,
            )
            with torch.inference_mode():
                audio = model.generate(
                    prompt=prompt or "",
                    negative_prompt=negative_prompt,
                    duration=duration,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    batch_size=1,
                    seed=seed,
                    chunked_decode=chunked_decode,
                )

            waveform = audio[0].detach().cpu().to(torch.float32).clamp(-1, 1)
            torchaudio.save(str(output_path), waveform, sample_rate)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        _set_job(job_id, status="completed", output_path=str(output_path))
        logger.info("Stable Audio 3 job %s completed: %s", job_id, output_path)
    except Exception as exc:
        logger.exception("Stable Audio 3 job %s failed", job_id)
        _set_job(job_id, status="failed", error=str(exc))


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "stable-audio-3-sfx",
        "model": MODEL_ID,
        "device": DEVICE,
        "sample_rate": sample_rate,
        "max_duration_seconds": MAX_DURATION_SECONDS,
        "default_steps": DEFAULT_STEPS,
    }


@app.post("/generate")
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _set_job(job_id, status="queued", output_path=None, error=None)
    background_tasks.add_task(
        _generate_stable_audio,
        job_id,
        request.prompt,
        request.negative_prompt,
        request.duration,
        request.seed,
        request.steps,
        request.cfg_scale,
        request.chunked_decode,
    )
    return {"job_id": job_id, "status": "queued"}


@app.post("/generate-text")
async def generate_text(request: GenerateRequest, background_tasks: BackgroundTasks):
    return await generate(request, background_tasks)


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {key: value for key, value in job.items() if key != "output_path"}


@app.get("/download/{job_id}")
async def download(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "completed" or not job.get("output_path"):
        raise HTTPException(status_code=400, detail="Job not completed")
    return FileResponse(
        job["output_path"],
        media_type="audio/wav",
        filename=f"stable_audio_3_sfx_{job_id}.wav",
    )


@app.delete("/job/{job_id}")
async def cleanup_job(job_id: str):
    with jobs_lock:
        job = jobs.pop(job_id, None)
    if job and job.get("output_path"):
        try:
            Path(job["output_path"]).unlink(missing_ok=True)
        except OSError:
            pass
    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
