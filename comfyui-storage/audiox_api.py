"""
AudioX REST API wrapper.

Runs the HKUSTAudio/AudioX model behind the same small REST contract used by
OpenFork's other non-ComfyUI audio services.
"""

import gc
import logging
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Dict, Optional

import torch
import torchaudio
from einops import rearrange
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from audiox import get_pretrained_model
from audiox.data.utils import read_video, load_and_process_audio
from audiox.inference.generation import generate_diffusion_cond


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenFork AudioX API")

INPUT_DIR = Path(os.environ.get("AUDIOX_INPUT_DIR", "/app/input"))
OUTPUT_DIR = Path(os.environ.get("AUDIOX_OUTPUT_DIR", "/app/output"))
MODEL_ID = os.environ.get("AUDIOX_MODEL_ID", "HKUSTAudio/AudioX")
MAX_DURATION_SECONDS = int(os.environ.get("AUDIOX_MAX_DURATION_SECONDS", "10"))
DEFAULT_STEPS = int(os.environ.get("AUDIOX_DEFAULT_STEPS", "100"))
DEFAULT_CFG_SCALE = float(os.environ.get("AUDIOX_DEFAULT_CFG_SCALE", "7.0"))
DEFAULT_SIGMA_MIN = float(os.environ.get("AUDIOX_DEFAULT_SIGMA_MIN", "0.03"))
DEFAULT_SIGMA_MAX = float(os.environ.get("AUDIOX_DEFAULT_SIGMA_MAX", "500"))
DEFAULT_SAMPLER = os.environ.get("AUDIOX_DEFAULT_SAMPLER", "dpmpp-3m-sde")
AUDIOX_CONDITIONING_SECONDS = float(os.environ.get("AUDIOX_CONDITIONING_SECONDS", "10"))
USE_MODEL_HALF = os.environ.get("AUDIOX_MODEL_HALF", "true").lower() in {
    "1",
    "true",
    "yes",
}

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class TextGenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    duration: float = 10.0
    seed: int = -1
    cfg_scale: float = DEFAULT_CFG_SCALE
    steps: int = DEFAULT_STEPS
    sampler_type: str = DEFAULT_SAMPLER
    sigma_min: float = DEFAULT_SIGMA_MIN
    sigma_max: float = DEFAULT_SIGMA_MAX


jobs: Dict[str, Dict[str, Optional[str]]] = {}
jobs_lock = threading.Lock()
model_lock = threading.Lock()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Loading AudioX model %s on %s", MODEL_ID, device)
model, model_config = get_pretrained_model(MODEL_ID)
model = model.to(device).eval().requires_grad_(False)
if USE_MODEL_HALF and device.type == "cuda":
    logger.info("Using float16 model weights for AudioX")
    model = model.to(torch.float16)

sample_rate = int(model_config["sample_rate"])
sample_size = int(model_config["sample_size"])
target_fps = int(model_config.get("video_fps", 5))
model_type = model_config.get("model_type", "diffusion_cond")
if model_type != "diffusion_cond":
    raise RuntimeError(f"Unsupported AudioX model type: {model_type}")

logger.info(
    "AudioX ready: sample_rate=%s sample_size=%s target_fps=%s",
    sample_rate,
    sample_size,
    target_fps,
)


def _set_job(job_id: str, **updates) -> None:
    with jobs_lock:
        jobs.setdefault(job_id, {}).update(updates)


def _clamp_duration(duration: float) -> float:
    try:
        value = float(duration)
    except (TypeError, ValueError):
        value = 10.0
    return max(1.0, min(float(MAX_DURATION_SECONDS), value))


def _model_window_seconds() -> float:
    return sample_size / sample_rate


def _conditioning_window_seconds() -> float:
    # AudioX's cross-conditioning path expects the prompt timeline to use the
    # training window, not necessarily the raw sample_size / sample_rate value.
    # Some checkpoints report a slightly longer audio sample window, which can
    # create 55 video conditioning frames against a 50-frame diffusion timeline.
    model_window = _model_window_seconds()
    requested_window = max(1.0, min(float(MAX_DURATION_SECONDS), AUDIOX_CONDITIONING_SECONDS))
    if requested_window > model_window:
        logger.info(
            "AudioX conditioning window %.2fs clipped to model window %.2fs",
            requested_window,
            model_window,
        )
        return model_window
    return requested_window


def _conditioning_tensor_dtype() -> torch.dtype:
    if USE_MODEL_HALF and device.type == "cuda":
        return torch.float16
    return torch.float32


def _normalize_audio(audio: torch.Tensor, duration: float) -> torch.Tensor:
    audio = rearrange(audio, "b d n -> d (b n)")
    target_samples = int(duration * sample_rate)
    audio = audio[:, :target_samples]
    peak = torch.max(torch.abs(audio))
    if peak > 0:
        audio = audio.to(torch.float32).div(peak)
    return audio.clamp(-1, 1).mul(32767).to(torch.int16).cpu()


def _generate_audiox(
    job_id: str,
    prompt: str,
    negative_prompt: Optional[str],
    video_path: Optional[Path],
    duration: float,
    seed: int,
    cfg_scale: float,
    steps: int,
    sampler_type: str,
    sigma_min: float,
    sigma_max: float,
) -> None:
    _set_job(job_id, status="processing", error=None)
    output_path = OUTPUT_DIR / f"{job_id}.wav"

    try:
        requested_duration = _clamp_duration(duration)
        model_duration = _model_window_seconds()
        conditioning_duration = _conditioning_window_seconds()
        conditioning_dtype = _conditioning_tensor_dtype()
        steps = max(1, min(500, int(steps)))
        seed = int(seed)

        with model_lock:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            if video_path:
                video_tensor = read_video(
                    str(video_path),
                    seek_time=0,
                    duration=conditioning_duration,
                    target_fps=target_fps,
                )
            else:
                video_tensor = torch.zeros(
                    (int(conditioning_duration * target_fps), 3, 224, 224),
                    dtype=conditioning_dtype,
                )
            video_tensor = video_tensor.to(dtype=conditioning_dtype)
            audio_tensor = load_and_process_audio(
                None,
                sample_rate,
                seconds_start=0,
                seconds_total=conditioning_duration,
            ).to(device=device, dtype=conditioning_dtype)
            video_prompt = {
                "video_tensors": video_tensor.unsqueeze(0),
                "video_sync_frames": torch.zeros(
                    1,
                    240,
                    768,
                    device=device,
                    dtype=conditioning_dtype,
                ),
            }

            logger.info(
                "AudioX job %s requested %.2fs; using %.2fs prompt window and %.2fs diffusion window",
                job_id,
                requested_duration,
                conditioning_duration,
                model_duration,
            )

            conditioning = [
                {
                    "video_prompt": video_prompt,
                    "text_prompt": prompt or "",
                    "audio_prompt": audio_tensor.unsqueeze(0),
                    "seconds_start": 0,
                    "seconds_total": conditioning_duration,
                }
            ]

            negative_conditioning = None
            if negative_prompt:
                negative_conditioning = [
                    {
                        "video_prompt": video_prompt,
                        "text_prompt": negative_prompt,
                        "audio_prompt": audio_tensor.unsqueeze(0),
                        "seconds_start": 0,
                        "seconds_total": conditioning_duration,
                    }
                ]

            with torch.inference_mode():
                audio = generate_diffusion_cond(
                    model,
                    conditioning=conditioning,
                    negative_conditioning=negative_conditioning,
                    steps=steps,
                    cfg_scale=float(cfg_scale),
                    batch_size=1,
                    sample_size=sample_size,
                    sample_rate=sample_rate,
                    seed=seed,
                    device=device,
                    sampler_type=sampler_type,
                    sigma_min=float(sigma_min),
                    sigma_max=float(sigma_max),
                    init_audio=None,
                    init_noise_level=0.1,
                    mask_args=None,
                    scale_phi=0.0,
                )

            audio = _normalize_audio(audio, requested_duration)
            torchaudio.save(str(output_path), audio, sample_rate)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        _set_job(job_id, status="completed", output_path=str(output_path))
        logger.info("AudioX job %s completed: %s", job_id, output_path)
    except Exception as exc:
        logger.exception("AudioX job %s failed", job_id)
        _set_job(job_id, status="failed", error=str(exc))
    finally:
        if video_path and video_path.exists():
            try:
                video_path.unlink()
            except OSError:
                pass


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "audiox",
        "device": str(device),
        "sample_rate": sample_rate,
        "model_window_seconds": _model_window_seconds(),
        "max_duration_seconds": MAX_DURATION_SECONDS,
    }


@app.post("/generate-text")
async def generate_text(request: TextGenerateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _set_job(job_id, status="queued", output_path=None, error=None)
    background_tasks.add_task(
        _generate_audiox,
        job_id,
        request.prompt,
        request.negative_prompt,
        None,
        request.duration,
        request.seed,
        request.cfg_scale,
        request.steps,
        request.sampler_type,
        request.sigma_min,
        request.sigma_max,
    )
    return {"job_id": job_id, "status": "queued"}


@app.post("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    video: Optional[UploadFile] = File(None),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    duration: float = Form(10.0),
    seed: int = Form(-1),
    cfg_scale: float = Form(DEFAULT_CFG_SCALE),
    num_steps: int = Form(DEFAULT_STEPS),
    sampler_type: str = Form(DEFAULT_SAMPLER),
    sigma_min: float = Form(DEFAULT_SIGMA_MIN),
    sigma_max: float = Form(DEFAULT_SIGMA_MAX),
):
    job_id = str(uuid.uuid4())
    video_path = None
    if video is not None:
        suffix = Path(video.filename or "input.mp4").suffix or ".mp4"
        video_path = INPUT_DIR / f"{job_id}{suffix}"
        with video_path.open("wb") as handle:
            shutil.copyfileobj(video.file, handle)

    _set_job(job_id, status="queued", output_path=None, error=None)
    background_tasks.add_task(
        _generate_audiox,
        job_id,
        prompt,
        negative_prompt or None,
        video_path,
        duration,
        seed,
        cfg_scale,
        num_steps,
        sampler_type,
        sigma_min,
        sigma_max,
    )
    return {"job_id": job_id, "status": "queued"}


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
        filename=f"audiox_{job_id}.wav",
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
