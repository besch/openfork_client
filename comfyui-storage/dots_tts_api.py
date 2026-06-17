"""
dots.tts FastAPI wrapper.

Provides a small REST service around rednote-hilab/dots.tts for experimental
TTS and zero-shot voice cloning.
"""

import asyncio
import inspect
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import soundfile as sf
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="dots.tts API", version="1.0.0")

INPUT_DIR = Path(os.environ.get("DOTS_TTS_INPUT_DIR", "/app/input"))
OUTPUT_DIR = Path(os.environ.get("DOTS_TTS_OUTPUT_DIR", "/app/output"))
MODEL_CACHE_DIR = Path(
    os.environ.get(
        "DOTS_TTS_MODEL_CACHE_DIR",
        os.environ.get("HF_HOME", "/app/models"),
    )
)

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = os.environ.get("DOTS_TTS_MODEL", "rednote-hilab/dots.tts-mf")
DEFAULT_PRECISION = os.environ.get("DOTS_TTS_PRECISION", "bfloat16")
DEFAULT_NUM_STEPS = int(os.environ.get("DOTS_TTS_NUM_STEPS", "4"))
DEFAULT_GUIDANCE_SCALE = float(os.environ.get("DOTS_TTS_GUIDANCE_SCALE", "1.2"))
DEFAULT_SEED = int(os.environ.get("DOTS_TTS_SEED", "42"))
OPTIMIZE = os.environ.get("DOTS_TTS_OPTIMIZE", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

runtime = None
model_load_lock = threading.Lock()
model_infer_lock = asyncio.Lock()
jobs: Dict[str, Dict[str, Any]] = {}
executor = ThreadPoolExecutor(max_workers=1)


class DotsTTSGenerateRequest(BaseModel):
    text: str
    prompt_audio_path: Optional[str] = None
    prompt_text: Optional[str] = None
    language: Optional[str] = None
    seed: Optional[int] = None
    num_steps: Optional[int] = Field(None, ge=1, le=64)
    guidance_scale: Optional[float] = Field(None, ge=0.0, le=5.0)
    normalize_text: bool = False


class DotsTTSVoiceClonePathRequest(DotsTTSGenerateRequest):
    prompt_audio_path: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    error: Optional[str] = None
    output_path: Optional[str] = None
    model: Optional[str] = None
    seed: Optional[int] = None
    requested_seed: Optional[int] = None
    num_steps: Optional[int] = None
    guidance_scale: Optional[float] = None


def _load_runtime():
    global runtime

    if runtime is not None:
        return runtime

    with model_load_lock:
        if runtime is None:
            logger.info(
                "Loading dots.tts model %s with precision=%s optimize=%s...",
                DEFAULT_MODEL,
                DEFAULT_PRECISION,
                OPTIMIZE,
            )
            from dots_tts.runtime import DotsTtsRuntime

            runtime = DotsTtsRuntime.from_pretrained(
                DEFAULT_MODEL,
                precision=DEFAULT_PRECISION,
                optimize=OPTIMIZE,
            )
            logger.info("dots.tts model loaded")

    return runtime


def _normalized_language(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() in {"none", "auto"}:
        return None
    if normalized.lower() in {"auto_detect", "auto-detect", "autodetect"}:
        return "auto_detect"
    return normalized


def _run_inference(job_id: str, request: DotsTTSGenerateRequest) -> tuple[str, Dict[str, Any]]:
    text = request.text.strip()
    if not text:
        raise ValueError("No text provided for dots.tts generation")

    if request.prompt_audio_path and not Path(request.prompt_audio_path).exists():
        raise FileNotFoundError(f"Reference audio not found: {request.prompt_audio_path}")

    loaded_runtime = _load_runtime()
    output_path = OUTPUT_DIR / f"{job_id}.wav"

    num_steps = request.num_steps or DEFAULT_NUM_STEPS
    guidance_scale = (
        request.guidance_scale
        if request.guidance_scale is not None
        else DEFAULT_GUIDANCE_SCALE
    )
    seed = request.seed if request.seed is not None else DEFAULT_SEED
    requested_seed = request.seed

    kwargs: Dict[str, Any] = {
        "text": text,
        "num_steps": num_steps,
        "guidance_scale": guidance_scale,
    }
    language = _normalized_language(request.language)
    if language:
        kwargs["language"] = language
    if request.normalize_text:
        kwargs["normalize_text"] = True
    if seed is not None:
        kwargs["seed"] = seed
    if request.prompt_audio_path:
        kwargs["prompt_audio_path"] = request.prompt_audio_path
    if request.prompt_text:
        kwargs["prompt_text"] = request.prompt_text

    supported_kwargs = set(inspect.signature(loaded_runtime.generate).parameters)
    unsupported_kwargs = sorted(key for key in kwargs if key not in supported_kwargs)
    if unsupported_kwargs:
        logger.info(
            "Ignoring unsupported dots.tts runtime kwargs for job %s: %s",
            job_id,
            ", ".join(unsupported_kwargs),
        )
    kwargs = {key: value for key, value in kwargs.items() if key in supported_kwargs}

    logger.info(
        "Running dots.tts job %s with model=%s steps=%s guidance=%s language=%s reference=%s",
        job_id,
        DEFAULT_MODEL,
        num_steps,
        guidance_scale,
        language or "none",
        bool(request.prompt_audio_path),
    )
    result = loaded_runtime.generate(**kwargs)

    audio = result["audio"].detach().float().cpu().squeeze().numpy()
    sample_rate = int(result.get("sample_rate") or loaded_runtime.sample_rate)
    sf.write(str(output_path), audio, sample_rate)

    metadata = {
        "model": DEFAULT_MODEL,
        "num_steps": num_steps,
        "guidance_scale": guidance_scale,
        "sample_rate": sample_rate,
    }
    if "seed" in supported_kwargs:
        metadata["seed"] = seed
    elif requested_seed is not None:
        metadata["requested_seed"] = requested_seed

    return str(output_path), metadata


async def _process_generation_job(job_id: str, request: DotsTTSGenerateRequest):
    try:
        jobs[job_id]["status"] = "processing"

        async with model_infer_lock:
            output_path, metadata = await asyncio.get_running_loop().run_in_executor(
                executor,
                _run_inference,
                job_id,
                request,
            )

        jobs[job_id].update(
            {
                "status": "completed",
                "output_path": output_path,
                **metadata,
            }
        )
        logger.info("dots.tts job %s completed", job_id)
    except Exception as exc:
        logger.error("dots.tts job %s failed: %s", job_id, exc, exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)


@app.on_event("startup")
async def startup_event():
    if os.environ.get("DOTS_TTS_PRELOAD", "1") != "0":
        asyncio.create_task(asyncio.to_thread(_load_runtime))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "dots-tts",
        "model": DEFAULT_MODEL,
        "loaded": runtime is not None,
    }


@app.get("/info")
async def info():
    return {
        "model": DEFAULT_MODEL,
        "modes": ["tts", "voice_clone"],
        "num_steps": DEFAULT_NUM_STEPS,
        "guidance_scale": DEFAULT_GUIDANCE_SCALE,
        "optimize": OPTIMIZE,
    }


@app.post("/generate", response_model=JobStatus)
async def generate(request: DotsTTSGenerateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "error": None, "output_path": None}
    background_tasks.add_task(_process_generation_job, job_id, request)
    return JobStatus(job_id=job_id, status="pending")


@app.post("/generate/voice-clone", response_model=JobStatus)
async def generate_voice_clone(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    prompt_text: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    seed: Optional[int] = Form(None),
    num_steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    normalize_text: bool = Form(False),
    prompt_audio: UploadFile = File(...),
):
    job_id = str(uuid.uuid4())
    suffix = Path(prompt_audio.filename or "").suffix or ".wav"
    suffix = suffix[:16]
    prompt_audio_path = INPUT_DIR / f"{job_id}_prompt{suffix}"

    with open(prompt_audio_path, "wb") as handle:
        handle.write(await prompt_audio.read())

    request = DotsTTSGenerateRequest(
        text=text,
        prompt_audio_path=str(prompt_audio_path),
        prompt_text=prompt_text,
        language=language,
        seed=seed,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        normalize_text=normalize_text,
    )
    jobs[job_id] = {
        "status": "pending",
        "error": None,
        "output_path": None,
        "prompt_audio_path": str(prompt_audio_path),
    }
    background_tasks.add_task(_process_generation_job, job_id, request)
    return JobStatus(job_id=job_id, status="pending")


@app.post("/generate/voice-clone-path", response_model=JobStatus)
async def generate_voice_clone_path(
    request: DotsTTSVoiceClonePathRequest,
    background_tasks: BackgroundTasks,
):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "error": None, "output_path": None}
    background_tasks.add_task(_process_generation_job, job_id, request)
    return JobStatus(job_id=job_id, status="pending")


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        error=job.get("error"),
        output_path=job.get("output_path"),
        model=job.get("model"),
        seed=job.get("seed"),
        requested_seed=job.get("requested_seed"),
        num_steps=job.get("num_steps"),
        guidance_scale=job.get("guidance_scale"),
    )


@app.get("/download/{job_id}")
async def download(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status is {job['status']}")

    output_path = job.get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(output_path, media_type="audio/wav", filename=f"{job_id}.wav")


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        return {"status": "not_found"}

    job = jobs.pop(job_id)
    for key in ("output_path", "prompt_audio_path"):
        path = job.get(key)
        if path and Path(path).exists():
            try:
                Path(path).unlink()
            except OSError:
                pass

    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting dots.tts API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
