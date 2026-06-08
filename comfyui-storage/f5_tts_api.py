"""
F5-TTS FastAPI wrapper.

Provides a small REST service around SWivid/F5-TTS for default-voice TTS and
reference-audio voice cloning.
"""

import asyncio
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, Optional

import soundfile as sf
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="F5-TTS API", version="1.0.0")

INPUT_DIR = Path(os.environ.get("F5_INPUT_DIR", "/app/input"))
OUTPUT_DIR = Path(os.environ.get("F5_OUTPUT_DIR", "/app/output"))
MODEL_CACHE_DIR = Path(
    os.environ.get(
        "F5_MODEL_CACHE_DIR",
        os.environ.get("HF_HOME", "/app/models"),
    )
)

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = os.environ.get("F5_TTS_MODEL", "F5TTS_v1_Base")
DEFAULT_REF_TEXT = os.environ.get(
    "F5_DEFAULT_REF_TEXT",
    "Some call me nature, others call me mother nature.",
)
DEFAULT_NFE_STEP = int(os.environ.get("F5_NFE_STEP", "32"))
DEFAULT_CFG_STRENGTH = float(os.environ.get("F5_CFG_STRENGTH", "2.0"))
DEFAULT_SPEED = float(os.environ.get("F5_SPEED", "1.0"))

f5_model = None
model_load_lock = threading.Lock()
model_infer_lock = asyncio.Lock()
jobs: Dict[str, Dict[str, Any]] = {}
executor = ThreadPoolExecutor(max_workers=1)


class F5GenerateRequest(BaseModel):
    text: str
    ref_audio_path: Optional[str] = None
    ref_text: Optional[str] = None
    speed: float = Field(DEFAULT_SPEED, ge=0.1, le=3.0)
    seed: Optional[int] = None
    nfe_step: Optional[int] = Field(None, ge=4, le=64)
    cfg_strength: Optional[float] = Field(None, ge=0.0, le=10.0)
    remove_silence: bool = False


class F5VoiceClonePathRequest(F5GenerateRequest):
    ref_audio_path: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    error: Optional[str] = None
    output_path: Optional[str] = None
    seed: Optional[int] = None


def _default_reference_audio() -> str:
    configured = os.environ.get("F5_DEFAULT_REF_AUDIO")
    if configured and Path(configured).exists():
        return configured

    try:
        packaged = files("f5_tts").joinpath(
            "infer/examples/basic/basic_ref_en.wav",
        )
        return str(packaged)
    except Exception as exc:
        logger.warning("Could not resolve packaged F5 reference audio: %s", exc)
        return ""


def _load_model():
    global f5_model

    if f5_model is not None:
        return f5_model

    with model_load_lock:
        if f5_model is None:
            logger.info("Loading F5-TTS model %s...", DEFAULT_MODEL)
            from f5_tts.api import F5TTS

            f5_model = F5TTS(
                model=DEFAULT_MODEL,
                hf_cache_dir=str(MODEL_CACHE_DIR),
            )
            logger.info("F5-TTS model loaded")

    return f5_model


def _resolve_reference(request: F5GenerateRequest) -> tuple[str, str]:
    ref_audio_path = request.ref_audio_path
    ref_text = request.ref_text

    if ref_audio_path:
        if not Path(ref_audio_path).exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")
        return ref_audio_path, ref_text or ""

    default_ref_audio = _default_reference_audio()
    if not default_ref_audio or not Path(default_ref_audio).exists():
        raise FileNotFoundError(
            "F5-TTS needs reference audio and no packaged default reference was found"
        )

    return default_ref_audio, ref_text or DEFAULT_REF_TEXT


def _run_inference(job_id: str, request: F5GenerateRequest) -> tuple[str, Optional[int]]:
    text = request.text.strip()
    if not text:
        raise ValueError("No text provided for F5-TTS generation")

    ref_audio_path, ref_text = _resolve_reference(request)
    model = _load_model()
    output_path = OUTPUT_DIR / f"{job_id}.wav"

    logger.info("Running F5-TTS job %s", job_id)
    wav, sample_rate, _ = model.infer(
        ref_file=ref_audio_path,
        ref_text=ref_text,
        gen_text=text,
        show_info=logger.info,
        progress=None,
        nfe_step=request.nfe_step or DEFAULT_NFE_STEP,
        cfg_strength=(
            request.cfg_strength
            if request.cfg_strength is not None
            else DEFAULT_CFG_STRENGTH
        ),
        speed=request.speed,
        remove_silence=request.remove_silence,
        file_wave=str(output_path),
        seed=request.seed,
    )

    if not output_path.exists():
        if wav is None:
            raise RuntimeError("F5-TTS returned no audio")
        sf.write(str(output_path), wav, sample_rate)

    return str(output_path), getattr(model, "seed", request.seed)


async def _process_generation_job(job_id: str, request: F5GenerateRequest):
    try:
        jobs[job_id]["status"] = "processing"

        async with model_infer_lock:
            output_path, seed = await asyncio.get_running_loop().run_in_executor(
                executor,
                _run_inference,
                job_id,
                request,
            )

        jobs[job_id].update(
            {
                "status": "completed",
                "output_path": output_path,
                "seed": seed,
            }
        )
        logger.info("F5-TTS job %s completed", job_id)
    except Exception as exc:
        logger.error("F5-TTS job %s failed: %s", job_id, exc, exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)


@app.on_event("startup")
async def startup_event():
    if os.environ.get("F5_PRELOAD", "1") != "0":
        asyncio.create_task(asyncio.to_thread(_load_model))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "f5-tts"}


@app.get("/info")
async def info():
    return {
        "model": DEFAULT_MODEL,
        "modes": ["tts", "voice_clone"],
        "default_ref_audio": bool(_default_reference_audio()),
        "nfe_step": DEFAULT_NFE_STEP,
        "cfg_strength": DEFAULT_CFG_STRENGTH,
    }


@app.post("/generate", response_model=JobStatus)
async def generate(request: F5GenerateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "error": None, "output_path": None}
    background_tasks.add_task(_process_generation_job, job_id, request)
    return JobStatus(job_id=job_id, status="pending")


@app.post("/generate/voice-clone", response_model=JobStatus)
async def generate_voice_clone(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    ref_text: Optional[str] = Form(None),
    speed: float = Form(DEFAULT_SPEED),
    seed: Optional[int] = Form(None),
    nfe_step: Optional[int] = Form(None),
    cfg_strength: Optional[float] = Form(None),
    remove_silence: bool = Form(False),
    ref_audio: UploadFile = File(...),
):
    job_id = str(uuid.uuid4())
    suffix = Path(ref_audio.filename or "").suffix or ".wav"
    suffix = suffix[:16]
    ref_audio_path = INPUT_DIR / f"{job_id}_ref{suffix}"

    with open(ref_audio_path, "wb") as handle:
        handle.write(await ref_audio.read())

    request = F5GenerateRequest(
        text=text,
        ref_audio_path=str(ref_audio_path),
        ref_text=ref_text,
        speed=speed,
        seed=seed,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        remove_silence=remove_silence,
    )
    jobs[job_id] = {
        "status": "pending",
        "error": None,
        "output_path": None,
        "ref_audio_path": str(ref_audio_path),
    }
    background_tasks.add_task(_process_generation_job, job_id, request)
    return JobStatus(job_id=job_id, status="pending")


@app.post("/generate/voice-clone-path", response_model=JobStatus)
async def generate_voice_clone_path(
    request: F5VoiceClonePathRequest,
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
        seed=job.get("seed"),
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
    for key in ("output_path", "ref_audio_path"):
        path = job.get(key)
        if path and Path(path).exists():
            try:
                Path(path).unlink()
            except OSError:
                pass

    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting F5-TTS API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
