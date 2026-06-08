"""
WavTTS FastAPI wrapper.

Exposes the WavTTS command-line inference path through the same REST shape used
by OpenFork's F5-TTS service: default-reference TTS and reference-audio voice
cloning.
"""

import asyncio
import logging
import os
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WavTTS API", version="1.0.0")

INPUT_DIR = Path(os.environ.get("WAVTTS_INPUT_DIR", "/app/input"))
OUTPUT_DIR = Path(os.environ.get("WAVTTS_OUTPUT_DIR", "/app/output"))
REPO_DIR = Path(os.environ.get("WAVTTS_REPO_DIR", "/opt/WavTTS"))

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = os.environ.get("WAVTTS_MODEL", "WavTTS")
DEFAULT_REF_AUDIO = os.environ.get(
    "WAVTTS_DEFAULT_REF_AUDIO",
    str(REPO_DIR / "src/wavtts/infer/examples/basic_ref_en.wav"),
)
DEFAULT_REF_TEXT = os.environ.get(
    "WAVTTS_DEFAULT_REF_TEXT",
    "Some call me nature, others call me mother nature.",
)
DEFAULT_NFE_STEP = int(os.environ.get("WAVTTS_NFE_STEP", "50"))
DEFAULT_CFG_STRENGTH = float(os.environ.get("WAVTTS_CFG_STRENGTH", "3.0"))
DEFAULT_SPEED = float(os.environ.get("WAVTTS_SPEED", "1.0"))
DEFAULT_SHIFT = float(os.environ.get("WAVTTS_SHIFT", "3.0"))
CLI_TIMEOUT_SECONDS = int(os.environ.get("WAVTTS_CLI_TIMEOUT_SECONDS", "1800"))

model_infer_lock = asyncio.Lock()
jobs: Dict[str, Dict[str, Any]] = {}
executor = ThreadPoolExecutor(max_workers=1)


class WavTTSGenerateRequest(BaseModel):
    text: str
    ref_audio_path: Optional[str] = None
    ref_text: Optional[str] = None
    speed: float = Field(DEFAULT_SPEED, ge=0.1, le=3.0)
    seed: Optional[int] = None
    nfe_step: Optional[int] = Field(None, ge=4, le=80)
    cfg_strength: Optional[float] = Field(None, ge=0.0, le=10.0)
    shift: Optional[float] = Field(None, ge=0.0, le=10.0)
    remove_silence: bool = False


class WavTTSVoiceClonePathRequest(WavTTSGenerateRequest):
    ref_audio_path: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    error: Optional[str] = None
    output_path: Optional[str] = None
    seed: Optional[int] = None


def _resolve_reference(request: WavTTSGenerateRequest) -> tuple[str, str]:
    ref_audio_path = request.ref_audio_path or DEFAULT_REF_AUDIO
    if not ref_audio_path or not Path(ref_audio_path).exists():
        raise FileNotFoundError(
            "WavTTS needs reference audio and no packaged default reference was found"
        )

    ref_text = request.ref_text
    if not request.ref_audio_path and not ref_text:
        ref_text = DEFAULT_REF_TEXT

    return ref_audio_path, ref_text or ""


def _run_inference(job_id: str, request: WavTTSGenerateRequest) -> tuple[str, Optional[int]]:
    text = request.text.strip()
    if not text:
        raise ValueError("No text provided for WavTTS generation")

    ref_audio_path, ref_text = _resolve_reference(request)
    output_path = OUTPUT_DIR / f"{job_id}.wav"

    cmd = [
        "wavtts_infer-cli",
        "--model",
        DEFAULT_MODEL,
        "--ref_audio",
        ref_audio_path,
        "--ref_text",
        ref_text,
        "--gen_text",
        text,
        "--output_dir",
        str(OUTPUT_DIR),
        "--output_file",
        output_path.name,
        "--nfe_step",
        str(request.nfe_step or DEFAULT_NFE_STEP),
        "--cfg_strength",
        str(
            request.cfg_strength
            if request.cfg_strength is not None
            else DEFAULT_CFG_STRENGTH
        ),
        "--shift",
        str(request.shift if request.shift is not None else DEFAULT_SHIFT),
        "--speed",
        str(request.speed),
    ]

    if request.remove_silence:
        cmd.append("--remove_silence")
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/app/models")
    env.setdefault("HUGGINGFACE_HUB_CACHE", "/app/models")
    env.setdefault("TRANSFORMERS_CACHE", "/app/models")
    src_path = str(REPO_DIR / "src")
    env["PYTHONPATH"] = (
        f"{src_path}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_path
    )

    logger.info("Running WavTTS job %s with command: %s", job_id, " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(REPO_DIR),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=CLI_TIMEOUT_SECONDS,
        check=False,
    )
    if result.stdout:
        logger.info("WavTTS output for %s:\n%s", job_id, result.stdout[-4000:])
    if result.returncode != 0:
        raise RuntimeError(f"WavTTS CLI failed with exit code {result.returncode}")

    if not output_path.exists():
        candidates = sorted(OUTPUT_DIR.glob(f"{job_id}*"))
        if candidates:
            candidates[0].rename(output_path)

    if not output_path.exists():
        raise RuntimeError("WavTTS completed but did not create an output file")

    return str(output_path), request.seed


async def _process_generation_job(job_id: str, request: WavTTSGenerateRequest):
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
        logger.info("WavTTS job %s completed", job_id)
    except Exception as exc:
        logger.error("WavTTS job %s failed: %s", job_id, exc, exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "wavtts",
        "default_ref_audio": Path(DEFAULT_REF_AUDIO).exists(),
    }


@app.get("/info")
async def info():
    return {
        "model": DEFAULT_MODEL,
        "modes": ["tts", "voice_clone"],
        "default_ref_audio": Path(DEFAULT_REF_AUDIO).exists(),
        "nfe_step": DEFAULT_NFE_STEP,
        "cfg_strength": DEFAULT_CFG_STRENGTH,
        "shift": DEFAULT_SHIFT,
    }


@app.post("/generate", response_model=JobStatus)
async def generate(request: WavTTSGenerateRequest, background_tasks: BackgroundTasks):
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
    shift: Optional[float] = Form(None),
    remove_silence: bool = Form(False),
    ref_audio: UploadFile = File(...),
):
    job_id = str(uuid.uuid4())
    suffix = Path(ref_audio.filename or "").suffix or ".wav"
    suffix = suffix[:16]
    ref_audio_path = INPUT_DIR / f"{job_id}_ref{suffix}"

    with open(ref_audio_path, "wb") as handle:
        handle.write(await ref_audio.read())

    request = WavTTSGenerateRequest(
        text=text,
        ref_audio_path=str(ref_audio_path),
        ref_text=ref_text,
        speed=speed,
        seed=seed,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        shift=shift,
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
    request: WavTTSVoiceClonePathRequest,
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

    logger.info("Starting WavTTS API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
