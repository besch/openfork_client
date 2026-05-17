#!/usr/bin/env python3
"""
DramaBox FastAPI wrapper.

Provides the same REST job contract used by the OpenFork client:
POST /generate, POST /generate/voice-clone, GET /status/{id},
GET /download/{id}, and DELETE /job/{id}.
"""

import logging
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel


APP_DIR = Path("/app")
SRC_DIR = APP_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference_server import TTSServer  # noqa: E402
from model_downloader import get_all_paths  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dramabox-api")

app = FastAPI(title="DramaBox API", version="1.0.0")

INPUT_DIR = Path(os.environ.get("DRAMABOX_INPUT_DIR", "/app/input"))
OUTPUT_DIR = Path(os.environ.get("DRAMABOX_OUTPUT_DIR", "/app/output"))
MODEL_CACHE_DIR = os.environ.get("DRAMABOX_MODEL_CACHE")
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()
model_lock = threading.Lock()
generation_lock = threading.Lock()
tts_server: Optional[TTSServer] = None


class DramaboxRequest(BaseModel):
    prompt: str
    cfg_scale: float = 2.5
    stg_scale: float = 1.5
    duration_multiplier: float = 1.1
    gen_duration: float = 0.0
    ref_duration: float = 10.0
    rescale_scale: str = "auto"
    seed: int = 42
    watermark: bool = True


def get_server() -> TTSServer:
    global tts_server

    if tts_server is not None:
        return tts_server

    with model_lock:
        if tts_server is not None:
            return tts_server

        logger.info("Fetching DramaBox checkpoints from HuggingFace cache...")
        paths = get_all_paths(cache_dir=MODEL_CACHE_DIR)

        logger.info("Loading DramaBox TTSServer...")
        tts_server = TTSServer(
            checkpoint=paths["transformer"],
            full_checkpoint=paths["audio_components"],
            gemma_root=paths["gemma_root"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=os.environ.get("LTX_DTYPE", "bf16"),
            compile_model=os.environ.get("DRAMABOX_COMPILE", "false").lower()
            in {"1", "true", "yes"},
            bnb_4bit=os.environ.get("DRAMABOX_BNB_4BIT", "true").lower()
            in {"1", "true", "yes"},
        )
        logger.info("DramaBox TTSServer ready.")
        return tts_server


def set_job_status(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        existing = jobs.get(job_id, {})
        existing.update(updates)
        jobs[job_id] = existing


def process_generation(job_id: str, request: DramaboxRequest, voice_ref: Optional[str]) -> None:
    output_path = OUTPUT_DIR / f"{job_id}.wav"
    started = time.time()

    try:
        set_job_status(job_id, status="processing", started_at=started)
        server = get_server()

        with generation_lock:
            server.generate_to_file(
                prompt=request.prompt,
                output=str(output_path),
                voice_ref=voice_ref,
                cfg_scale=request.cfg_scale,
                stg_scale=request.stg_scale,
                duration_multiplier=request.duration_multiplier,
                seed=int(request.seed),
                ref_duration=request.ref_duration,
                rescale_scale=request.rescale_scale,
                gen_duration=request.gen_duration,
                watermark=request.watermark,
            )

        set_job_status(
            job_id,
            status="completed",
            output_path=str(output_path),
            processing_seconds=time.time() - started,
        )
    except Exception as exc:
        logger.exception("DramaBox job %s failed", job_id)
        set_job_status(job_id, status="failed", error=str(exc))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "loaded": tts_server is not None,
        "cuda": torch.cuda.is_available(),
    }


@app.post("/generate")
async def generate(request: DramaboxRequest, background_tasks: BackgroundTasks):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    job_id = str(uuid.uuid4())
    set_job_status(job_id, status="queued")
    background_tasks.add_task(process_generation, job_id, request, None)
    return {"job_id": job_id, "status": "queued"}


@app.post("/generate/voice-clone")
async def generate_voice_clone(
    background_tasks: BackgroundTasks,
    voice_ref: UploadFile = File(...),
    prompt: str = Form(...),
    cfg_scale: float = Form(2.5),
    stg_scale: float = Form(1.5),
    duration_multiplier: float = Form(1.1),
    gen_duration: float = Form(0.0),
    ref_duration: float = Form(10.0),
    rescale_scale: str = Form("auto"),
    seed: int = Form(42),
    watermark: bool = Form(True),
):
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    job_id = str(uuid.uuid4())
    suffix = Path(voice_ref.filename or "reference.wav").suffix or ".wav"
    ref_path = INPUT_DIR / f"{job_id}_reference{suffix}"
    with ref_path.open("wb") as handle:
        shutil.copyfileobj(voice_ref.file, handle)

    request = DramaboxRequest(
        prompt=prompt,
        cfg_scale=cfg_scale,
        stg_scale=stg_scale,
        duration_multiplier=duration_multiplier,
        gen_duration=gen_duration,
        ref_duration=ref_duration,
        rescale_scale=rescale_scale,
        seed=seed,
        watermark=watermark,
    )

    set_job_status(job_id, status="queued", reference_path=str(ref_path))
    background_tasks.add_task(process_generation, job_id, request, str(ref_path))
    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
async def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **job}


@app.get("/download/{job_id}")
async def download(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "completed" or not job.get("output_path"):
        raise HTTPException(status_code=400, detail="Job not completed")

    output_path = Path(job["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(str(output_path), media_type="audio/wav", filename=f"{job_id}.wav")


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    with jobs_lock:
        job = jobs.pop(job_id, None)
    if not job:
        return {"status": "not_found"}

    for path_key in ("output_path", "reference_path"):
        path = job.get(path_key)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
