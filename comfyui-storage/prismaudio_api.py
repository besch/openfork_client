#!/usr/bin/env python3
"""
Prism Audio (PRiSM) REST API Server for OpenFork DGN Client
Provides an HTTP API for video-to-audio synthesis using PRiSM.
Accepts video file upload + prompt, returns generated audio.
"""

import os
import sys
import uuid
import logging
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import torch
import torchaudio
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Job storage
jobs = {}

# Directories
WORK_DIR = Path("/app")
OUTPUT_DIR = WORK_DIR / "output"
INPUT_DIR = WORK_DIR / "input"
CKPT_DIR = WORK_DIR / "ckpts"

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Read 16GB-variant env vars (set in Dockerfile.prismaudio-16gb)
MAX_DURATION = float(os.environ.get("PRISMAUDIO_MAX_DURATION", "10"))
BATCH_SIZE = int(os.environ.get("PRISMAUDIO_BATCH_SIZE", "1"))

# Thread pool for blocking GPU inference (keeps FastAPI event loop free)
_executor = ThreadPoolExecutor(max_workers=1)

# Globals for loaded model
_model = None


def wrap_cot(prompt: str) -> str:
    """Wrap simple prompt into PRiSM Chain-of-Thought format if tags are missing."""
    tags = ["<Semantic>", "<Temporal>", "<Aesthetic>", "<Spatial>"]
    if any(tag in prompt for tag in tags):
        return prompt

    return (
        f"<Semantic>{prompt}</Semantic>"
        f"<Temporal>Natural flow synchronous with video events.</Temporal>"
        f"<Aesthetic>Professional quality, clear audio, natural ambiance.</Aesthetic>"
        f"<Spatial>Centered spatial placement.</Spatial>"
    )


def load_model():
    """Load PRiSM model at startup using ThinkSound's public API."""
    global _model

    logger.info("Loading PRiSM (Prism Audio) model...")

    # The cloned repo exposes create_model_from_config_path and get_pretrained_model
    # via ThinkSound/__init__.py → models/factory.py + models/pretrained.py
    # Checkpoint layout in /app/ckpts/ mirrors what snapshot_download places there:
    #   ckpts/prismaudio.ckpt  — main diffusion weights
    #   ckpts/vae.ckpt         — pretransform / VAE weights
    #   ckpts/model_config.json — architecture config
    sys.path.insert(0, str(WORK_DIR))

    try:
        from ThinkSound import create_model_from_config_path
        from ThinkSound.models.utils import load_ckpt_state_dict

        config_path = CKPT_DIR / "model_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")

        model = create_model_from_config_path(str(config_path))

        # Load main diffusion weights
        main_ckpt = CKPT_DIR / "prismaudio.ckpt"
        if main_ckpt.exists():
            load_ckpt_state_dict(model, str(main_ckpt), prefix="diffusion.")
            logger.info(f"Loaded diffusion weights from {main_ckpt}")

        # Load VAE / pretransform weights
        vae_ckpt = CKPT_DIR / "vae.ckpt"
        if vae_ckpt.exists():
            load_ckpt_state_dict(model, str(vae_ckpt), prefix="autoencoder.")
            logger.info(f"Loaded VAE weights from {vae_ckpt}")

        model = model.to(DEVICE, dtype=DTYPE)
        model.eval()
        _model = model
        logger.info(f"PRiSM model loaded on {DEVICE} ({DTYPE})")

    except Exception as e:
        logger.error(f"Failed to load PRiSM model: {e}", exc_info=True)
        # Raise so the lifespan handler can surface the failure clearly
        raise


def _run_inference(
    job_id: str,
    video_path: str,
    prompt: str,
    negative_prompt: str,
    duration: float,
    seed: int,
    cfg_strength: float,
    num_steps: int,
):
    """Run PRiSM generation (blocking — called inside ThreadPoolExecutor)."""
    try:
        jobs[job_id]["status"] = "processing"

        cot_prompt = wrap_cot(prompt)
        logger.info(f"[{job_id}] Generating audio: prompt='{prompt}', duration={duration}s, seed={seed}")
        logger.info(f"[{job_id}] CoT prompt: {cot_prompt}")

        # Clamp duration to the VRAM envelope for this image variant
        duration = min(duration, MAX_DURATION)

        from ThinkSound.inference.generation import generate_diffusion_cond

        sample_rate = 44100
        sample_size = int(sample_rate * duration)

        conditioning = {
            "video_path": video_path,
            "prompt": cot_prompt,
        }
        if negative_prompt:
            conditioning["negative_prompt"] = negative_prompt

        with torch.inference_mode():
            audio = generate_diffusion_cond(
                model=_model,
                steps=num_steps,
                cfg_scale=cfg_strength,
                conditioning=conditioning,
                sample_size=sample_size,
                seed=seed,
                device=DEVICE,
                batch_size=BATCH_SIZE,
            )

        # audio shape: [batch, channels, samples] — take first item
        if audio.dim() == 3:
            audio = audio[0]

        output_path = OUTPUT_DIR / f"{job_id}.wav"
        torchaudio.save(str(output_path), audio.cpu().float(), sample_rate)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_path"] = str(output_path)
        logger.info(f"[{job_id}] Completed → {output_path}")

    except Exception as e:
        logger.error(f"[{job_id}] Failed: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        try:
            Path(video_path).unlink(missing_ok=True)
        except Exception:
            pass


async def generate_audio_async(
    job_id: str,
    video_path: str,
    prompt: str,
    negative_prompt: str,
    duration: float,
    seed: int,
    cfg_strength: float,
    num_steps: int,
):
    """Dispatch blocking inference to the thread pool without blocking the event loop."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        _executor,
        _run_inference,
        job_id,
        video_path,
        prompt,
        negative_prompt,
        duration,
        seed,
        cfg_strength,
        num_steps,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Prism Audio API starting...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GB")
    logger.info(f"MAX_DURATION={MAX_DURATION}s  BATCH_SIZE={BATCH_SIZE}")

    load_model()
    yield


app = FastAPI(title="Prism Audio API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "prismaudio",
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": _model is not None,
    }


@app.post("/generate")
async def generate_endpoint(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    duration: str = Form("8"),
    seed: str = Form("42"),
    cfg_strength: str = Form("4.5"),
    num_steps: str = Form("25"),
):
    """Start video-to-audio generation via PRiSM."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    job_id = str(uuid.uuid4())

    video_path = INPUT_DIR / f"{job_id}_{video.filename}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    jobs[job_id] = {
        "status": "pending",
        "output_path": None,
        "error": None,
    }

    background_tasks.add_task(
        generate_audio_async,
        job_id,
        str(video_path),
        prompt,
        negative_prompt,
        float(duration),
        int(seed),
        float(cfg_strength),
        int(num_steps),
    )

    return {"job_id": job_id, "status": "pending"}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get the status of a generation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "output_path": job.get("output_path"),
        "error": job.get("error"),
    }


@app.get("/download/{job_id}")
async def download(job_id: str):
    """Download the generated audio file (.wav)."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job['status']}")

    output_path = job.get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename=f"{job_id}.wav",
    )


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its output file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs.pop(job_id)
    if job.get("output_path"):
        Path(job["output_path"]).unlink(missing_ok=True)

    return {"message": "Job deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)