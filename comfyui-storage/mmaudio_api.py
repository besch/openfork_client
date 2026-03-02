#!/usr/bin/env python3
"""
MMAudio REST API Server for OpenFork DGN Client
Provides an HTTP API for video-to-audio synthesis using MMAudio.
Accepts video file upload + prompt, returns generated audio.
"""

import os
import sys
import uuid
import logging
import shutil
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

# Model variant - set via env var (default: small_44k for 8GB)
MODEL_VARIANT = os.environ.get("MMAUDIO_VARIANT", "small_44k")

# Globals for loaded model
_net = None
_feature_utils = None
_model_cfg = None
_device = None
_dtype = None


def load_model():
    """Load MMAudio model at startup."""
    global _net, _feature_utils, _model_cfg, _device, _dtype

    from mmaudio.eval_utils import all_model_cfg, ModelConfig
    from mmaudio.model.networks import get_my_mmaudio

    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _dtype = torch.bfloat16

    logger.info(f"Loading MMAudio model variant: {MODEL_VARIANT}")
    if MODEL_VARIANT not in all_model_cfg:
        raise ValueError(f"Unknown model variant: {MODEL_VARIANT}. Available: {list(all_model_cfg.keys())}")

    _model_cfg = all_model_cfg[MODEL_VARIANT]
    _model_cfg.download_if_needed()

    _net = get_my_mmaudio(_model_cfg.model_name).to(_device, _dtype).eval()
    _net.load_weights(torch.load(_model_cfg.model_path, map_location=_device, weights_only=True))
    logger.info(f"Loaded model weights from {_model_cfg.model_path}")

    from mmaudio.model.utils.features_utils import FeaturesUtils
    _feature_utils = FeaturesUtils(
        tod_vae_ckpt=_model_cfg.vae_path,
        synchformer_ckpt=_model_cfg.synchformer_ckpt,
        enable_conditions=True,
        mode=_model_cfg.mode,
        bigvgan_vocoder_ckpt=_model_cfg.bigvgan_16k_path,
        need_vae_encoder=False
    )
    _feature_utils = _feature_utils.to(_device, _dtype).eval()
    logger.info("MMAudio model loaded successfully")
    if torch.cuda.is_available():
        logger.info(f"Memory usage: {torch.cuda.max_memory_allocated() / (2**30):.2f} GB")


@torch.inference_mode()
def generate_audio_sync(
    job_id: str,
    video_path: str,
    prompt: str,
    negative_prompt: str,
    duration: float,
    seed: int,
    cfg_strength: float,
    num_steps: int,
):
    """Run MMAudio generation synchronously."""
    try:
        jobs[job_id]["status"] = "processing"

        from mmaudio.eval_utils import load_video, generate
        from mmaudio.model.flow_matching import FlowMatching

        seq_cfg = _model_cfg.seq_cfg
        rng = torch.Generator(device=_device)
        rng.manual_seed(seed)
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

        # Load video
        video_info = load_video(Path(video_path), duration)
        clip_frames = video_info.clip_frames.unsqueeze(0)
        sync_frames = video_info.sync_frames.unsqueeze(0)
        actual_duration = video_info.duration_sec

        seq_cfg.duration = actual_duration
        _net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

        logger.info(f"Generating audio for job {job_id}: prompt='{prompt}', duration={actual_duration}s, seed={seed}")

        audios = generate(
            clip_frames,
            sync_frames,
            [prompt],
            negative_text=[negative_prompt] if negative_prompt else None,
            feature_utils=_feature_utils,
            net=_net,
            fm=fm,
            rng=rng,
            cfg_strength=cfg_strength,
        )
        audio = audios.float().cpu()[0]

        # Save output
        output_path = OUTPUT_DIR / f"{job_id}.flac"
        torchaudio.save(str(output_path), audio, seq_cfg.sampling_rate)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_path"] = str(output_path)
        logger.info(f"Job {job_id} completed: {output_path}")
        if torch.cuda.is_available():
            logger.info(f"Memory usage: {torch.cuda.max_memory_allocated() / (2**30):.2f} GB")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        # Clean up input video
        try:
            Path(video_path).unlink(missing_ok=True)
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and clean up on shutdown."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("MMAudio API starting...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name}")
        logger.info(f"VRAM: {props.total_memory / 1024**3:.1f} GB")

    load_model()
    yield


app = FastAPI(title="MMAudio API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_VARIANT,
        "cuda_available": torch.cuda.is_available(),
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
    """
    Start video-to-audio generation.
    Accepts video file upload with generation parameters.
    """
    job_id = str(uuid.uuid4())

    # Save uploaded video
    video_path = INPUT_DIR / f"{job_id}_{video.filename}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    jobs[job_id] = {
        "status": "pending",
        "output_path": None,
        "error": None,
    }

    background_tasks.add_task(
        generate_audio_sync,
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
    """Download the generated audio file."""
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
        media_type="audio/flac",
        filename=f"{job_id}.flac",
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
