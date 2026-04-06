#!/usr/bin/env python3
"""
Prism Audio (PRiSM) REST API Server for OpenFork DGN Client
Provides an HTTP API for video-to-audio synthesis using PRiSM.
Accepts video file upload + prompt, returns generated audio.
"""

import os
import sys
import json
import uuid
import logging
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder
from moviepy.editor import VideoFileClip
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Job storage
jobs = {}

# Directories
WORK_DIR = Path("/app")
OUTPUT_DIR = WORK_DIR / "output"
INPUT_DIR = WORK_DIR / "input"
CKPT_DIR = WORK_DIR / "ckpts"

# Configuration — mirrors app.py constants exactly
SAMPLE_RATE = 44100
_CLIP_FPS = 4
_CLIP_SIZE = 288
_SYNC_FPS = 25
_SYNC_SIZE = 224

MODEL_CONFIG_PATH = WORK_DIR / "PrismAudio" / "configs" / "model_configs" / "prismaudio.json"
CKPT_PATH        = CKPT_DIR / "prismaudio.ckpt"
VAE_CKPT_PATH    = CKPT_DIR / "vae.ckpt"
VAE_CONFIG_PATH  = WORK_DIR / "PrismAudio" / "configs" / "model_configs" / "stable_audio_2_0_vae.json"
SYNCHFORMER_PATH = CKPT_DIR / "synchformer_state_dict.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Read 16GB-variant env vars (set in Dockerfile.prismaudio-16gb)
MAX_DURATION = float(os.environ.get("PRISMAUDIO_MAX_DURATION", "10"))
BATCH_SIZE   = int(os.environ.get("PRISMAUDIO_BATCH_SIZE", "1"))

# Thread pool for blocking GPU inference (keeps FastAPI event loop free)
_executor = ThreadPoolExecutor(max_workers=1)

# Globals for loaded models
_diffusion         = None
_feature_extractor = None
_sync_transform    = None
_model_config      = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


def _log_ckpts_contents():
    try:
        files = sorted(CKPT_DIR.rglob("*"))
        if files:
            logger.info(f"Contents of {CKPT_DIR}:")
            for f in files:
                if f.is_file():
                    size_mb = f.stat().st_size / 1024 / 1024
                    logger.info(f"  {f.relative_to(CKPT_DIR)}  ({size_mb:.1f} MB)")
        else:
            logger.warning(f"{CKPT_DIR} is empty — weights were not downloaded at build time")
    except Exception as e:
        logger.warning(f"Could not list {CKPT_DIR}: {e}")


def pad_to_square(video_tensor: torch.Tensor) -> torch.Tensor:
    """(L, C, H, W) -> (L, C, _CLIP_SIZE, _CLIP_SIZE)"""
    l, c, h, w = video_tensor.shape
    max_side = max(h, w)
    pad_h = max_side - h
    pad_w = max_side - w
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    padded = F.pad(video_tensor, pad=padding, mode="constant", value=0)
    return F.interpolate(padded, size=(_CLIP_SIZE, _CLIP_SIZE), mode="bilinear", align_corners=False)


def get_video_duration(video_path: str) -> float:
    clip = VideoFileClip(str(video_path))
    duration = clip.duration
    clip.close()
    return duration


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

def load_model():
    """Load PRiSM diffusion model and feature extractor at startup."""
    global _diffusion, _feature_extractor, _sync_transform, _model_config

    logger.info("Loading PRiSM (Prism Audio) model...")
    _log_ckpts_contents()

    sys.path.insert(0, str(WORK_DIR))

    # Verify required files
    missing = [p for p in (MODEL_CONFIG_PATH, CKPT_PATH, VAE_CKPT_PATH, SYNCHFORMER_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Required PRiSM files not found: {[str(m) for m in missing]}")

    with open(MODEL_CONFIG_PATH) as f:
        _model_config = json.load(f)

    from PrismAudio.models import create_model_from_config
    from PrismAudio.models.utils import load_ckpt_state_dict
    from data_utils.v2a_utils.feature_utils_288 import FeaturesUtils

    # --- Diffusion model ---
    diffusion = create_model_from_config(_model_config)
    diffusion.load_state_dict(torch.load(str(CKPT_PATH), map_location="cpu"))
    logger.info(f"Loaded diffusion weights from {CKPT_PATH}")

    # --- VAE / pretransform ---
    if diffusion.pretransform is not None:
        vae_state = load_ckpt_state_dict(str(VAE_CKPT_PATH), prefix="autoencoder.")
        diffusion.pretransform.load_state_dict(vae_state)
        logger.info(f"Loaded VAE weights from {VAE_CKPT_PATH}")

    diffusion = diffusion.to(DEVICE, dtype=DTYPE).eval()
    _diffusion = diffusion

    # --- Feature extractor (VideoPrism + Synchformer + T5) ---
    feature_extractor = FeaturesUtils(
        vae_ckpt=None,
        vae_config=str(VAE_CONFIG_PATH),
        enable_conditions=True,
        synchformer_ckpt=str(SYNCHFORMER_PATH),
    )
    _feature_extractor = feature_extractor.eval().to(DEVICE)

    # --- Sync transform ---
    _sync_transform = v2.Compose([
        v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(_SYNC_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    logger.info(f"PRiSM model loaded on {DEVICE} ({DTYPE})")
    logger.info(f"MAX_DURATION={MAX_DURATION}s  BATCH_SIZE={BATCH_SIZE}")


# ---------------------------------------------------------------------------
# video frame extraction
# ---------------------------------------------------------------------------

def extract_video_frames(video_path: str):
    """
    Decode clip_chunk and sync_chunk from a video.

    Returns:
        clip_chunk : (L, H, W, C) float32 [0, 1]    — for VideoPrism
        sync_chunk : (L, C, H, W) float32 normalized — for Synchformer
        duration   : float (seconds)
    """
    duration_sec = get_video_duration(video_path)

    reader = StreamingMediaDecoder(video_path)
    reader.add_basic_video_stream(
        frames_per_chunk=int(_CLIP_FPS * duration_sec),
        frame_rate=_CLIP_FPS,
        format="rgb24",
    )
    reader.add_basic_video_stream(
        frames_per_chunk=int(_SYNC_FPS * duration_sec),
        frame_rate=_SYNC_FPS,
        format="rgb24",
    )
    reader.fill_buffer()
    data_chunk = reader.pop_chunks()

    clip_chunk = data_chunk[0]
    sync_chunk = data_chunk[1]

    if clip_chunk is None:
        raise RuntimeError("CLIP video stream returned None")
    if sync_chunk is None:
        raise RuntimeError("Sync video stream returned None")

    # --- clip_chunk: (L, C, H, W) uint8 -> (L, H, W, C) float32 [0,1] ---
    clip_expected = int(_CLIP_FPS * duration_sec)
    clip_chunk = clip_chunk[:clip_expected]
    if clip_chunk.shape[0] < clip_expected:
        pad_n = clip_expected - clip_chunk.shape[0]
        clip_chunk = torch.cat([clip_chunk, clip_chunk[-1:].repeat(pad_n, 1, 1, 1)], dim=0)
    clip_chunk = pad_to_square(clip_chunk)          # (L, C, _CLIP_SIZE, _CLIP_SIZE)
    clip_chunk = clip_chunk.permute(0, 2, 3, 1)    # (L, H, W, C)
    clip_chunk = clip_chunk.float() / 255.0        # [0, 1]

    # --- sync_chunk: (L, C, H, W) uint8 -> (L, C, _SYNC_SIZE, _SYNC_SIZE) normalized ---
    sync_expected = int(_SYNC_FPS * duration_sec)
    sync_chunk = sync_chunk[:sync_expected]
    if sync_chunk.shape[0] < sync_expected:
        pad_n = sync_expected - sync_chunk.shape[0]
        sync_chunk = torch.cat([sync_chunk, sync_chunk[-1:].repeat(pad_n, 1, 1, 1)], dim=0)
    sync_chunk = _sync_transform(sync_chunk)

    logger.info(f"clip_chunk: {clip_chunk.shape}, sync_chunk: {sync_chunk.shape}, duration: {duration_sec:.2f}s")
    return clip_chunk, sync_chunk, duration_sec


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------

def extract_features(clip_chunk: torch.Tensor, sync_chunk: torch.Tensor, caption: str) -> dict:
    info = {}
    with torch.no_grad():
        info["text_features"] = _feature_extractor.encode_t5_text([caption])[0].cpu()

        clip_input = torch.from_numpy(np.array(clip_chunk)).unsqueeze(0)
        video_feat, frame_embed, _, text_feat = \
            _feature_extractor.encode_video_and_text_with_videoprism(clip_input, [caption])
        info["global_video_features"] = torch.tensor(np.array(video_feat)).squeeze(0).cpu()
        info["video_features"]        = torch.tensor(np.array(frame_embed)).squeeze(0).cpu()
        info["global_text_features"]  = torch.tensor(np.array(text_feat)).squeeze(0).cpu()

        sync_input = sync_chunk.unsqueeze(0).to(DEVICE)
        info["sync_features"] = _feature_extractor.encode_video_with_sync(sync_input)[0].cpu()

    return info


# ---------------------------------------------------------------------------
# diffusion sampling
# ---------------------------------------------------------------------------

def run_diffusion(meta: dict, duration: float) -> torch.Tensor:
    """
    Run the PRiSM diffusion pipeline.
    Returns float32 CPU tensor of shape (channels, samples).
    """
    from PrismAudio.inference.sampling import sample, sample_discrete_euler

    diffusion_objective = _model_config["model"]["diffusion"]["diffusion_objective"]
    latent_length = round(SAMPLE_RATE * duration / 2048)

    meta_on_device = {
        k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
        for k, v in meta.items()
    }
    metadata = (meta_on_device,)

    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            conditioning = _diffusion.conditioner(metadata, DEVICE)

        video_exist = torch.stack([item["video_exist"] for item in metadata], dim=0)
        if "metaclip_features" in conditioning:
            conditioning["metaclip_features"][~video_exist] = \
                _diffusion.model.model.empty_clip_feat
        if "sync_features" in conditioning:
            conditioning["sync_features"][~video_exist] = \
                _diffusion.model.model.empty_sync_feat

        cond_inputs = _diffusion.get_conditioning_inputs(conditioning)
        noise = torch.randn([1, _diffusion.io_channels, latent_length]).to(DEVICE)

        with torch.amp.autocast("cuda"):
            if diffusion_objective == "v":
                fakes = sample(
                    _diffusion.model, noise, 24, 0,
                    **cond_inputs, cfg_scale=5, batch_cfg=True,
                )
            else:  # rectified_flow
                fakes = sample_discrete_euler(
                    _diffusion.model, noise, 24,
                    **cond_inputs, cfg_scale=5, batch_cfg=True,
                )

            if _diffusion.pretransform is not None:
                fakes = _diffusion.pretransform.decode(fakes)

    return (
        fakes.to(torch.float32)
             .div(torch.max(torch.abs(fakes)))
             .clamp(-1, 1)
             .cpu()
    )


# ---------------------------------------------------------------------------
# inference entry point
# ---------------------------------------------------------------------------

def _run_inference(
    job_id: str,
    video_path: str,
    prompt: str,
    duration_override: Optional[float],
    seed: int,
):
    """Run PRiSM generation (blocking — called inside ThreadPoolExecutor)."""
    try:
        jobs[job_id]["status"] = "processing"
        torch.manual_seed(seed)

        cot_prompt = wrap_cot(prompt)
        logger.info(f"[{job_id}] Extracting video frames: {video_path}")

        clip_chunk, sync_chunk, video_duration = extract_video_frames(video_path)

        # Clamp to VRAM envelope for this image variant
        duration = min(duration_override or video_duration, MAX_DURATION)

        logger.info(f"[{job_id}] Extracting features, prompt='{prompt}', duration={duration:.1f}s, seed={seed}")
        info = extract_features(clip_chunk, sync_chunk, cot_prompt)

        # Build metadata dict (mirrors app.py build_meta)
        meta = dict(info)
        meta["id"]          = job_id
        meta["relpath"]     = "demo.npz"
        meta["path"]        = "demo.npz"
        meta["caption_cot"] = cot_prompt
        meta["video_exist"] = torch.tensor(True)

        logger.info(f"[{job_id}] Running diffusion...")
        audio = run_diffusion(meta, duration)  # (1, channels, samples) float32 CPU

        output_path = OUTPUT_DIR / f"{job_id}.wav"
        torchaudio.save(str(output_path), audio[0], SAMPLE_RATE)

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
    duration_override: Optional[float],
    seed: int,
):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        _executor,
        _run_inference,
        job_id,
        video_path,
        prompt,
        duration_override,
        seed,
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Prism Audio API starting...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GB")

    load_model()
    yield


app = FastAPI(title="Prism Audio API", version="2.0.0", lifespan=lifespan)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "prismaudio",
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": _diffusion is not None,
    }


@app.post("/generate")
async def generate_endpoint(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    prompt: str = Form(""),
    duration: str = Form("0"),    # 0 = use video's natural duration
    seed: str = Form("42"),
    # kept for API compat but unused — model uses fixed cfg/steps
    negative_prompt: str = Form(""),
    cfg_strength: str = Form("5"),
    num_steps: str = Form("24"),
):
    """Start video-to-audio generation via PRiSM."""
    if _diffusion is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    job_id = str(uuid.uuid4())

    video_path = INPUT_DIR / f"{job_id}_{video.filename}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    duration_val = float(duration)
    duration_override = duration_val if duration_val > 0 else None

    jobs[job_id] = {"status": "pending", "output_path": None, "error": None}

    background_tasks.add_task(
        generate_audio_async,
        job_id,
        str(video_path),
        prompt,
        duration_override,
        int(seed),
    )

    return {"job_id": job_id, "status": "pending"}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return {"job_id": job_id, "status": job["status"], "error": job.get("error")}


@app.get("/download/{job_id}")
async def download(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job['status']}")
    output_path = job.get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(output_path, media_type="audio/wav", filename=f"{job_id}.wav")


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs.pop(job_id)
    if job.get("output_path"):
        Path(job["output_path"]).unlink(missing_ok=True)
    return {"message": "Job deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
