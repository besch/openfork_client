"""
Ideogram 4 REST API server for OpenFork DGN Client.

Loads Ideogram 4 once, then serves text-to-image jobs through a small FastAPI
surface compatible with the existing REST image processors.
"""

import asyncio
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

WORK_DIR = Path("/app")
OUTPUT_DIR = WORK_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUANTIZATION_REPOS = {
    "nf4": "ideogram-ai/ideogram-4-nf4",
    "fp8": "ideogram-ai/ideogram-4-fp8",
}

DEFAULT_QUANTIZATION = os.environ.get("IDEOGRAM_QUANTIZATION", "nf4").lower()
MODEL_REPO = os.environ.get(
    "IDEOGRAM_MODEL_REPO",
    QUANTIZATION_REPOS.get(DEFAULT_QUANTIZATION, QUANTIZATION_REPOS["nf4"]),
)
DEFAULT_SAMPLER_PRESET = os.environ.get(
    "IDEOGRAM_SAMPLER_PRESET",
    "V4_QUALITY_48",
)
DEFAULT_USE_MAGIC_PROMPT = os.environ.get("IDEOGRAM_USE_MAGIC_PROMPT", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
DEFAULT_MAGIC_PROMPT_MODEL = os.environ.get("IDEOGRAM_MAGIC_PROMPT_MODEL")

pipe = None
model_loading = False
model_error: Optional[str] = None
model_load_lock = threading.Lock()
model_infer_lock = asyncio.Lock()
jobs: Dict[str, Dict[str, Any]] = {}
executor = ThreadPoolExecutor(max_workers=1)

app = FastAPI(title="Ideogram 4 API", version="1.0.0")


class GenerateRequest(BaseModel):
    prompt: str
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    sampler_preset: Optional[str] = None
    seed: Optional[int] = None
    use_magic_prompt: Optional[bool] = None
    warn_on_caption_issues: bool = True


class HealthResponse(BaseModel):
    status: str
    model_id: str
    model_loaded: bool
    error: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    status: str
    error: Optional[str] = None
    output_path: Optional[str] = None
    seed: Optional[int] = None


def _round_to_multiple_of_16(value: int) -> int:
    return max(256, min(2048, int(round(value / 16) * 16)))


def _resolve_seed(seed: Optional[int]) -> int:
    if seed is not None:
        return int(seed)
    return int(time.time_ns() % (2**31 - 1))


def _load_model():
    global pipe, model_loading, model_error

    if pipe is not None:
        return pipe

    with model_load_lock:
        if pipe is not None:
            return pipe

        model_loading = True
        model_error = None
        try:
            from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                "Loading Ideogram 4 model repo=%s quantization=%s device=%s",
                MODEL_REPO,
                DEFAULT_QUANTIZATION,
                device,
            )
            pipe = Ideogram4Pipeline.from_pretrained(
                config=Ideogram4PipelineConfig(weights_repo=MODEL_REPO),
                device=device,
                dtype=torch.bfloat16,
            )
            logger.info("Ideogram 4 model loaded")
            return pipe
        except Exception as exc:
            model_error = str(exc)
            logger.error("Failed to load Ideogram 4 model: %s", exc, exc_info=True)
            raise
        finally:
            model_loading = False


def _expand_prompt_if_requested(prompt: str, width: int, height: int, use_magic: bool) -> str:
    if not use_magic:
        return prompt

    api_key = os.environ.get("MAGIC_PROMPT_API_KEY") or os.environ.get("IDEOGRAM_API_KEY")
    if not api_key:
        raise ValueError(
            "Ideogram magic prompt requested but no MAGIC_PROMPT_API_KEY or IDEOGRAM_API_KEY is set"
        )

    from ideogram4 import DEFAULT_MAGIC_PROMPT, MAGIC_PROMPTS, aspect_ratio_from_size

    model_name = DEFAULT_MAGIC_PROMPT_MODEL or DEFAULT_MAGIC_PROMPT
    aspect_ratio = aspect_ratio_from_size(width, height)
    magic = MAGIC_PROMPTS[model_name](api_key=api_key)
    expanded = magic.expand(prompt, aspect_ratio=aspect_ratio)
    logger.info("Expanded Ideogram prompt with %s for %s", model_name, aspect_ratio)
    return expanded


def _run_generation(job_id: str, request: GenerateRequest) -> tuple[str, int]:
    prompt = request.prompt.strip()
    if not prompt:
        raise ValueError("No prompt provided for Ideogram 4 generation")

    from ideogram4 import PRESETS

    model = _load_model()
    width = _round_to_multiple_of_16(request.width)
    height = _round_to_multiple_of_16(request.height)
    preset_name = request.sampler_preset or DEFAULT_SAMPLER_PRESET
    if preset_name not in PRESETS:
        raise ValueError(
            f"Unknown Ideogram sampler preset {preset_name}. Available: {', '.join(sorted(PRESETS))}"
        )

    use_magic_prompt = (
        request.use_magic_prompt
        if request.use_magic_prompt is not None
        else DEFAULT_USE_MAGIC_PROMPT
    )
    prompt = _expand_prompt_if_requested(prompt, width, height, use_magic_prompt)
    preset = PRESETS[preset_name]
    seed = _resolve_seed(request.seed)
    output_path = OUTPUT_DIR / f"{job_id}.png"

    logger.info(
        "Running Ideogram 4 job %s size=%sx%s preset=%s seed=%s magic=%s",
        job_id,
        width,
        height,
        preset_name,
        seed,
        use_magic_prompt,
    )
    images = model(
        prompt,
        height=height,
        width=width,
        num_steps=preset.num_steps,
        guidance_schedule=preset.guidance_schedule,
        mu=preset.mu,
        std=preset.std,
        seed=seed,
        raise_on_caption_issues=not request.warn_on_caption_issues,
    )
    if not images:
        raise RuntimeError("Ideogram 4 returned no images")

    images[0].save(output_path)
    return str(output_path), seed


async def _process_generation_job(job_id: str, request: GenerateRequest):
    try:
        jobs[job_id]["status"] = "processing"
        async with model_infer_lock:
            output_path, seed = await asyncio.get_running_loop().run_in_executor(
                executor,
                _run_generation,
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
        logger.info("Ideogram 4 job %s completed", job_id)
    except Exception as exc:
        logger.error("Ideogram 4 job %s failed: %s", job_id, exc, exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)


@app.on_event("startup")
async def startup_event():
    if os.environ.get("IDEOGRAM_PRELOAD", "1") != "0":
        asyncio.create_task(asyncio.to_thread(_load_model))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    status = "healthy" if pipe is not None else "loading"
    if model_error:
        status = "error"
    return HealthResponse(
        status=status,
        model_id=MODEL_REPO,
        model_loaded=pipe is not None,
        error=model_error,
    )


@app.get("/info")
async def info():
    from ideogram4 import PRESETS

    return {
        "model_repo": MODEL_REPO,
        "quantization": DEFAULT_QUANTIZATION,
        "sampler_preset": DEFAULT_SAMPLER_PRESET,
        "available_sampler_presets": sorted(PRESETS.keys()),
        "magic_prompt_default": DEFAULT_USE_MAGIC_PROMPT,
    }


@app.post("/generate", response_model=JobStatus)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    if pipe is None and model_error:
        raise HTTPException(status_code=503, detail=model_error)
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


@app.get("/output/{job_id}")
async def output(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status is {job['status']}")

    output_path = job.get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(output_path, media_type="image/png", filename=f"{job_id}.png")


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        return {"status": "not_found"}

    job = jobs.pop(job_id)
    output_path = job.get("output_path")
    if output_path and Path(output_path).exists():
        try:
            Path(output_path).unlink()
        except OSError:
            pass

    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Ideogram 4 API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
