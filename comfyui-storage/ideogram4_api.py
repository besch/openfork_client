"""
Ideogram 4 REST API server for OpenFork DGN Client.

Loads Ideogram 4 once, then serves text-to-image jobs through a small FastAPI
surface compatible with the existing REST image processors.
"""

import asyncio
import gc
import json
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
from PIL import Image
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
LOW_VRAM_MODE = os.environ.get("IDEOGRAM_LOW_VRAM_MODE", "auto").lower()
LOW_VRAM_OFFLOAD_MAX_MB = int(
    os.environ.get("IDEOGRAM_LOW_VRAM_OFFLOAD_MAX_MB", "24576")
)
WRAP_PLAIN_PROMPT_JSON = os.environ.get(
    "IDEOGRAM_WRAP_PLAIN_PROMPT_JSON", "1"
).lower() in ("1", "true", "yes", "on")

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


def _gpu_memory_snapshot() -> Optional[str]:
    if not torch.cuda.is_available():
        return None

    try:
        device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        gib = 1024**3
        return (
            f"gpu={torch.cuda.get_device_name(device)} "
            f"free={free / gib:.2f}GiB total={total / gib:.2f}GiB "
            f"allocated={allocated / gib:.2f}GiB reserved={reserved / gib:.2f}GiB"
        )
    except Exception as exc:
        logger.debug("Could not read CUDA memory stats: %s", exc)
        return None


def _gpu_total_memory_mb() -> Optional[int]:
    if not torch.cuda.is_available():
        return None

    try:
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory
        return int(total / 1024 / 1024)
    except Exception as exc:
        logger.debug("Could not read CUDA total memory: %s", exc)
        return None


def _should_use_low_vram_loader(device: str) -> bool:
    if device != "cuda":
        return False
    if LOW_VRAM_MODE in ("1", "true", "yes", "on", "force"):
        return True
    if LOW_VRAM_MODE in ("0", "false", "no", "off", "disabled"):
        return False

    total_mb = _gpu_total_memory_mb()
    return total_mb is not None and total_mb <= LOW_VRAM_OFFLOAD_MAX_MB


def _clear_cuda_cache(reason: str):
    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.empty_cache()
        snapshot = _gpu_memory_snapshot()
        if snapshot:
            logger.info("Cleared CUDA cache %s; %s", reason, snapshot)
    except Exception as exc:
        logger.debug("Could not clear CUDA cache %s: %s", reason, exc)


def _patch_low_vram_pipeline(model, text_device: torch.device, vae_device: torch.device):
    """Keep bulky side modules off CUDA while the denoisers stay resident."""
    original_encode_text = model._encode_text

    def _encode_text_low_vram(token_ids, text_position_ids, indicator):
        encoded = original_encode_text(
            token_ids.to(text_device),
            text_position_ids.to(text_device),
            indicator.to(text_device),
        )
        return encoded.to(device=model.device, dtype=torch.float32)

    def _decode_low_vram(z: torch.Tensor, *, grid_h: int, grid_w: int) -> list[Image.Image]:
        batch_size = z.shape[0]
        patch = model.config.patch_size

        latent_scale = model.latent_scale.to(device=z.device)
        latent_shift = model.latent_shift.to(device=z.device)
        z = z * latent_scale + latent_shift

        ae_channels = z.shape[-1] // (patch * patch)
        z = z.view(batch_size, grid_h, grid_w, patch, patch, ae_channels)
        z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
        z = z.view(batch_size, ae_channels, grid_h * patch, grid_w * patch)

        try:
            vae_dtype = next(model.autoencoder.parameters()).dtype
        except StopIteration:
            vae_dtype = model.dtype
        z = z.to(device=vae_device, dtype=vae_dtype)
        decoded = model.autoencoder.decoder(z)
        decoded = decoded.float().clamp(-1.0, 1.0)
        decoded = ((decoded + 1.0) * 127.5).round().to(torch.uint8)
        decoded = decoded.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(arr) for arr in decoded]

    model._encode_text = _encode_text_low_vram
    model._decode = _decode_low_vram
    model.low_vram_offload = {
        "text_encoder_device": str(text_device),
        "autoencoder_device": str(vae_device),
    }
    return model


def _load_model_low_vram(device: str, dtype: torch.dtype):
    from huggingface_hub import hf_hub_download
    from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig
    from ideogram4.modeling_ideogram4 import Ideogram4Config
    from ideogram4.pipeline_ideogram4 import (
        _build_transformer,
        _load_autoencoder,
        _load_indexed_or_single_state_dict,
        _load_qwen3_vl,
    )

    cuda_device = torch.device(device)
    cpu_device = torch.device("cpu")
    config = Ideogram4PipelineConfig(weights_repo=MODEL_REPO)
    transformer_config = Ideogram4Config()
    logger.info(
        "Loading Ideogram 4 with low-VRAM offload repo=%s quantization=%s "
        "denoisers=%s text_encoder=cpu autoencoder=cpu",
        MODEL_REPO,
        DEFAULT_QUANTIZATION,
        device,
    )

    conditional_state_dict = _load_indexed_or_single_state_dict(
        config.weights_repo, config.conditional_index_filename
    )
    conditional_transformer = _build_transformer(
        transformer_config, conditional_state_dict, cuda_device, dtype
    )
    del conditional_state_dict
    gc.collect()
    _clear_cuda_cache("after conditional transformer load")

    unconditional_state_dict = _load_indexed_or_single_state_dict(
        config.weights_repo, config.unconditional_index_filename
    )
    unconditional_transformer = _build_transformer(
        transformer_config, unconditional_state_dict, cuda_device, dtype
    )
    del unconditional_state_dict
    gc.collect()
    _clear_cuda_cache("after unconditional transformer load")

    autoencoder_weights = hf_hub_download(
        repo_id=config.weights_repo, filename=config.autoencoder_filename
    )

    try:
        text_tokenizer, text_encoder = _load_qwen3_vl(
            config.weights_repo,
            cpu_device,
            dtype,
            tokenizer_subfolder=config.tokenizer_subfolder,
            text_encoder_subfolder=config.text_encoder_subfolder,
        )
        text_device = cpu_device
    except Exception as exc:
        logger.warning(
            "CPU text-encoder load failed (%s); falling back to CUDA text encoder",
            exc,
        )
        text_tokenizer, text_encoder = _load_qwen3_vl(
            config.weights_repo,
            cuda_device,
            dtype,
            tokenizer_subfolder=config.tokenizer_subfolder,
            text_encoder_subfolder=config.text_encoder_subfolder,
        )
        text_device = cuda_device

    autoencoder = _load_autoencoder(autoencoder_weights, cpu_device, dtype)
    model = Ideogram4Pipeline(
        conditional_transformer=conditional_transformer,
        unconditional_transformer=unconditional_transformer,
        text_encoder=text_encoder,
        text_tokenizer=text_tokenizer,
        autoencoder=autoencoder,
        config=config,
        device=cuda_device,
        dtype=dtype,
    )
    return _patch_low_vram_pipeline(model, text_device, cpu_device)


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
            if _should_use_low_vram_loader(device):
                pipe = _load_model_low_vram(device=device, dtype=torch.bfloat16)
            else:
                pipe = Ideogram4Pipeline.from_pretrained(
                    config=Ideogram4PipelineConfig(weights_repo=MODEL_REPO),
                    device=device,
                    dtype=torch.bfloat16,
                )
            snapshot = _gpu_memory_snapshot()
            if snapshot:
                logger.info("Ideogram 4 model loaded; %s", snapshot)
            else:
                logger.info("Ideogram 4 model loaded")
            return pipe
        except Exception as exc:
            model_error = str(exc)
            logger.error("Failed to load Ideogram 4 model: %s", exc, exc_info=True)
            _clear_cuda_cache("after model load failure")
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


def _is_json_caption(prompt: str) -> bool:
    stripped = prompt.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return False

    try:
        return isinstance(json.loads(stripped), dict)
    except json.JSONDecodeError:
        return False


def _wrap_plain_prompt_as_json(prompt: str) -> str:
    caption = {
        "high_level_description": prompt,
        "style_description": {
            "aesthetics": "clean, polished, high quality, design-forward",
            "lighting": "balanced, natural, readable",
            "medium": "graphic_design",
            "art_style": "professional commercial visual design with clear composition",
        },
        "compositional_deconstruction": {
            "background": "A coherent background that supports the requested subject without distracting clutter.",
            "elements": [
                {
                    "type": "obj",
                    "desc": prompt,
                }
            ],
        },
    }
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def _prepare_prompt(prompt: str, width: int, height: int, use_magic_prompt: bool) -> str:
    prompt = _expand_prompt_if_requested(prompt, width, height, use_magic_prompt)
    if WRAP_PLAIN_PROMPT_JSON and not use_magic_prompt and not _is_json_caption(prompt):
        prompt = _wrap_plain_prompt_as_json(prompt)
        logger.info("Wrapped plain Ideogram prompt in structured JSON caption")
    return prompt


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
    prompt = _prepare_prompt(prompt, width, height, use_magic_prompt)
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
    snapshot = _gpu_memory_snapshot()
    if snapshot:
        logger.info("Ideogram 4 job %s pre-generation memory: %s", job_id, snapshot)

    with torch.inference_mode():
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
    finally:
        _clear_cuda_cache(f"after job {job_id}")


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
