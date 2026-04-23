#!/usr/bin/env python3
"""
ERNIE-Image REST API Server for OpenFork DGN Client
Baidu ERNIE-Image text-to-image generation via diffusers pipeline.

Supports two model variants via ERNIE_MODEL_ID env var:
  - baidu/ERNIE-Image      (standard, 50 steps)
  - baidu/ERNIE-Image-Turbo (turbo, 8 steps)

Model weights are expected to be pre-baked into the Docker image at /app/models.
The server starts immediately and loads the model in a background thread so
that /health is reachable right away and the client can start polling.
"""

import os
import uuid
import logging
import time
import inspect
import gc
import threading
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

WORK_DIR = Path("/app")
OUTPUT_DIR = WORK_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

jobs: dict = {}

pipe = None
model_loading = False
model_error: Optional[str] = None
_inference_semaphore = threading.Semaphore(1)

MODEL_ID = os.environ.get("ERNIE_MODEL_ID", "baidu/ERNIE-Image-Turbo")
# local_files_only=True because weights are baked into the image.
HF_HOME = os.environ.get("HF_HOME", "/app/models")
DEFAULT_STEPS = int(os.environ.get("ERNIE_DEFAULT_STEPS", "8"))
MODEL_DTYPE = os.environ.get("ERNIE_DTYPE", "fp16")
# Turbo needs guidance_scale=1.0; standard ERNIE-Image uses 4.0-5.0
_IS_TURBO = "turbo" in MODEL_ID.lower()
DEFAULT_CFG = float(os.environ.get("ERNIE_DEFAULT_CFG", "1.0" if _IS_TURBO else "4.0"))


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


DEFAULT_USE_PE = _env_flag("ERNIE_USE_PE", True)
ALLOW_MODEL_DOWNLOAD = _env_flag("ERNIE_ALLOW_MODEL_DOWNLOAD", False)
ENABLE_CPU_OFFLOAD = _env_flag("ERNIE_ENABLE_CPU_OFFLOAD", False)
ENABLE_ATTENTION_SLICING = _env_flag("ERNIE_ENABLE_ATTENTION_SLICING", False)
ENABLE_VAE_TILING = _env_flag("ERNIE_ENABLE_VAE_TILING", False)
ENABLE_SEQUENTIAL_CPU_OFFLOAD = _env_flag("ERNIE_SEQUENTIAL_CPU_OFFLOAD", False)
ENABLE_TORCH_COMPILE = _env_flag("ERNIE_TORCH_COMPILE", False)
GENERATOR_DEVICE = os.environ.get(
    "ERNIE_GENERATOR_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: Optional[int] = None
    # -1 sentinel → use model default (1.0 for Turbo, 4.0 for standard)
    guidance_scale: float = -1.0
    seed: Optional[int] = None
    use_pe: Optional[bool] = None


class HealthResponse(BaseModel):
    status: str
    model_id: str
    model_loaded: bool
    error: Optional[str] = None


def _is_oom_error(exc: Exception) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _log_cuda_memory(context: str) -> None:
    if not torch.cuda.is_available():
        return

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated_bytes = torch.cuda.memory_allocated()
        reserved_bytes = torch.cuda.memory_reserved()
        logger.info(
            "[%s] CUDA memory: allocated=%.2f GiB reserved=%.2f GiB free=%.2f GiB total=%.2f GiB",
            context,
            allocated_bytes / 1024**3,
            reserved_bytes / 1024**3,
            free_bytes / 1024**3,
            total_bytes / 1024**3,
        )
    except Exception as exc:
        logger.warning("Failed to query CUDA memory during %s: %s", context, exc)


def _cleanup_cuda(context: str, synchronize: bool = False) -> None:
    for _ in range(2):
        gc.collect()

    if not torch.cuda.is_available():
        return

    _log_cuda_memory(f"{context} (before cleanup)")

    if synchronize:
        try:
            torch.cuda.synchronize()
        except Exception as exc:
            logger.warning("CUDA synchronize failed during %s: %s", context, exc)

    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning("torch.cuda.empty_cache() failed during %s: %s", context, exc)

    if hasattr(torch.cuda, "ipc_collect"):
        try:
            torch.cuda.ipc_collect()
        except Exception as exc:
            logger.warning("torch.cuda.ipc_collect() failed during %s: %s", context, exc)

    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    _log_cuda_memory(f"{context} (after cleanup)")


def _fallback_load_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _resolve_dtype() -> torch.dtype:
    if MODEL_DTYPE == "bf16":
        return torch.bfloat16
    if MODEL_DTYPE == "fp16":
        return torch.float16
    if MODEL_DTYPE == "fp8":
        # fp8 halves VRAM vs fp16 (~8GB for the 8B DiT vs ~16GB).
        # Supported for storage+compute on Ada/Ampere (RTX 30xx/40xx) with
        # PyTorch 2.1+. If the GPU/driver rejects it we fall back to a
        # compatibility dtype for the current runtime.
        if hasattr(torch, "float8_e4m3fn"):
            return torch.float8_e4m3fn
        fallback_dtype = _fallback_load_dtype()
        logger.warning(
            "torch.float8_e4m3fn not available on this build; falling back to %s",
            fallback_dtype,
        )
        return fallback_dtype
    # auto
    return _fallback_load_dtype()


def _configure_pipeline_memory(pipe_instance, device: str) -> None:
    if ENABLE_VAE_TILING and hasattr(pipe_instance, "enable_vae_tiling"):
        try:
            pipe_instance.enable_vae_tiling()
            logger.info("Enabled VAE tiling for ERNIE-Image pipeline")
        except Exception as exc:
            logger.warning(f"enable_vae_tiling() failed: {exc}")

    if ENABLE_ATTENTION_SLICING and hasattr(pipe_instance, "enable_attention_slicing"):
        try:
            pipe_instance.enable_attention_slicing()
            logger.info("Enabled attention slicing for ERNIE-Image pipeline")
        except Exception as exc:
            logger.warning(f"enable_attention_slicing() failed: {exc}")

    if ENABLE_CPU_OFFLOAD:
        offload_order = (
            ("enable_sequential_cpu_offload", "enable_model_cpu_offload")
            if ENABLE_SEQUENTIAL_CPU_OFFLOAD
            else ("enable_model_cpu_offload", "enable_sequential_cpu_offload")
        )
        for method_name in offload_order:
            if not hasattr(pipe_instance, method_name):
                continue
            try:
                method = getattr(pipe_instance, method_name)
                params = inspect.signature(method).parameters
                if "gpu_id" in params:
                    method(gpu_id=0)
                else:
                    method()
                logger.info("Enabled ERNIE-Image CPU offload via %s()", method_name)
                return
            except Exception as exc:
                logger.warning("%s() failed: %s", method_name, exc)

        logger.warning(
            "ERNIE_ENABLE_CPU_OFFLOAD=true but no offload method could be applied; "
            "falling back to full GPU placement."
        )

    pipe_instance.to(device)
    logger.info("Placed ERNIE-Image pipeline on %s without CPU offload", device)


def _try_load_pipeline(
    model_id: str, load_kwargs: dict, device: str, local_only: bool
) -> tuple:
    """
    Attempt to load the pipeline, with OOM-aware dtype fallback.

    Load order:
      1. Requested dtype (fp8 or caller's choice) — full GPU placement if
         ENABLE_CPU_OFFLOAD=false, otherwise model_cpu_offload.
      2. On OOM or fp8 rejection: retry with a safer dtype + model_cpu_offload.
    """
    from diffusers import ErnieImagePipeline

    def _do_load(kwargs):
        if local_only:
            return ErnieImagePipeline.from_pretrained(
                model_id, local_files_only=True, **kwargs
            )
        return ErnieImagePipeline.from_pretrained(model_id, **kwargs)

    try:
        p = _do_load(load_kwargs)
        _configure_pipeline_memory(p, device)
        return p, load_kwargs.get("torch_dtype")
    except (TypeError, RuntimeError, torch.cuda.OutOfMemoryError) as first_err:
        # fp8 may be rejected by the pipeline internals on some driver versions;
        # OOM means the dtype didn't save enough VRAM — both warrant a safer retry.
        err_str = str(first_err).lower()
        is_fp8_issue = any(
            token in err_str
            for token in ("float8", "fp8", "e4m3", "storage object", "default dtype")
        )
        is_oom = "out of memory" in err_str or isinstance(
            first_err, torch.cuda.OutOfMemoryError
        )

        if is_fp8_issue or is_oom:
            reason = "fp8 unsupported by pipeline" if is_fp8_issue else "OOM with requested dtype"
            logger.warning(
                "Initial load failed (%s: %s). Retrying with %s + cpu_offload.",
                reason, first_err,
                _fallback_load_dtype(),
            )
            torch.cuda.empty_cache()
            fallback_dtype = _fallback_load_dtype()
            fallback_kwargs = {**load_kwargs, "torch_dtype": fallback_dtype}
            p = _do_load(fallback_kwargs)
            # Force cpu_offload on the fallback path regardless of env setting
            if hasattr(p, "enable_model_cpu_offload"):
                try:
                    p.enable_model_cpu_offload(gpu_id=0)
                except TypeError:
                    p.enable_model_cpu_offload()
            else:
                p.to(device)
            logger.info("Fallback load succeeded: %s + cpu_offload", fallback_dtype)
            return p, fallback_dtype
        raise


def load_model() -> None:
    global pipe, model_loading, model_error
    model_loading = True
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = _resolve_dtype()

        logger.info(
            "Loading ERNIE-Image model: %s | device=%s | dtype=%s | "
            "use_pe_default=%s | cpu_offload=%s | attention_slicing=%s | "
            "vae_tiling=%s | torch_compile=%s",
            MODEL_ID, device, dtype, DEFAULT_USE_PE,
            ENABLE_CPU_OFFLOAD, ENABLE_ATTENTION_SLICING,
            ENABLE_VAE_TILING, ENABLE_TORCH_COMPILE,
        )

        load_kwargs = {
            "torch_dtype": dtype,
            "cache_dir": HF_HOME,
        }
        if not DEFAULT_USE_PE:
            load_kwargs["pe"] = None
            load_kwargs["pe_tokenizer"] = None

        try:
            pipe, loaded_dtype = _try_load_pipeline(
                MODEL_ID, load_kwargs, device, local_only=True
            )
            logger.info("Model loaded from local cache.")
        except Exception as local_err:
            if not ALLOW_MODEL_DOWNLOAD:
                raise RuntimeError(
                    "Local ERNIE-Image model cache load failed and runtime "
                    "downloads are disabled. Rebuild the Docker image with the "
                    f"model baked into HF_HOME={HF_HOME}, or set "
                    "ERNIE_ALLOW_MODEL_DOWNLOAD=true for a dev-only fallback. "
                    f"Original error: {local_err}"
                ) from local_err

            logger.warning(
                "Local cache load failed (%s). Falling back to HF download "
                "because ERNIE_ALLOW_MODEL_DOWNLOAD=true...", local_err,
            )
            pipe, loaded_dtype = _try_load_pipeline(
                MODEL_ID, load_kwargs, device, local_only=False
            )
            logger.info("Model downloaded and loaded from HuggingFace.")

        # Optional torch.compile — significant speedup after a one-time warmup.
        # Only applied to the transformer (the hot path), not the full pipeline.
        if ENABLE_TORCH_COMPILE and hasattr(pipe, "transformer"):
            try:
                logger.info("Applying torch.compile to transformer (first generation will be slow)...")
                pipe.transformer = torch.compile(
                    pipe.transformer,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                logger.info("torch.compile applied successfully.")
            except Exception as compile_err:
                logger.warning("torch.compile failed (%s); continuing without it.", compile_err)

        logger.info("ERNIE-Image model ready on %s with dtype %s", device, loaded_dtype)
    except Exception as e:
        logger.error("Failed to load ERNIE-Image model: %s", e, exc_info=True)
        model_error = str(e)
    finally:
        model_loading = False


def _make_step_callback(job_id: str, total_steps: int):
    """
    Returns a diffusers-compatible step callback that logs progress and updates
    the in-memory job dict so GET /status/{id} exposes step/progress_pct.

    Diffusers >=0.27 uses callback_on_step_end(pipe, step, timestep, kwargs→dict).
    Older versions use callback(step, timestep, latents).
    One function handles both signatures.
    """
    def _cb(pipeline_or_step, step_or_ts=None, ts_or_latents=None, callback_kwargs=None):
        step = step_or_ts if hasattr(pipeline_or_step, "__call__") else pipeline_or_step
        pct = int((step + 1) / total_steps * 100)
        jobs[job_id]["step"] = step + 1
        jobs[job_id]["total_steps"] = total_steps
        jobs[job_id]["progress_pct"] = pct
        logger.info("ERNIE-Image job %s: step %d/%d (%d%%)", job_id, step + 1, total_steps, pct)
        return callback_kwargs or {}
    return _cb


def run_generation(job_id: str, request: GenerateRequest) -> None:
    logger.info("Job %s waiting for ERNIE-Image inference slot", job_id)
    _inference_semaphore.acquire()
    try:
        jobs[job_id]["status"] = "processing"
        logger.info(f"Starting generation for job {job_id}: {request.prompt[:80]}...")
        _cleanup_cuda(f"job {job_id} preflight")

        steps = request.num_inference_steps or DEFAULT_STEPS
        cfg = request.guidance_scale if request.guidance_scale >= 0 else DEFAULT_CFG
        use_pe = DEFAULT_USE_PE if request.use_pe is None else request.use_pe
        generator = None
        generator_device_name = "none"
        if request.seed is not None:
            generator_device = GENERATOR_DEVICE
            if generator_device == "cuda" and not torch.cuda.is_available():
                generator_device = "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(request.seed)
            generator_device_name = generator_device

        start_time = time.time()
        logger.info(
            "ERNIE-Image job %s config: steps=%s cfg=%s size=%sx%s use_pe=%s generator_device=%s",
            job_id,
            steps,
            cfg,
            request.width,
            request.height,
            use_pe,
            generator_device_name,
        )

        # Wire in per-step progress callback. Detect the right kwarg name so we
        # don't pass an unknown argument and silently swallow the TypeError.
        step_cb = _make_step_callback(job_id, steps)
        call_sig = inspect.signature(pipe.__call__)
        if "callback_on_step_end" in call_sig.parameters:
            cb_kwargs = {"callback_on_step_end": step_cb}
        elif "callback" in call_sig.parameters:
            cb_kwargs = {"callback": step_cb, "callback_steps": 1}
        else:
            logger.warning("Pipeline has no recognised callback parameter; step progress will not be logged.")
            cb_kwargs = {}

        def _invoke_pipeline(use_pe_value: bool):
            with torch.inference_mode():
                return pipe(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt or None,
                    width=request.width,
                    height=request.height,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=generator,
                    use_pe=use_pe_value,
                    **cb_kwargs,
                )

        try:
            result = _invoke_pipeline(use_pe)
        except Exception as exc:
            if _is_oom_error(exc) and use_pe:
                logger.warning(
                    "ERNIE-Image job %s hit CUDA OOM with use_pe=True. "
                    "Retrying once with use_pe=False.",
                    job_id,
                )
                jobs[job_id]["warning"] = (
                    "Prompt enhancer disabled after CUDA OOM; retried automatically."
                )
                _cleanup_cuda(f"job {job_id} after OOM retry trigger", synchronize=True)
                use_pe = False
                result = _invoke_pipeline(use_pe)
            else:
                raise
        elapsed = time.time() - start_time
        logger.info(
            f"Generation completed for job {job_id} in {elapsed:.1f}s ({steps} steps)"
        )

        image = result.images[0]
        output_path = OUTPUT_DIR / f"{job_id}.png"
        image.save(str(output_path), format="PNG")

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_path"] = str(output_path)
        jobs[job_id]["elapsed_seconds"] = elapsed
        jobs[job_id]["steps"] = steps
    except Exception as e:
        logger.error(f"Generation failed for job {job_id}: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        if _is_oom_error(e):
            jobs[job_id]["error_code"] = "cuda_oom"
    finally:
        _cleanup_cuda(f"job {job_id} teardown", synchronize=True)
        _inference_semaphore.release()


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    import threading

    thread = threading.Thread(target=load_model, daemon=True)
    thread.start()
    yield


app = FastAPI(title="ERNIE-Image API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if pipe is not None:
        status = "ok"
    elif model_loading:
        status = "loading"
    else:
        status = "error"

    return HealthResponse(
        status=status,
        model_id=MODEL_ID,
        model_loaded=pipe is not None,
        error=model_error,
    )


@app.post("/generate")
async def generate(request: GenerateRequest):
    if model_loading:
        raise HTTPException(status_code=503, detail="Model is still loading")
    if pipe is None:
        raise HTTPException(
            status_code=503, detail=f"Model not loaded: {model_error}"
        )

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "prompt": request.prompt,
        "created_at": time.time(),
    }

    import threading

    thread = threading.Thread(
        target=run_generation, args=(job_id, request), daemon=True
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/output/{job_id}")
async def get_output(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status is {job['status']}")
    output_path = job.get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(output_path, media_type="image/png", filename=f"{job_id}.png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
