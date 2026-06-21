"""
TeleStyleV2 REST API server for OpenFork DGN Client.

Loads Qwen-Image-Edit-2509 with the TeleStyleV2 style and DMD LoRAs, then
serves reference-image style transfer jobs through the same REST surface used by
other image services.
"""

import asyncio
import logging
import os
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

WORK_DIR = Path("/app")
INPUT_DIR = WORK_DIR / "input"
OUTPUT_DIR = WORK_DIR / "output"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TELESTYLE_REPO_DIR = Path(os.environ.get("TELESTYLE_REPO_DIR", "/opt/TeleStyleV2"))
MODEL_PATH = os.environ.get(
    "TELESTYLE_MODEL_PATH",
    "/app/models/Qwen-Image-Edit-2509",
)
LORA_DIR = Path(os.environ.get("TELESTYLE_LORA_DIR", "/app/models/TeleStyleV2"))
STYLE_LORA = os.environ.get(
    "TELESTYLE_STYLE_LORA",
    str(LORA_DIR / "diffusers-TeleStyleV2-QIE-2509-Lora-bf16.safetensors"),
)
DMD_LORA = os.environ.get(
    "TELESTYLE_DMD_LORA",
    str(LORA_DIR / "QIE-2509-Lightning-4steps-V1.0-bf16.safetensors"),
)
DEFAULT_PROMPT = os.environ.get(
    "TELESTYLE_DEFAULT_PROMPT",
    "Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1.",
)
DEFAULT_MIN_EDGE = int(os.environ.get("TELESTYLE_DEFAULT_MIN_EDGE", "1024"))
DEFAULT_STEPS = int(os.environ.get("TELESTYLE_DEFAULT_STEPS", "4"))
DEFAULT_GUIDANCE = float(os.environ.get("TELESTYLE_DEFAULT_GUIDANCE", "1.0"))
DEFAULT_USE_CONTENT_PROMPT = os.environ.get(
    "TELESTYLE_USE_CONTENT_PROMPT",
    "0",
).lower() in {"1", "true", "yes", "on"}
DEFAULT_USE_STYLE_PROMPT = os.environ.get(
    "TELESTYLE_USE_STYLE_PROMPT",
    "0",
).lower() in {"1", "true", "yes", "on"}
ENABLE_CPU_OFFLOAD = os.environ.get(
    "TELESTYLE_ENABLE_CPU_OFFLOAD",
    "0",
).lower() in {"1", "true", "yes", "on"}
FUSE_LORA = os.environ.get("TELESTYLE_FUSE_LORA", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MAX_SEED = 2**31 - 1

if TELESTYLE_REPO_DIR.exists():
    import sys

    sys.path.insert(0, str(TELESTYLE_REPO_DIR))

pipe = None
model_loading = False
model_error: Optional[str] = None
model_load_lock = threading.Lock()
model_infer_lock = asyncio.Lock()
jobs: Dict[str, Dict[str, Any]] = {}
executor = ThreadPoolExecutor(max_workers=1)

app = FastAPI(title="TeleStyleV2 API", version="1.0.0")


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
    content_prompt: Optional[str] = None
    style_prompt: Optional[str] = None
    final_prompt: Optional[str] = None


def _clear_cuda_cache(reason: str) -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache %s", reason)
    except Exception as exc:
        logger.debug("Could not clear CUDA cache %s: %s", reason, exc)


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
            from pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                "Loading TeleStyleV2 pipeline model=%s style_lora=%s dmd_lora=%s device=%s cpu_offload=%s",
                MODEL_PATH,
                STYLE_LORA,
                DMD_LORA,
                device,
                ENABLE_CPU_OFFLOAD,
            )
            model = QwenImageEditPlusPipeline.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
            )
            if device == "cuda" and ENABLE_CPU_OFFLOAD:
                model.enable_model_cpu_offload()
            else:
                model.to(device)

            model.set_progress_bar_config(disable=None)
            model.load_lora_weights(STYLE_LORA, adapter_name="style")
            model.load_lora_weights(DMD_LORA, adapter_name="dmd")
            model.set_adapters(["style", "dmd"], adapter_weights=[1.0, 1.0])
            if FUSE_LORA:
                model.fuse_lora(adapter_names=["style", "dmd"], lora_scale=1.0)
                model.unload_lora_weights()

            pipe = model
            snapshot = _gpu_memory_snapshot()
            if snapshot:
                logger.info("TeleStyleV2 model loaded; %s", snapshot)
            else:
                logger.info("TeleStyleV2 model loaded")
            return pipe
        except Exception as exc:
            model_error = str(exc)
            logger.error("Failed to load TeleStyleV2 model: %s", exc, exc_info=True)
            _clear_cuda_cache("after model load failure")
            raise
        finally:
            model_loading = False


def _safe_seed(seed: Optional[int], randomize_seed: bool) -> int:
    if randomize_seed or seed is None:
        return random.randint(0, MAX_SEED)
    return max(0, min(MAX_SEED, int(seed)))


def _round_to_16(value: int) -> int:
    return max(16, int(value) - (int(value) % 16))


def _resize_for_min_edge(image: Image.Image, min_edge: int) -> Image.Image:
    min_edge = max(256, min(2048, _round_to_16(min_edge)))
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Input image has invalid dimensions")
    if width > height:
        new_height = min_edge
        new_width = _round_to_16((width / height) * new_height)
    else:
        new_width = min_edge
        new_height = _round_to_16((height / width) * new_width)
    return image.resize((new_width, new_height), Image.LANCZOS)


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _describe_reference_image(image: Image.Image, instruction: str) -> str:
    from qwen_vl_utils import process_vision_info

    model = _load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    text = model.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = model.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    generated_ids = model.text_encoder.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return model.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def _merge_prompt(
    prompt: str,
    content_prompt: str,
    style_prompt: str,
    use_content_prompt: bool,
    use_style_prompt: bool,
) -> str:
    pieces = [prompt.strip() or DEFAULT_PROMPT]
    if use_content_prompt and content_prompt and content_prompt not in pieces[0]:
        pieces.append(content_prompt)
    if use_style_prompt and style_prompt and style_prompt not in pieces[0]:
        pieces.append(style_prompt)
    return ",".join(piece.strip(", ") for piece in pieces if piece.strip(", "))


def _run_generation(
    job_id: str,
    content_path: Optional[str],
    style_path: Optional[str],
    prompt: str,
    seed: Optional[int],
    randomize_seed: bool,
    true_guidance_scale: float,
    num_inference_steps: int,
    min_edge: int,
    use_content_prompt: bool,
    use_style_prompt: bool,
) -> Dict[str, Any]:
    model = _load_model()
    content_image = _open_rgb(Path(content_path)) if content_path else None
    style_image = _open_rgb(Path(style_path)) if style_path else None
    if content_image is None and style_image is None:
        raise ValueError("At least one content or style reference image is required")

    content_prompt = ""
    style_prompt = ""
    if content_image is not None and use_content_prompt:
        content_prompt = _describe_reference_image(
            content_image,
            "describe main objects (fewer than 3) with separated words, each word is separated by comma, the total number of words is strictly fewer than 3",
        )
        logger.info("TeleStyleV2 content prompt: %s", content_prompt)
    if style_image is not None and use_style_prompt:
        style_prompt = _describe_reference_image(
            style_image,
            "describe only the artistic style, material and stroke, lighting, color in 5 words, not objects.",
        )
        logger.info("TeleStyleV2 style prompt: %s", style_prompt)

    images = []
    if content_image is not None:
        content_image = _resize_for_min_edge(content_image, min_edge)
        images.append(content_image)
    if style_image is not None:
        style_image = _resize_for_min_edge(style_image, min_edge)
        images.append(style_image)

    base_image = content_image or style_image
    if base_image is None:
        raise ValueError("No resized reference image available")

    resolved_seed = _safe_seed(seed, randomize_seed)
    final_prompt = _merge_prompt(
        prompt,
        content_prompt,
        style_prompt,
        use_content_prompt,
        use_style_prompt,
    )
    output_path = OUTPUT_DIR / f"{job_id}.png"

    logger.info(
        "Running TeleStyleV2 job %s size=%sx%s seed=%s steps=%s guidance=%s refs=%s prompt=%r",
        job_id,
        base_image.size[0],
        base_image.size[1],
        resolved_seed,
        num_inference_steps,
        true_guidance_scale,
        len(images),
        final_prompt,
    )
    snapshot = _gpu_memory_snapshot()
    if snapshot:
        logger.info("TeleStyleV2 job %s pre-generation memory: %s", job_id, snapshot)

    payload = {
        "image": images,
        "prompt": final_prompt,
        "generator": torch.manual_seed(resolved_seed),
        "true_cfg_scale": true_guidance_scale,
        "negative_prompt": " ",
        "num_inference_steps": max(1, min(50, int(num_inference_steps))),
        "guidance_scale": true_guidance_scale,
        "num_images_per_prompt": 1,
        "width": base_image.size[0],
        "height": base_image.size[1],
    }
    with torch.inference_mode():
        result = model(**payload)
    if not result.images:
        raise RuntimeError("TeleStyleV2 returned no images")

    result.images[0].save(output_path)
    return {
        "output_path": str(output_path),
        "seed": resolved_seed,
        "content_prompt": content_prompt,
        "style_prompt": style_prompt,
        "final_prompt": final_prompt,
    }


async def _process_generation_job(job_id: str, request: Dict[str, Any]) -> None:
    try:
        jobs[job_id]["status"] = "processing"
        async with model_infer_lock:
            result = await asyncio.get_running_loop().run_in_executor(
                executor,
                _run_generation,
                job_id,
                request.get("content_path"),
                request.get("style_path"),
                request.get("prompt", ""),
                request.get("seed"),
                request.get("randomize_seed", False),
                request.get("true_guidance_scale", DEFAULT_GUIDANCE),
                request.get("num_inference_steps", DEFAULT_STEPS),
                request.get("min_edge", DEFAULT_MIN_EDGE),
                request.get("use_content_prompt", DEFAULT_USE_CONTENT_PROMPT),
                request.get("use_style_prompt", DEFAULT_USE_STYLE_PROMPT),
            )
        jobs[job_id].update({"status": "completed", **result})
        logger.info("TeleStyleV2 job %s completed", job_id)
    except Exception as exc:
        logger.error("TeleStyleV2 job %s failed: %s", job_id, exc, exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)
    finally:
        _clear_cuda_cache(f"after job {job_id}")


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


async def _save_upload(upload: Optional[UploadFile], job_id: str, label: str) -> Optional[str]:
    if upload is None:
        return None
    suffix = Path(upload.filename or f"{label}.png").suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"
    destination = INPUT_DIR / f"{job_id}_{label}{suffix}"
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"{label} image is empty")
    destination.write_bytes(data)
    try:
        with Image.open(destination) as image:
            image.verify()
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"{label} image is not a valid image: {exc}",
        ) from exc
    return str(destination)


@app.on_event("startup")
async def startup_event() -> None:
    if os.environ.get("TELESTYLE_PRELOAD", "1") != "0":
        asyncio.create_task(asyncio.to_thread(_load_model))


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    status = "healthy" if pipe is not None else "loading"
    if model_error:
        status = "error"
    return HealthResponse(
        status=status,
        model_id=str(MODEL_PATH),
        model_loaded=pipe is not None,
        error=model_error,
    )


@app.post("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    content_image: Optional[UploadFile] = File(None),
    style_image: Optional[UploadFile] = File(None),
    prompt: str = Form(DEFAULT_PROMPT),
    seed: Optional[int] = Form(None),
    randomize_seed: bool = Form(False),
    true_guidance_scale: float = Form(DEFAULT_GUIDANCE),
    num_inference_steps: int = Form(DEFAULT_STEPS),
    min_edge: int = Form(DEFAULT_MIN_EDGE),
    use_content_prompt: Optional[str] = Form(None),
    use_style_prompt: Optional[str] = Form(None),
) -> Dict[str, str]:
    if pipe is None and model_error:
        raise HTTPException(status_code=503, detail=f"Model failed to load: {model_error}")

    job_id = str(uuid.uuid4())
    content_path = await _save_upload(content_image, job_id, "content")
    style_path = await _save_upload(style_image, job_id, "style")
    if not content_path and not style_path:
        raise HTTPException(
            status_code=400,
            detail="At least one content_image or style_image file is required",
        )

    jobs[job_id] = {"job_id": job_id, "status": "queued"}
    request = {
        "content_path": content_path,
        "style_path": style_path,
        "prompt": prompt,
        "seed": seed,
        "randomize_seed": randomize_seed,
        "true_guidance_scale": true_guidance_scale,
        "num_inference_steps": num_inference_steps,
        "min_edge": min_edge,
        "use_content_prompt": _parse_bool(
            use_content_prompt,
            DEFAULT_USE_CONTENT_PROMPT,
        ),
        "use_style_prompt": _parse_bool(use_style_prompt, DEFAULT_USE_STYLE_PROMPT),
    }
    background_tasks.add_task(_process_generation_job, job_id, request)
    return {"job_id": job_id}


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str) -> JobStatus:
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**jobs[job_id])


@app.get("/output/{job_id}")
async def get_output(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Job status: {job.get('status')}")
    output_path = job.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(output_path, media_type="image/png", filename=f"{job_id}.png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
