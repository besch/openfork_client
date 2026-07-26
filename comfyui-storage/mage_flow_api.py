"""OpenFork REST adapter for Microsoft Mage-Flow generation and editing."""

from __future__ import annotations

import gc
import os
import threading
import time
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

OUTPUT_DIR = Path(os.getenv("MAGE_FLOW_OUTPUT_DIR", "/app/output"))
INPUT_DIR = Path(os.getenv("MAGE_FLOW_INPUT_DIR", "/app/input"))
MODEL_ROOT = Path(os.getenv("MAGE_FLOW_MODEL_ROOT", "/app/models"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    ("generate", "quality"): Path(
        os.getenv("MAGE_FLOW_MODEL", str(MODEL_ROOT / "Mage-Flow"))
    ),
    ("generate", "turbo"): Path(
        os.getenv("MAGE_FLOW_TURBO_MODEL", str(MODEL_ROOT / "Mage-Flow-Turbo"))
    ),
    ("edit", "quality"): Path(
        os.getenv("MAGE_FLOW_EDIT_MODEL", str(MODEL_ROOT / "Mage-Flow-Edit"))
    ),
    ("edit", "turbo"): Path(
        os.getenv(
            "MAGE_FLOW_EDIT_TURBO_MODEL",
            str(MODEL_ROOT / "Mage-Flow-Edit-Turbo"),
        )
    ),
}

app = FastAPI(title="OpenFork Mage-Flow API")
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
generation_lock = threading.Lock()
active_pipeline = None
active_pipeline_key: tuple[str, str] | None = None
model_error: str | None = None


def _clamp_dimension(value: int) -> int:
    return max(512, min(2048, int(round(value / 16) * 16)))


def _load_pipeline(operation: str, variant: str):
    global active_pipeline, active_pipeline_key
    key = (operation, variant)
    if active_pipeline is not None and active_pipeline_key == key:
        return active_pipeline

    active_pipeline = None
    active_pipeline_key = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_path = MODEL_PATHS[key]
    if not model_path.exists():
        raise RuntimeError(f"Mage-Flow model is not baked at {model_path}")

    from mage_flow import MageFlowPipeline

    active_pipeline = MageFlowPipeline.from_pretrained(
        str(model_path),
        device="cuda",
    )
    active_pipeline_key = key
    return active_pipeline


def _run_job(
    job_id: str,
    operation: str,
    variant: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    reference_paths: list[Path],
):
    output_path = OUTPUT_DIR / f"{job_id}.png"
    try:
        with generation_lock:
            with jobs_lock:
                jobs[job_id]["status"] = "processing"
            pipe = _load_pipeline(operation, variant)
            common = {
                "steps": steps,
                "cfg": cfg,
                "seeds": [seed],
                "neg_prompts": [negative_prompt or " "],
            }
            if operation == "edit":
                if not reference_paths:
                    raise RuntimeError("Mage-Flow edit requires at least one image")
                refs: object = (
                    str(reference_paths[0])
                    if len(reference_paths) == 1
                    else [str(path) for path in reference_paths]
                )
                image = pipe.edit(
                    [prompt],
                    [refs],
                    heights=[height],
                    widths=[width],
                    **common,
                )[0]
            else:
                image = pipe.generate(
                    [prompt],
                    heights=[height],
                    widths=[width],
                    **common,
                )[0]
            image.save(output_path)
            with jobs_lock:
                jobs[job_id].update(
                    status="completed",
                    output=str(output_path),
                    reference_count=len(reference_paths),
                )
    except Exception as exc:
        with jobs_lock:
            jobs[job_id].update(status="failed", error=str(exc))
    finally:
        for path in reference_paths:
            path.unlink(missing_ok=True)


@app.get("/health")
def health():
    missing = [str(path) for path in MODEL_PATHS.values() if not path.exists()]
    return {
        "status": "error" if model_error or missing else "ok",
        "model_loaded": not missing,
        "active_model": "-".join(active_pipeline_key) if active_pipeline_key else None,
        "error": model_error or (f"Missing baked models: {', '.join(missing)}" if missing else None),
    }


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    operation: str = Form("generate"),
    variant: str = Form("quality"),
    negative_prompt: str = Form(""),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(20),
    cfg: float = Form(5.0),
    seed: int = Form(42),
    reference_images: list[UploadFile] = File(default=[]),
):
    if operation not in {"generate", "edit"}:
        raise HTTPException(400, "operation must be generate or edit")
    if variant not in {"quality", "turbo"}:
        raise HTTPException(400, "variant must be quality or turbo")
    if operation == "edit" and not reference_images:
        raise HTTPException(400, "Mage-Flow edit requires reference_images")

    width, height = _clamp_dimension(width), _clamp_dimension(height)
    if max(width / height, height / width) > 4:
        raise HTTPException(400, "Mage-Flow aspect ratio may not exceed 4:1")
    if variant == "turbo":
        steps, cfg = 4, 1.0
    else:
        default_steps = 30 if operation == "edit" else 20
        steps = max(1, min(75, steps or default_steps))
        cfg = max(0.0, min(20.0, cfg))

    job_id = str(uuid.uuid4())
    reference_paths: list[Path] = []
    for index, upload in enumerate(reference_images[:8]):
        suffix = Path(upload.filename or "").suffix or ".png"
        path = INPUT_DIR / f"{job_id}_{index}{suffix}"
        path.write_bytes(await upload.read())
        try:
            Image.open(path).verify()
        except Exception as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(400, f"Invalid reference image {index + 1}: {exc}")
        reference_paths.append(path)

    with jobs_lock:
        jobs[job_id] = {
            "status": "pending",
            "created_at": time.time(),
            "operation": operation,
            "variant": variant,
        }
    threading.Thread(
        target=_run_job,
        args=(
            job_id,
            operation,
            variant,
            prompt,
            negative_prompt,
            width,
            height,
            steps,
            cfg,
            seed,
            reference_paths,
        ),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return {key: value for key, value in job.items() if key != "output"}


@app.get("/output/{job_id}")
def output(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(404, "Output is not ready")
    return FileResponse(job["output"], media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
