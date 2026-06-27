#!/usr/bin/env python3
"""FastAPI wrapper for official LTX-2 pipelines inference."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

app = FastAPI(title="OpenFork LTX-2 Official Pipelines API", version="1.0.0")

LTX2_REPO = Path(os.environ.get("LTX2_REPO", "/opt/LTX-2"))
PIPELINES_DIR = Path(
    os.environ.get("LTX2_PIPELINES_DIR", str(LTX2_REPO / "packages" / "ltx-pipelines"))
)
MODEL_PATH = Path(
    os.environ.get("LTX2_MODEL_PATH", "/models/ltx2/ltx-2.3-22b-dev.safetensors")
)
DISTILLED_CHECKPOINT_PATH = Path(
    os.environ.get(
        "LTX2_DISTILLED_CHECKPOINT_PATH",
        "/models/ltx2/ltx-2.3-22b-distilled-1.1.safetensors",
    )
)
GEMMA_ROOT = Path(
    os.environ.get(
        "LTX2_GEMMA_ROOT",
        os.environ.get("LTX2_TEXT_ENCODER_PATH", "/models/gemma-3-12b-it-qat-q4_0-unquantized"),
    )
)
DISTILLED_LORA_PATH = Path(
    os.environ.get(
        "LTX2_DISTILLED_LORA_PATH",
        "/models/ltx2/ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
    )
)
SPATIAL_UPSAMPLER_PATH = Path(
    os.environ.get(
        "LTX2_SPATIAL_UPSAMPLER_PATH",
        "/models/ltx2/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    )
)
JOBS_ROOT = Path(os.environ.get("LTX2_PIPELINES_JOBS_ROOT", "/app/ltx2_pipeline_jobs"))
OUTPUT_DIR = Path(os.environ.get("LTX2_OUTPUT_DIR", "/app/output"))
TARGET_VRAM_GB = int(os.environ.get("LTX2_TARGET_VRAM_GB", "32"))
VRAM_PROFILE = os.environ.get("LTX2_VRAM_PROFILE", f"{TARGET_VRAM_GB}gb")
DEFAULT_WIDTH = int(os.environ.get("LTX2_PIPELINES_DEFAULT_WIDTH", "1024"))
DEFAULT_HEIGHT = int(os.environ.get("LTX2_PIPELINES_DEFAULT_HEIGHT", "576"))
DEFAULT_NUM_FRAMES = int(os.environ.get("LTX2_PIPELINES_DEFAULT_NUM_FRAMES", "49"))
DEFAULT_STEPS = int(os.environ.get("LTX2_PIPELINES_DEFAULT_STEPS", "30"))
DEFAULT_OFFLOAD_MODE = os.environ.get("LTX2_PIPELINES_DEFAULT_OFFLOAD", "cpu")
DEFAULT_QUANTIZATION = os.environ.get("LTX2_PIPELINES_DEFAULT_QUANTIZATION", "fp8-cast")
PIPELINE_MODULE = os.environ.get("LTX2_PIPELINES_MODULE", "ltx_pipelines.ti2vid_two_stages")

SUPPORTED_PIPELINE_MODULES = {
    "ltx_pipelines.ti2vid_two_stages",
    "ltx_pipelines.distilled",
}

JOBS_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()


def _set_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        current = jobs.setdefault(job_id, {})
        current.update(updates)
        current["updated_at"] = time.time()


def _get_job(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(job)


def _snap_to_multiple(value: int, multiple: int, minimum: int) -> int:
    return max(minimum, int(round(value / multiple)) * multiple)


def _snap_frames(value: int) -> int:
    value = max(1, int(value))
    return value if value % 8 == 1 else max(1, ((value - 1) // 8) * 8 + 1)


def _extract_lora_if_needed(lora_path: Path, workspace: Path) -> Path:
    if lora_path.suffix.lower() != ".zip":
        return lora_path

    extract_dir = workspace / "lora_package"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(lora_path) as archive:
        root = extract_dir.resolve()
        for member in archive.infolist():
            target = (extract_dir / member.filename).resolve()
            if root not in (target, *target.parents):
                raise ValueError(f"Unsafe LoRA archive path: {member.filename}")
        archive.extractall(extract_dir)

    candidates = sorted(
        [path for path in extract_dir.rglob("*.safetensors") if path.is_file()],
        key=lambda path: (path.stat().st_mtime, path.stat().st_size),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No .safetensors LoRA file found in ZIP package")
    return candidates[0]


def _is_distilled_pipeline() -> bool:
    return PIPELINE_MODULE == "ltx_pipelines.distilled"


def _required_model_paths() -> list[Path]:
    if _is_distilled_pipeline():
        return [DISTILLED_CHECKPOINT_PATH, GEMMA_ROOT, SPATIAL_UPSAMPLER_PATH]
    return [MODEL_PATH, GEMMA_ROOT, DISTILLED_LORA_PATH, SPATIAL_UPSAMPLER_PATH]


def _run_pipeline(
    job_id: str,
    *,
    image_path: Path | None,
    lora_path: Path | None,
    prompt: str,
    negative_prompt: str,
    trigger: str,
    lora_strength: float,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: float,
    steps: int,
    seed: int,
    image_strength: float,
    offload_mode: str,
    quantization: str,
    distilled_lora_strength: float,
) -> None:
    workspace = JOBS_ROOT / job_id
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    log_path = workspace / "pipeline.log"

    try:
        workspace.mkdir(parents=True, exist_ok=True)
        selected_lora = _extract_lora_if_needed(lora_path, workspace) if lora_path else None
        final_prompt = " ".join(part for part in [trigger.strip(), prompt.strip()] if part)
        width = _snap_to_multiple(width, 64, 512)
        height = _snap_to_multiple(height, 64, 288)
        num_frames = _snap_frames(num_frames)
        offload_mode = offload_mode if offload_mode in {"none", "cpu", "disk"} else "cpu"

        if PIPELINE_MODULE not in SUPPORTED_PIPELINE_MODULES:
            raise ValueError(f"Unsupported LTX-2 pipeline module: {PIPELINE_MODULE}")

        command = [
            "uv",
            "run",
            "python",
            "-m",
            PIPELINE_MODULE,
        ]
        if _is_distilled_pipeline():
            command.extend(["--distilled-checkpoint-path", str(DISTILLED_CHECKPOINT_PATH)])
        else:
            command.extend(
                [
                    "--checkpoint-path",
                    str(MODEL_PATH),
                    "--distilled-lora",
                    str(DISTILLED_LORA_PATH),
                    str(distilled_lora_strength),
                ]
            )

        command.extend(
            [
                "--gemma-root",
                str(GEMMA_ROOT),
                "--spatial-upsampler-path",
                str(SPATIAL_UPSAMPLER_PATH),
                "--prompt",
                final_prompt,
                "--width",
                str(width),
                "--height",
                str(height),
                "--num-frames",
                str(num_frames),
                "--frame-rate",
                str(frame_rate),
                "--seed",
                str(seed),
                "--offload",
                offload_mode,
                "--max-batch-size",
                "1",
                "--output-path",
                str(output_path),
            ]
        )
        if selected_lora:
            command.extend(["--lora", str(selected_lora), str(lora_strength)])
        if image_path:
            command.extend(["--image", str(image_path), "0", str(image_strength)])
        if not _is_distilled_pipeline():
            command.extend(["--negative-prompt", negative_prompt])
        if not _is_distilled_pipeline():
            command.extend(["--num-inference-steps", str(max(1, steps))])
        if quantization:
            command.extend(["--quantization", quantization])

        _set_job(
            job_id,
            status="running",
            phase="generate",
            command=" ".join(command),
            settings={
                "mode": "image-to-video-lora" if image_path else "text-to-video",
                "pipeline_module": PIPELINE_MODULE,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "steps": "distilled-fixed" if _is_distilled_pipeline() else steps,
                "seed": seed,
                "has_lora": selected_lora is not None,
                "lora_strength": lora_strength if selected_lora else None,
                "image_strength": image_strength if image_path else None,
                "offload_mode": offload_mode,
                "quantization": quantization,
            },
        )
        with log_path.open("a", encoding="utf-8", errors="replace") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(PIPELINES_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            assert process.stdout is not None
            tail: list[str] = []
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                tail.append(line.rstrip())
                tail = tail[-80:]
                if len(tail) % 20 == 0:
                    _set_job(job_id, log_tail=tail)
            return_code = process.wait()
            _set_job(job_id, log_tail=tail)
        if return_code != 0:
            raise RuntimeError(f"LTX-2 pipeline failed with exit code {return_code}")
        if not output_path.exists():
            raise FileNotFoundError(f"Pipeline did not create {output_path}")
        _set_job(
            job_id,
            status="completed",
            phase="completed",
            output_path=str(output_path),
            output_size_bytes=output_path.stat().st_size,
        )
    except Exception as exc:
        _set_job(job_id, status="failed", phase="failed", error=str(exc))


@app.get("/health")
def health() -> dict[str, Any]:
    required = _required_model_paths()
    module_supported = PIPELINE_MODULE in SUPPORTED_PIPELINE_MODULES
    model_ready = all(path.exists() for path in required)
    return {
        "status": "ok" if module_supported and model_ready else "error",
        "model_loaded": module_supported and model_ready,
        "models_present": model_ready,
        "module_supported": module_supported,
        "pipeline_module": PIPELINE_MODULE,
        "pipeline_dir": str(PIPELINES_DIR),
        "checkpoint": str(MODEL_PATH),
        "distilled_checkpoint": str(DISTILLED_CHECKPOINT_PATH),
        "gemma_root": str(GEMMA_ROOT),
        "distilled_lora": str(DISTILLED_LORA_PATH),
        "spatial_upsampler": str(SPATIAL_UPSAMPLER_PATH),
        "missing_paths": [str(path) for path in required if not path.exists()],
        "target_vram_gb": TARGET_VRAM_GB,
        "vram_profile": VRAM_PROFILE,
        "default_width": DEFAULT_WIDTH,
        "default_height": DEFAULT_HEIGHT,
        "default_num_frames": DEFAULT_NUM_FRAMES,
        "default_steps": DEFAULT_STEPS,
        "default_offload_mode": DEFAULT_OFFLOAD_MODE,
        "default_quantization": DEFAULT_QUANTIZATION,
        "experimental": TARGET_VRAM_GB < 32,
        "notes": "Official LTX-2 ltx-pipelines two-stage T2V plus I2V LoRA support.",
    }


def _assert_generation_ready() -> None:
    if PIPELINE_MODULE not in SUPPORTED_PIPELINE_MODULES:
        raise HTTPException(status_code=503, detail=f"Unsupported LTX-2 pipeline module: {PIPELINE_MODULE}")
    if not all(path.exists() for path in _required_model_paths()):
        raise HTTPException(status_code=503, detail="One or more LTX-2 pipeline model files are missing.")
    if not PIPELINES_DIR.exists():
        raise HTTPException(status_code=503, detail="LTX-2 pipelines directory is missing.")


@app.post("/generate/i2v-lora")
async def generate_i2v_lora(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    lora: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(
        "worst quality, low quality, blurry, jittery, distorted anatomy, text, subtitles, "
        "captions, logos, watermarks, slow motion, frozen motion, reverse loop, ping-pong motion"
    ),
    trigger: str = Form(""),
    lora_strength: float = Form(0.8),
    width: int = Form(DEFAULT_WIDTH),
    height: int = Form(DEFAULT_HEIGHT),
    num_frames: int = Form(DEFAULT_NUM_FRAMES),
    frame_rate: float = Form(25.0),
    steps: int = Form(DEFAULT_STEPS),
    seed: int = Form(42),
    image_strength: float = Form(1.0),
    offload_mode: str = Form(DEFAULT_OFFLOAD_MODE),
    quantization: str = Form(DEFAULT_QUANTIZATION),
    distilled_lora_strength: float = Form(0.8),
) -> dict[str, str]:
    _assert_generation_ready()

    job_id = uuid.uuid4().hex
    workspace = JOBS_ROOT / job_id
    workspace.mkdir(parents=True, exist_ok=True)
    image_suffix = Path(image.filename or "image.png").suffix or ".png"
    lora_suffix = Path(lora.filename or "lora.safetensors").suffix or ".safetensors"
    image_path = workspace / f"input{image_suffix}"
    lora_path = workspace / f"lora{lora_suffix}"
    with image_path.open("wb") as handle:
        shutil.copyfileobj(image.file, handle)
    with lora_path.open("wb") as handle:
        shutil.copyfileobj(lora.file, handle)

    _set_job(job_id, status="queued", phase="queued", created_at=time.time())
    background_tasks.add_task(
        _run_pipeline,
        job_id,
        image_path=image_path,
        lora_path=lora_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        trigger=trigger,
        lora_strength=float(lora_strength),
        width=int(width),
        height=int(height),
        num_frames=int(num_frames),
        frame_rate=float(frame_rate),
        steps=int(steps),
        seed=int(seed),
        image_strength=float(image_strength),
        offload_mode=offload_mode,
        quantization=quantization,
        distilled_lora_strength=float(distilled_lora_strength),
    )
    return {"job_id": job_id, "status": "queued"}


@app.post("/generate/t2v")
async def generate_t2v(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    lora: Optional[UploadFile] = File(None),
    negative_prompt: str = Form(
        "worst quality, low quality, blurry, jittery, distorted anatomy, text, subtitles, "
        "captions, logos, watermarks, slow motion, frozen motion, reverse loop, ping-pong motion"
    ),
    trigger: str = Form(""),
    lora_strength: float = Form(0.8),
    width: int = Form(DEFAULT_WIDTH),
    height: int = Form(DEFAULT_HEIGHT),
    num_frames: int = Form(DEFAULT_NUM_FRAMES),
    frame_rate: float = Form(25.0),
    steps: int = Form(DEFAULT_STEPS),
    seed: int = Form(42),
    offload_mode: str = Form(DEFAULT_OFFLOAD_MODE),
    quantization: str = Form(DEFAULT_QUANTIZATION),
    distilled_lora_strength: float = Form(0.8),
) -> dict[str, str]:
    _assert_generation_ready()

    job_id = uuid.uuid4().hex
    workspace = JOBS_ROOT / job_id
    workspace.mkdir(parents=True, exist_ok=True)
    lora_path: Path | None = None
    if lora:
        lora_suffix = Path(lora.filename or "lora.safetensors").suffix or ".safetensors"
        lora_path = workspace / f"lora{lora_suffix}"
        with lora_path.open("wb") as handle:
            shutil.copyfileobj(lora.file, handle)

    _set_job(job_id, status="queued", phase="queued", created_at=time.time())
    background_tasks.add_task(
        _run_pipeline,
        job_id,
        image_path=None,
        lora_path=lora_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        trigger=trigger,
        lora_strength=float(lora_strength),
        width=int(width),
        height=int(height),
        num_frames=int(num_frames),
        frame_rate=float(frame_rate),
        steps=int(steps),
        seed=int(seed),
        image_strength=0.0,
        offload_mode=offload_mode,
        quantization=quantization,
        distilled_lora_strength=float(distilled_lora_strength),
    )
    return {"job_id": job_id, "status": "queued"}


@app.post("/generate/text-to-video")
async def generate_text_to_video(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    lora: Optional[UploadFile] = File(None),
    negative_prompt: str = Form(
        "worst quality, low quality, blurry, jittery, distorted anatomy, text, subtitles, "
        "captions, logos, watermarks, slow motion, frozen motion, reverse loop, ping-pong motion"
    ),
    trigger: str = Form(""),
    lora_strength: float = Form(0.8),
    width: int = Form(DEFAULT_WIDTH),
    height: int = Form(DEFAULT_HEIGHT),
    num_frames: int = Form(DEFAULT_NUM_FRAMES),
    frame_rate: float = Form(25.0),
    steps: int = Form(DEFAULT_STEPS),
    seed: int = Form(42),
    offload_mode: str = Form(DEFAULT_OFFLOAD_MODE),
    quantization: str = Form(DEFAULT_QUANTIZATION),
    distilled_lora_strength: float = Form(0.8),
) -> dict[str, str]:
    return await generate_t2v(
        background_tasks,
        prompt=prompt,
        lora=lora,
        negative_prompt=negative_prompt,
        trigger=trigger,
        lora_strength=lora_strength,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=frame_rate,
        steps=steps,
        seed=seed,
        offload_mode=offload_mode,
        quantization=quantization,
        distilled_lora_strength=distilled_lora_strength,
    )


@app.get("/status/{job_id}")
def status(job_id: str) -> dict[str, Any]:
    return _get_job(job_id)


@app.get("/output/{job_id}")
def output(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Job is not completed")
    output_path = Path(str(job.get("output_path", "")))
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output not found")
    return FileResponse(output_path, media_type="video/mp4", filename=output_path.name)


@app.delete("/job/{job_id}")
def delete_job(job_id: str) -> dict[str, bool]:
    with jobs_lock:
        jobs.pop(job_id, None)
    shutil.rmtree(JOBS_ROOT / job_id, ignore_errors=True)
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    if output_path.exists():
        output_path.unlink()
    return {"deleted": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
