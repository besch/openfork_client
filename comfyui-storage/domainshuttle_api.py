"""DomainShuttle REST API wrapper for OpenFork.

The upstream project ships a batch JSONL runner. This wrapper turns that runner
into the small REST surface used by OpenFork REST processors.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

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
INPUT_DIR = Path(os.environ.get("DOMAINSHUTTLE_INPUT_DIR", str(WORK_DIR / "input")))
OUTPUT_DIR = Path(os.environ.get("DOMAINSHUTTLE_OUTPUT_DIR", str(WORK_DIR / "output")))
REPO_DIR = Path(os.environ.get("DOMAINSHUTTLE_REPO_DIR", "/opt/DomainShuttle"))
MODEL_PATH = Path(
    os.environ.get(
        "DOMAINSHUTTLE_MODEL_PATH",
        "/models/DomainShuttle/Wan2.2-DomainShuttle-A14B",
    )
)
PREDICT_SCRIPT = REPO_DIR / "examples" / "wan2.2_domainshuttle" / "predict_r2v_batch.py"
CONVERT_SCRIPT = REPO_DIR / "scripts" / "wan2.2_domainshuttle" / "convert_mindspeed_domainshuttle.py"

DEFAULT_WIDTH = int(os.environ.get("DOMAINSHUTTLE_DEFAULT_WIDTH", "832"))
DEFAULT_HEIGHT = int(os.environ.get("DOMAINSHUTTLE_DEFAULT_HEIGHT", "480"))
DEFAULT_VIDEO_LENGTH = int(os.environ.get("DOMAINSHUTTLE_DEFAULT_VIDEO_LENGTH", "81"))
DEFAULT_FPS = int(os.environ.get("DOMAINSHUTTLE_DEFAULT_FPS", "24"))
DEFAULT_STEPS = int(os.environ.get("DOMAINSHUTTLE_DEFAULT_STEPS", "40"))
DEFAULT_SHIFT = float(os.environ.get("DOMAINSHUTTLE_DEFAULT_SHIFT", "5"))
DEFAULT_GUIDANCE_HIGH = float(os.environ.get("DOMAINSHUTTLE_DEFAULT_GUIDANCE_HIGH", "4.0"))
DEFAULT_GUIDANCE_LOW = float(os.environ.get("DOMAINSHUTTLE_DEFAULT_GUIDANCE_LOW", "3.0"))
DEFAULT_NPROC_PER_NODE = int(os.environ.get("DOMAINSHUTTLE_NPROC_PER_NODE", "1"))
DEFAULT_RING_DEGREE = int(os.environ.get("DOMAINSHUTTLE_RING_DEGREE", "1"))
MASTER_PORT_BASE = int(os.environ.get("DOMAINSHUTTLE_MASTER_PORT", "12345"))
MAX_LOG_LINES = int(os.environ.get("DOMAINSHUTTLE_MAX_LOG_LINES", "240"))

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

jobs: Dict[str, Dict[str, Any]] = {}
job_lock = threading.Lock()
convert_lock = threading.Lock()
model_error: Optional[str] = None

app = FastAPI(title="DomainShuttle API", version="1.0.0")


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
    log_tail: Optional[List[str]] = None


def _safe_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _safe_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _safe_seed(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return max(0, min(2**31 - 1, int(value)))


def _converted_model_ready() -> bool:
    required = [
        MODEL_PATH / "high_noise_model" / "diffusion_pytorch_model.safetensors",
        MODEL_PATH / "low_noise_model" / "diffusion_pytorch_model.safetensors",
        MODEL_PATH / "configuration.json",
        MODEL_PATH / "Wan2.1_VAE.pth",
        MODEL_PATH / "models_t5_umt5-xxl-enc-bf16.pth",
        MODEL_PATH / "google",
    ]
    return PREDICT_SCRIPT.exists() and all(path.exists() for path in required)


def _raw_model_ready() -> bool:
    required = [
        MODEL_PATH / "high_noise_model",
        MODEL_PATH / "low_noise_model",
        MODEL_PATH / "configuration.json",
        MODEL_PATH / "Wan2.1_VAE.pth",
        MODEL_PATH / "models_t5_umt5-xxl-enc-bf16.pth",
        MODEL_PATH / "google",
    ]
    return PREDICT_SCRIPT.exists() and CONVERT_SCRIPT.exists() and all(path.exists() for path in required)


def _model_ready() -> bool:
    return _converted_model_ready() or _raw_model_ready()


def _append_log(job_id: str, line: str) -> None:
    with job_lock:
        job = jobs.get(job_id)
        if not job:
            return
        log_tail = job.setdefault("log_tail", [])
        log_tail.append(line.rstrip())
        if len(log_tail) > MAX_LOG_LINES:
            del log_tail[: len(log_tail) - MAX_LOG_LINES]


def _set_job(job_id: str, **updates: Any) -> None:
    with job_lock:
        if job_id in jobs:
            jobs[job_id].update(updates)


def _run_command(job_id: str, command: List[str], cwd: Path, env: Dict[str, str]) -> None:
    logger.info("Running DomainShuttle command for %s: %s", job_id, " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.info("[domainshuttle:%s] %s", job_id, line.rstrip())
        _append_log(job_id, line)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"DomainShuttle command exited with code {return_code}")


def _ensure_converted(job_id: str) -> None:
    global model_error
    if _converted_model_ready():
        return
    if not CONVERT_SCRIPT.exists():
        raise RuntimeError(
            f"DomainShuttle converted model files are missing and converter was not found at {CONVERT_SCRIPT}"
        )

    with convert_lock:
        if _converted_model_ready():
            return
        try:
            _append_log(job_id, "Converting DomainShuttle MindSpeed weights to diffusers layout")
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{REPO_DIR}:{env.get('PYTHONPATH', '')}"
            _run_command(
                job_id,
                [
                    "python",
                    str(CONVERT_SCRIPT),
                    "--output",
                    str(MODEL_PATH),
                ],
                REPO_DIR,
                env,
            )
            if not _converted_model_ready():
                raise RuntimeError("DomainShuttle conversion finished but required files are still missing")
        except Exception as exc:
            model_error = str(exc)
            raise


def _domain_codes(value: str, reference_count: int) -> List[str]:
    if not value:
        codes = ["Human"]
    else:
        value = value.strip()
        if value.startswith("["):
            try:
                parsed = json.loads(value)
                codes = [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:
                codes = [value]
        else:
            codes = [item.strip() for item in value.split(",") if item.strip()]
    if not codes:
        codes = ["Human"]
    if len(codes) < reference_count:
        codes.extend([codes[-1]] * (reference_count - len(codes)))
    return codes[:reference_count]


def _write_jsonl(
    job_id: str,
    prompt: str,
    reference_paths: List[Path],
    domain_code: List[str],
) -> Path:
    task_input_dir = INPUT_DIR / job_id
    jsonl_path = task_input_dir / "input.jsonl"
    relative_names = [path.name for path in reference_paths]
    record = {
        "prompt": prompt,
        "prompts": prompt,
        "cap": prompt,
        "domain_code": domain_code,
        "seg_root": str(task_input_dir),
        "seg_meta": [{"seg_file": relative_names}],
    }
    if len(reference_paths) == 1:
        record["image_path"] = str(reference_paths[0])
    jsonl_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return jsonl_path


def _find_output_video(output_dir: Path) -> Optional[Path]:
    candidates = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_size, path.stat().st_mtime))


def _generation_command(request: Dict[str, Any]) -> List[str]:
    nproc = _safe_int(request.get("nproc_per_node"), DEFAULT_NPROC_PER_NODE, 1, 8)
    ulysses_degree = _safe_int(request.get("ulysses_degree"), nproc, 1, 8)
    ring_degree = _safe_int(request.get("ring_degree"), DEFAULT_RING_DEGREE, 1, 8)
    master_port = MASTER_PORT_BASE + (int(request["job_id"].replace("-", "")[:4], 16) % 1000)
    seed = request.get("seed")

    command = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        f"--master_port={master_port}",
        str(PREDICT_SCRIPT),
        "--input_json",
        str(request["input_json"]),
        "--output_dir",
        str(request["output_dir"]),
        "--domain_model_name",
        str(MODEL_PATH),
        "--height",
        str(request["height"]),
        "--width",
        str(request["width"]),
        "--video_length",
        str(request["video_length"]),
        "--fps",
        str(request["fps"]),
        "--num_inference_steps",
        str(request["num_inference_steps"]),
        "--guidance_scale",
        str(request["guidance_scale_high"]),
        str(request["guidance_scale_low"]),
        "--shift",
        str(request["shift"]),
        "--ulysses_degree",
        str(ulysses_degree),
        "--ring_degree",
        str(ring_degree),
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    return command


def _run_generation(job_id: str, request: Dict[str, Any]) -> None:
    try:
        _set_job(job_id, status="processing")
        _ensure_converted(job_id)

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{REPO_DIR}:{env.get('PYTHONPATH', '')}"
        env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        command = _generation_command(request)
        _run_command(job_id, command, REPO_DIR, env)

        output_path = _find_output_video(Path(request["output_dir"]))
        if output_path is None:
            raise RuntimeError(f"No DomainShuttle output video found in {request['output_dir']}")
        _set_job(job_id, status="completed", output_path=str(output_path))
        logger.info("DomainShuttle job %s completed: %s", job_id, output_path)
    except Exception as exc:
        logger.error("DomainShuttle job %s failed: %s", job_id, exc, exc_info=True)
        _set_job(job_id, status="failed", error=str(exc))


async def _save_uploads(reference_images: List[UploadFile], job_id: str) -> List[Path]:
    task_input_dir = INPUT_DIR / job_id
    task_input_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for index, upload in enumerate(reference_images):
        suffix = Path(upload.filename or f"reference_{index}.png").suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            suffix = ".png"
        destination = task_input_dir / f"reference_{index:02d}{suffix}"
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"reference image {index + 1} is empty")
        destination.write_bytes(data)
        try:
            with Image.open(destination) as image:
                image.verify()
        except Exception as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail=f"reference image {index + 1} is not a valid image: {exc}",
            ) from exc
        paths.append(destination)
    return paths


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("DomainShuttle API starting: repo=%s model=%s", REPO_DIR, MODEL_PATH)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    ready = _model_ready()
    status = "healthy" if ready else "loading"
    if model_error:
        status = "error"
    return HealthResponse(
        status=status,
        model_id=str(MODEL_PATH),
        model_loaded=ready,
        error=model_error,
    )


@app.post("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    reference_images: List[UploadFile] = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    domain_code: str = Form("Human"),
    height: int = Form(DEFAULT_HEIGHT),
    width: int = Form(DEFAULT_WIDTH),
    video_length: int = Form(DEFAULT_VIDEO_LENGTH),
    fps: int = Form(DEFAULT_FPS),
    num_inference_steps: int = Form(DEFAULT_STEPS),
    guidance_scale_high: float = Form(DEFAULT_GUIDANCE_HIGH),
    guidance_scale_low: float = Form(DEFAULT_GUIDANCE_LOW),
    shift: float = Form(DEFAULT_SHIFT),
    seed: Optional[int] = Form(None),
    nproc_per_node: int = Form(DEFAULT_NPROC_PER_NODE),
    ring_degree: int = Form(DEFAULT_RING_DEGREE),
) -> Dict[str, str]:
    if not reference_images:
        raise HTTPException(status_code=400, detail="At least one reference image is required")
    if len(reference_images) > 5:
        raise HTTPException(status_code=400, detail="At most five reference images are supported")

    job_id = str(uuid.uuid4())
    reference_paths = await _save_uploads(reference_images, job_id)
    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    codes = _domain_codes(domain_code, len(reference_paths))
    input_json = _write_jsonl(job_id, prompt, reference_paths, codes)
    resolved_seed = _safe_seed(seed)

    request = {
        "job_id": job_id,
        "input_json": input_json,
        "output_dir": output_dir,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "domain_code": codes,
        "height": _safe_int(height, DEFAULT_HEIGHT, 256, 1024),
        "width": _safe_int(width, DEFAULT_WIDTH, 256, 1024),
        "video_length": _safe_int(video_length, DEFAULT_VIDEO_LENGTH, 17, 145),
        "fps": _safe_int(fps, DEFAULT_FPS, 8, 30),
        "num_inference_steps": _safe_int(num_inference_steps, DEFAULT_STEPS, 8, 50),
        "guidance_scale_high": _safe_float(guidance_scale_high, DEFAULT_GUIDANCE_HIGH, 1.0, 12.0),
        "guidance_scale_low": _safe_float(guidance_scale_low, DEFAULT_GUIDANCE_LOW, 1.0, 12.0),
        "shift": _safe_float(shift, DEFAULT_SHIFT, 0.1, 20.0),
        "seed": resolved_seed,
        "nproc_per_node": _safe_int(nproc_per_node, DEFAULT_NPROC_PER_NODE, 1, 8),
        "ring_degree": _safe_int(ring_degree, DEFAULT_RING_DEGREE, 1, 8),
    }
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "seed": resolved_seed,
        "log_tail": [],
    }
    background_tasks.add_task(asyncio.to_thread, _run_generation, job_id, request)
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
    return FileResponse(output_path, media_type="video/mp4", filename=f"{job_id}.mp4")


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str) -> Dict[str, str]:
    with job_lock:
        jobs.pop(job_id, None)
    for directory in (INPUT_DIR / job_id, OUTPUT_DIR / job_id):
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
    return {"status": "deleted", "job_id": job_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
