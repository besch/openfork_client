"""OpenFork REST adapter for ShotPlan's released Wan2.2 inference script."""

from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

SHOTPLAN_ROOT = Path(os.getenv("SHOTPLAN_ROOT", "/app/ShotPlan"))
WAN22_ROOT = Path(os.getenv("SHOTPLAN_WAN22_ROOT", "/app/models/Wan2.2-T2V-A14B"))
CHECKPOINT_ROOT = Path(
    os.getenv(
        "SHOTPLAN_CHECKPOINT_ROOT",
        "/app/models/ShotPlan-Wan2.2-T2V-A14B-HighNoise",
    )
)
OUTPUT_DIR = Path(os.getenv("SHOTPLAN_OUTPUT_DIR", "/app/output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="OpenFork ShotPlan API")
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
generation_lock = threading.Lock()


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    cut_frames: list[int] = Field(min_length=1, max_length=5)
    negative_prompt: str = Field(default="", max_length=4000)
    width: int = Field(default=832, ge=256, le=1280)
    height: int = Field(default=480, ge=256, le=1280)
    num_frames: int = Field(default=81)
    steps: int = Field(default=50, ge=1, le=100)
    seed: int = 42

    @model_validator(mode="after")
    def validate_cut_frames(self):
        if self.num_frames != 81:
            raise ValueError("The released ShotPlan Wan2.2 lane requires 81 frames")
        if self.cut_frames != sorted(set(self.cut_frames)):
            raise ValueError("cut_frames must be unique and strictly increasing")
        if any(frame <= 0 or frame >= self.num_frames - 1 for frame in self.cut_frames):
            raise ValueError("cut_frames must be between frames 1 and 79")
        return self


def _checkpoint_path() -> Path:
    configured = os.getenv("SHOTPLAN_CHECKPOINT")
    if configured:
        return Path(configured)
    candidates = sorted(CHECKPOINT_ROOT.rglob("*.safetensors"))
    if not candidates:
        raise RuntimeError(f"No ShotPlan safetensors checkpoint found under {CHECKPOINT_ROOT}")
    return candidates[0]


def _run_job(job_id: str, request: GenerateRequest):
    task_dir = OUTPUT_DIR / job_id
    task_dir.mkdir(parents=True, exist_ok=True)
    log_path = task_dir / "inference.log"
    try:
        with generation_lock:
            with jobs_lock:
                jobs[job_id]["status"] = "processing"
            command = [
                "python",
                str(SHOTPLAN_ROOT / "inference" / "infer_wan22.py"),
                "--wan22_root",
                str(WAN22_ROOT),
                "--ckpt",
                str(_checkpoint_path()),
                "--prompt",
                request.prompt,
                "--cut_at",
                ",".join(str(frame) for frame in request.cut_frames),
                "--output_dir",
                str(task_dir),
                "--gpus",
                os.getenv("SHOTPLAN_GPUS", "0"),
                "--height",
                str(request.height),
                "--width",
                str(request.width),
                "--num_frames",
                "81",
                "--steps",
                str(request.steps),
                "--seed",
                str(request.seed),
            ]
            if request.negative_prompt:
                command.extend(["--negative_prompt", request.negative_prompt])
            with open(log_path, "w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    cwd=SHOTPLAN_ROOT,
                    env={**os.environ, "PYTHONPATH": str(SHOTPLAN_ROOT)},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            candidates = sorted(task_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
            if not candidates:
                raise RuntimeError("ShotPlan inference produced no MP4 output")
            output_path = task_dir / "output.mp4"
            candidates[-1].replace(output_path)
            with jobs_lock:
                jobs[job_id].update(status="completed", output=str(output_path))
    except Exception as exc:
        with jobs_lock:
            jobs[job_id].update(
                status="failed",
                error=str(exc),
                log_tail=log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                if log_path.exists()
                else None,
            )


@app.get("/health")
def health():
    missing = [
        str(path)
        for path in (
            SHOTPLAN_ROOT / "inference" / "infer_wan22.py",
            WAN22_ROOT,
            CHECKPOINT_ROOT,
        )
        if not path.exists()
    ]
    try:
        checkpoint = str(_checkpoint_path()) if not missing else None
    except Exception as exc:
        missing.append(str(exc))
        checkpoint = None
    return {
        "status": "error" if missing else "ok",
        "model_loaded": not missing,
        "checkpoint": checkpoint,
        "error": "; ".join(missing) if missing else None,
    }


@app.post("/generate")
def generate(request: GenerateRequest):
    health_state = health()
    if not health_state["model_loaded"]:
        raise HTTPException(503, health_state["error"])
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "status": "pending",
            "created_at": time.time(),
            "cut_frames": request.cut_frames,
        }
    threading.Thread(target=_run_job, args=(job_id, request), daemon=True).start()
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
    return FileResponse(job["output"], media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
