"""
InSpatio-World REST API Server

Wraps the InSpatio-World novel-view synthesis pipeline (3 steps):
  Step 1 — Video captioning (Florence-2)
  Step 2 — Depth estimation (DA3) + point cloud rendering
  Step 3 — V2V inference (InSpatio-World 1.3B)

Exposes a FastAPI server at port 8000 with /health, /generate, /status/{task_id}, /download/{task_id}
"""

import os
import uuid
import shutil
import subprocess
import threading
import time
import logging
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("inspatio-api")

app = FastAPI(title="InSpatio-World API")

PIPELINE_ROOT = os.environ.get("INSPATIO_ROOT", "/opt/inspatio-world")
CHECKPOINT_DIR = os.environ.get(
    "INSPATIO_CHECKPOINTS", os.path.join(PIPELINE_ROOT, "checkpoints")
)
TRAJ_DIR = os.environ.get("INSPATIO_TRAJ_DIR", os.path.join(PIPELINE_ROOT, "traj"))
OUTPUT_BASE = os.environ.get("INSPATIO_OUTPUT_DIR", "/data/inspatio_output")
TASKS_DIR = os.environ.get("INSPATIO_TASKS_DIR", "/data/inspatio_tasks")

MODEL_READY = False
MODEL_LOAD_LOCK = threading.Lock()

tasks: dict[str, dict] = {}


class GenerateRequest(BaseModel):
    video_url: str
    trajectory: str = "x_y_circle_cycle"
    rotation_only: bool = False
    relative_to_source: bool = False
    disable_adaptive_frame: bool = False
    freeze_repeat: int = 0
    freeze_frame: Optional[int] = None
    use_tae: bool = False
    compile_dit: bool = False
    seed: Optional[int] = None


TRAJECTORY_MAP = {
    "x_y_circle_cycle": "x_y_circle_cycle.txt",
    "zoom_out_in": "zoom_out_in.txt",
}


def _get_python():
    for candidate in ("/usr/bin/python3", "/usr/bin/python", "python3", "python"):
        if os.path.isfile(candidate) or shutil.which(candidate):
            return candidate
    return "python3"


def _ensure_dirs():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    os.makedirs(TASKS_DIR, exist_ok=True)


def _resolve_trajectory(traj_name: str) -> Optional[str]:
    traj_file = TRAJECTORY_MAP.get(traj_name)
    if traj_file:
        return os.path.join(TRAJ_DIR, traj_file)
    custom = os.path.join(
        TRAJ_DIR, traj_name if traj_name.endswith(".txt") else f"{traj_name}.txt"
    )
    if os.path.isfile(custom):
        return custom
    return None


def _run_pipeline(task_id: str, request: GenerateRequest):
    python = _get_python()
    task = tasks[task_id]
    task_input_dir = os.path.join(TASKS_DIR, task_id, "input")
    task_output_dir = os.path.join(TASKS_DIR, task_id, "output")
    os.makedirs(task_input_dir, exist_ok=True)
    os.makedirs(task_output_dir, exist_ok=True)

    try:
        task["status"] = "downloading"
        log.info(f"[{task_id}] Downloading input video: {request.video_url}")

        import requests as req

        resp = req.get(request.video_url, timeout=120, stream=True)
        resp.raise_for_status()
        input_video = os.path.join(task_input_dir, "input.mp4")
        with open(input_video, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        log.info(f"[{task_id}] Video downloaded: {input_video}")

        task["status"] = "processing"
        task["progress_text"] = "Running InSpatio-World pipeline..."

        traj_path = _resolve_trajectory(request.trajectory)
        if not traj_path:
            raise ValueError(f"Unknown trajectory: {request.trajectory}")

        config_path = os.path.join(PIPELINE_ROOT, "configs", "inference_1.3b.yaml")
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR, "InSpatio-World-1.3B", "InSpatio-World-1.3B.safetensors"
        )

        cmd = [
            "bash",
            os.path.join(PIPELINE_ROOT, "run_test_pipeline.sh"),
            "--input_dir",
            task_input_dir,
            "--traj_txt_path",
            traj_path,
            "--config_path",
            config_path,
            "--checkpoint_path",
            checkpoint_path,
            "--output_folder",
            task_output_dir,
            "--step3_gpus",
            "0",
        ]

        if request.rotation_only:
            cmd.append("--rotation_only")
        if request.relative_to_source:
            cmd.append("--relative_to_source")
        if request.disable_adaptive_frame:
            cmd.append("--disable_adaptive_frame")
        if request.freeze_repeat > 0:
            cmd.extend(["--freeze_repeat", str(request.freeze_repeat)])
        if request.freeze_frame is not None:
            cmd.extend(["--freeze_frame", str(request.freeze_frame)])
        if request.use_tae:
            cmd.append("--use_tae")
        if request.compile_dit:
            cmd.append("--compile_dit")

        log.info(f"[{task_id}] Running pipeline: {' '.join(cmd)}")

        env = os.environ.copy()
        env["PYTHONPATH"] = PIPELINE_ROOT

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=PIPELINE_ROOT,
            env=env,
        )

        output_lines = []
        for line in iter(process.stdout.readline, b""):
            decoded = line.decode("utf-8", errors="replace").strip()
            output_lines.append(decoded)
            if decoded:
                log.info(f"[{task_id}] {decoded}")
            task["progress_text"] = (
                decoded[-200:] if decoded else task.get("progress_text", "")
            )

        return_code = process.wait(timeout=600)
        task["log"] = "\n".join(output_lines[-50:])

        if return_code != 0:
            raise RuntimeError(f"Pipeline failed with exit code {return_code}")

        output_video = _find_output_video(task_output_dir)
        if not output_video:
            raise FileNotFoundError(f"No output video found in {task_output_dir}")

        task["status"] = "completed"
        task["output_path"] = str(output_video)
        log.info(f"[{task_id}] Completed: {output_video}")

    except Exception as e:
        log.error(f"[{task_id}] Failed: {e}", exc_info=True)
        task["status"] = "failed"
        task["error"] = str(e)


def _find_output_video(output_dir: str) -> Optional[Path]:
    output_path = Path(output_dir)
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        matches = list(output_path.rglob(ext))
        if matches:
            return sorted(matches, key=lambda p: p.stat().st_size, reverse=True)[0]
    return None


@app.on_event("startup")
def _startup():
    _ensure_dirs()


@app.get("/health")
def health():
    checkpoints_ok = os.path.isdir(CHECKPOINT_DIR)
    pipeline_ok = os.path.isfile(os.path.join(PIPELINE_ROOT, "run_test_pipeline.sh"))
    return {
        "status": "ok" if (checkpoints_ok and pipeline_ok) else "loading",
        "models_initialized": checkpoints_ok and pipeline_ok,
        "checkpoints_dir": CHECKPOINT_DIR,
        "pipeline_ready": pipeline_ok,
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {
        "status": "queued",
        "progress_text": "Queued",
        "output_path": None,
        "error": None,
        "log": "",
        "request": req.dict(),
    }

    thread = threading.Thread(target=_run_pipeline, args=(task_id, req), daemon=True)
    thread.start()

    return {"data": {"task_id": task_id}}


@app.get("/status/{task_id}")
def status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    status_code_map = {
        "queued": 0,
        "downloading": 0,
        "processing": 0,
        "completed": 1,
        "failed": 2,
    }

    return {
        "data": [
            {
                "task_id": task_id,
                "status": status_code_map.get(task["status"], 0),
                "progress_text": task.get("progress_text", ""),
                "error": task.get("error"),
            }
        ]
    }


@app.get("/download/{task_id}")
def download(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    if task["status"] != "completed" or not task.get("output_path"):
        raise HTTPException(status_code=400, detail="Output not ready")

    output_path = Path(task["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        str(output_path),
        media_type="video/mp4",
        filename=f"inspatio_{task_id}.mp4",
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("INSPATIO_API_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
