"""Serialized REST adapter around ABot-World's stateful streaming demo."""

import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from web_client import config as world_config
from web_client.inference import start_stream
from web_client.pipeline_loader import get_pipeline
from web_client.state import state

OUTPUT_DIR = Path(os.getenv("ABOT_OUTPUT_DIR", "/data/abot_tasks"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ACTION_KEYS = {
    "forward": "W", "backward": "S", "left": "A", "right": "D",
    "look_up": "I", "look_down": "K", "look_left": "J", "look_right": "L",
}

app = FastAPI(title="OpenFork ABot-World API")
jobs = {}
jobs_lock = threading.Lock()
generation_lock = threading.Lock()
model_loaded = False
model_error = None


class Action(BaseModel):
    action: str
    durationSeconds: float = Field(ge=0.25, le=10)


class GenerateRequest(BaseModel):
    source_video_url: str
    prompt: str = Field(min_length=1, max_length=10000)
    action_sequence: list[Action] = Field(min_length=1, max_length=32)
    world_event: str | None = Field(default=None, max_length=1000)
    seed: int | None = None


def load_model():
    global model_loaded, model_error
    try:
        get_pipeline()
        state.model_ready = True
        model_loaded = True
    except Exception as exc:
        model_error = str(exc)


def download_reference(source_url: str, task_dir: Path):
    video_path = task_dir / "source.mp4"
    image_path = task_dir / "reference.png"
    with requests.get(source_url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with open(video_path, "wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
    subprocess.run(
        ["ffmpeg", "-y", "-sseof", "-0.1", "-i", str(video_path), "-frames:v", "1", str(image_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return image_path


def set_action(action: str):
    key = ACTION_KEYS.get(action)
    with state._key_lock:
        state.frontend_pressed = {key} if key else set()
        state.frontend_activated = set()


def run_job(job_id: str, request: GenerateRequest):
    task_dir = OUTPUT_DIR / job_id
    task_dir.mkdir(parents=True, exist_ok=True)
    try:
        with generation_lock:
            with jobs_lock:
                jobs[job_id]["status"] = "processing"
            reference = download_reference(request.source_video_url, task_dir)
            prompt = request.prompt
            if request.world_event:
                prompt = f"{prompt}. During the rollout: {request.world_event}"

            total_seconds = sum(item.durationSeconds for item in request.action_sequence)
            fps = int(getattr(world_config, "VIDEO_FPS", 16))
            frames_per_block = max(1, int(getattr(world_config, "FRAMES_PER_BLOCK", 4)))
            max_blocks = max(1, round(total_seconds * fps / frames_per_block))
            world_config.MAX_BLOCKS = max_blocks
            import web_client.inference as inference_module
            inference_module.MAX_BLOCKS = max_blocks

            boundaries = []
            elapsed = 0.0
            for item in request.action_sequence:
                elapsed += item.durationSeconds
                boundaries.append((elapsed, item.action))
            set_action(boundaries[0][1])
            frame_index = 0
            for _ in start_stream(prompt, str(reference)):
                frame_index += 1
                current_seconds = frame_index / fps
                for boundary, action in boundaries:
                    if current_seconds <= boundary:
                        set_action(action)
                        break
            set_action("idle")

            candidates = sorted(
                Path(world_config.OUTPUT_DIR).glob("stream_*.mp4"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                raise RuntimeError("ABot-World produced no video")
            output_path = task_dir / "output.mp4"
            candidates[0].replace(output_path)
            with jobs_lock:
                jobs[job_id].update(status="completed", output=str(output_path))
    except Exception as exc:
        with jobs_lock:
            jobs[job_id].update(status="failed", error=str(exc))
    finally:
        set_action("idle")


@app.on_event("startup")
def startup():
    threading.Thread(target=load_model, daemon=True).start()


@app.get("/health")
def health():
    return {"status": "error" if model_error else "ok", "model_loaded": model_loaded, "error": model_error}


@app.post("/generate")
def generate(request: GenerateRequest):
    if not model_loaded:
        raise HTTPException(503, model_error or "Model is still loading")
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {"status": "pending", "created_at": time.time()}
    threading.Thread(target=run_job, args=(job_id, request), daemon=True).start()
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
