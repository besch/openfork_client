"""OpenFork REST adapter for HOMIE subject-consistent reference-to-video."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

HOMIE_ROOT = Path(os.getenv("HOMIE_ROOT", "/app/HOMIE"))
WAN_ROOT = Path(
    os.getenv("HOMIE_WAN_ROOT", "/app/models/Wan2.1-T2V-14B-Diffusers")
)
HOMIE_CKPT = Path(
    os.getenv("HOMIE_CHECKPOINT", "/app/models/HOMIE-Wan-Model")
)
MLLM_CKPT = Path(
    os.getenv("HOMIE_MLLM_CHECKPOINT", "/app/models/Qwen3-VL-2B-Thinking")
)
OUTPUT_DIR = Path(os.getenv("HOMIE_OUTPUT_DIR", "/app/output"))
INPUT_DIR = Path(os.getenv("HOMIE_INPUT_DIR", "/app/input"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="OpenFork HOMIE API")
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
generation_lock = threading.Lock()


def _run(command: list[str], log_handle):
    subprocess.run(
        command,
        cwd=HOMIE_ROOT,
        env={**os.environ, "PYTHONPATH": str(HOMIE_ROOT)},
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        check=True,
    )


def _run_job(job_id: str, payload: dict, reference_paths: list[Path]):
    task_dir = OUTPUT_DIR / job_id
    task_dir.mkdir(parents=True, exist_ok=True)
    log_path = task_dir / "inference.log"
    try:
        with generation_lock:
            with jobs_lock:
                jobs[job_id]["status"] = "processing"
            group_sizes = payload["group_sizes"]
            groups = []
            cursor = 0
            for size in group_sizes:
                groups.append(
                    [str(path) for path in reference_paths[cursor : cursor + size]]
                )
                cursor += size
            meta_path = task_dir / "input.jsonl"
            feature_meta_path = task_dir / "input_with_mllm.jsonl"
            feature_dir = task_dir / "mllm_features"
            meta_path.write_text(
                json.dumps(
                    {"reference_paths": groups, "prompt": payload["prompt"]},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = task_dir / "output.mp4"
            with open(log_path, "w", encoding="utf-8") as log:
                _run(
                    [
                        "python",
                        "generate_mllm_feature.py",
                        "--meta_file",
                        str(meta_path),
                        "--output_meta",
                        str(feature_meta_path),
                        "--feature_dir",
                        str(feature_dir),
                        "--model_path",
                        str(MLLM_CKPT),
                    ],
                    log,
                )
                _run(
                    [
                        "python",
                        "generate.py",
                        "--task",
                        "s2v-14B",
                        "--size",
                        payload["size"],
                        "--frame_num",
                        str(payload["frame_num"]),
                        "--sample_fps",
                        str(payload["fps"]),
                        "--ckpt_dir",
                        str(WAN_ROOT),
                        "--homie_ckpt",
                        str(HOMIE_CKPT),
                        "--input_json",
                        str(feature_meta_path),
                        "--save_path",
                        str(task_dir),
                        "--save_file",
                        str(output_path),
                        "--base_seed",
                        str(payload["seed"]),
                        "--sample_steps",
                        str(payload["steps"]),
                        "--sample_shift",
                        str(payload["flow_shift"]),
                        "--sample_guide_scale",
                        str(payload["cfg"]),
                    ],
                    log,
                )
            if not output_path.exists():
                candidates = sorted(
                    task_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime
                )
                if not candidates:
                    raise RuntimeError("HOMIE inference produced no MP4 output")
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
    finally:
        for path in reference_paths:
            path.unlink(missing_ok=True)


@app.get("/health")
def health():
    required = [
        HOMIE_ROOT / "generate.py",
        HOMIE_ROOT / "generate_mllm_feature.py",
        WAN_ROOT,
        HOMIE_CKPT,
        MLLM_CKPT,
    ]
    missing = [str(path) for path in required if not path.exists()]
    return {
        "status": "error" if missing else "ok",
        "model_loaded": not missing,
        "error": f"Missing baked HOMIE resources: {', '.join(missing)}"
        if missing
        else None,
    }


@app.post("/generate")
async def generate(
    payload: str = Form(...),
    reference_images: list[UploadFile] = File(...),
):
    health_state = health()
    if not health_state["model_loaded"]:
        raise HTTPException(503, health_state["error"])
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"Invalid payload JSON: {exc}")

    prompt = str(data.get("prompt") or "").strip()
    group_sizes = [int(size) for size in data.get("group_sizes") or []]
    if not prompt:
        raise HTTPException(400, "prompt is required")
    if not group_sizes or any(size < 1 or size > 9 for size in group_sizes):
        raise HTTPException(400, "group_sizes must describe 1–8 non-empty groups")
    if len(group_sizes) > 8 or sum(group_sizes) != len(reference_images):
        raise HTTPException(400, "group_sizes do not match uploaded reference_images")

    size = str(data.get("size") or "832*480")
    if size not in {"832*480", "1280*720"}:
        raise HTTPException(400, "Unsupported HOMIE size")
    normalized = {
        "prompt": prompt,
        "group_sizes": group_sizes,
        "size": size,
        "frame_num": 97,
        "fps": max(8, min(30, int(data.get("fps") or 24))),
        "steps": max(1, min(75, int(data.get("steps") or 50))),
        "flow_shift": max(0.1, min(20, float(data.get("flow_shift") or 3))),
        "cfg": max(0, min(20, float(data.get("cfg") or 5))),
        "seed": int(data.get("seed") if data.get("seed") is not None else 6666),
    }
    job_id = str(uuid.uuid4())
    paths = []
    for index, upload in enumerate(reference_images):
        suffix = Path(upload.filename or "").suffix or ".png"
        path = INPUT_DIR / f"{job_id}_{index}{suffix}"
        path.write_bytes(await upload.read())
        paths.append(path)
    with jobs_lock:
        jobs[job_id] = {
            "status": "pending",
            "created_at": time.time(),
            "subject_count": len(group_sizes),
            "reference_count": len(paths),
        }
    threading.Thread(
        target=_run_job, args=(job_id, normalized, paths), daemon=True
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
    return FileResponse(job["output"], media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
