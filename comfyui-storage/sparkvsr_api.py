import os
import uuid
import shutil
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI(title="SparkVSR API")

JOBS = {}
INPUT_DIR = Path("/app/input")
OUTPUT_DIR = Path("/app/output")

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_sparkvsr_model_path() -> str:
    """Resolve the SparkVSR Stage-2 checkpoint from the HuggingFace cache."""
    try:
        from huggingface_hub import snapshot_download
        repo_root = snapshot_download("JiongzeYu/SparkVSR")
        ckpt = os.path.join(repo_root, "checkpoints", "sparkvsr-s2", "ckpt-500-sft")
        if os.path.exists(ckpt):
            print(f"[SparkVSR] Resolved model path: {ckpt}")
            return ckpt
        print(f"[SparkVSR] Warning: expected checkpoint not found at {ckpt}, using repo root")
        return repo_root
    except Exception as e:
        print(f"[SparkVSR] Warning: could not resolve model path via HF hub: {e}")
        return "checkpoints/sparkvsr-s2/ckpt-500-sft"


SPARKVSR_MODEL_PATH = _resolve_sparkvsr_model_path()


async def _process_video(
    task_id: str,
    input_path: str,
    upscale_factor: int,
    ref_mode: str,
    ref_guidance_scale: float,
):
    JOBS[task_id] = {"status": "processing", "progress": 0}

    # sparkvsr_inference_script.py expects --input_dir (a directory), not a file path.
    # Create per-task input/output dirs so concurrent jobs don't collide.
    task_input_dir = INPUT_DIR / task_id
    task_output_dir = OUTPUT_DIR / task_id
    task_input_dir.mkdir(parents=True, exist_ok=True)
    task_output_dir.mkdir(parents=True, exist_ok=True)

    input_filename = os.path.basename(input_path)
    shutil.copy2(input_path, task_input_dir / input_filename)

    final_output = OUTPUT_DIR / f"{task_id}_out.mp4"

    try:
        print(f"[Task {task_id}] Running SparkVSR on {input_path}")

        cmd = [
            "python", "/app/sparkvsr_inference_script.py",
            "--input_dir", str(task_input_dir),
            "--model_path", SPARKVSR_MODEL_PATH,
            "--output_path", str(task_output_dir),
            "--upscale", str(upscale_factor),
            "--ref_mode", ref_mode,
            "--ref_guidance_scale", str(ref_guidance_scale),
            "--is_vae_st",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/app",
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            print(f"[Task {task_id}] Failed:\n{error_msg}")
            JOBS[task_id] = {"status": "failed", "error": error_msg}
            return

        # Find the output video produced in task_output_dir
        output_videos = sorted(
            list(task_output_dir.rglob("*.mp4")), key=os.path.getmtime
        )
        if not output_videos:
            JOBS[task_id] = {
                "status": "failed",
                "error": "No output video found after inference",
            }
            return

        shutil.move(str(output_videos[-1]), str(final_output))
        print(f"[Task {task_id}] Completed → {final_output}")
        JOBS[task_id] = {"status": "completed", "output": str(final_output)}

    except Exception as e:
        print(f"[Task {task_id}] Exception: {e}")
        JOBS[task_id] = {"status": "failed", "error": str(e)}
    finally:
        shutil.rmtree(task_input_dir, ignore_errors=True)
        shutil.rmtree(task_output_dir, ignore_errors=True)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upscale")
async def upscale(
    video: UploadFile,
    upscale: int = Form(4),
    ref_mode: str = Form("no_ref"),
    ref_guidance_scale: float = Form(1.0),
):
    task_id = str(uuid.uuid4())
    input_path = INPUT_DIR / f"{task_id}_{video.filename}"

    with open(input_path, "wb") as f:
        f.write(await video.read())

    JOBS[task_id] = {"status": "queued", "progress": 0}
    asyncio.create_task(
        _process_video(task_id, str(input_path), upscale, ref_mode, ref_guidance_scale)
    )
    return {"task_id": task_id}


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in JOBS:
        raise HTTPException(status_code=404, detail="Task not found")
    return JOBS[task_id]


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    if task_id not in JOBS:
        raise HTTPException(status_code=404, detail="Task not found")

    job = JOBS[task_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task is in state {job['status']}")

    out_file = job["output"]
    if os.path.exists(out_file):
        return FileResponse(out_file, media_type="video/mp4", filename=f"{task_id}_upscaled.mp4")

    # Fallback: search output dir for any file matching this task
    for f in OUTPUT_DIR.glob(f"*{task_id}*.mp4"):
        return FileResponse(str(f), media_type="video/mp4", filename=f.name)

    raise HTTPException(status_code=500, detail="Generated file not found")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
