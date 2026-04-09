import os
import uuid
import shutil
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import uvicorn

INPUT_DIR = Path("/app/input")
OUTPUT_DIR = Path("/app/output")

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JOBS = {}

# Wrapper script written to disk once at startup.
# Patches enable_sequential_cpu_offload → enable_model_cpu_offload before running
# the inference script.  Newer accelerate versions place VAE sub-modules on the
# meta device during sequential offload, which breaks CogVideoX VAE forward passes
# with "Cannot copy out of meta tensor; no data!".  Model-level offload uses real
# CPU tensors instead and avoids this bug while still freeing ~15-18 GiB of VRAM.
_INFERENCE_WRAPPER = Path("/tmp/sparkvsr_run.py")
_INFERENCE_WRAPPER.write_text(
    "import sys, diffusers\n"
    "try:\n"
    "    _cls = diffusers.CogVideoXImageToVideoPipeline\n"
    "    _cls.enable_sequential_cpu_offload = _cls.enable_model_cpu_offload\n"
    "except Exception as _e:\n"
    "    print('[sparkvsr-patch] could not patch:', _e, flush=True)\n"
    "import runpy\n"
    "runpy.run_path('/app/sparkvsr_inference_script.py', run_name='__main__')\n"
)

# Resolved at startup in the background; None means still downloading.
SPARKVSR_MODEL_PATH: str | None = None
_model_ready = asyncio.Event()


async def _download_model():
    """Download SparkVSR weights in the background so the API can start immediately."""
    global SPARKVSR_MODEL_PATH
    try:
        from huggingface_hub import snapshot_download
        hf_token = os.environ.get("HF_TOKEN")
        print("[SparkVSR] Downloading model weights from JiongzeYu/SparkVSR ...")
        repo_root = await asyncio.to_thread(
            snapshot_download,
            "JiongzeYu/SparkVSR",
            token=hf_token,
        )
        ckpt = os.path.join(repo_root, "checkpoints", "sparkvsr-s2", "ckpt-500-sft")
        if os.path.exists(ckpt):
            SPARKVSR_MODEL_PATH = ckpt
            print(f"[SparkVSR] Model ready: {ckpt}")
        else:
            SPARKVSR_MODEL_PATH = repo_root
            print(f"[SparkVSR] Warning: expected checkpoint not found at {ckpt}, using repo root")
    except Exception as e:
        print(f"[SparkVSR] ERROR: could not download model: {e}")
        SPARKVSR_MODEL_PATH = "checkpoints/sparkvsr-s2/ckpt-500-sft"
    finally:
        _model_ready.set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_download_model())
    yield


app = FastAPI(title="SparkVSR API", lifespan=lifespan)


async def _process_video(
    task_id: str,
    input_path: str,
    upscale_factor: int,
    ref_mode: str,
    ref_guidance_scale: float,
    cpu_offload: bool = True,
    chunk_len: int = 49,
):
    JOBS[task_id] = {"status": "processing", "progress": 0}

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
            "python", str(_INFERENCE_WRAPPER),
            "--input_dir", str(task_input_dir),
            "--model_path", SPARKVSR_MODEL_PATH,
            "--output_path", str(task_output_dir),
            "--upscale", str(upscale_factor),
            "--ref_mode", ref_mode,
            "--ref_guidance_scale", str(ref_guidance_scale),
            "--is_vae_st",
            "--chunk_len", str(chunk_len),
            *(["--is_cpu_offload"] if cpu_offload else []),
        ]

        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/app",
            env=env,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            print(f"[Task {task_id}] Failed:\n{error_msg}")
            JOBS[task_id] = {"status": "failed", "error": error_msg}
            return

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
    return {"status": "ok", "model_ready": _model_ready.is_set()}


@app.post("/upscale")
async def upscale(
    video: UploadFile,
    upscale: int = Form(4),
    ref_mode: str = Form("no_ref"),
    ref_guidance_scale: float = Form(1.0),
    cpu_offload: bool = Form(True),
    chunk_len: int = Form(49),
):
    if not _model_ready.is_set():
        raise HTTPException(status_code=503, detail="Model is still downloading, please retry shortly")

    task_id = str(uuid.uuid4())
    input_path = INPUT_DIR / f"{task_id}_{video.filename}"

    with open(input_path, "wb") as f:
        f.write(await video.read())

    JOBS[task_id] = {"status": "queued", "progress": 0}
    asyncio.create_task(
        _process_video(task_id, str(input_path), upscale, ref_mode, ref_guidance_scale, cpu_offload, chunk_len)
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

    for f in OUTPUT_DIR.glob(f"*{task_id}*.mp4"):
        return FileResponse(str(f), media_type="video/mp4", filename=f.name)

    raise HTTPException(status_code=500, detail="Generated file not found")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
