import os
import uuid
import shutil
import asyncio
import time
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
#
# Memory strategy for 24GB GPU (CogVideoX1.5-5B-I2V, ~23 GB loaded):
#
# Phase 1 — VAE encode (input video → latents):
#   Move transformer + text_encoder to CPU before calling vae.encode().
#   This frees ~16 GB, giving the VAE room to run.
#   Do NOT restore them afterward.
#
# Phase 2 — Transformer denoising:
#   On the very first transformer.forward() call, lazily move the transformer
#   back to GPU.  Text encoder stays on CPU permanently, keeping ~6 GB free
#   throughout denoising (enough headroom for full-resolution activations).
#
# This avoids the accelerate hook bugs (meta-tensor / device-mismatch) by
# managing device placement explicitly with plain .to() calls.
_INFERENCE_WRAPPER = Path("/tmp/sparkvsr_run.py")
_INFERENCE_WRAPPER.write_text(
    "import gc, os, torch, diffusers\n"
    "import imageio.v3 as _spark_iio\n"
    "from diffusers.models.autoencoders.autoencoder_kl_cogvideox import AutoencoderKLCogVideoX\n"
    "from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel as T3D\n"
    "_GPU = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "_orig_imwrite = _spark_iio.imwrite\n"
    "_orig_enc = AutoencoderKLCogVideoX.encode\n"
    "_orig_dec = AutoencoderKLCogVideoX.decode\n"
    "_orig_fwd = T3D.forward\n"
    "_pipe = None\n"
    "def _patched_imwrite(uri, image, *a, **kw):\n"
    "    if kw.get('codec') == 'libx264':\n"
    "        target_crf = os.environ.get('SPARKVSR_OUTPUT_CRF', '20')\n"
    "        target_preset = os.environ.get('SPARKVSR_OUTPUT_PRESET', 'medium')\n"
    "        target_pixfmt = os.environ.get('SPARKVSR_OUTPUT_PIXEL_FORMAT', 'yuv420p')\n"
    "        ffmpeg_params = list(kw.get('ffmpeg_params') or [])\n"
    "        if '-crf' in ffmpeg_params:\n"
    "            idx = ffmpeg_params.index('-crf')\n"
    "            if idx + 1 < len(ffmpeg_params):\n"
    "                ffmpeg_params[idx + 1] = target_crf\n"
    "            else:\n"
    "                ffmpeg_params.append(target_crf)\n"
    "        else:\n"
    "            ffmpeg_params.extend(['-crf', target_crf])\n"
    "        if '-preset' not in ffmpeg_params:\n"
    "            ffmpeg_params.extend(['-preset', target_preset])\n"
    "        if '-movflags' not in ffmpeg_params:\n"
    "            ffmpeg_params.extend(['-movflags', '+faststart'])\n"
    "        kw['ffmpeg_params'] = ffmpeg_params\n"
    "        kw['pixelformat'] = target_pixfmt\n"
    "        print(f'[sparkvsr-patch] mp4 encode tuned: crf={target_crf} pixfmt={target_pixfmt}', flush=True)\n"
    "    return _orig_imwrite(uri, image, *a, **kw)\n"
    "_spark_iio.imwrite = _patched_imwrite\n"
    "def _find_pipe():\n"
    "    global _pipe\n"
    "    if _pipe is not None:\n"
    "        return _pipe\n"
    "    for obj in gc.get_objects():\n"
    "        if isinstance(obj, diffusers.CogVideoXImageToVideoPipeline):\n"
    "            _pipe = obj\n"
    "            return obj\n"
    "    return None\n"
    "def _module_state(module):\n"
    "    if module is None:\n"
    "        return None, None\n"
    "    try:\n"
    "        param = next(module.parameters())\n"
    "        return param.device, param.dtype\n"
    "    except StopIteration:\n"
    "        return None, None\n"
    "def _move_module(module, device, label):\n"
    "    current_device, _ = _module_state(module)\n"
    "    if current_device is None or current_device.type == device:\n"
    "        return\n"
    "    module.to(device)\n"
    "    if device == 'cpu':\n"
    "        gc.collect()\n"
    "        torch.cuda.empty_cache()\n"
    "    print(f'[sparkvsr-patch] {label}→{device.upper()}', flush=True)\n"
    "def _align_tensor(tensor, module, label):\n"
    "    target_device, target_dtype = _module_state(module)\n"
    "    if target_device is None:\n"
    "        return tensor\n"
    "    needs_device = tensor.device != target_device\n"
    "    needs_dtype = torch.is_floating_point(tensor) and tensor.dtype != target_dtype\n"
    "    if needs_device or needs_dtype:\n"
    "        if torch.is_floating_point(tensor):\n"
    "            tensor = tensor.to(device=target_device, dtype=target_dtype)\n"
    "        else:\n"
    "            tensor = tensor.to(device=target_device)\n"
    "        print(f'[sparkvsr-patch] {label} aligned to {target_device.type.upper()}', flush=True)\n"
    "    return tensor\n"
    "def _ensure_text_encoder_patch(pipe):\n"
    "    te = getattr(pipe, 'text_encoder', None)\n"
    "    if te is None or getattr(te, '_sparkvsr_wrapped', False):\n"
    "        return\n"
    "    _orig_te = te.forward\n"
    "    def _te_fwd(*fa, **fkw):\n"
    "        _move_module(getattr(pipe, 'transformer', None), 'cpu', 'transformer')\n"
    "        _move_module(getattr(pipe, 'vae', None), 'cpu', 'vae')\n"
    "        _move_module(te, _GPU, 'text_encoder')\n"
    "        if fa:\n"
    "            fa = (_align_tensor(fa[0], te, 'text_encoder input'),) + fa[1:]\n"
    "        elif 'input_ids' in fkw:\n"
    "            fkw['input_ids'] = _align_tensor(fkw['input_ids'], te, 'text_encoder input')\n"
    "        try:\n"
    "            return _orig_te(*fa, **fkw)\n"
    "        finally:\n"
    "            _move_module(te, 'cpu', 'text_encoder')\n"
    "    te.forward = _te_fwd\n"
    "    te._sparkvsr_wrapped = True\n"
    "def _vae_enc(self, video, *a, **kw):\n"
    "    pipe = _find_pipe()\n"
    "    if pipe is not None:\n"
    "        _ensure_text_encoder_patch(pipe)\n"
    "        _move_module(getattr(pipe, 'transformer', None), 'cpu', 'transformer')\n"
    "        _move_module(getattr(pipe, 'text_encoder', None), 'cpu', 'text_encoder')\n"
    "    _move_module(self, _GPU, 'vae')\n"
    "    video = _align_tensor(video, self, 'vae.encode input')\n"
    "    return _orig_enc(self, video, *a, **kw)\n"
    "def _vae_dec(self, latents, *a, **kw):\n"
    "    pipe = _find_pipe()\n"
    "    if pipe is not None:\n"
    "        _move_module(getattr(pipe, 'transformer', None), 'cpu', 'transformer')\n"
    "        _move_module(getattr(pipe, 'text_encoder', None), 'cpu', 'text_encoder')\n"
    "    _move_module(self, _GPU, 'vae')\n"
    "    latents = _align_tensor(latents, self, 'vae.decode input')\n"
    "    result = _orig_dec(self, latents, *a, **kw)\n"
    "    if hasattr(result, 'sample') and torch.is_tensor(result.sample):\n"
    "        result.sample = result.sample.cpu()\n"
    "        print('[sparkvsr-patch] decoded sample→CPU', flush=True)\n"
    "    _move_module(self, 'cpu', 'vae')\n"
    "    return result\n"
    "def _trans_fwd(self, *a, **kw):\n"
    "    pipe = _find_pipe()\n"
    "    if pipe is not None:\n"
    "        _ensure_text_encoder_patch(pipe)\n"
    "        _move_module(getattr(pipe, 'vae', None), 'cpu', 'vae')\n"
    "        _move_module(getattr(pipe, 'text_encoder', None), 'cpu', 'text_encoder')\n"
    "    _move_module(self, _GPU, 'transformer')\n"
    "    if 'hidden_states' in kw:\n"
    "        kw['hidden_states'] = _align_tensor(kw['hidden_states'], self, 'transformer hidden_states')\n"
    "    if 'encoder_hidden_states' in kw:\n"
    "        kw['encoder_hidden_states'] = _align_tensor(kw['encoder_hidden_states'], self, 'transformer encoder_hidden_states')\n"
    "    if 'timestep' in kw:\n"
    "        kw['timestep'] = _align_tensor(kw['timestep'], self, 'transformer timestep')\n"
    "    if 'image_rotary_emb' in kw and kw['image_rotary_emb'] is not None:\n"
    "        kw['image_rotary_emb'] = tuple(_align_tensor(x, self, 'transformer rotary_emb') for x in kw['image_rotary_emb'])\n"
    "    if 'ofs' in kw and kw['ofs'] is not None:\n"
    "        kw['ofs'] = _align_tensor(kw['ofs'], self, 'transformer ofs')\n"
    "    try:\n"
    "        return _orig_fwd(self, *a, **kw)\n"
    "    finally:\n"
    "        _move_module(self, 'cpu', 'transformer')\n"
    "AutoencoderKLCogVideoX.encode = _vae_enc\n"
    "AutoencoderKLCogVideoX.decode = _vae_dec\n"
    "T3D.forward = _trans_fwd\n"
    "import runpy\n"
    "runpy.run_path('/app/sparkvsr_inference_script.py', run_name='__main__')\n"
)

# Resolved at startup in the background; None means still downloading.
SPARKVSR_MODEL_PATH: str | None = None
_model_ready = asyncio.Event()


def _merge_job_status(task_id: str, **updates):
    job = dict(JOBS.get(task_id, {}))
    job.update(updates)
    JOBS[task_id] = job


def _set_job_progress(task_id: str, progress: int):
    current = JOBS.get(task_id, {})
    current_progress = int(current.get("progress", 0) or 0)
    _merge_job_status(task_id, progress=max(current_progress, max(0, min(100, int(progress)))))


def _progress_from_output(task_id: str, text: str):
    lowered = text.lower()

    if "loading weights" in lowered:
        _set_job_progress(task_id, 8)
    if "loading pipeline components" in lowered:
        _set_job_progress(task_id, 15)
    if "processing videos" in lowered:
        _set_job_progress(task_id, 25)
    if "[sparkvsr-patch] transformer" in lowered or "[sparkvsr-patch] vae" in lowered:
        _set_job_progress(task_id, 45)
    if "completed" in lowered and "processing videos" not in lowered:
        _set_job_progress(task_id, 100)


async def _consume_process_stream(stream, sink, task_id: str):
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            break
        text = chunk.decode(errors="replace")
        sink.append(text)
        _progress_from_output(task_id, text)


async def _progress_heartbeat(task_id: str, process, expected_seconds: int):
    started_at = time.monotonic()
    while process.returncode is None:
        elapsed = time.monotonic() - started_at
        # Estimated progress only: SparkVSR does not emit machine-readable task %
        estimated = 5 + int((elapsed / max(expected_seconds, 1)) * 90)
        _set_job_progress(task_id, min(95, estimated))
        await asyncio.sleep(5)


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
    tile_size: int = 0,
):
    JOBS[task_id] = {"status": "processing", "progress": 1}

    task_input_dir = INPUT_DIR / task_id
    task_output_dir = OUTPUT_DIR / task_id
    task_input_dir.mkdir(parents=True, exist_ok=True)
    task_output_dir.mkdir(parents=True, exist_ok=True)

    input_filename = os.path.basename(input_path)
    shutil.copy2(input_path, task_input_dir / input_filename)

    final_output = OUTPUT_DIR / f"{task_id}_out.mp4"

    try:
        print(f"[Task {task_id}] Running SparkVSR on {input_path}")

        inference_entrypoint = str(_INFERENCE_WRAPPER if cpu_offload else Path("/app/sparkvsr_inference_script.py"))

        cmd = [
            "python", inference_entrypoint,
            "--input_dir", str(task_input_dir),
            "--model_path", SPARKVSR_MODEL_PATH,
            "--output_path", str(task_output_dir),
            "--upscale", str(upscale_factor),
            "--ref_mode", ref_mode,
            "--ref_guidance_scale", str(ref_guidance_scale),
            "--is_vae_st",
            "--chunk_len", str(chunk_len),
            *(["--tile_size_hw", str(tile_size), str(tile_size), "--overlap_hw", "32", "32"] if tile_size > 0 else []),
        ]

        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env.setdefault("SPARKVSR_OUTPUT_CRF", "20")
        env.setdefault("SPARKVSR_OUTPUT_PRESET", "medium")
        env.setdefault("SPARKVSR_OUTPUT_PIXEL_FORMAT", "yuv420p")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/app",
            env=env,
        )

        _set_job_progress(task_id, 3)

        stdout_chunks = []
        stderr_chunks = []
        expected_seconds = 1200 if upscale_factor >= 4 else 720

        stdout_task = asyncio.create_task(
            _consume_process_stream(process.stdout, stdout_chunks, task_id)
        )
        stderr_task = asyncio.create_task(
            _consume_process_stream(process.stderr, stderr_chunks, task_id)
        )
        progress_task = asyncio.create_task(
            _progress_heartbeat(task_id, process, expected_seconds)
        )

        returncode = await process.wait()
        await stdout_task
        await stderr_task
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

        stdout_text = "".join(stdout_chunks)
        stderr_text = "".join(stderr_chunks)

        if returncode != 0:
            error_msg = "\n".join(
                part for part in (stdout_text.strip(), stderr_text.strip()) if part
            ) or "Unknown error"
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
        file_size = final_output.stat().st_size if final_output.exists() else 0
        JOBS[task_id] = {
            "status": "completed",
            "progress": 100,
            "output": str(final_output),
            "file_size_bytes": file_size,
        }

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
    tile_size: int = Form(0),
):
    if not _model_ready.is_set():
        raise HTTPException(status_code=503, detail="Model is still downloading, please retry shortly")

    task_id = str(uuid.uuid4())
    input_path = INPUT_DIR / f"{task_id}_{video.filename}"

    with open(input_path, "wb") as f:
        f.write(await video.read())

    JOBS[task_id] = {"status": "queued", "progress": 0}
    asyncio.create_task(
        _process_video(task_id, str(input_path), upscale, ref_mode, ref_guidance_scale, cpu_offload, chunk_len, tile_size)
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
