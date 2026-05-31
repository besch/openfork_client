"""
TurboDiffusion REST API Server

FastAPI server that wraps TurboDiffusion inference scripts for T2V and I2V generation.
Runs inside the TurboDiffusion Docker container.
"""

import os
import sys
import uuid
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TurboDiffusion API", version="1.0.0")

# Configuration
CHECKPOINTS_DIR = Path("/opt/TurboDiffusion/checkpoints")
OUTPUTS_DIR = Path("/opt/TurboDiffusion/outputs")
INPUTS_DIR = Path("/opt/TurboDiffusion/inputs")
TURBODIFFUSION_DIR = Path("/opt/TurboDiffusion")

# Job storage
jobs = {}
executor = ThreadPoolExecutor(max_workers=1)


def _python_cmd() -> str:
    return sys.executable or "python"


def _format_process_error(result: subprocess.CompletedProcess) -> str:
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    parts = []
    if stderr:
        parts.append(f"stderr:\n{stderr[-3500:]}")
    if stdout:
        parts.append(f"stdout:\n{stdout[-1500:]}")
    return "\n\n".join(parts) or f"Process exited with code {result.returncode}"


def _inference_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(TURBODIFFUSION_DIR),
            str(TURBODIFFUSION_DIR / "turbodiffusion"),
            env.get("PYTHONPATH", ""),
        ]
    )
    return env


class T2VRequest(BaseModel):
    prompt: str
    resolution: str = "480p"
    num_steps: int = 4
    seed: int = 0


class I2VRequest(BaseModel):
    prompt: str
    resolution: str = "480p"
    num_steps: int = 4
    num_frames: int = 49
    seed: int = 0


class JobStatus(BaseModel):
    job_id: str
    status: str
    error: Optional[str] = None
    output_path: Optional[str] = None


def run_t2v_inference(job_id: str, prompt: str, resolution: str, num_steps: int, seed: int):
    """Run T2V inference using TurboDiffusion CLI."""
    try:
        jobs[job_id]["status"] = "processing"
        
        output_dir = OUTPUTS_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "generated_video.mp4"
        
        cmd = [
            _python_cmd(), "turbodiffusion/inference/wan2.1_t2v_infer.py",
            "--model", "Wan2.1-1.3B",
            "--dit_path", str(CHECKPOINTS_DIR / "TurboWan2.1-T2V-1.3B-480P-quant.pth"),
            "--vae_path", str(CHECKPOINTS_DIR / "Wan2.1_VAE.pth"),
            "--text_encoder_path", str(CHECKPOINTS_DIR / "models_t5_umt5-xxl-enc-bf16.pth"),
            "--resolution", resolution,
            "--prompt", prompt,
            "--num_samples", "1",
            "--num_steps", str(num_steps),
            "--seed", str(seed),
            "--quant_linear",
            "--attention_type", "sla",
            "--sla_topk", "0.1",
            "--save_path", str(output_file),
        ]
        
        logger.info(f"Running T2V command: {' '.join(cmd)}")
        
        env = _inference_env()
        
        result = subprocess.run(
            cmd,
            cwd=str(TURBODIFFUSION_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            error = _format_process_error(result)
            logger.error(f"T2V failed: {error}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = error[-4000:]
            return
        
        # Find output video
        output_files = list(output_dir.glob("*.mp4"))
        if output_files:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["output_path"] = str(output_files[0])
            logger.info(f"T2V completed: {output_files[0]}")
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = "No output file generated"
            
    except subprocess.TimeoutExpired:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = "Generation timeout"
    except Exception as e:
        logger.exception(f"T2V error: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


def run_i2v_inference(job_id: str, image_path: str, prompt: str, resolution: str, num_steps: int, num_frames: int, seed: int):
    """Run I2V inference using TurboDiffusion CLI."""
    try:
        jobs[job_id]["status"] = "processing"
        
        output_dir = OUTPUTS_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "generated_video.mp4"
        
        cmd = [
            _python_cmd(), "turbodiffusion/inference/wan2.2_i2v_infer.py",
            "--model", "Wan2.2-A14B",
            "--low_noise_model_path", str(CHECKPOINTS_DIR / "TurboWan2.2-I2V-A14B-low-720P-quant.pth"),
            "--high_noise_model_path", str(CHECKPOINTS_DIR / "TurboWan2.2-I2V-A14B-high-720P-quant.pth"),
            "--vae_path", str(CHECKPOINTS_DIR / "Wan2.1_VAE.pth"),
            "--text_encoder_path", str(CHECKPOINTS_DIR / "models_t5_umt5-xxl-enc-bf16.pth"),
            "--resolution", resolution,
            "--adaptive_resolution",
            "--image_path", image_path,
            "--prompt", prompt,
            "--num_samples", "1",
            "--num_steps", str(num_steps),
            "--num_frames", str(num_frames),
            "--seed", str(seed),
            "--quant_linear",
            "--attention_type", "sla",
            "--sla_topk", "0.1",
            "--ode",
            "--save_path", str(output_file),
        ]
        
        logger.info(f"Running I2V command: {' '.join(cmd)}")
        
        env = _inference_env()
        
        result = subprocess.run(
            cmd,
            cwd=str(TURBODIFFUSION_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            error = _format_process_error(result)
            logger.error(f"I2V failed: {error}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = error[-4000:]
            return
        
        # Find output video
        output_files = list(output_dir.glob("*.mp4"))
        if output_files:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["output_path"] = str(output_files[0])
            logger.info(f"I2V completed: {output_files[0]}")
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = "No output file generated"
            
    except subprocess.TimeoutExpired:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = "Generation timeout"
    except Exception as e:
        logger.exception(f"I2V error: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/generate/t2v")
async def generate_t2v(request: T2VRequest):
    """Submit a text-to-video generation job."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "type": "t2v"}
    
    executor.submit(
        run_t2v_inference,
        job_id,
        request.prompt,
        request.resolution,
        request.num_steps,
        request.seed
    )
    
    return {"job_id": job_id}


@app.post("/generate/i2v")
async def generate_i2v(
    prompt: str = Form(...),
    resolution: str = Form("480p"),
    num_steps: int = Form(4),
    num_frames: int = Form(49),
    seed: int = Form(0),
    image: UploadFile = File(...)
):
    """Submit an image-to-video generation job."""
    job_id = str(uuid.uuid4())
    
    # Save uploaded image
    image_path = INPUTS_DIR / f"{job_id}_{image.filename}"
    with open(image_path, "wb") as f:
        content = await image.read()
        f.write(content)
    
    jobs[job_id] = {"status": "pending", "type": "i2v", "image_path": str(image_path)}
    
    executor.submit(
        run_i2v_inference,
        job_id,
        str(image_path),
        prompt,
        resolution,
        num_steps,
        num_frames,
        seed
    )
    
    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get the status of a generation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        error=job.get("error"),
        output_path=job.get("output_path")
    )


@app.get("/download/{job_id}")
async def download_output(job_id: str):
    """Download the generated video file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    output_path = job.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(output_path, media_type="video/mp4", filename=f"{job_id}.mp4")


@app.delete("/job/{job_id}")
async def cleanup_job(job_id: str):
    """Clean up a job and its files."""
    if job_id not in jobs:
        return {"status": "not_found"}
    
    job = jobs[job_id]
    
    # Clean up output directory
    output_dir = OUTPUTS_DIR / job_id
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    
    # Clean up input image if I2V
    if "image_path" in job:
        image_path = Path(job["image_path"])
        if image_path.exists():
            image_path.unlink()
    
    del jobs[job_id]
    return {"status": "cleaned"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
