"""
FastAPI Server for DiagDistill

Exposes DiagDistill (HunyuanVideo distillation) as a REST API.
"""

import os
import time
import uuid
import logging
import subprocess
import torch
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="DiagDistill API")

# Configuration
MODELS_DIR = "/opt/DiagDistill/models"
OUTPUT_DIR = "/tmp/diagdistill_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    error: Optional[str] = None
    output_path: Optional[str] = None

jobs = {}

def run_diagdistill(job_id: str, prompt: str, negative_prompt: str, resolution: str, seed: int, start_image_path: Optional[str] = None):
    try:
        jobs[job_id]["status"] = "processing"
        output_file = f"{OUTPUT_DIR}/{job_id}.mp4"
        
        # DiagDistill uses torchrun with inference.py and a YAML config file
        # We need to create a custom config for each request
        config_content = f"""# Auto-generated config for job {job_id}
denoising_step_list:
  - 1000
  - 100
warp_denoising_step: true
use_diagonal_denoising: true
enable_torch_compile: false
use_taehv: true
taehv_checkpoint_path: /opt/DiagDistill/checkpoints/taew2_1.pth
num_frame_per_block: 3
model_name: Wan2.1-T2V-1.3B
model_kwargs:
  local_attend: 12
  timestep_shift: 5.0
  sink_size: 3

# Inference settings
data_path: /tmp/{job_id}_prompt.txt
output_folder: {OUTPUT_DIR}
inference_iter: -1
num_output_frames: 21
use_ema: false
seed: {seed}
num_samples: 1
save_with_index: true
global_sink: true
context_noise: 0

# Model checkpoints - these need to be provided
generator_ckpt: 
lora_ckpt: 

adapter:
  type: "lora"
  rank: 256
  alpha: 256
  dropout: 0.0
  dtype: "bfloat16"
  verbose: false
"""
        
        # Write prompt to a text file (DiagDistill reads prompts from file)
        prompt_file = f"/tmp/{job_id}_prompt.txt"
        with open(prompt_file, "w") as f:
            f.write(prompt)
        
        # Write config file
        config_file = f"/tmp/{job_id}_config.yaml"
        with open(config_file, "w") as f:
            f.write(config_content)
        
        # Construct command for DiagDistill using torchrun
        # Note: This requires the DiagDistill models to be downloaded
        cmd = [
            "torchrun",
            "--nproc_per_node=1",
            "--master_port=29500",
            "inference.py",
            "--config_path", config_file,
        ]
        
        logging.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/opt/DiagDistill", check=True)
        
        # Find the output video (DiagDistill creates timestamped folders)
        # Look for the most recent video file in the output directory
        output_folder = OUTPUT_DIR
        video_files = []
        if os.path.exists(output_folder):
            for f in os.listdir(output_folder):
                if f.endswith('.mp4'):
                    video_files.append(os.path.join(output_folder, f))
        
        if video_files:
            # Use the most recent video file
            latest_video = max(video_files, key=os.path.getmtime)
            # Move to expected location
            if latest_video != output_file:
                import shutil
                shutil.move(latest_video, output_file)
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["output_path"] = output_file
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = f"Output file not generated. {result.stderr}"

    except subprocess.CalledProcessError as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = f"Process failed: {e.stderr}"
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

class T2VRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    resolution: str = "720p"
    seed: int = 0

@app.get("/health")
def health():
    try:
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "None"
    except Exception as e:
        logging.error(f"Health check error: {e}")
        cuda_ok = False
        gpu_name = "Error"
    return {"status": "ok", "cuda": cuda_ok, "gpu_name": gpu_name}

@app.post("/generate/t2v")
def generate_t2v(req: T2VRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "job_id": job_id}
    background_tasks.add_task(run_diagdistill, job_id, req.prompt, req.negative_prompt, req.resolution, req.seed)
    return {"job_id": job_id}

@app.post("/generate/i2v")
async def generate_i2v(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    resolution: str = Form("720p"),
    seed: int = Form(0)
):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "job_id": job_id}
    
    # Save uploaded image
    input_image_path = f"/tmp/{job_id}_input.png"
    contents = await image.read()
    with open(input_image_path, "wb") as f:
        f.write(contents)
    
    background_tasks.add_task(run_diagdistill, job_id, prompt, negative_prompt, resolution, seed, input_image_path)
    return {"job_id": job_id}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return jobs[job_id]

@app.get("/download/{job_id}")
def download(job_id: str):
    if job_id not in jobs or jobs[job_id]["status"] != "completed":
        return JSONResponse(status_code=404, content={"error": "Output not ready"})
    return FileResponse(jobs[job_id]["output_path"])

@app.delete("/job/{job_id}")
def delete_job(job_id: str):
    if job_id in jobs:
        out_path = jobs[job_id].get("output_path")
        if out_path and os.path.exists(out_path):
            os.remove(out_path)
        del jobs[job_id]
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
