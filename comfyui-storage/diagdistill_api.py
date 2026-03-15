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
        
        # Construct command for DiagDistill predict_video.py
        # Based on: python predict_video.py --prompt "..." --output_path "..." --seed ...
        cmd = [
            "python", "predict_video.py",
            "--prompt", prompt,
            "--negative_prompt", negative_prompt,
            "--output_path", output_file,
            "--seed", str(seed),
            "--steps", "16",
        ]
        
        if start_image_path:
            cmd.extend(["--image_path", start_image_path])
            cmd.extend(["--mode", "i2v"])
        else:
            cmd.extend(["--mode", "t2v"])

        logging.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if os.path.exists(output_file):
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

@app.get("/health")
def health():
    return {"status": "ok", "cuda": torch.cuda.is_available()}

@app.post("/generate/t2v")
def generate_t2v(background_tasks: BackgroundTasks, prompt: str, negative_prompt: str = "", resolution: str = "720p", seed: int = 0):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "job_id": job_id}
    background_tasks.add_task(run_diagdistill, job_id, prompt, negative_prompt, resolution, seed)
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
