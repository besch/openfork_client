import os
import uuid
import time
import asyncio
import argparse
import subprocess
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

async def _process_video(task_id: str, input_path: str, upscale_factor: int, ref_mode: str, ref_guidance_scale: float):
    JOBS[task_id] = {"status": "processing", "progress": 0}
    output_path = OUTPUT_DIR / f"{task_id}_out.mp4"
    
    try:
        # Here we invoke the sparkvsr inference script according to the parameters.
        # Assuming taco-group/SparkVSR comes with an inference.py or similar
        print(f"[Task {task_id}] Running SparkVSR on {input_path}")
        
        # Example command, will need adjustment based on actual repo structure
        cmd = [
            "python", "inference.py",
            "--video_path", str(input_path),
            "--save_dir", str(output_path.parent),
            "--upscale_factor", str(upscale_factor),
            "--ref_mode", ref_mode,
            "--ref_guidance_scale", str(ref_guidance_scale)
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            print(f"[Task {task_id}] Failed. Error:\n{error_msg}")
            JOBS[task_id] = {"status": "failed", "error": error_msg}
            return
            
        print(f"[Task {task_id}] Successfully generated {output_path}")
        JOBS[task_id] = {
            "status": "completed", 
            "output": str(output_path)
        }
            
    except Exception as e:
        print(f"[Task {task_id}] Exception: {e}")
        JOBS[task_id] = {"status": "failed", "error": str(e)}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upscale")
async def upscale(
    video: UploadFile,
    upscale: int = Form(4),
    ref_mode: str = Form("no_ref"),
    ref_guidance_scale: float = Form(1.0)
):
    task_id = str(uuid.uuid4())
    input_path = INPUT_DIR / f"{task_id}_{video.filename}"
    
    with open(input_path, "wb") as f:
        f.write(await video.read())
        
    JOBS[task_id] = {"status": "queued", "progress": 0}
    
    # Run processing loop in background
    asyncio.create_task(_process_video(task_id, str(input_path), upscale, ref_mode, ref_guidance_scale))
    
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
        
    # Standard fallback on the way SparkVSR dumps files into save_dir
    # Usually it's named derived from input file, so we may need to find the mp4.
    out_file = job["output"]
    if os.path.exists(out_file):
         return FileResponse(out_file, media_type="video/mp4", filename=f"{task_id}_upscaled.mp4")
    else:
         # Find any mp4 generated in output dir tracking this task
         for file in OUTPUT_DIR.glob(f"*{task_id}*.mp4"):
             return FileResponse(str(file), media_type="video/mp4", filename=file.name)
         
         # Otherwise fallback just the newest mp4 (for single-tenant docker instances)
         videos = sorted(OUTPUT_DIR.glob("*.mp4"), key=os.path.getmtime)
         if videos:
             return FileResponse(str(videos[-1]), media_type="video/mp4", filename=videos[-1].name)
             
         raise HTTPException(status_code=500, detail="Generated file not found")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
