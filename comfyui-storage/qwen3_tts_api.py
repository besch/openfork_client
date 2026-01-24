"""
Qwen3-TTS FastAPI Wrapper

This provides a REST API for Qwen3-TTS generation, supporting:
- Custom Voice TTS (using built-in speakers)
- Voice Design (using natural language voice descriptions)
- Voice Cloning (using reference audio)
"""

import os
import uuid
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen3-TTS API", version="1.0.0")

# Global model instances
custom_voice_model = None
voice_design_model = None
base_model = None
model_lock = asyncio.Lock()

# Job storage
jobs: Dict[str, Dict[str, Any]] = {}

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=2)

INPUT_DIR = Path("/app/input")
OUTPUT_DIR = Path("/app/output")
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Supported languages and speakers for CustomVoice model
SUPPORTED_LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "German", 
                       "French", "Russian", "Portuguese", "Spanish", "Italian"]

SUPPORTED_SPEAKERS = [
    # Chinese speakers
    "Vivian", "Chelsie", "Austin", "Ethan",
    # English speakers  
    "Ryan", "Olivia", "Lily", "Emma",
    # Japanese speakers
    "Haruto", "Yuki",
    # Korean speakers
    "Minjun", "Sora",
    # German speakers
    "Leon", "Mia",
    # French speakers
    "Louis", "Chloe",
    # Russian speakers
    "Ivan", "Anna",
    # Portuguese speakers
    "Miguel", "Sofia",
    # Spanish speakers
    "Mateo", "Lucia",
    # Italian speakers
    "Leonardo", "Giulia"
]


class TTSRequest(BaseModel):
    """Request model for TTS generation."""
    text: str
    language: str = "Auto"
    speaker: str = "Olivia"
    instruct: Optional[str] = None
    mode: str = "custom_voice"  # custom_voice, voice_design, voice_clone
    voice_design_instruct: Optional[str] = None


class VoiceCloneRequest(BaseModel):
    """Request model for voice cloning."""
    text: str
    language: str = "Auto"
    ref_audio_path: str
    ref_text: Optional[str] = None
    x_vector_only: bool = False


class JobStatus(BaseModel):
    """Job status response model."""
    job_id: str
    status: str
    error: Optional[str] = None
    output_path: Optional[str] = None


def load_custom_voice_model():
    """Load the CustomVoice model."""
    global custom_voice_model
    if custom_voice_model is None:
        logger.info("Loading Qwen3-TTS CustomVoice model...")
        from qwen_tts import Qwen3TTSModel
        
        # Use environment variable to select model size (default 0.6B)
        # 16GB VRAM service can set QWEN_MODEL_SIZE=1.7B
        model_size = os.getenv("QWEN_MODEL_SIZE", "0.6B")
        model_name = f"Qwen/Qwen3-TTS-12Hz-{model_size}-CustomVoice"
        logger.info(f"Loading CustomVoice model: {model_name}")
        
        # Check if FlashAttention is available
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            logger.info("Using FlashAttention 2")
        except ImportError:
            attn_impl = "sdpa"
            logger.info("FlashAttention not available, using SDPA")
        
        custom_voice_model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        logger.info("CustomVoice model loaded successfully")
    return custom_voice_model


def load_voice_design_model():
    """Load the VoiceDesign model."""
    global voice_design_model
    if voice_design_model is None:
        logger.info("Loading Qwen3-TTS VoiceDesign model...")
        from qwen_tts import Qwen3TTSModel
        
        # Use environment variable to select model size (default 0.6B)
        # Note: Dockerfile only pre-downloads VoiceDesign for 1.7B
        model_size = os.getenv("QWEN_MODEL_SIZE", "0.6B")
        model_name = f"Qwen/Qwen3-TTS-12Hz-{model_size}-VoiceDesign"
        
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
        
        voice_design_model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        logger.info("VoiceDesign model loaded successfully")
    return voice_design_model


def load_base_model():
    """Load the Base model for voice cloning."""
    global base_model
    if base_model is None:
        logger.info("Loading Qwen3-TTS Base model for voice cloning...")
        from qwen_tts import Qwen3TTSModel
        
        # Use environment variable to select model size (default 0.6B)
        model_size = os.getenv("QWEN_MODEL_SIZE", "0.6B")
        model_name = f"Qwen/Qwen3-TTS-12Hz-{model_size}-Base"
        logger.info(f"Loading Base model: {model_name}")
        
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
        
        base_model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )

        logger.info("Base model loaded successfully")
    return base_model


def generate_custom_voice(text: str, language: str, speaker: str, instruct: Optional[str] = None) -> str:
    """Generate speech using CustomVoice model."""
    model = load_custom_voice_model()
    
    if instruct:
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
        )
    else:
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
        )
    
    output_path = OUTPUT_DIR / f"{uuid.uuid4()}.wav"
    sf.write(str(output_path), wavs[0], sr)
    return str(output_path)


def generate_voice_design(text: str, language: str, instruct: str) -> str:
    """Generate speech using VoiceDesign model with natural language voice description."""
    model = load_voice_design_model()
    
    wavs, sr = model.generate_voice_design(
        text=text,
        language=language,
        instruct=instruct,
    )
    
    output_path = OUTPUT_DIR / f"{uuid.uuid4()}.wav"
    sf.write(str(output_path), wavs[0], sr)
    return str(output_path)


def generate_voice_clone(text: str, language: str, ref_audio_path: str, 
                         ref_text: Optional[str] = None, x_vector_only: bool = False) -> str:
    """Generate speech using Base model with voice cloning."""
    model = load_base_model()
    
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only,
    )
    
    output_path = OUTPUT_DIR / f"{uuid.uuid4()}.wav"
    sf.write(str(output_path), wavs[0], sr)
    return str(output_path)


async def process_tts_job(job_id: str, request: TTSRequest):
    """Process a TTS generation job asynchronously."""
    try:
        jobs[job_id]["status"] = "processing"
        
        if request.mode == "custom_voice":
            output_path = await asyncio.get_event_loop().run_in_executor(
                executor,
                generate_custom_voice,
                request.text,
                request.language,
                request.speaker,
                request.instruct
            )
        elif request.mode == "voice_design":
            if not request.voice_design_instruct:
                raise ValueError("voice_design_instruct is required for voice_design mode")
            output_path = await asyncio.get_event_loop().run_in_executor(
                executor,
                generate_voice_design,
                request.text,
                request.language,
                request.voice_design_instruct
            )
        else:
            raise ValueError(f"Unknown mode: {request.mode}")
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_path"] = output_path
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


async def process_voice_clone_job(job_id: str, text: str, language: str, 
                                   ref_audio_path: str, ref_text: Optional[str], 
                                   x_vector_only: bool):
    """Process a voice cloning job asynchronously."""
    try:
        jobs[job_id]["status"] = "processing"
        
        output_path = await asyncio.get_event_loop().run_in_executor(
            executor,
            generate_voice_clone,
            text,
            language,
            ref_audio_path,
            ref_text,
            x_vector_only
        )
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_path"] = output_path
        logger.info(f"Voice clone job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Voice clone job {job_id} failed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "qwen3-tts"}


@app.get("/info")
async def get_info():
    """Get supported languages and speakers."""
    return {
        "languages": SUPPORTED_LANGUAGES,
        "speakers": SUPPORTED_SPEAKERS,
        "modes": ["custom_voice", "voice_design", "voice_clone"],
        "models": {
            "custom_voice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        }
    }


@app.post("/generate", response_model=JobStatus)
async def generate_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    """Submit a TTS generation job."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "error": None, "output_path": None}
    
    background_tasks.add_task(process_tts_job, job_id, request)
    
    return JobStatus(job_id=job_id, status="pending")


@app.post("/generate/voice-clone", response_model=JobStatus)
async def generate_voice_clone_endpoint(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    language: str = Form("Auto"),
    ref_text: Optional[str] = Form(None),
    x_vector_only: bool = Form(False),
    ref_audio: UploadFile = File(...)
):
    """Submit a voice cloning job with reference audio upload."""
    job_id = str(uuid.uuid4())
    
    # Save uploaded reference audio
    ref_audio_path = INPUT_DIR / f"{job_id}_ref.wav"
    with open(ref_audio_path, "wb") as f:
        content = await ref_audio.read()
        f.write(content)
    
    jobs[job_id] = {"status": "pending", "error": None, "output_path": None}
    
    background_tasks.add_task(
        process_voice_clone_job, 
        job_id, 
        text, 
        language, 
        str(ref_audio_path), 
        ref_text, 
        x_vector_only
    )
    
    return JobStatus(job_id=job_id, status="pending")


@app.post("/generate/voice-clone-path", response_model=JobStatus)
async def generate_voice_clone_from_path(request: VoiceCloneRequest, background_tasks: BackgroundTasks):
    """Submit a voice cloning job with reference audio path."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "error": None, "output_path": None}
    
    background_tasks.add_task(
        process_voice_clone_job,
        job_id,
        request.text,
        request.language,
        request.ref_audio_path,
        request.ref_text,
        request.x_vector_only
    )
    
    return JobStatus(job_id=job_id, status="pending")


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
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
    """Download the generated audio file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status is {job['status']}")
    
    output_path = job.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(output_path, media_type="audio/wav", filename=f"{job_id}.wav")


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its output files."""
    if job_id not in jobs:
        return {"status": "not_found"}
    
    job = jobs.pop(job_id)
    
    # Clean up output file
    if job.get("output_path") and os.path.exists(job["output_path"]):
        try:
            os.remove(job["output_path"])
        except OSError:
            pass
    
    # Clean up reference audio if exists
    ref_path = INPUT_DIR / f"{job_id}_ref.wav"
    if ref_path.exists():
        try:
            os.remove(ref_path)
        except OSError:
            pass
    
    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Qwen3-TTS API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
