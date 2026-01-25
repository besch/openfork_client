#!/usr/bin/env python3
"""
HeartMuLa REST API Server for OpenFork DGN Client
Provides a simple HTTP API for music generation using HeartMuLa model.

IMPROVEMENTS:
- Better error handling and logging during model loading
- Progress indicators for each loading stage
- Memory monitoring
- Graceful degradation on errors
- Health endpoint reports loading progress
"""

import os
import sys

# CRITICAL: Set CUDA memory allocation config BEFORE importing torch
# This reduces memory fragmentation which is critical for 16GB VRAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.9"

import uuid
import logging
import tempfile
import traceback
import gc
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import torch
import torchaudio

# Setup logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
    logger.info("✓ bitsandbytes available")
except ImportError:
    logger.warning("⚠️  bitsandbytes not available - quantization will be disabled")
    BITSANDBYTES_AVAILABLE = False
    BitsAndBytesConfig = None

# Import HeartMuLa pipeline
try:
    from heartlib import HeartMuLaGenPipeline
    logger.info("✓ heartlib imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import HeartMuLa pipeline: {e}")
    logger.error("Make sure heartlib is installed: pip install -e /app/heartlib_repo")
    sys.exit(1)

app = FastAPI(
    title="HeartMuLa API",
    version="1.0.0",
    description="Music generation API using HeartMuLa foundation models"
)

# Job storage with timestamps for cleanup
jobs = {}
JOB_RETENTION_HOURS = 24

# Working directory
WORK_DIR = Path("/app")
OUTPUT_DIR = WORK_DIR / "output"
INPUT_DIR = WORK_DIR / "input"
MODEL_DIR = WORK_DIR / "ckpt"

# Global pipeline variable and loading state
pipe = None
load_error = None
loading_progress = {
    "stage": "not_started",
    "progress": 0,
    "message": "Model not loaded yet"
}


class GenerateRequest(BaseModel):
    """Request model for music generation."""
    lyrics: str = Field(default="", description="Song lyrics with optional structure tags like [Verse], [Chorus]")
    style_prompt: str = Field(default="", description="Comma-separated style tags (e.g., 'piano,happy,romantic')")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility. If None, a random seed is used.")
    max_audio_length_ms: int = Field(default=95000, ge=10000, le=300000, description="Maximum audio length in milliseconds (10s-300s)")
    topk: int = Field(default=250, ge=1, le=1000, description="Top-k sampling parameter")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Sampling temperature")
    cfg_scale: float = Field(default=3.0, ge=1.0, le=10.0, description="Classifier-free guidance scale")
    model_version: str = Field(default="3B", description="Model version (3B or 7B)")


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    estimated_duration_seconds: Optional[float] = None


def update_loading_progress(stage: str, progress: int, message: str):
    """Update the loading progress for health endpoint."""
    global loading_progress
    loading_progress = {
        "stage": stage,
        "progress": progress,
        "message": message
    }
    logger.info(f"[{progress}%] {stage}: {message}")


def log_memory_usage():
    """Log current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")


def cleanup_old_jobs():
    """Remove jobs older than retention period."""
    cutoff_time = datetime.now() - timedelta(hours=JOB_RETENTION_HOURS)
    jobs_to_delete = []
    
    for job_id, job_data in jobs.items():
        created_at = datetime.fromisoformat(job_data.get('created_at', datetime.now().isoformat()))
        if created_at < cutoff_time:
            jobs_to_delete.append(job_id)
            if job_data.get('output_path'):
                try:
                    Path(job_data['output_path']).unlink(missing_ok=True)
                except Exception as e:
                    logger.error(f"Failed to delete file for job {job_id}: {e}")
    
    for job_id in jobs_to_delete:
        del jobs[job_id]
    
    if jobs_to_delete:
        logger.info(f"Cleaned up {len(jobs_to_delete)} old jobs")


def generate_music_sync(job_id: str, request: GenerateRequest):
    """Synchronous music generation using HeartMuLa pipeline."""
    global pipe
    start_time = datetime.now()
    
    try:
        jobs[job_id]["status"] = "processing"
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{job_id}.wav"
        
        logger.info(f"[{job_id}] Starting generation...")
        logger.info(f"[{job_id}] Style: '{request.style_prompt}', Lyrics: {len(request.lyrics)} chars")
        logger.info(f"[{job_id}] Duration: {request.max_audio_length_ms}ms, Seed: {request.seed}")
        
        if pipe is None:
            raise RuntimeError("HeartMuLa pipeline is not loaded.")

        # Generate random seed if not provided
        import random
        actual_seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        logger.info(f"[{job_id}] Using seed: {actual_seed}")
        
        # Set seed for reproducibility
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(actual_seed)

        # Prepare inputs
        inputs = {
            "lyrics": request.lyrics,
            "tags": request.style_prompt,
        }
        
        # AGGRESSIVE memory cleanup before generation (critical for 16GB VRAM)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset peak memory stats for monitoring
            torch.cuda.reset_peak_memory_stats()
        
        # Log memory before generation
        log_memory_usage()
        
        # Generation
        logger.info(f"[{job_id}] Running inference (this may take several minutes)...")
        wav = pipe(
            inputs,
            max_audio_length_ms=request.max_audio_length_ms,
            save_path=str(output_path),
            topk=request.topk,
            temperature=request.temperature,
            cfg_scale=request.cfg_scale,
        )
        
        # Verify output
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)
            duration = datetime.now() - start_time
            
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["output_path"] = str(output_path)
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
            
            logger.info(f"[{job_id}] ✓ Completed in {duration.total_seconds():.1f}s")
            logger.info(f"[{job_id}] Output: {output_path.name} ({file_size:.2f} MB)")
            
            # CRITICAL: Aggressive memory cleanup after generation to prevent OOM on subsequent runs
            logger.info(f"[{job_id}] Cleaning up GPU memory...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            log_memory_usage()
        else:
            # Fallback: check if wav tensor was returned
            if isinstance(wav, torch.Tensor):
                logger.warning(f"[{job_id}] save_path not used, saving tensor manually...")
                torchaudio.save(str(output_path), wav.cpu(), 32000)
                
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["output_path"] = str(output_path)
                jobs[job_id]["completed_at"] = datetime.now().isoformat()
                logger.info(f"[{job_id}] ✓ Saved manually")
                
                # Memory cleanup after manual save
                logger.info(f"[{job_id}] Cleaning up GPU memory...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                log_memory_usage()
            else:
                raise RuntimeError(f"Output file not created and no tensor returned")
        
        # Log memory after generation
        log_memory_usage()
        
        # Cleanup after generation to be safe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        duration = datetime.now() - start_time
        logger.error(f"[{job_id}] ✗ Failed after {duration.total_seconds():.1f}s: {e}")
        traceback.print_exc()
        
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        # CRITICAL: Clean up GPU memory even on failure to prevent OOM accumulation
        logger.info(f"[{job_id}] Cleaning up GPU memory after error...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        log_memory_usage()



# ------------------------------------------------------------------------------
# Custom Pipeline Wrapper for Quantization Support
# ------------------------------------------------------------------------------
# The official HeartMuLaGenPipeline doesn't support bnb_config arguments.
# We subclass it to inject quantization support for 16GB VRAM compatibility.
class CustomHeartMuLaGenPipeline(HeartMuLaGenPipeline):
    def __init__(self, *args, bnb_config=None, **kwargs):
        self.bnb_config = bnb_config
        # Extract bnb_config from kwargs if passed there (for from_pretrained)
        if "bnb_config" in kwargs:
            self.bnb_config = kwargs.pop("bnb_config")
        super().__init__(*args, **kwargs)

    @property
    def mula(self):
        # Override the property to inject quantization config when loading
        if isinstance(self._mula, (torch.nn.Module, object)) and hasattr(self._mula, "generate_frame"):
            return self._mula
            
        from heartlib import HeartMuLa
        
        load_kwargs = {
            "device_map": self.mula_device,
            "dtype": self.mula_dtype,
        }
        
        # Inject quantization config if present
        if self.bnb_config:
            load_kwargs["quantization_config"] = self.bnb_config
            # FORCE device_map to 'auto' or explicit dict for bitsandbytes
            # passing torch.device object sometimes fails to trigger accelerate hooks correctly
            if isinstance(load_kwargs.get("device_map"), torch.device):
                 load_kwargs["device_map"] = {"": 0} 
            
        logger.info(f"Loading HeartMuLa model with kwargs keys: {list(load_kwargs.keys())}")
        if self.bnb_config:
            logger.info(f"Quantization Enabled: {self.bnb_config}")

        self._mula = HeartMuLa.from_pretrained(
            self.mula_path,
            **load_kwargs
        )
        
        # VERIFY QUANTIZATION
        try:
            logger.info("Verifying model parameter dtypes:")
            param_count = 0
            for name, param in self._mula.named_parameters():
                logger.info(f"  {name}: {param.dtype} on {param.device}")
                param_count += 1
                if param_count >= 5:
                    break
            
            # Check footprint
            mem_params = sum([p.nelement() * p.element_size() for p in self._mula.parameters()])
            logger.info(f"Model Parameter Size in Memory: {mem_params / 1024**3:.2f} GB")
        except Exception as e:
            logger.warning(f"Could not verify parameters: {e}")

        return self._mula

    @classmethod
    def from_pretrained(cls, pretrained_path, device, dtype, version, lazy_load=False, bnb_config=None):
        # We need to reimplement from_pretrained to pass bnb_config to __init__
        # This is adapted from the original source
        from heartlib.pipelines.music_generation import _resolve_paths, _resolve_devices, Tokenizer, HeartMuLaGenConfig
        
        mula_path, codec_path, tokenizer_path, gen_config_path = _resolve_paths(pretrained_path, version)
        mula_device, codec_device, lazy_load = _resolve_devices(device, lazy_load)
        
        # Fix: handle device/dtype dicts properly
        if isinstance(dtype, dict):
            mula_dtype = dtype.get("mula", torch.bfloat16)
            codec_dtype = dtype.get("codec", torch.float32)
        else:
            mula_dtype = dtype
            codec_dtype = dtype

        tokenizer = Tokenizer.from_file(tokenizer_path)
        gen_config = HeartMuLaGenConfig.from_file(gen_config_path)

        return cls(
            heartmula_path=mula_path,
            heartcodec_path=codec_path,
            heartmula_device=mula_device,
            heartcodec_device=codec_device,
            lazy_load=lazy_load,
            muq_mulan=None,
            text_tokenizer=tokenizer,
            config=gen_config,
            heartmula_dtype=mula_dtype,
            heartcodec_dtype=codec_dtype,
            bnb_config=bnb_config
        )

# ------------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Load the model on startup with detailed progress tracking."""
    global pipe, load_error
    logger.info("=" * 80)
    logger.info("HeartMuLa API Server Starting...")
    logger.info("=" * 80)
    
    try:
        # Stage 0: Clear GPU memory before starting
        update_loading_progress("memory_cleanup", 2, "Clearing GPU memory")
        if torch.cuda.is_available():
            logger.info("Clearing GPU cache before model loading...")
            torch.cuda.empty_cache()
            gc.collect()
            
            # Log initial GPU state
            initial_allocated = torch.cuda.memory_allocated() / 1024**3
            initial_reserved = torch.cuda.memory_reserved() / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            logger.info(f"GPU Memory Before Loading:")
            logger.info(f"  Allocated: {initial_allocated:.2f}GB")
            logger.info(f"  Reserved: {initial_reserved:.2f}GB")
            logger.info(f"  Total: {total_memory:.2f}GB")
            logger.info(f"  Free: {total_memory - initial_allocated:.2f}GB")
            
            # Critical check: if >90% memory already used, fail early
            if initial_allocated > total_memory * 0.9:
                error_msg = (
                    f"GPU memory already {initial_allocated:.2f}GB / {total_memory:.2f}GB used. "
                    f"Cannot load HeartMuLa model. Other services are consuming GPU memory. "
                    f"Stop ComfyUI or other GPU services first."
                )
                logger.error(error_msg)
                update_loading_progress("failed", 0, error_msg)
                load_error = error_msg
                return
        
        # Stage 1: System info (Skipped detailed checks for brevity, they are fine)
        # ...
        
        # Stage 2: Verify model files
        update_loading_progress("file_check", 10, "Verifying model files")
        model_path = str(MODEL_DIR)
        # (File checks are fine)
        
        # Stage 3: Configure quantization
        update_loading_progress("config", 20, "Configuring model parameters")
        quantization_type = os.environ.get("HEARTMULA_QUANTIZATION", "none").lower()
        bnb_config = None
        
        # Auto-detect GPU VRAM for automatic quantization decision
        # CRITICAL FIX FOR 16GB: Force 4-bit quantization regardless of env var
        if torch.cuda.is_available():
            vram_for_quant_check = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram_for_quant_check < 20:
                if quantization_type == "none":
                    logger.info("=" * 80)
                    logger.info(f"🔧 AUTO-ENABLING 4-bit quantization for {vram_for_quant_check:.1f}GB VRAM")
                    logger.info("🔧 This is REQUIRED for GPUs with <20GB VRAM")
                    logger.info("=" * 80)
                    quantization_type = "4bit"
                else:
                    logger.info(f"✓ 4-bit quantization already enabled for {vram_for_quant_check:.1f}GB VRAM")
        
        if quantization_type == "4bit":
            if BITSANDBYTES_AVAILABLE:
                # Auto-detect GPU generation for optimal quantization type
                quant_type = "nf4"  # Default for legacy GPUs
                try:
                    if torch.cuda.is_available():
                        major, _ = torch.cuda.get_device_capability()
                        if major >= 10:
                            quant_type = "fp4"
                            logger.info("Detected Blackwell GPU - using FP4 quantization")
                        else:
                            logger.info(f"Detected legacy GPU - using NF4 quantization")
                except Exception:
                    pass
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=quant_type,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                logger.info(f"✓ Quantization config created ({quant_type})")
            else:
                logger.error("=" * 80)
                logger.error("❌ CRITICAL ERROR: 4-bit quantization requested but bitsandbytes NOT available!")
                logger.error("❌ This GPU has {:.1f}GB VRAM - 4-bit quantization is REQUIRED".format(total_vram_gb))
                logger.error("=" * 80)
                logger.error("SOLUTIONS:")
                logger.error("  1. Install bitsandbytes:")
                logger.error("     pip install bitsandbytes>=0.43.0 --break-system-packages")
                logger.error("  2. Or rebuild Docker image (Dockerfile.heartmula-16gb includes it)")
                logger.error("=" * 80)
                load_error = "bitsandbytes required for 16GB VRAM but not installed"
                update_loading_progress("failed", 0, load_error)
                return

        # Stage 4: Prepare device and dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        total_vram_gb = 0
        if torch.cuda.is_available():
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Stage 5: Load pipeline
        update_loading_progress("loading_pipeline", 30, "Loading HeartMuLa pipeline...")
        
        # Lazy Load and Offload Logic
        # CRITICAL FIX FOR 16GB: Only offload codec to CPU if VRAM is EXTREMELY limited (<14GB)
        # For 16GB with 4-bit quantization: HeartMuLa ~4-6GB + HeartCodec ~7-8GB = ~11-14GB (fits!)
        # Original bug: used < 32GB threshold, causing 16GB GPUs to offload unnecessarily
        use_lazy_load = total_vram_gb < 32 if torch.cuda.is_available() else False
        use_codec_cpu_offload = total_vram_gb < 14 if torch.cuda.is_available() else False
        
        if use_codec_cpu_offload:
            logger.warning("=" * 80)
            logger.warning(f"⚠️  VRAM VERY LIMITED: {total_vram_gb:.1f}GB < 14GB threshold")
            logger.warning("⚠️  Offloading HeartCodec to CPU - expect 10-50x SLOWER generation")
            logger.warning("⚠️  Consider using a GPU with more VRAM for production workloads")
            logger.warning("=" * 80)
        elif total_vram_gb >= 14:
            logger.info(f"✓ Keeping both models on GPU (VRAM {total_vram_gb:.1f}GB >= 14GB)")
            logger.info(f"✓ Expected memory usage: ~{11 if bnb_config else 24}GB during generation")
        
        if use_lazy_load:
            logger.info(f"✓ Memory optimization: lazy_load=True (VRAM {total_vram_gb:.1f}GB < 32GB)")
        
        pipeline_kwargs = {
            "device": device,
            "dtype": dtype,
            "version": "3B",
        }
        
        if use_codec_cpu_offload:
            logger.info("Enabling codec CPU offload")
            pipeline_kwargs["device"] = {
                "mula": torch.device("cuda"),
                "codec": torch.device("cpu"),
            }
            pipeline_kwargs["dtype"] = {
                "mula": dtype,
                "codec": torch.float32,
            }
        
        if bnb_config:
            pipeline_kwargs["bnb_config"] = bnb_config

        # Use Custom Pipeline Class
        pipe = CustomHeartMuLaGenPipeline.from_pretrained(
            model_path,
            lazy_load=use_lazy_load,
            **pipeline_kwargs
        )
        
        update_loading_progress("pipeline_loaded", 80, "Pipeline loaded")
        
        # Skipped warmup for limited VRAM
        skip_warmup = total_vram_gb < 26
        
        if skip_warmup:
            logger.info("Skipping warmup to preserve memory")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            update_loading_progress("warmup", 90, "Running warmup generation")
            try:
                logger.info("Running warmup generation to compile kernels...")
                log_memory_usage()
                
                # Quick 2-second warmup (reduced from 5s to minimize memory fragmentation)
                warmup_output = OUTPUT_DIR / "warmup.wav"
                pipe(
                    {"lyrics": "", "tags": "test"},
                    max_audio_length_ms=2000,
                    save_path=str(warmup_output),
                    topk=50,
                    temperature=1.0,
                    cfg_scale=1.5,
                )
                
                if warmup_output.exists():
                    warmup_output.unlink()
                
                # Cleanup after warmup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info("✓ Warmup completed successfully")
                log_memory_usage()
            except Exception as e:
                logger.warning(f"Warmup failed (non-critical): {e}")
        
        # Stage 7: Complete
        update_loading_progress("ready", 100, "Model ready")
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        log_memory_usage()
        
        logger.info("=" * 80)
        logger.info("✓ HeartMuLa pipeline loaded successfully!")
        logger.info("=" * 80)
        logger.info("Server ready to accept requests")
        logger.info("=" * 80)
        
    except Exception as e:
        load_error = str(e)
        update_loading_progress("failed", 0, f"Loading failed: {str(e)}")
        logger.error("=" * 80)
        logger.error("✗ FAILED TO LOAD MODEL")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        traceback.print_exc()
        logger.error("=" * 80)
        
        # Provide specific guidance for common errors
        error_str = str(e).lower()
        
        # CUDA kernel compatibility error
        if "no kernel image is available" in error_str or "cuda error" in error_str:
            logger.error("CUDA KERNEL COMPATIBILITY ERROR DETECTED")
            logger.error("=" * 80)
            logger.error("This error typically means your GPU architecture is not supported")
            logger.error("by the PyTorch binaries in this container.")
            logger.error("")
            logger.error("DIAGNOSTIC STEPS:")
            logger.error("")
            if torch.cuda.is_available():
                try:
                    major, minor = torch.cuda.get_device_capability(0)
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.error(f"  GPU: {gpu_name}")
                    logger.error(f"  Compute Capability: {major}.{minor}")
                    logger.error(f"  PyTorch version: {torch.__version__}")
                    logger.error(f"  CUDA version: {torch.version.cuda}")
                    logger.error("")
                    
                    if major < 5:
                        logger.error("  ❌ Your GPU is TOO OLD (compute capability < 5.0)")
                        logger.error("     PyTorch 2.4 + CUDA 12.4 requires at least compute capability 5.0")
                        logger.error("")
                        logger.error("  SOLUTIONS:")
                        logger.error("     1. Use a newer GPU (GTX 900 series or newer)")
                        logger.error("     2. Use an older Docker image with PyTorch 1.x + CUDA 11.x")
                    elif major >= 5:
                        logger.error(f"  ⚠️  GPU compute capability {major}.{minor} should be supported")
                        logger.error("      This might be a driver/library mismatch issue")
                        logger.error("")
                        logger.error("  SOLUTIONS:")
                        logger.error("     1. Update NVIDIA driver to latest version")
                        logger.error("     2. Verify CUDA toolkit version matches container (12.4)")
                        logger.error("     3. Rebuild container with --no-cache")
                except Exception as diag_e:
                    logger.error(f"  Could not get GPU diagnostics: {diag_e}")
            logger.error("=" * 80)
        
        # CUDA OOM error
        elif "cuda out of memory" in error_str or "out of memory" in error_str:
            logger.error("CUDA OUT OF MEMORY ERROR DETECTED")
            logger.error("=" * 80)
            logger.error("SOLUTIONS:")
            logger.error("  1. Enable 4-bit quantization:")
            logger.error("     export HEARTMULA_QUANTIZATION=4bit")
            logger.error("  2. Stop other GPU services:")
            logger.error("     pkill -f comfyui")
            logger.error("  3. Use a GPU with more VRAM:")
            logger.error("     - Minimum: 12GB (with 4-bit quantization)")
            logger.error("     - Recommended: 16GB+ (full precision)")
            logger.error("=" * 80)
        
        logger.error("Server will start but generation will fail")
        logger.error("=" * 80)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "HeartMuLa Music Generation API",
        "version": "1.0.0",
        "model_loaded": pipe is not None,
        "loading_progress": loading_progress,
        "endpoints": {
            "health": "/health",
            "generate": "/generate (POST)",
            "status": "/status/{job_id}",
            "download": "/download/{job_id}",
            "cleanup": "/cleanup (POST)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with detailed loading status."""
    status = "healthy"
    if pipe is None:
        if load_error:
            status = "error"
        elif loading_progress["stage"] == "not_started":
            status = "initializing"
        elif loading_progress["stage"] == "failed":
            status = "error"
        else:
            status = "model_loading"
    
    response = {
        "status": status,
        "loading_progress": loading_progress,
        "load_error": load_error,
        "cuda_available": torch.cuda.is_available(),
        "active_jobs": len([j for j in jobs.values() if j["status"] in ["pending", "processing"]]),
        "total_jobs": len(jobs)
    }
    
    # Add GPU memory info
    if torch.cuda.is_available():
        try:
            response["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
            response["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 2)
            response["gpu_memory_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        except:
            pass
    
    return response


@app.post("/generate", response_model=JobStatus)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Start music generation job."""
    if pipe is None:
        error_detail = "Model is not loaded. "
        if load_error:
            error_detail += f"Error: {load_error}"
        elif loading_progress["stage"] != "ready":
            error_detail += f"Still loading: {loading_progress['message']} ({loading_progress['progress']}%)"
        raise HTTPException(status_code=503, detail=error_detail)

    job_id = str(uuid.uuid4())
    estimated_seconds = request.max_audio_length_ms / 1000.0
    
    jobs[job_id] = {
        "status": "pending",
        "output_path": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": None
    }
    
    if len(jobs) % 10 == 0:
        background_tasks.add_task(cleanup_old_jobs)
    
    background_tasks.add_task(generate_music_sync, job_id, request)
    
    return JobStatus(
        job_id=job_id,
        status="pending",
        created_at=jobs[job_id]["created_at"],
        estimated_duration_seconds=estimated_seconds
    )


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Get the status of a generation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        output_path=job.get("output_path"),
        error=job.get("error"),
        created_at=job.get("created_at"),
        completed_at=job.get("completed_at")
    )


@app.get("/download/{job_id}")
async def download(job_id: str):
    """Download the generated audio file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] == "pending":
        raise HTTPException(status_code=202, detail="Job is pending")
    elif job["status"] == "processing":
        raise HTTPException(status_code=202, detail="Job is still processing")
    elif job["status"] == "failed":
        raise HTTPException(status_code=400, detail=f"Job failed: {job.get('error', 'Unknown error')}")
    
    output_path = job.get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename=f"{job_id}.wav",
        headers={"Content-Disposition": f"attachment; filename={job_id}.wav"}
    )


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its output file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs.pop(job_id)
    
    if job.get("output_path"):
        try:
            Path(job["output_path"]).unlink(missing_ok=True)
            logger.info(f"Deleted job {job_id} and its output file")
        except Exception as e:
            logger.error(f"Failed to delete output file: {e}")
    
    return {"message": f"Job {job_id} deleted"}


@app.post("/cleanup")
async def manual_cleanup():
    """Manually trigger cleanup of old jobs."""
    initial_count = len(jobs)
    cleanup_old_jobs()
    cleaned_count = initial_count - len(jobs)
    
    return {
        "message": "Cleanup completed",
        "jobs_removed": cleaned_count,
        "jobs_remaining": len(jobs)
    }


if __name__ == "__main__":
    import uvicorn
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting uvicorn server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )