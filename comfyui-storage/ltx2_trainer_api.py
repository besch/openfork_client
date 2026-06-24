#!/usr/bin/env python3
"""FastAPI wrapper for the official Lightricks LTX-2 trainer."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import yaml
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

app = FastAPI(title="OpenFork LTX-2 Official Trainer API", version="1.0.0")

LTX2_REPO = Path(os.environ.get("LTX2_REPO", "/opt/LTX-2"))
TRAINER_DIR = Path(
    os.environ.get("LTX2_TRAINER_DIR", str(LTX2_REPO / "packages" / "ltx-trainer"))
)
MODEL_PATH = Path(
    os.environ.get("LTX2_MODEL_PATH", "/models/ltx2/ltx-2.3-22b-dev.safetensors")
)
TEXT_ENCODER_PATH = Path(
    os.environ.get(
        "LTX2_TEXT_ENCODER_PATH",
        "/models/gemma-3-12b-it-qat-q4_0-unquantized",
    )
)
JOBS_ROOT = Path(os.environ.get("LTX2_JOBS_ROOT", "/app/ltx2_jobs"))
OUTPUT_DIR = Path(os.environ.get("LTX2_OUTPUT_DIR", "/app/output"))
DEFAULT_BUCKETS = os.environ.get("LTX2_TRAINER_RESOLUTION_BUCKETS", "512x288x49")
DEFAULT_STEPS = int(os.environ.get("LTX2_TRAINER_STEPS", "2000"))
DEFAULT_RANK = int(os.environ.get("LTX2_TRAINER_RANK", "16"))
DEFAULT_ALPHA = int(os.environ.get("LTX2_TRAINER_ALPHA", str(DEFAULT_RANK)))
DEFAULT_QUANTIZATION = os.environ.get("LTX2_TRAINER_QUANTIZATION", "int8-quanto")
TARGET_VRAM_GB = int(os.environ.get("LTX2_TARGET_VRAM_GB", "32"))
VRAM_PROFILE = os.environ.get("LTX2_VRAM_PROFILE", f"{TARGET_VRAM_GB}gb")

SUPPORTED_TRAINER_QUANTIZATION = {
    "",
    "none",
    "int8-quanto",
    "int4-quanto",
    "int2-quanto",
    "fp8-quanto",
    "fp8uz-quanto",
}

JOBS_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()


def _set_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        current = jobs.setdefault(job_id, {})
        current.update(updates)
        current["updated_at"] = time.time()


def _get_job(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(job)


def _safe_extract_zip(archive_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        dest_root = dest_dir.resolve()
        for member in archive.infolist():
            member_path = (dest_dir / member.filename).resolve()
            if dest_root not in (member_path, *member_path.parents):
                raise ValueError(f"Unsafe archive path: {member.filename}")
        archive.extractall(dest_dir)


def _safe_extract_tar(archive_path: Path, dest_dir: Path) -> None:
    with tarfile.open(archive_path) as archive:
        dest_root = dest_dir.resolve()
        for member in archive.getmembers():
            member_path = (dest_dir / member.name).resolve()
            if dest_root not in (member_path, *member_path.parents):
                raise ValueError(f"Unsafe archive path: {member.name}")
        archive.extractall(dest_dir)


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    suffixes = "".join(archive_path.suffixes).lower()
    if suffixes.endswith(".zip"):
        _safe_extract_zip(archive_path, dest_dir)
        return
    if suffixes.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz")):
        _safe_extract_tar(archive_path, dest_dir)
        return
    raise ValueError("Dataset archive must be .zip, .tar, .tar.gz, .tgz, .tar.bz2, or .tar.xz")


def _find_dataset_file(root: Path) -> Path:
    preferred = [
        root / "dataset.json",
        root / "dataset.jsonl",
        root / "dataset.csv",
        root / "metadata.json",
        root / "metadata.jsonl",
        root / "metadata.csv",
    ]
    for path in preferred:
        if path.exists():
            return path

    matches = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv"}
    ]
    if not matches:
        raise FileNotFoundError(
            "No dataset metadata file found. Expected dataset.json, dataset.jsonl, "
            "dataset.csv, or another JSON/JSONL/CSV metadata file."
        )
    return sorted(matches, key=lambda p: (len(p.parts), str(p)))[0]


def _parse_video_dims(resolution_buckets: str) -> list[int]:
    first_bucket = resolution_buckets.split(";", 1)[0].strip()
    parts = first_bucket.lower().split("x")
    if len(parts) != 3:
        return [512, 288, 49]
    width, height, frames = (int(part) for part in parts)
    width = max(32, (width // 32) * 32)
    height = max(32, (height // 32) * 32)
    frames = max(1, int(frames))
    frames = frames if frames % 8 == 1 else max(1, ((frames - 1) // 8) * 8 + 1)
    return [width, height, frames]


def _normalize_trainer_quantization(value: str | None) -> str | None:
    quantization = (value or "").strip()
    if quantization not in SUPPORTED_TRAINER_QUANTIZATION:
        raise ValueError(
            "Unsupported trainer quantization. Expected one of: "
            + ", ".join(sorted(item or "none" for item in SUPPORTED_TRAINER_QUANTIZATION))
        )
    if quantization in {"", "none"}:
        return None
    return quantization


def _run_command(
    job_id: str,
    command: list[str],
    cwd: Path,
    log_path: Path,
    phase: str,
) -> None:
    _set_job(job_id, status="running", phase=phase, command=" ".join(command))
    with log_path.open("a", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"\n\n===== {phase}: {' '.join(command)} =====\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert process.stdout is not None
        tail: list[str] = []
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            tail.append(line.rstrip())
            tail = tail[-80:]
            if len(tail) % 20 == 0:
                _set_job(job_id, log_tail=tail)
        return_code = process.wait()
        _set_job(job_id, log_tail=tail)
    if return_code != 0:
        raise RuntimeError(f"{phase} failed with exit code {return_code}")


def _build_training_config(
    *,
    output_dir: Path,
    precomputed_dir: Path,
    resolution_buckets: str,
    steps: int,
    rank: int,
    alpha: int,
    learning_rate: float,
    first_frame_probability: float,
    gradient_accumulation_steps: int,
    quantization: str | None,
    validation_prompt: str | None,
    seed: int,
    skip_audio: bool,
) -> dict[str, Any]:
    video_strategy: dict[str, Any] = {
        "is_generated": True,
        "latents_dir": "latents",
        "conditions": [
            {
                "type": "first_frame",
                "probability": max(0.0, min(1.0, first_frame_probability)),
            }
        ],
    }
    strategy: dict[str, Any] = {
        "name": "flexible",
        "video": video_strategy,
    }
    if not skip_audio:
        strategy["audio"] = {
            "is_generated": True,
            "latents_dir": "audio_latents",
        }

    validation_samples = []
    if validation_prompt:
        validation_samples.append({"prompt": validation_prompt})

    return {
        "model": {
            "model_path": str(MODEL_PATH),
            "text_encoder_path": str(TEXT_ENCODER_PATH),
            "training_mode": "lora",
            "load_checkpoint": None,
        },
        "lora": {
            "rank": rank,
            "alpha": alpha,
            "dropout": 0.0,
            "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        },
        "training_strategy": strategy,
        "optimization": {
            "learning_rate": learning_rate,
            "steps": steps,
            "batch_size": 1,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_grad_norm": 1.0,
            "optimizer_type": "adamw8bit",
            "scheduler_type": "linear",
            "scheduler_params": {},
            "enable_gradient_checkpointing": True,
        },
        "acceleration": {
            "mixed_precision_mode": "bf16",
            "quantization": quantization,
            "load_text_encoder_in_8bit": True,
            "offload_optimizer_during_validation": True,
        },
        "data": {
            "preprocessed_data_root": str(precomputed_dir),
            "num_dataloader_workers": 2,
        },
        "validation": {
            "samples": validation_samples,
            "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
            "video_dims": _parse_video_dims(resolution_buckets),
            "frame_rate": 25.0,
            "seed": seed,
            "inference_steps": 30,
            "interval": None,
            "guidance_scale": 4.0,
            "stg_scale": 1.0,
            "stg_blocks": [29],
            "stg_mode": "stg_v" if skip_audio else "stg_av",
            "generate_audio": not skip_audio,
            "skip_initial_validation": True,
        },
        "checkpoints": {
            "interval": max(100, min(500, max(1, steps // 4))),
            "keep_last_n": 2,
            "precision": "bfloat16",
        },
        "flow_matching": {
            "timestep_sampling_mode": "shifted_logit_normal",
            "timestep_sampling_params": {},
        },
        "hub": {"push_to_hub": False, "hub_model_id": None},
        "wandb": {
            "enabled": False,
            "project": "ltx-2-trainer",
            "entity": None,
            "tags": ["openfork", "ltx2", "i2v-lora", "low-vram"],
            "log_validation_videos": False,
        },
        "seed": seed,
        "output_dir": str(output_dir),
    }


def _zip_output(job_id: str, run_dir: Path, config_path: Path, log_path: Path) -> Path:
    archive_path = OUTPUT_DIR / f"{job_id}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if config_path.exists():
            archive.write(config_path, "openfork_training_config.yaml")
        if log_path.exists():
            archive.write(log_path, "openfork_training.log")
        for path in run_dir.rglob("*"):
            if path.is_file():
                archive.write(path, f"trainer_output/{path.relative_to(run_dir)}")
    return archive_path


def _train_job(
    job_id: str,
    archive_path: Path,
    *,
    lora_trigger: str,
    resolution_buckets: str,
    steps: int,
    rank: int,
    alpha: int,
    learning_rate: float,
    first_frame_probability: float,
    gradient_accumulation_steps: int,
    quantization: str | None,
    skip_audio: bool,
    validation_prompt: str | None,
    seed: int,
) -> None:
    workspace = JOBS_ROOT / job_id
    dataset_dir = workspace / "dataset"
    precomputed_dir = workspace / "precomputed"
    run_dir = workspace / "output"
    log_path = workspace / "train.log"
    config_path = workspace / "config.yaml"

    try:
        workspace.mkdir(parents=True, exist_ok=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        precomputed_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        _set_job(job_id, status="running", phase="extracting_dataset")
        _extract_archive(archive_path, dataset_dir)
        dataset_file = _find_dataset_file(dataset_dir)

        preprocess_cmd = [
            "uv",
            "run",
            "python",
            "scripts/process_dataset.py",
            str(dataset_file),
            "--resolution-buckets",
            resolution_buckets,
            "--model-path",
            str(MODEL_PATH),
            "--text-encoder-path",
            str(TEXT_ENCODER_PATH),
            "--output-dir",
            str(precomputed_dir),
            "--batch-size",
            "1",
            "--load-text-encoder-in-8bit",
            "--overwrite",
        ]
        if lora_trigger:
            preprocess_cmd.extend(["--lora-trigger", lora_trigger])
        if skip_audio:
            preprocess_cmd.append("--skip-audio")
        _run_command(job_id, preprocess_cmd, TRAINER_DIR, log_path, "preprocess")

        config = _build_training_config(
            output_dir=run_dir,
            precomputed_dir=precomputed_dir,
            resolution_buckets=resolution_buckets,
            steps=steps,
            rank=rank,
            alpha=alpha,
            learning_rate=learning_rate,
            first_frame_probability=first_frame_probability,
            gradient_accumulation_steps=gradient_accumulation_steps,
            quantization=quantization,
            validation_prompt=validation_prompt,
            seed=seed,
            skip_audio=skip_audio,
        )
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        train_cmd = [
            "uv",
            "run",
            "python",
            "scripts/train.py",
            str(config_path),
        ]
        _run_command(job_id, train_cmd, TRAINER_DIR, log_path, "train")

        output_archive = _zip_output(job_id, run_dir, config_path, log_path)
        lora_files = [
            str(path.relative_to(run_dir))
            for path in run_dir.rglob("*.safetensors")
            if path.is_file()
        ]
        _set_job(
            job_id,
            status="completed",
            phase="completed",
            output_path=str(output_archive),
            output_size_bytes=output_archive.stat().st_size,
            lora_files=lora_files,
            settings={
                "resolution_buckets": resolution_buckets,
                "steps": steps,
                "rank": rank,
                "alpha": alpha,
                "learning_rate": learning_rate,
                "first_frame_probability": first_frame_probability,
                "quantization": quantization,
                "skip_audio": skip_audio,
                "seed": seed,
            },
        )
    except Exception as exc:
        _set_job(job_id, status="failed", phase="failed", error=str(exc))


@app.get("/health")
def health() -> dict[str, Any]:
    model_ready = MODEL_PATH.exists() and TEXT_ENCODER_PATH.exists()
    return {
        "status": "ok" if model_ready else "error",
        "model_loaded": model_ready,
        "model_path": str(MODEL_PATH),
        "text_encoder_path": str(TEXT_ENCODER_PATH),
        "trainer_dir": str(TRAINER_DIR),
        "trainer_dir_exists": TRAINER_DIR.exists(),
        "standard_vram_gb": 80,
        "low_vram_gb": 32,
        "target_vram_gb": TARGET_VRAM_GB,
        "vram_profile": VRAM_PROFILE,
        "default_resolution_buckets": DEFAULT_BUCKETS,
        "default_rank": DEFAULT_RANK,
        "default_alpha": DEFAULT_ALPHA,
        "default_quantization": DEFAULT_QUANTIZATION,
        "experimental": TARGET_VRAM_GB < 32,
        "notes": (
            "Official LTX-2 Trainer: 80GB+ standard config, 32GB low-VRAM INT8 config. "
            "Profiles below 32GB are experimental and use more aggressive quantization/resolution defaults."
        ),
    }


@app.post("/train/i2v-lora")
async def train_i2v_lora(
    background_tasks: BackgroundTasks,
    dataset_archive: UploadFile = File(...),
    lora_trigger: str = Form(""),
    resolution_buckets: str = Form(DEFAULT_BUCKETS),
    steps: int = Form(DEFAULT_STEPS),
    rank: int = Form(DEFAULT_RANK),
    alpha: int = Form(DEFAULT_ALPHA),
    learning_rate: float = Form(1e-4),
    first_frame_probability: float = Form(0.5),
    gradient_accumulation_steps: int = Form(1),
    quantization: str = Form(DEFAULT_QUANTIZATION),
    skip_audio: bool = Form(True),
    validation_prompt: str = Form(""),
    seed: int = Form(42),
) -> dict[str, str]:
    if not MODEL_PATH.exists() or not TEXT_ENCODER_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="LTX-2 checkpoint or Gemma text encoder is missing.",
        )
    if not TRAINER_DIR.exists():
        raise HTTPException(status_code=503, detail="LTX-2 trainer directory is missing.")

    try:
        normalized_quantization = _normalize_trainer_quantization(quantization)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = uuid.uuid4().hex
    workspace = JOBS_ROOT / job_id
    workspace.mkdir(parents=True, exist_ok=True)
    archive_suffix = "".join(Path(dataset_archive.filename or "dataset.zip").suffixes) or ".zip"
    archive_path = workspace / f"dataset{archive_suffix}"
    with archive_path.open("wb") as handle:
        shutil.copyfileobj(dataset_archive.file, handle)

    _set_job(
        job_id,
        status="queued",
        phase="queued",
        created_at=time.time(),
        archive_path=str(archive_path),
    )
    background_tasks.add_task(
        _train_job,
        job_id,
        archive_path,
        lora_trigger=lora_trigger.strip(),
        resolution_buckets=resolution_buckets.strip() or DEFAULT_BUCKETS,
        steps=max(1, int(steps)),
        rank=max(1, int(rank)),
        alpha=max(1, int(alpha)),
        learning_rate=float(learning_rate),
        first_frame_probability=float(first_frame_probability),
        gradient_accumulation_steps=max(1, int(gradient_accumulation_steps)),
        quantization=normalized_quantization,
        skip_audio=bool(skip_audio),
        validation_prompt=validation_prompt.strip() or None,
        seed=int(seed),
    )
    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def status(job_id: str) -> dict[str, Any]:
    return _get_job(job_id)


@app.get("/output/{job_id}")
def output(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Job is not completed")
    output_path = Path(str(job.get("output_path", "")))
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output not found")
    return FileResponse(
        output_path,
        media_type="application/zip",
        filename=output_path.name,
    )


@app.delete("/job/{job_id}")
def delete_job(job_id: str) -> dict[str, bool]:
    with jobs_lock:
        jobs.pop(job_id, None)
    shutil.rmtree(JOBS_ROOT / job_id, ignore_errors=True)
    output_path = OUTPUT_DIR / f"{job_id}.zip"
    if output_path.exists():
        output_path.unlink()
    return {"deleted": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
