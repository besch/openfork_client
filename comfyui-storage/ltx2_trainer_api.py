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

SHORT_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]
VIDEO_TARGET_MODULES = [
    "attn1.to_k",
    "attn1.to_q",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_k",
    "attn2.to_q",
    "attn2.to_v",
    "attn2.to_out.0",
]
VIDEO_IC_TARGET_MODULES = [*VIDEO_TARGET_MODULES, "ff.net.0.proj", "ff.net.2"]
AUDIO_TARGET_MODULES = [
    "audio_attn1.to_k",
    "audio_attn1.to_q",
    "audio_attn1.to_v",
    "audio_attn1.to_out.0",
    "audio_attn2.to_k",
    "audio_attn2.to_q",
    "audio_attn2.to_v",
    "audio_attn2.to_out.0",
    "audio_ff.net.0.proj",
    "audio_ff.net.2",
]

SUPPORTED_TRAINING_MODES = {
    "t2v": "text-to-video",
    "t2a": "text-to-audio",
    "i2v": "image-to-video",
    "video_extend": "video extension",
    "audio_extend": "audio extension",
    "video_inpainting": "video inpainting",
    "audio_inpainting": "audio inpainting",
    "video_outpainting": "video outpainting",
    "v2v_ic_lora": "video IC-LoRA",
    "a2a_ic_lora": "audio IC-LoRA",
    "av2av_ic_lora": "joint audio-video IC-LoRA",
    "a2v": "audio-to-video",
    "v2a": "video-to-audio",
}

MODE_ALIASES = {
    "text-to-video": "t2v",
    "text_to_video": "t2v",
    "txt2vid": "t2v",
    "text-to-audio": "t2a",
    "text_to_audio": "t2a",
    "txt2audio": "t2a",
    "image-to-video": "i2v",
    "image_to_video": "i2v",
    "img2vid": "i2v",
    "video-extension": "video_extend",
    "video_extension": "video_extend",
    "video-extend": "video_extend",
    "video_prefix": "video_extend",
    "video_suffix": "video_extend",
    "audio-extension": "audio_extend",
    "audio_extension": "audio_extend",
    "audio-extend": "audio_extend",
    "audio_prefix": "audio_extend",
    "audio_suffix": "audio_extend",
    "video-inpainting": "video_inpainting",
    "video_inpaint": "video_inpainting",
    "audio-inpainting": "audio_inpainting",
    "audio_inpaint": "audio_inpainting",
    "video-outpainting": "video_outpainting",
    "video_outpaint": "video_outpainting",
    "video-ic-lora": "v2v_ic_lora",
    "video_ic_lora": "v2v_ic_lora",
    "v2v": "v2v_ic_lora",
    "audio-ic-lora": "a2a_ic_lora",
    "audio_ic_lora": "a2a_ic_lora",
    "a2a": "a2a_ic_lora",
    "av-ic-lora": "av2av_ic_lora",
    "av_ic_lora": "av2av_ic_lora",
    "audio-video-ic-lora": "av2av_ic_lora",
    "joint_ic_lora": "av2av_ic_lora",
    "audio-to-video": "a2v",
    "audio_to_video": "a2v",
    "video-to-audio": "v2a",
    "video_to_audio": "v2a",
    "foley": "v2a",
}

VIDEO_DATA_MODES = {
    "t2v",
    "i2v",
    "video_extend",
    "video_inpainting",
    "video_outpainting",
    "v2v_ic_lora",
    "av2av_ic_lora",
    "a2v",
    "v2a",
}
AUDIO_DATA_MODES = {
    "t2v",
    "i2v",
    "video_extend",
    "t2a",
    "audio_extend",
    "audio_inpainting",
    "a2a_ic_lora",
    "av2av_ic_lora",
    "a2v",
    "v2a",
}
AUDIO_ONLY_MODES = {"t2a", "audio_extend", "audio_inpainting", "a2a_ic_lora"}
OPTIONAL_AUDIO_MODES = {"t2v", "i2v", "video_extend"}
AUDIO_REQUIRED_MODES = {
    "t2a",
    "audio_extend",
    "audio_inpainting",
    "a2a_ic_lora",
    "av2av_ic_lora",
    "a2v",
    "v2a",
}
VIDEO_ONLY_MODES = {"video_inpainting", "video_outpainting", "v2v_ic_lora"}

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


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_training_mode(value: str | None) -> str:
    raw = (value or "i2v").strip().lower().replace(" ", "_")
    raw = MODE_ALIASES.get(raw, raw)
    if raw not in SUPPORTED_TRAINING_MODES:
        raise ValueError(
            "Unsupported LTX-2 training mode. Expected one of: "
            + ", ".join(sorted(SUPPORTED_TRAINING_MODES))
        )
    return raw


def _normalize_condition_type(value: str | None, default: str = "prefix") -> str:
    condition = (value or default).strip().lower().replace("_", "-")
    if condition in {"suffix", "backward", "reverse"}:
        return "suffix"
    return "prefix"


def _parse_spatial_region(value: str | None) -> list[int]:
    if not value:
        return [0, 0, 288, 576]
    parsed: Any
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [part for part in value.replace("x", ",").replace(" ", ",").split(",") if part]
    if not isinstance(parsed, list) or len(parsed) != 4:
        raise ValueError("spatial_region must be a JSON list or comma-separated y1,x1,y2,x2 value")
    try:
        region = [int(part) for part in parsed]
    except (TypeError, ValueError) as exc:
        raise ValueError("spatial_region values must be integers") from exc
    if region[2] <= region[0] or region[3] <= region[1]:
        raise ValueError("spatial_region must be ordered as [y1, x1, y2, x2]")
    return region


def _mode_skip_audio(training_mode: str, skip_audio: bool) -> bool:
    if training_mode in VIDEO_ONLY_MODES:
        return True
    if training_mode in AUDIO_REQUIRED_MODES:
        return False
    if training_mode in OPTIONAL_AUDIO_MODES:
        return bool(skip_audio)
    return bool(skip_audio)


def _mode_uses_video(training_mode: str) -> bool:
    return training_mode in VIDEO_DATA_MODES


def _mode_uses_audio(training_mode: str, skip_audio: bool) -> bool:
    if training_mode not in AUDIO_DATA_MODES:
        return False
    return not _mode_skip_audio(training_mode, skip_audio)


def _target_modules_for_mode(training_mode: str, skip_audio: bool) -> list[str]:
    if training_mode in AUDIO_ONLY_MODES:
        return AUDIO_TARGET_MODULES
    if training_mode == "v2v_ic_lora":
        return VIDEO_IC_TARGET_MODULES
    if training_mode in VIDEO_ONLY_MODES:
        return VIDEO_TARGET_MODULES
    if training_mode in OPTIONAL_AUDIO_MODES and _mode_skip_audio(training_mode, skip_audio):
        return VIDEO_TARGET_MODULES
    return SHORT_TARGET_MODULES


def _build_mode_strategy(
    training_mode: str,
    *,
    skip_audio: bool,
    first_frame_probability: float,
    condition_type: str | None,
    condition_probability: float,
    temporal_boundary: int,
    spatial_region: str | None,
    reference_probability: float,
    mask_probability: float,
) -> dict[str, Any]:
    actual_skip_audio = _mode_skip_audio(training_mode, skip_audio)
    condition_probability = _clamp_probability(condition_probability)
    first_frame_probability = _clamp_probability(first_frame_probability)
    reference_probability = _clamp_probability(reference_probability)
    mask_probability = _clamp_probability(mask_probability)
    temporal_boundary = max(1, int(temporal_boundary))
    extension_condition = _normalize_condition_type(condition_type)

    strategy: dict[str, Any] = {"name": "flexible"}

    if training_mode == "t2v":
        strategy["video"] = {"is_generated": True, "latents_dir": "latents"}
        if not actual_skip_audio:
            strategy["audio"] = {"is_generated": True, "latents_dir": "audio_latents"}
    elif training_mode == "i2v":
        strategy["video"] = {
            "is_generated": True,
            "latents_dir": "latents",
            "conditions": [
                {
                    "type": "first_frame",
                    "probability": first_frame_probability,
                }
            ],
        }
        if not actual_skip_audio:
            strategy["audio"] = {"is_generated": True, "latents_dir": "audio_latents"}
    elif training_mode == "video_extend":
        strategy["video"] = {
            "is_generated": True,
            "latents_dir": "latents",
            "conditions": [
                {
                    "type": extension_condition,
                    "temporal_boundary": temporal_boundary,
                    "probability": condition_probability,
                }
            ],
        }
        if not actual_skip_audio:
            strategy["audio"] = {"is_generated": True, "latents_dir": "audio_latents"}
    elif training_mode == "video_inpainting":
        strategy["video"] = {
            "is_generated": True,
            "latents_dir": "latents",
            "conditions": [
                {
                    "type": "mask",
                    "mask_dir": "video_masks",
                    "probability": mask_probability,
                }
            ],
        }
    elif training_mode == "video_outpainting":
        strategy["video"] = {
            "is_generated": True,
            "latents_dir": "latents",
            "conditions": [
                {
                    "type": "spatial_crop",
                    "spatial_region": _parse_spatial_region(spatial_region),
                    "probability": condition_probability,
                }
            ],
        }
    elif training_mode == "v2v_ic_lora":
        conditions: list[dict[str, Any]] = [
            {
                "type": "reference",
                "latents_dir": "reference_latents",
                "probability": reference_probability,
            }
        ]
        if first_frame_probability > 0:
            conditions.append({"type": "first_frame", "probability": first_frame_probability})
        strategy["video"] = {
            "is_generated": True,
            "latents_dir": "latents",
            "conditions": conditions,
        }
    elif training_mode == "a2v":
        strategy["video"] = {"is_generated": True, "latents_dir": "latents"}
        strategy["audio"] = {"is_generated": False, "latents_dir": "audio_latents"}
    elif training_mode == "v2a":
        strategy["video"] = {"is_generated": False, "latents_dir": "latents"}
        strategy["audio"] = {"is_generated": True, "latents_dir": "audio_latents"}
    elif training_mode == "t2a":
        strategy["audio"] = {"is_generated": True, "latents_dir": "audio_latents"}
    elif training_mode == "audio_extend":
        strategy["audio"] = {
            "is_generated": True,
            "latents_dir": "audio_latents",
            "conditions": [
                {
                    "type": extension_condition,
                    "temporal_boundary": temporal_boundary,
                    "probability": condition_probability,
                }
            ],
        }
    elif training_mode == "audio_inpainting":
        strategy["audio"] = {
            "is_generated": True,
            "latents_dir": "audio_latents",
            "conditions": [
                {
                    "type": "mask",
                    "mask_dir": "audio_masks",
                    "probability": mask_probability,
                }
            ],
        }
    elif training_mode == "a2a_ic_lora":
        strategy["audio"] = {
            "is_generated": True,
            "latents_dir": "audio_latents",
            "conditions": [
                {
                    "type": "reference",
                    "latents_dir": "reference_audio_latents",
                    "probability": reference_probability,
                }
            ],
        }
    elif training_mode == "av2av_ic_lora":
        strategy["video"] = {
            "is_generated": True,
            "latents_dir": "latents",
            "conditions": [
                {
                    "type": "reference",
                    "latents_dir": "reference_latents",
                    "probability": reference_probability,
                }
            ],
        }
        strategy["audio"] = {
            "is_generated": True,
            "latents_dir": "audio_latents",
            "conditions": [
                {
                    "type": "reference",
                    "latents_dir": "reference_audio_latents",
                    "probability": reference_probability,
                }
            ],
        }
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")

    return strategy


def _generated_modalities(strategy: dict[str, Any]) -> tuple[bool, bool]:
    video_generated = bool(strategy.get("video", {}).get("is_generated"))
    audio_generated = bool(strategy.get("audio", {}).get("is_generated"))
    return video_generated, audio_generated


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
    training_mode: str,
    output_dir: Path,
    precomputed_dir: Path,
    resolution_buckets: str,
    steps: int,
    rank: int,
    alpha: int,
    learning_rate: float,
    first_frame_probability: float,
    condition_type: str | None,
    condition_probability: float,
    temporal_boundary: int,
    spatial_region: str | None,
    reference_probability: float,
    mask_probability: float,
    gradient_accumulation_steps: int,
    quantization: str | None,
    validation_prompt: str | None,
    seed: int,
    skip_audio: bool,
) -> dict[str, Any]:
    training_mode = _normalize_training_mode(training_mode)
    actual_skip_audio = _mode_skip_audio(training_mode, skip_audio)
    strategy = _build_mode_strategy(
        training_mode,
        skip_audio=actual_skip_audio,
        first_frame_probability=first_frame_probability,
        condition_type=condition_type,
        condition_probability=condition_probability,
        temporal_boundary=temporal_boundary,
        spatial_region=spatial_region,
        reference_probability=reference_probability,
        mask_probability=mask_probability,
    )
    video_generated, audio_generated = _generated_modalities(strategy)

    validation_samples = []
    if validation_prompt:
        validation_samples.append({"prompt": validation_prompt})

    validation: dict[str, Any] = {
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
        "stg_mode": "stg_v" if video_generated and not audio_generated else "stg_av",
        "generate_audio": audio_generated,
        "skip_initial_validation": True,
    }

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
            "target_modules": _target_modules_for_mode(training_mode, actual_skip_audio),
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
        "validation": validation,
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
            "tags": ["openfork", "ltx2", f"{training_mode}-lora", "low-vram"],
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
    training_mode: str,
    lora_trigger: str,
    resolution_buckets: str,
    audio_durations: str | None,
    steps: int,
    rank: int,
    alpha: int,
    learning_rate: float,
    first_frame_probability: float,
    condition_type: str | None,
    condition_probability: float,
    temporal_boundary: int,
    spatial_region: str | None,
    reference_probability: float,
    mask_probability: float,
    reference_downscale_factor: int,
    reference_temporal_scale_factor: int,
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
        training_mode = _normalize_training_mode(training_mode)
        actual_skip_audio = _mode_skip_audio(training_mode, skip_audio)
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
        if _mode_uses_video(training_mode):
            preprocess_cmd.extend(["--resolution-buckets", resolution_buckets])
        if audio_durations:
            preprocess_cmd.extend(["--audio-durations", audio_durations])
        if reference_downscale_factor > 1:
            preprocess_cmd.extend(["--reference-downscale-factor", str(reference_downscale_factor)])
        if reference_temporal_scale_factor > 1:
            preprocess_cmd.extend(
                ["--reference-temporal-scale-factor", str(reference_temporal_scale_factor)]
            )
        if lora_trigger:
            preprocess_cmd.extend(["--lora-trigger", lora_trigger])
        if actual_skip_audio:
            preprocess_cmd.append("--skip-audio")
        _run_command(job_id, preprocess_cmd, TRAINER_DIR, log_path, "preprocess")

        config = _build_training_config(
            training_mode=training_mode,
            output_dir=run_dir,
            precomputed_dir=precomputed_dir,
            resolution_buckets=resolution_buckets,
            steps=steps,
            rank=rank,
            alpha=alpha,
            learning_rate=learning_rate,
            first_frame_probability=first_frame_probability,
            condition_type=condition_type,
            condition_probability=condition_probability,
            temporal_boundary=temporal_boundary,
            spatial_region=spatial_region,
            reference_probability=reference_probability,
            mask_probability=mask_probability,
            gradient_accumulation_steps=gradient_accumulation_steps,
            quantization=quantization,
            validation_prompt=validation_prompt,
            seed=seed,
            skip_audio=actual_skip_audio,
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
                "training_mode": training_mode,
                "training_mode_name": SUPPORTED_TRAINING_MODES[training_mode],
                "resolution_buckets": resolution_buckets,
                "audio_durations": audio_durations,
                "steps": steps,
                "rank": rank,
                "alpha": alpha,
                "learning_rate": learning_rate,
                "first_frame_probability": first_frame_probability,
                "condition_type": condition_type,
                "condition_probability": condition_probability,
                "temporal_boundary": temporal_boundary,
                "spatial_region": spatial_region,
                "reference_probability": reference_probability,
                "mask_probability": mask_probability,
                "reference_downscale_factor": reference_downscale_factor,
                "reference_temporal_scale_factor": reference_temporal_scale_factor,
                "quantization": quantization,
                "skip_audio": actual_skip_audio,
                "uses_video": _mode_uses_video(training_mode),
                "uses_audio": _mode_uses_audio(training_mode, actual_skip_audio),
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
        "supported_training_modes": SUPPORTED_TRAINING_MODES,
        "experimental": TARGET_VRAM_GB < 32,
        "notes": (
            "Official LTX-2 Trainer: 80GB+ standard config, 32GB low-VRAM INT8 config. "
            "Profiles below 32GB are experimental and use more aggressive quantization/resolution defaults."
        ),
    }


async def _queue_lora_training(
    background_tasks: BackgroundTasks,
    dataset_archive: UploadFile,
    *,
    training_mode: str,
    lora_trigger: str,
    resolution_buckets: str,
    audio_durations: str,
    steps: int,
    rank: int,
    alpha: int,
    learning_rate: float,
    first_frame_probability: float,
    condition_type: str,
    condition_probability: float,
    temporal_boundary: int,
    spatial_region: str,
    reference_probability: float,
    mask_probability: float,
    reference_downscale_factor: int,
    reference_temporal_scale_factor: int,
    gradient_accumulation_steps: int,
    quantization: str,
    skip_audio: bool,
    validation_prompt: str,
    seed: int,
) -> dict[str, str]:
    if not MODEL_PATH.exists() or not TEXT_ENCODER_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="LTX-2 checkpoint or Gemma text encoder is missing.",
        )
    if not TRAINER_DIR.exists():
        raise HTTPException(status_code=503, detail="LTX-2 trainer directory is missing.")

    try:
        normalized_mode = _normalize_training_mode(training_mode)
        normalized_quantization = _normalize_trainer_quantization(quantization)
        _parse_spatial_region(spatial_region) if normalized_mode == "video_outpainting" else None
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
        training_mode=normalized_mode,
        lora_trigger=lora_trigger.strip(),
        resolution_buckets=resolution_buckets.strip() or DEFAULT_BUCKETS,
        audio_durations=audio_durations.strip() or None,
        steps=max(1, int(steps)),
        rank=max(1, int(rank)),
        alpha=max(1, int(alpha)),
        learning_rate=float(learning_rate),
        first_frame_probability=float(first_frame_probability),
        condition_type=condition_type.strip() or None,
        condition_probability=float(condition_probability),
        temporal_boundary=max(1, int(temporal_boundary)),
        spatial_region=spatial_region.strip() or None,
        reference_probability=float(reference_probability),
        mask_probability=float(mask_probability),
        reference_downscale_factor=max(1, int(reference_downscale_factor)),
        reference_temporal_scale_factor=max(1, int(reference_temporal_scale_factor)),
        gradient_accumulation_steps=max(1, int(gradient_accumulation_steps)),
        quantization=normalized_quantization,
        skip_audio=bool(skip_audio),
        validation_prompt=validation_prompt.strip() or None,
        seed=int(seed),
    )
    return {"job_id": job_id, "status": "queued"}


@app.post("/train/lora")
async def train_lora(
    background_tasks: BackgroundTasks,
    dataset_archive: UploadFile = File(...),
    training_mode: str = Form("i2v"),
    lora_trigger: str = Form(""),
    resolution_buckets: str = Form(DEFAULT_BUCKETS),
    audio_durations: str = Form(""),
    steps: int = Form(DEFAULT_STEPS),
    rank: int = Form(DEFAULT_RANK),
    alpha: int = Form(DEFAULT_ALPHA),
    learning_rate: float = Form(1e-4),
    first_frame_probability: float = Form(0.5),
    condition_type: str = Form("prefix"),
    condition_probability: float = Form(1.0),
    temporal_boundary: int = Form(8),
    spatial_region: str = Form(""),
    reference_probability: float = Form(1.0),
    mask_probability: float = Form(1.0),
    reference_downscale_factor: int = Form(1),
    reference_temporal_scale_factor: int = Form(1),
    gradient_accumulation_steps: int = Form(1),
    quantization: str = Form(DEFAULT_QUANTIZATION),
    skip_audio: bool = Form(False),
    validation_prompt: str = Form(""),
    seed: int = Form(42),
) -> dict[str, str]:
    return await _queue_lora_training(
        background_tasks,
        dataset_archive,
        training_mode=training_mode,
        lora_trigger=lora_trigger,
        resolution_buckets=resolution_buckets,
        audio_durations=audio_durations,
        steps=steps,
        rank=rank,
        alpha=alpha,
        learning_rate=learning_rate,
        first_frame_probability=first_frame_probability,
        condition_type=condition_type,
        condition_probability=condition_probability,
        temporal_boundary=temporal_boundary,
        spatial_region=spatial_region,
        reference_probability=reference_probability,
        mask_probability=mask_probability,
        reference_downscale_factor=reference_downscale_factor,
        reference_temporal_scale_factor=reference_temporal_scale_factor,
        gradient_accumulation_steps=gradient_accumulation_steps,
        quantization=quantization,
        skip_audio=skip_audio,
        validation_prompt=validation_prompt,
        seed=seed,
    )


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
    return await _queue_lora_training(
        background_tasks,
        dataset_archive,
        training_mode="i2v",
        lora_trigger=lora_trigger,
        resolution_buckets=resolution_buckets,
        audio_durations="",
        steps=steps,
        rank=rank,
        alpha=alpha,
        learning_rate=learning_rate,
        first_frame_probability=first_frame_probability,
        condition_type="prefix",
        condition_probability=1.0,
        temporal_boundary=8,
        spatial_region="",
        reference_probability=1.0,
        mask_probability=1.0,
        reference_downscale_factor=1,
        reference_temporal_scale_factor=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        quantization=quantization,
        skip_audio=skip_audio,
        validation_prompt=validation_prompt,
        seed=seed,
    )


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
