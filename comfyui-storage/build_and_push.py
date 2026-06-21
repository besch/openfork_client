#!/usr/bin/env python3
"""
Docker Build and Push Script with Retry Logic

Builds and pushes Docker images one at a time with retry capability.
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


@dataclass
class ImageConfig:
    dockerfile: str
    tag: str
    build: bool = False
    push: bool = False
    build_args: dict = None
    direct_push: bool = False  # Stream layers directly to registry during build when push is requested.


# Default rebuild queue for image/content changes.
# Wan2GP server-only changes do not belong here: start_cloud.sh refreshes
# comfyui-storage/wan2gp_server.py from the public OpenFork client repo at
# runtime, with the image-baked copy as a fallback.
IMAGES: List[ImageConfig] = [
    ImageConfig("Dockerfile.qwen", "beschiak/openfork-qwen-12gb:latest", build=True, push=True, direct_push=True),
    # ImageConfig("Dockerfile.qwen-8gb", "beschiak/openfork-qwen-8gb:latest", build=True, push=True, direct_push=True),
    # ImageConfig("Dockerfile.qwen-turbo-8gb", "beschiak/openfork-qwen-image-turbo-8gb:latest", build=True, push=True, direct_push=True),
    # ImageConfig("Dockerfile.flux-kontext-dev-16gb", "beschiak/openfork-flux-kontext-dev-16gb:latest", build=True, push=True, direct_push=True),
    # ImageConfig("Dockerfile.ltx23-wan2gp-hdr", "beschiak/openfork-ltx23-wan2gp-hdr:latest", build=True, push=True, direct_push=True),
    # ImageConfig("Dockerfile.ltx23-wan2gp-12gb-hdr", "beschiak/openfork-ltx23-wan2gp-12gb-hdr:latest", build=True, push=True, direct_push=True),
    # ImageConfig("Dockerfile.dots-tts", "beschiak/openfork-dots-tts-6gb:latest", build=True, push=True, direct_push=True, build_args={"DOTS_TTS_MODEL": "rednote-hilab/dots.tts-mf"}),
    # ImageConfig("Dockerfile.scail-wan2gp-24gb", "beschiak/openfork-scail-wan2gp-24gb:latest", build=True, push=True, direct_push=True),
    # ImageConfig("Dockerfile.davinci-magihuman-wan2gp-24gb", "beschiak/openfork-davinci-magihuman-wan2gp-24gb:latest", build=True, push=True, direct_push=True),
    # ImageConfig("Dockerfile.qwen-8gb", "beschiak/openfork-qwen-image-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.qwen-turbo-8gb", "beschiak/openfork-qwen-image-turbo-8gb:latest", build=True, push=True),
    # Previous production audio rebuilds:
    # ImageConfig("Dockerfile.qwen3-tts", "beschiak/openfork-qwen3-tts-8gb-v1:latest", build=True, push=True, direct_push=True, build_args={"MODEL_SIZE": "0.6B"}),
    # ImageConfig("Dockerfile.qwen3-tts", "beschiak/openfork-qwen3-tts-16gb-v1:latest", build=True, push=True, direct_push=True, build_args={"MODEL_SIZE": "1.7B"}),
    # ImageConfig("Dockerfile.scenema-audio", "beschiak/openfork-scenema-audio-16gb:latest", build=True, push=True, direct_push=True),
    # ImageConfig("Dockerfile.heartmula-16gb", "beschiak/openfork-heartmula-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.heartmula-24gb", "beschiak/openfork-heartmula-24gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.qwen3-tts", "beschiak/openfork-qwen3-tts-8gb-v1:latest", build=True, build_args={"MODEL_SIZE": "0.6B"}),
    # ImageConfig("Dockerfile.qwen3-tts", "beschiak/openfork-qwen3-tts-16gb-v1:latest", build=True, push=True, build_args={"MODEL_SIZE": "1.7B"}),
    # ImageConfig("Dockerfile.hunyuan-video-16gb", "beschiak/openfork-hunyuan-video-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.acestep-8gb", "beschiak/openfork-acestep-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.acestep-16gb", "beschiak/openfork-acestep-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.ltx2-8gb", "beschiak/openfork-ltx2-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.ltx2-16gb", "beschiak/openfork-ltx2-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.ltx2-24gb", "beschiak/openfork-ltx2-24gb:latest", build=True, push=True),
    # Redundant tier-specific images (Consolidated into -24gb)
    # ImageConfig("Dockerfile.mmaudio-8gb", "beschiak/openfork-mmaudio-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.mmaudio-16gb", "beschiak/openfork-mmaudio-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.lavasr", "beschiak/openfork-lavasr:latest", build=True, push=True),
    # ImageConfig("Dockerfile.prismaudio-8gb", "beschiak/openfork-prismaudio-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.prismaudio-16gb", "beschiak/openfork-prismaudio-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.sparkvsr-24gb", "beschiak/openfork-sparkvsr-24gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.inspatio-world-16gb", "beschiak/openfork-inspatio-world-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.inspatio-world-24gb", "beschiak/openfork-inspatio-world-24gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.anima-16gb", "beschiak/openfork-anima-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.anima-8gb", "beschiak/openfork-anima-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.llm", "beschiak/openfork-llm-4gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.llm-gemma4-12b", "beschiak/openfork-llm-gemma4-12b-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.f5-tts", "beschiak/openfork-f5-tts-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.wavtts", "beschiak/openfork-wavtts-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.ltx23-wan2gp-8gb", "beschiak/openfork-ltx23-wan2gp-8gb:latest", build=True, push=True),
    # LTX-2.3 ComfyUI tiers (colour-fixed via distilled LoRA at strength 0.5)
    # ImageConfig("Dockerfile.ltx23-comfyui-12gb", "beschiak/openfork-ltx23-comfyui-12gb:latest", build=True, push=True),
    # ImageConfig( "Dockerfile.ernie-image-8gb", "beschiak/openfork-ernie-image-8gb:latest", build=True, direct_push=True, ),
    # ImageConfig( "Dockerfile.ernie-image-24gb", "beschiak/openfork-ernie-image-24gb:latest", build=True, push=True, ),
    # ImageConfig( "Dockerfile.ernie-image-16gb", "beschiak/openfork-ernie-image-16gb:latest", build=True, push=True, ),
    # ImageConfig( "Dockerfile.ideogram4", "beschiak/openfork-ideogram4-16gb:latest", build=True, push=True, build_args={"IDEOGRAM_QUANTIZATION": "nf4"} ),
    # ImageConfig( "Dockerfile.ideogram4", "beschiak/openfork-ideogram4-24gb:latest", build=True, push=True, build_args={"IDEOGRAM_QUANTIZATION": "fp8"} ),
    # ImageConfig("Dockerfile.ltx23-wan2gp-hdr", "beschiak/openfork-ltx23-wan2gp-hdr:latest", build=True, push=True),
    # ImageConfig("Dockerfile.ltx23-wan2gp-12gb-hdr", "beschiak/openfork-ltx23-wan2gp-12gb-hdr:latest", build=True, push=True),
    # ImageConfig("Dockerfile.ltx23-wan2gp-8gb", "beschiak/openfork-ltx23-wan2gp-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.ltx23-wan2gp-original", "beschiak/openfork-ltx23-wan2gp-original:latest", build=True, push=True),
    # ImageConfig("Dockerfile.ltx23-comfyui-24gb", "beschiak/openfork-ltx23-comfyui-24gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.ltx23-comfyui-8gb", "beschiak/openfork-ltx23-comfyui-8gb:latest", build=True),
    # ImageConfig("Dockerfile.ltx23-comfyui-16gb", "beschiak/openfork-ltx23-comfyui-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.ltx23-comfyui-8gb", "beschiak/openfork-ltx23-comfyui-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.stream-diffvsr-8gb", "beschiak/openfork-stream-diffvsr-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.zimage-turbo-8gb", "beschiak/openfork-zimage-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.zimage-full-8gb", "beschiak/openfork-zimage-full-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.davinci-magihuman-wan2gp-16gb", "beschiak/openfork-davinci-magihuman-wan2gp-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.davinci-magihuman-wan2gp-24gb", "beschiak/openfork-davinci-magihuman-wan2gp-24gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.davinci-magihuman-wan2gp-32gb", "beschiak/openfork-davinci-magihuman-wan2gp-32gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.ltx23-wan2gp-12gb", "beschiak/openfork-ltx23-wan2gp-12gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.vista4d-wan2gp-24gb", "beschiak/openfork-vista4d-wan2gp-24gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.audiox", "beschiak/openfork-audiox:latest", build=True, push=True),
    # ImageConfig("Dockerfile.dreamid-omni-24gb", "beschiak/openfork-dreamid-omni-24gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.zimage-full-16gb", "beschiak/openfork-zimage-full-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.zimage-full-24gb", "beschiak/openfork-zimage-full-24gb:latest", build=True, push=True),
    # ImageConfig( "Dockerfile.qwen", "beschiak/openfork-qwen-12gb:latest", build=True, push=True, direct_push=True, ),
    # ImageConfig( "Dockerfile.qwen-8gb", "beschiak/openfork-qwen-8gb:latest", build=True, push=True, direct_push=True, ),
    # ImageConfig( "Dockerfile.qwen-turbo-8gb", "beschiak/openfork-qwen-turbo-8gb:latest", build=True, push=True, direct_push=True, ),
    # ImageConfig( "Dockerfile.turbodiffusion", "beschiak/openfork-turbodiffusion:latest", build=True, push=True, direct_push=True, ),
    # ImageConfig("Dockerfile.pid-zimage-upscaler-16gb", "beschiak/openfork-pid-zimage-upscaler-16gb:latest", build=True, push=True, direct_push=True),
    # ImageConfig("Dockerfile.flux-kontext-dev-8gb", "beschiak/openfork-flux-kontext-dev-8gb:latest", build=True, push=False),
    # ImageConfig("Dockerfile.flux-kontext-dev-12gb", "beschiak/openfork-flux-kontext-dev-12gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.flux-kontext-dev-16gb", "beschiak/openfork-flux-kontext-dev-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.flux-kontext-dev-24gb", "beschiak/openfork-flux-kontext-dev-24gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.scenema-audio", "beschiak/openfork-scenema-audio-16gb:latest", build=True, push=True, direct_push=True),
    # ImageConfig("Dockerfile.dramabox", "beschiak/openfork-dramabox-24gb:latest", build=True, push=True, direct_push=True),
    # ImageConfig("Dockerfile.stable.audio", "beschiak/openfork-stable-audio-3-sfx:latest", build=True, push=True),
    # ImageConfig("Dockerfile.scail-wan2gp-16gb", "beschiak/openfork-scail-wan2gp-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.scail-wan2gp-24gb", "beschiak/openfork-scail-wan2gp-24gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.wan22-wan2gp-8gb", "beschiak/openfork-wan22-wan2gp-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.wan22-wan2gp-16gb", "beschiak/openfork-wan22-wan2gp-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.llm-gemma4-12b", "beschiak/openfork-llm-gemma4-12b-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.wavtts", "beschiak/openfork-wavtts-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.ideogram4", "beschiak/openfork-ideogram4-16gb:latest", build=True, push=True, build_args={"IDEOGRAM_QUANTIZATION": "nf4"} ),
    # ImageConfig("Dockerfile.ideogram4", "beschiak/openfork-ideogram4-24gb:latest", build=True, push=True, build_args={"IDEOGRAM_QUANTIZATION": "fp8"} ),
    # ImageConfig("Dockerfile.f5-tts", "beschiak/openfork-f5-tts-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.wan22-wan2gp-10gb", "beschiak/openfork-wan22-wan2gp-10gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.wan22-wan2gp-12gb", "beschiak/openfork-wan22-wan2gp-12gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.wan22-wan2gp-24gb", "beschiak/openfork-wan22-wan2gp-24gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.scail2-wan2gp-16gb", "beschiak/openfork-scail2-wan2gp-16gb:latest", build=True, push=True, direct_push=True),
    # ImageConfig("Dockerfile.scail2-wan2gp-24gb", "beschiak/openfork-scail2-wan2gp-24gb:latest", build=True, push=True, direct_push=True),
]

OPTIONAL_IMAGES: List[ImageConfig] = [
    ImageConfig("Dockerfile.qwen", "beschiak/openfork-qwen-12gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.qwen-2512-lora-24gb", "beschiak/openfork-qwen-2512-lora-24gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.qwen-8gb", "beschiak/openfork-qwen-8gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.qwen-turbo-8gb", "beschiak/openfork-qwen-image-turbo-8gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.flux-kontext-dev-12gb", "beschiak/openfork-flux-kontext-dev-12gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.flux-kontext-dev-16gb", "beschiak/openfork-flux-kontext-dev-16gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.flux-kontext-dev-24gb", "beschiak/openfork-flux-kontext-dev-24gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.wan22", "beschiak/openfork-wan22-8gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.wan22-16gb", "beschiak/openfork-wan22-16gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.wan22-24gb", "beschiak/openfork-wan22-24gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.ltx23-wan2gp-12gb", "beschiak/openfork-ltx23-wan2gp-12gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.ltx23-wan2gp-12gb-hdr", "beschiak/openfork-ltx23-wan2gp-12gb-hdr:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.ltx23-wan2gp-hdr", "beschiak/openfork-ltx23-wan2gp-hdr:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.ltx23-wan2gp", "beschiak/openfork-ltx23-wan2gp:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.scail2-wan2gp-16gb", "beschiak/openfork-scail2-wan2gp-16gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.scail2-wan2gp-24gb", "beschiak/openfork-scail2-wan2gp-24gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.scail-wan2gp-16gb", "beschiak/openfork-scail-wan2gp-16gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.scail-wan2gp-24gb", "beschiak/openfork-scail-wan2gp-24gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.wan22-wan2gp-8gb", "beschiak/openfork-wan22-wan2gp-8gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.wan22-wan2gp-10gb", "beschiak/openfork-wan22-wan2gp-10gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.wan22-wan2gp-12gb", "beschiak/openfork-wan22-wan2gp-12gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.wan22-wan2gp-16gb", "beschiak/openfork-wan22-wan2gp-16gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.wan22-wan2gp-24gb", "beschiak/openfork-wan22-wan2gp-24gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.vista4d-wan2gp-24gb", "beschiak/openfork-vista4d-wan2gp-24gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.dreamid-omni-24gb", "beschiak/openfork-dreamid-omni-24gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.telestylev2-80gb", "beschiak/openfork-telestylev2-80gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.davinci-magihuman-wan2gp-16gb", "beschiak/openfork-davinci-magihuman-wan2gp-16gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.davinci-magihuman-wan2gp-24gb", "beschiak/openfork-davinci-magihuman-wan2gp-24gb:latest", build=True, push=True, direct_push=True),
    ImageConfig("Dockerfile.davinci-magihuman-wan2gp-32gb", "beschiak/openfork-davinci-magihuman-wan2gp-32gb:latest", build=True, push=True, direct_push=True),
]

WAN2GP_IMAGE_REFRESH_PRESET: List[str] = [
    "openfork-ltx23-wan2gp-hdr",
    "openfork-ltx23-wan2gp-12gb-hdr",
    "beschiak/openfork-ltx23-wan2gp-12gb:latest",
    "beschiak/openfork-ltx23-wan2gp:latest",
    "scail-wan2gp-16gb",
    "scail-wan2gp-24gb",
    "scail2-wan2gp-16gb",
    "scail2-wan2gp-24gb",
    "davinci-magihuman-wan2gp-16gb",
    "davinci-magihuman-wan2gp-24gb",
    "davinci-magihuman-wan2gp-32gb",
    "openfork-wan22-wan2gp-8gb",
    "openfork-wan22-wan2gp-10gb",
    "openfork-wan22-wan2gp-12gb",
    "openfork-wan22-wan2gp-16gb",
    "openfork-wan22-wan2gp-24gb",
    "vista4d-wan2gp-24gb",
]

BUILD_PRESETS: Dict[str, List[str]] = {
    "current-rebuild": [
        "openfork-qwen-12gb",
        "openfork-qwen-8gb",
        "openfork-qwen-image-turbo-8gb",
        "flux-kontext-dev-16gb",
    ],
    # Explicit image rebuild for Wan2GP Docker/model/runtime-fallback changes.
    # Do not use this for wan2gp_server.py-only changes; start_cloud.sh pulls
    # that file directly from GitHub at runtime.
    "wan2gp-image-refresh": WAN2GP_IMAGE_REFRESH_PRESET,
    # Legacy alias kept so older run commands still work.
    "wan2gp-server-refresh": WAN2GP_IMAGE_REFRESH_PRESET,
    "scail2": [
        "openfork-scail2-wan2gp-16gb",
        "openfork-scail2-wan2gp-24gb",
    ],
    "character-consistency": [
        "openfork-qwen-12gb",
        "openfork-qwen-2512-lora-24gb",
        "openfork-qwen-8gb",
        "flux-kontext-dev-16gb",
        "flux-kontext-dev-24gb",
        "openfork-wan22-16gb",
        "openfork-wan22-24gb",
        "openfork-ltx23-wan2gp-hdr",
        "openfork-scail2-wan2gp-24gb",
        "scail-wan2gp-24gb",
        "dreamid-omni-24gb",
        "davinci-magihuman-wan2gp-24gb",
    ],
    "character-consistency-full": [
        "openfork-qwen-12gb",
        "openfork-qwen-2512-lora-24gb",
        "openfork-qwen-8gb",
        "openfork-qwen-image-turbo-8gb",
        "flux-kontext-dev-12gb",
        "flux-kontext-dev-16gb",
        "flux-kontext-dev-24gb",
        "openfork-wan22-8gb",
        "openfork-wan22-16gb",
        "openfork-wan22-24gb",
        "ltx23-wan2gp-12gb",
        "openfork-ltx23-wan2gp-hdr",
        "openfork-scail2-wan2gp-16gb",
        "openfork-scail2-wan2gp-24gb",
        "scail-wan2gp-16gb",
        "scail-wan2gp-24gb",
        "dreamid-omni-24gb",
        "davinci-magihuman-wan2gp-16gb",
        "davinci-magihuman-wan2gp-24gb",
        "davinci-magihuman-wan2gp-32gb",
    ],
    "telestylev2": [
        "openfork-telestylev2-80gb",
    ],
}

PUSH_ATTEMPTS = 4
RETRY_DELAY_SECONDS = 1200  # 20 minutes


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _format_command_for_log(command: List[str]) -> str:
    redacted = []
    redact_next = False
    for part in command:
        if redact_next:
            if part.startswith("HF_TOKEN="):
                redacted.append("HF_TOKEN=<redacted>")
            else:
                redacted.append(part)
            redact_next = False
            continue

        redacted.append(part)
        if part == "--build-arg":
            redact_next = True

    return " ".join(redacted)


def _dockerfile_declares_hf_arg(dockerfile: str) -> bool:
    try:
        with open(dockerfile, "r", encoding="utf-8") as handle:
            return any(line.strip().startswith("ARG HF_TOKEN") for line in handle)
    except OSError:
        return True


def _dockerfile_declares_arg(dockerfile: str, arg_name: str) -> bool:
    try:
        with open(dockerfile, "r", encoding="utf-8") as handle:
            return any(
                line.strip().startswith(f"ARG {arg_name}")
                for line in handle
            )
    except OSError:
        return False


def _dockerfile_declares_hf_secret(dockerfile: str) -> bool:
    try:
        with open(dockerfile, "r", encoding="utf-8") as handle:
            return any(
                "--mount=type=secret" in line and "id=hf_token" in line
                for line in handle
            )
    except OSError:
        return False


def _dockerfile_requires_hf_secret(dockerfile: str) -> bool:
    try:
        with open(dockerfile, "r", encoding="utf-8") as handle:
            return any(
                "--mount=type=secret" in line
                and "id=hf_token" in line
                and "required=false" not in line
                for line in handle
            )
    except OSError:
        return False


def run_command(command: List[str], description: str, extra_env: dict = None) -> bool:
    """
    Run a command and return True if successf ul, False otherwise.
    """
    print(f"\n{'=' * 60}")
    print(f"🔹 {description}")
    print(f"   Command: {_format_command_for_log(command)}")
    print("=" * 60)

    env = {**os.environ, **(extra_env or {})}

    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            env=env,
        )
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED (exit code: {e.returncode})")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def cleanup_after_image(
    tag: str,
    remove_image: bool = True,
    prune_build_cache: bool = False,
    trim_disk: bool = False,
) -> None:
    """
    Best-effort cleanup for large sequential image builds.

    This intentionally does not fail the whole build if cleanup cannot remove a
    layer or trim the filesystem. A failed push keeps the local image for retry.
    """
    print(f"\n{'=' * 60}")
    print(f"Cleanup after {tag}")
    print("=" * 60)

    if remove_image:
        run_command(
            ["docker", "image", "rm", "--force", tag],
            f"Removing local image {tag}",
        )
    else:
        print("Skipping local image removal; direct push did not export a local tag.")

    run_command(
        ["docker", "image", "prune", "--force"],
        "Pruning dangling Docker images",
    )

    if prune_build_cache:
        run_command(
            ["docker", "builder", "prune", "--force", "--all"],
            "Pruning Docker build cache",
        )

    if trim_disk:
        trim_script = (
            "sync && "
            "if command -v fstrim >/dev/null 2>&1; then "
            "if [ \"$(id -u)\" -eq 0 ]; then "
            "fstrim -av; "
            "elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then "
            "sudo fstrim -av; "
            "else "
            "echo 'sudo is not available without a password; skipping fstrim.'; "
            "fi; "
            "else "
            "echo 'fstrim is not installed; skipping filesystem trim.'; "
            "fi"
        )
        run_command(["bash", "-lc", trim_script], "Trimming free WSL disk blocks")


def build_image(
    dockerfile: str,
    tag: str,
    hf_token: str = None,
    rebuild: bool = False,
    build_args: dict = None,
    fresh_clone: bool = False,
    direct_push: bool = False,
) -> bool:
    """
    Build a Docker image. No retry logic for builds - failure is ignored.

    direct_push=True uses --output type=registry to stream layers directly to
    the registry during the build step, completely bypassing the local containerd
    image store export.  This avoids the BuildKit lease-expiry crash that occurs
    when exporting very large images (≥100 GB) on Docker Desktop for Windows.
    When direct_push is used the separate push step must be skipped (the image
    is already in the registry).
    """
    command = ["docker", "build"]

    if rebuild:
        command.append("--no-cache")

    if fresh_clone:
        import time as _time

        command.extend(["--build-arg", f"CACHEBUST={int(_time.time())}"])

    uses_hf_secret = _dockerfile_declares_hf_secret(dockerfile)
    if uses_hf_secret:
        if hf_token:
            command.extend(["--secret", "id=hf_token,env=HF_TOKEN"])
        elif _dockerfile_requires_hf_secret(dockerfile):
            print(
                f"❌ {dockerfile} requires HF_TOKEN as a BuildKit secret, "
                "but no token was provided."
            )
            return False
    elif _dockerfile_declares_hf_arg(dockerfile):
        command.extend(["--build-arg", "HF_TOKEN"])

    if (
        os.environ.get("OPENFORK_CLIENT_SCRIPT_REF")
        and _dockerfile_declares_arg(dockerfile, "OPENFORK_CLIENT_SCRIPT_REF")
    ):
        command.extend(["--build-arg", "OPENFORK_CLIENT_SCRIPT_REF"])

    if hf_token and direct_push and not uses_hf_secret:
        command.extend(["--secret", "id=hf_token,env=HF_TOKEN"])

    if build_args:
        for key, value in build_args.items():
            command.extend(["--build-arg", f"{key}={value}"])

    if direct_push or uses_hf_secret:
        command.extend(["--progress", os.environ.get("DOCKER_BUILD_PROGRESS", "plain")])

    command.extend(["-f", dockerfile])

    if direct_push:
        # Stream layers straight to the registry — no local export needed.
        # Requires `docker login` to have been run beforehand.
        command.extend(["--output", f"type=registry", "--tag", tag])
        print(f"\n📦 Building {tag} with direct registry push (--output type=registry)")
    else:
        command.extend(["-t", tag])
        print(f"\n📦 Building {tag} (single attempt)")

    command.append(".")
    # Legacy builder avoids BuildKit daemon OOM on large downloads, but secret
    # mounts and direct registry output require BuildKit.
    extra_env = (
        {"DOCKER_BUILDKIT": "1"}
        if (direct_push or uses_hf_secret)
        else {"DOCKER_BUILDKIT": "0"}
    )
    if hf_token:
        extra_env["HF_TOKEN"] = hf_token
    return run_command(command, f"Building {tag}", extra_env=extra_env)


def push_image(tag: str) -> bool:
    """
    Push a Docker image with one retry after unsuccessful attempt (2 total).
    """
    command = ["docker", "push", tag]

    for attempt in range(1, PUSH_ATTEMPTS + 1):
        print(f"\n🚀 Push attempt {attempt}/{PUSH_ATTEMPTS} for {tag}")

        if run_command(command, f"Pushing {tag}"):
            return True

        if attempt < PUSH_ATTEMPTS:
            print(f"⏳ Waiting {RETRY_DELAY_SECONDS} seconds before retry...")
            time.sleep(RETRY_DELAY_SECONDS)

    return False


def build_and_push_image(
    config: ImageConfig,
    hf_token: str = None,
    rebuild: bool = False,
    global_push: bool = False,
    global_build: bool = False,
    fresh_clone: bool = False,
    force_direct_push: bool = False,
    explicit_actions: bool = False,
    cleanup_after_success: bool = False,
    prune_build_cache_after_success: bool = False,
    trim_after_success: bool = False,
) -> str:
    """
    Build and push a single image based on its configuration and global flags.
    Returns: "success", "skipped", "build_failed", or "push_failed"
    """
    if explicit_actions:
        should_build = global_build
        should_push = global_push
    else:
        should_build = config.build
        should_push = config.push

    direct_push_requested = config.direct_push or force_direct_push
    use_direct_push = direct_push_requested and should_build and should_push

    print(f"\n{'#' * 60}")
    print(f"# Processing: {config.dockerfile} -> {config.tag}")
    print(
        f"# Config: build={config.build}, push={config.push}, direct_push={config.direct_push}"
    )
    print(
        f"# Effective: build={should_build}, push={should_push}, direct_push={use_direct_push}"
    )
    print("#" * 60)

    if use_direct_push:
        print(
            "ℹ️  direct_push mode: layers will be streamed to the registry during build."
        )
        print("   Ensure you are logged in (`docker login`) before proceeding.")
    elif direct_push_requested and should_build and not should_push:
        print("ℹ️  direct_push disabled because push was not requested.")
    elif direct_push_requested and should_push and not should_build:
        print("ℹ️  direct_push only applies when build and push are both requested.")

    # Build the image if configured (per-image or global)
    if should_build:
        if not build_image(
            config.dockerfile,
            config.tag,
            hf_token,
            rebuild,
            config.build_args,
            fresh_clone,
            direct_push=use_direct_push,
        ):
            print(
                f"\n⚠️ FAILED to build {config.tag}. Ignoring build failure as requested."
            )
            return "build_failed"
    else:
        print(f"⏭️ Skipping build for {config.tag} as configured")

    # Push the image if configured (per-image or global)
    # In direct_push mode the image is already in the registry after a build.
    if use_direct_push:
        print(
            f"\n🎉 {config.tag} was pushed to the registry during build (direct_push mode)."
        )
        if cleanup_after_success:
            cleanup_after_image(
                config.tag,
                remove_image=False,
                prune_build_cache=prune_build_cache_after_success,
                trim_disk=trim_after_success,
            )
        return "success"

    if should_push:
        if not push_image(config.tag):
            print(f"\n💥 FAILED to push {config.tag} after {PUSH_ATTEMPTS} attempts")
            return "push_failed"
        print(f"\n🎉 Successfully pushed {config.tag}")
    else:
        print(f"⏭️ Skipping push for {config.tag}")

    result = "success" if should_build or should_push else "skipped"
    if cleanup_after_success and result == "success":
        cleanup_after_image(
            config.tag,
            remove_image=should_build,
            prune_build_cache=prune_build_cache_after_success,
            trim_disk=trim_after_success,
        )

    return result


def select_images(
    images: List[ImageConfig],
    image_indexes: List[int] = None,
    image_filters: List[str] = None,
) -> List[Tuple[int, ImageConfig]]:
    selected = list(enumerate(images))

    if image_indexes:
        valid_indexes = set(range(len(images)))
        invalid = [index for index in image_indexes if index not in valid_indexes]
        if invalid:
            raise ValueError(
                f"Invalid image index {invalid[0]}; valid range is 0-{len(images) - 1}."
            )
        requested = set(image_indexes)
        selected = [(index, config) for index, config in selected if index in requested]

    if image_filters:
        selected = [
            (index, config)
            for index, config in selected
            if any(
                needle == config.dockerfile
                or needle == config.tag
                or needle in config.dockerfile
                or needle in config.tag
                for needle in image_filters
            )
        ]

    deduped: List[Tuple[int, ImageConfig]] = []
    seen = set()
    for index, config in selected:
        key = (config.dockerfile, config.tag)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((index, config))

    return deduped


def parse_delay(delay_str: str) -> int:
    """
    Parse a delay string into seconds.
    Supports formats:
    - "10" -> 10 seconds
    - "10s" -> 10 seconds
    - "10m" -> 600 seconds
    - "10h" -> 36000 seconds
    - "in 10 minutes" -> 600 seconds
    """
    import re

    # Normalize string
    delay_str = delay_str.lower().strip()

    # Remove "in " prefix if present
    if delay_str.startswith("in "):
        delay_str = delay_str[3:].strip()

    # Simple number check
    if delay_str.isdigit():
        return int(delay_str)

    # Parse units
    # Match number followed by optional space and unit
    match = re.match(
        r"^(\d+)\s*(s|sec|seconds?|m|min|minutes?|h|hrs?|hours?)$", delay_str
    )

    if not match:
        raise ValueError(f"Invalid delay format: {delay_str}")

    value = int(match.group(1))
    unit = match.group(2)

    if unit.startswith("s"):
        return value
    elif unit.startswith("m"):
        return value * 60
    elif unit.startswith("h"):
        return value * 3600

    return value


def main():
    """
    Main entry point - builds and pushes all configured images.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Docker Build and Push Script with Retry Logic"
    )
    parser.add_argument(
        "--delay",
        type=str,
        help="Msg to start delay e.g. 'in 10 minutes', '10m', '300s'",
    )
    parser.add_argument(
        "--hf-token", type=str, help="Hugging Face token for gated models"
    )
    parser.add_argument(
        "--build", action="store_true", help="Build images (overrides per-image config)"
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Force rebuild by using --no-cache"
    )
    parser.add_argument(
        "--fresh-clone",
        dest="fresh_clone",
        action="store_true",
        help="Bust cache only at git clone layers (ensures latest ComfyUI/extensions without full rebuild)",
    )
    parser.add_argument(
        "--push", action="store_true", help="Push images (overrides per-image config)"
    )
    parser.add_argument(
        "--direct-push",
        dest="direct_push",
        action="store_true",
        help=(
            "Stream layers directly to the registry during build using "
            "'--output type=registry'. Bypasses the local containerd image store "
            "export step, which fixes the BuildKit lease-expiry crash on Docker "
            "Desktop for Windows when building very large images (≥100 GB). "
            "Requires prior `docker login`. Overrides per-image direct_push setting."
        ),
    )
    parser.add_argument(
        "--cleanup-after-each",
        action="store_true",
        default=_env_flag("OPENFORK_CLEANUP_AFTER_EACH"),
        help=(
            "After each successful image, remove the local tag and prune dangling "
            "Docker images. Use this for large sequential push builds when you do "
            "not need to keep the image locally."
        ),
    )
    parser.add_argument(
        "--prune-build-cache-after-each",
        action="store_true",
        default=_env_flag("OPENFORK_PRUNE_BUILD_CACHE_AFTER_EACH"),
        help=(
            "With --cleanup-after-each, also run 'docker builder prune --all'. "
            "This frees more disk but makes later builds slower."
        ),
    )
    parser.add_argument(
        "--trim-after-each",
        action="store_true",
        default=_env_flag("OPENFORK_TRIM_AFTER_EACH"),
        help=(
            "With --cleanup-after-each, run sync/fstrim after Docker cleanup. "
            "On a sparse WSL VHDX this lets Windows reclaim free blocks."
        ),
    )
    parser.add_argument(
        "--image-index",
        type=int,
        action="append",
        help="Only process a zero-based image index from IMAGES. Can be repeated.",
    )
    parser.add_argument(
        "--image",
        action="append",
        help=(
            "Only process images whose Dockerfile or tag matches this value "
            "or contains it. Can be repeated."
        ),
    )
    parser.add_argument(
        "--preset",
        action="append",
        choices=sorted(BUILD_PRESETS.keys()),
        help=(
            "Include an optional build preset. Example: "
            "--preset character-consistency"
        ),
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print optional build presets and exit.",
    )
    parser.add_argument(
        "--list-image-indexes",
        action="store_true",
        help="Print selected image indexes and exit. Used by wrapper scripts.",
    )
    parser.add_argument(
        "actions",
        nargs="*",
        help="Optional action aliases. Example: build push",
    )

    args = parser.parse_args()
    invalid_actions = [
        action for action in args.actions if action not in {"build", "push"}
    ]
    if invalid_actions:
        parser.error(
            f"invalid action '{invalid_actions[0]}' (choose from 'build', 'push')"
        )

    if args.list_presets:
        for preset_name, filters in BUILD_PRESETS.items():
            print(f"{preset_name}:")
            for image_filter in filters:
                print(f"  - {image_filter}")
        sys.exit(0)

    preset_filters: List[str] = []
    for preset_name in args.preset or []:
        preset_filters.extend(BUILD_PRESETS[preset_name])

    image_filters = [*(args.image or []), *preset_filters]
    image_pool = IMAGES if not image_filters else [*IMAGES, *OPTIONAL_IMAGES]

    try:
        selected_images = select_images(image_pool, args.image_index, image_filters)
    except ValueError as exc:
        parser.error(str(exc))

    if args.list_image_indexes:
        print(" ".join(str(index) for index, _config in selected_images))
        sys.exit(0)

    if not selected_images:
        print("❌ No images matched the requested filters.")
        sys.exit(1)

    if "build" in args.actions:
        args.build = True
    if "push" in args.actions:
        args.push = True
    explicit_actions = args.build or args.push

    print("\n" + "=" * 60)
    print("🐳 Docker Build and Push Script")
    print("=" * 60)

    if args.delay:
        try:
            delay_seconds = parse_delay(args.delay)
            print(f"⏳ Delayed start requested: {args.delay}")
            print(f"   Waiting {delay_seconds} seconds before starting...")

            # Show countdown for longer delays
            if delay_seconds > 60:
                while delay_seconds > 0:
                    if delay_seconds % 60 == 0:
                        print(f"   {delay_seconds // 60} minutes remaining...")
                    time.sleep(1)
                    delay_seconds -= 1
            else:
                time.sleep(delay_seconds)

            print("\n⏰ Delay finished. Starting build process now.")
        except ValueError as e:
            print(f"❌ Error parsing delay: {e}")
            sys.exit(1)

    print(f"Images to process: {len(selected_images)}")
    print(f"Push attempts: {PUSH_ATTEMPTS} (1 initial + 1 retry)")
    print(f"Retry delay: {RETRY_DELAY_SECONDS} seconds")
    if args.cleanup_after_each:
        print("Cleanup after each success: enabled")
        print(
            "Build cache prune after each success: "
            f"{'enabled' if args.prune_build_cache_after_each else 'disabled'}"
        )
        print(
            "Filesystem trim after each success: "
            f"{'enabled' if args.trim_after_each else 'disabled'}"
        )

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    successful = []
    skipped = []
    build_failed = []
    push_failed = []

    for _index, config in selected_images:
        result = build_and_push_image(
            config,
            hf_token,
            args.rebuild,
            args.push,
            args.build,
            getattr(args, "fresh_clone", False),
            force_direct_push=getattr(args, "direct_push", False),
            explicit_actions=explicit_actions,
            cleanup_after_success=args.cleanup_after_each,
            prune_build_cache_after_success=args.prune_build_cache_after_each,
            trim_after_success=args.trim_after_each,
        )
        if result == "success":
            successful.append(config.tag)
        elif result == "skipped":
            skipped.append(config.tag)
        elif result == "build_failed":
            build_failed.append(config.tag)
        else:
            push_failed.append(config.tag)

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    if successful:
        print(f"\n✅ Successfully built and pushed ({len(successful)}):")
        for tag in successful:
            print(f"   - {tag}")

    if skipped:
        print(f"\n⏭️ Skipped ({len(skipped)}):")
        for tag in skipped:
            print(f"   - {tag}")

    if build_failed:
        print(f"\n⚠️ Build failed (ignored) ({len(build_failed)}):")
        for tag in build_failed:
            print(f"   - {tag}")

    if push_failed:
        print(f"\n❌ Push failed after retries ({len(push_failed)}):")
        for tag in push_failed:
            print(f"   - {tag}")

    print("\n" + "=" * 60)

    if push_failed:
        print("⚠️  Some images failed to push after retries!")
        sys.exit(1)
    elif build_failed:
        print("💡 Some images failed to build but were ignored.")
        sys.exit(0)
    else:
        print("🎉 All images processed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
