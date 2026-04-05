#!/usr/bin/env python3
"""
Docker Build and Push Script with Retry Logic

Builds and pushes Docker images one at a time with retry capability.
"""

import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List


@dataclass
class ImageConfig:
    dockerfile: str
    tag: str
    build: bool = False
    push: bool = False
    build_args: dict = None
    direct_push: bool = True  # Stream layers directly to registry during build (bypasses local image store)


# Define the images to build and push
IMAGES: List[ImageConfig] = [
    # ImageConfig("Dockerfile.qwen-8gb", "beschiak/openfork-qwen-image-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.qwen-turbo-8gb", "beschiak/openfork-qwen-image-turbo-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.heartmula-16gb", "beschiak/openfork-heartmula-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.heartmula-24gb", "beschiak/openfork-heartmula-24gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.qwen3-tts", "beschiak/openfork-qwen3-tts-8gb:latest", build=True, build_args={"MODEL_SIZE": "0.6B"}),
    # ImageConfig("Dockerfile.qwen3-tts", "beschiak/openfork-qwen3-tts-16gb:latest", build=True, push=True, build_args={"MODEL_SIZE": "1.7B"}),
    # ImageConfig("Dockerfile.hunyuan-video-16gb", "beschiak/openfork-hunyuan-video-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.acestep-8gb", "beschiak/openfork-acestep-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.acestep-16gb", "beschiak/openfork-acestep-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.ltx2-8gb", "beschiak/openfork-ltx2-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.ltx2-16gb", "beschiak/openfork-ltx2-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.ltx2-24gb", "beschiak/openfork-ltx2-24gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.ltx23-wan2gp", "beschiak/openfork-ltx23-wan2gp:latest", build=True, push=True),
    # ImageConfig("Dockerfile.sparkvsr-24gb", "beschiak/openfork-sparkvsr-24gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.davinci-magihuman-32gb", "beschiak/openfork-davinci-magihuman-32gb:latest", build=True, push=True),
    # Redundant tier-specific images (Consolidated into -24gb)
    ImageConfig("Dockerfile.ltx23-wan2gp", "beschiak/openfork-ltx23:latest", build=True, push=True),
    # ImageConfig("Dockerfile.mmaudio-8gb", "beschiak/openfork-mmaudio-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.mmaudio-16gb", "beschiak/openfork-mmaudio-16gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.lavasr", "beschiak/openfork-lavasr:latest", build=True, push=True),
    # ImageConfig("Dockerfile.prismaudio-8gb", "beschiak/openfork-prismaudio-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.prismaudio-16gb", "beschiak/openfork-prismaudio-16gb:latest", build=True, push=True),
]

PUSH_ATTEMPTS = 4
RETRY_DELAY_SECONDS = 1200  # 20 minutes


def run_command(command: List[str], description: str) -> bool:
    """
    Run a command and return True if successf ul, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"🔹 {description}")
    print(f"   Command: {' '.join(command)}")
    print('='*60)
    
    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
        )
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED (exit code: {e.returncode})")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def build_image(dockerfile: str, tag: str, hf_token: str = None, rebuild: bool = False, build_args: dict = None, fresh_clone: bool = False, direct_push: bool = False) -> bool:
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

    if hf_token:
        command.extend(["--build-arg", f"HF_TOKEN={hf_token}"])
    else:
        command.extend(["--build-arg", "HF_TOKEN"])
        
    if build_args:
        for key, value in build_args.items():
            command.extend(["--build-arg", f"{key}={value}"])

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
    return run_command(command, f"Building {tag}")


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


def build_and_push_image(config: ImageConfig, hf_token: str = None, rebuild: bool = False, global_push: bool = False, global_build: bool = False, fresh_clone: bool = False, force_direct_push: bool = False) -> str:
    """
    Build and push a single image based on its configuration and global flags.
    Returns: "success", "build_failed", or "push_failed"
    """
    use_direct_push = config.direct_push or force_direct_push

    print(f"\n{'#'*60}")
    print(f"# Processing: {config.dockerfile} -> {config.tag}")
    print(f"# Config: build={config.build}, push={config.push}, direct_push={use_direct_push}")
    print('#'*60)

    if use_direct_push:
        print("ℹ️  direct_push mode: layers will be streamed to the registry during build.")
        print("   Ensure you are logged in (`docker login`) before proceeding.")
    
    # Build the image if configured (per-image or global)
    should_build = config.build or global_build
    if should_build:
        if not build_image(config.dockerfile, config.tag, hf_token, rebuild, config.build_args, fresh_clone, direct_push=use_direct_push):
            print(f"\n⚠️ FAILED to build {config.tag}. Ignoring build failure as requested.")
            return "build_failed"
    else:
        print(f"⏭️ Skipping build for {config.tag} as configured")
    
    # In direct_push mode the image is already in the registry — skip the push step.
    if use_direct_push:
        print(f"\n🎉 {config.tag} was pushed to the registry during build (direct_push mode).")
        return "success"

    # Push the image if configured (per-image or global)
    should_push = config.push or global_push
    if should_push:
        if not push_image(config.tag):
            print(f"\n💥 FAILED to push {config.tag} after {PUSH_ATTEMPTS} attempts")
            return "push_failed"
        print(f"\n🎉 Successfully pushed {config.tag}")
    else:
        print(f"⏭️ Skipping push for {config.tag}")
        
    return "success"



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
    match = re.match(r'^(\d+)\s*(s|sec|seconds?|m|min|minutes?|h|hrs?|hours?)$', delay_str)
    
    if not match:
        raise ValueError(f"Invalid delay format: {delay_str}")
        
    value = int(match.group(1))
    unit = match.group(2)
    
    if unit.startswith('s'):
        return value
    elif unit.startswith('m'):
        return value * 60
    elif unit.startswith('h'):
        return value * 3600
        
    return value


def main():
    """
    Main entry point - builds and pushes all configured images.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Docker Build and Push Script with Retry Logic")
    parser.add_argument("--delay", type=str, help="Msg to start delay e.g. 'in 10 minutes', '10m', '300s'")
    parser.add_argument("--hf-token", type=str, help="Hugging Face token for gated models")
    parser.add_argument("--build", action="store_true", help="Build images (overrides per-image config)")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild by using --no-cache")
    parser.add_argument("--fresh-clone", dest="fresh_clone", action="store_true", help="Bust cache only at git clone layers (ensures latest ComfyUI/extensions without full rebuild)")
    parser.add_argument("--push", action="store_true", help="Push images (overrides per-image config)")
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
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🐳 Docker Build and Push Script")
    print("="*60)
    
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
            
    print(f"Images to process: {len(IMAGES)}")
    print(f"Push attempts: {PUSH_ATTEMPTS} (1 initial + 1 retry)")
    print(f"Retry delay: {RETRY_DELAY_SECONDS} seconds")
    
    successful = []
    build_failed = []
    push_failed = []
    
    for config in IMAGES:
        result = build_and_push_image(config, args.hf_token, args.rebuild, args.push, args.build, getattr(args, "fresh_clone", False), force_direct_push=getattr(args, "direct_push", False))
        if result == "success":
            successful.append(config.tag)
        elif result == "build_failed":
            build_failed.append(config.tag)
        else:
            push_failed.append(config.tag)
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    if successful:
        print(f"\n✅ Successfully built and pushed ({len(successful)}):")
        for tag in successful:
            print(f"   - {tag}")
    
    if build_failed:
        print(f"\n⚠️ Build failed (ignored) ({len(build_failed)}):")
        for tag in build_failed:
            print(f"   - {tag}")
            
    if push_failed:
        print(f"\n❌ Push failed after retries ({len(push_failed)}):")
        for tag in push_failed:
            print(f"   - {tag}")
    
    print("\n" + "="*60)
    
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
