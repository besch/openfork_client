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
    build: bool = True
    push: bool = True


# Define the images to build and push
IMAGES: List[ImageConfig] = [
    ImageConfig("Dockerfile.heartmula", "beschiak/openfork-heartmula:latest", build=True, push=True),
    ImageConfig("Dockerfile.hunyuan-video-16gb", "beschiak/openfork-hunyuan-video-16gb:latest", build=True, push=True),
    ImageConfig("Dockerfile.wan22-24gb", "beschiak/openfork-wan22-24gb:latest", build=True, push=True),
    ImageConfig("Dockerfile.ltx2-24gb", "beschiak/openfork-ltx2-24gb:latest", build=True, push=True),
    ImageConfig("Dockerfile.ltx2-8gb", "beschiak/openfork-ltx2-8gb:latest", build=True, push=True),
    # ImageConfig("Dockerfile.ltx2-16gb", "beschiak/openfork-ltx2-16gb:latest", build=True, push=True),
]

PUSH_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 600  # 10 minutes


def run_command(command: List[str], description: str) -> bool:
    """
    Run a command and return True if successful, False otherwise.
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


def build_image(dockerfile: str, tag: str, hf_token: str = None, rebuild: bool = False) -> bool:
    """
    Build a Docker image. No retry logic for builds - failure is ignored.
    """
    command = ["docker", "build"]
    
    if rebuild:
        command.append("--no-cache")
        
    if hf_token:
        command.extend(["--build-arg", f"HF_TOKEN={hf_token}"])
    else:
        command.extend(["--build-arg", "HF_TOKEN"])
        
    command.extend(["-f", dockerfile, "-t", tag, "."])
    
    print(f"\n📦 Building {tag} (single attempt)")
    return run_command(command, f"Building {tag}")


def push_image(tag: str) -> bool:
    """
    Push a Docker image with one retry after unsuccessful attempt (2 total).
    """
    # Note: --max-concurrent-uploads 3 is added to help with stability on some environments
    command = ["docker", "push", "--max-concurrent-uploads", "3", tag]
    
    for attempt in range(1, PUSH_ATTEMPTS + 1):
        print(f"\n🚀 Push attempt {attempt}/{PUSH_ATTEMPTS} for {tag}")
        
        if run_command(command, f"Pushing {tag}"):
            return True
        
        if attempt < PUSH_ATTEMPTS:
            print(f"⏳ Waiting {RETRY_DELAY_SECONDS} seconds before retry...")
            time.sleep(RETRY_DELAY_SECONDS)
    
    return False


def build_and_push_image(config: ImageConfig, hf_token: str = None, rebuild: bool = False, global_push: bool = False) -> str:
    """
    Build and push a single image based on its configuration and global flags.
    Returns: "success", "build_failed", or "push_failed"
    """
    print(f"\n{'#'*60}")
    print(f"# Processing: {config.dockerfile} -> {config.tag}")
    print(f"# Config: build={config.build}, push={config.push}")
    print('#'*60)
    
    # Build the image if configured
    if config.build:
        if not build_image(config.dockerfile, config.tag, hf_token, rebuild):
            print(f"\n⚠️ FAILED to build {config.tag}. Ignoring build failure as requested.")
            return "build_failed"
    else:
        print(f"⏭️ Skipping build for {config.tag} as configured")
    
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
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild by using --no-cache")
    parser.add_argument("--push", action="store_true", help="Push images after building (default: build only)")
    
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
        result = build_and_push_image(config, args.hf_token, args.rebuild, args.push)
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
