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


# Define the images to build and push
IMAGES: List[ImageConfig] = [
    ImageConfig("Dockerfile.ltx2-16gb", "beschiak/openfork-ltx2-16gb:latest"),
    ImageConfig("Dockerfile.ltx2-8gb", "beschiak/openfork-ltx2-8gb:latest"),
    ImageConfig("Dockerfile.yume-16gb", "beschiak/openfork-yume-16gb:latest"),
    # ImageConfig("Dockerfile.ltx2-24gb", "beschiak/openfork-ltx2-24gb:latest"),
]

MAX_RETRIES = 2
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
    Build a Docker image with retry logic.
    """
    command = ["docker", "build"]
    
    if rebuild:
        command.append("--no-cache")
        
    if hf_token:
        command.extend(["--build-arg", f"HF_TOKEN={hf_token}"])
    else:
        command.extend(["--build-arg", "HF_TOKEN"])
        
    command.extend(["-f", dockerfile, "-t", tag, "."])
    
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n📦 Build attempt {attempt}/{MAX_RETRIES} for {tag}")
        
        if run_command(command, f"Building {tag}"):
            return True
        
        if attempt < MAX_RETRIES:
            print(f"⏳ Waiting {RETRY_DELAY_SECONDS} seconds before retry...")
            time.sleep(RETRY_DELAY_SECONDS)
    
    return False


def push_image(tag: str) -> bool:
    """
    Push a Docker image with retry logic.
    """
    command = ["docker", "push", tag]
    
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n🚀 Push attempt {attempt}/{MAX_RETRIES} for {tag}")
        
        if run_command(command, f"Pushing {tag}"):
            return True
        
        if attempt < MAX_RETRIES:
            print(f"⏳ Waiting {RETRY_DELAY_SECONDS} seconds before retry...")
            time.sleep(RETRY_DELAY_SECONDS)
    
    return False


def build_and_push_image(config: ImageConfig, hf_token: str = None, rebuild: bool = False) -> bool:
    """
    Build and push a single image. Returns True if both operations succeed.
    """
    print(f"\n{'#'*60}")
    print(f"# Processing: {config.dockerfile} -> {config.tag}")
    print('#'*60)
    
    # Build the image
    if not build_image(config.dockerfile, config.tag, hf_token, rebuild):
        print(f"\n💥 FAILED to build {config.tag} after {MAX_RETRIES} attempts")
        return False
    
    # Push the image
    if not push_image(config.tag):
        print(f"\n💥 FAILED to push {config.tag} after {MAX_RETRIES} attempts")
        return False
    
    print(f"\n🎉 Successfully built and pushed {config.tag}")
    return True



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
    print(f"Max retries per operation: {MAX_RETRIES}")
    print(f"Retry delay: {RETRY_DELAY_SECONDS} seconds")
    
    successful = []
    failed = []
    
    for config in IMAGES:
        if build_and_push_image(config, args.hf_token, args.rebuild):
            successful.append(config.tag)
        else:
            failed.append(config.tag)
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    if successful:
        print(f"\n✅ Successfully built and pushed ({len(successful)}):")
        for tag in successful:
            print(f"   - {tag}")
    
    if failed:
        print(f"\n❌ Failed ({len(failed)}):")
        for tag in failed:
            print(f"   - {tag}")
    
    print("\n" + "="*60)
    
    if failed:
        print("⚠️  Some images failed to build or push!")
        sys.exit(1)
    else:
        print("🎉 All images built and pushed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
