#!/bin/bash
# Run build_and_push.py via Git Bash calling WSL Ubuntu distro
# Works with Git Bash on Windows
# Usage: ./run_build_git_bash.sh [build] [push] [options]
# Example: export HF_TOKEN="hf_xxxxx" && ./run_build_git_bash.sh build
# Example: export HF_TOKEN="hf_xxxxx" && ./run_build_git_bash.sh build push

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DISTRO="${WSL_DISTRO:-Openfork}"

echo ""
echo "========================================"
echo "Docker Build and Push via Git Bash"
echo "========================================"
echo ""

# Verify WSL is available
if ! command -v wsl.exe &> /dev/null; then
    echo "❌ WSL not found. Make sure you have WSL2 installed."
    echo "   Install with: wsl --install"
    exit 1
fi

# Convert Windows path to WSL path
WSL_SCRIPT_PATH="/mnt/d/openfork/client/comfyui-storage"

echo "📍 Running in Git Bash"
echo "🐧 Target WSL distro: $WSL_DISTRO"
echo "📂 Script path in WSL: $WSL_SCRIPT_PATH"
echo ""

# Check and start Docker service in WSL
echo "🔧 Checking Docker service..."
wsl.exe -d "$WSL_DISTRO" bash -c "sudo service docker status > /dev/null 2>&1" 2>/dev/null || {
    echo "⏳ Starting Docker service in WSL..."
    wsl.exe -d "$WSL_DISTRO" bash -c "sudo service docker start" || {
        echo "⚠️  Warning: Could not start Docker service"
    }
}

# Check Docker socket permissions
echo "🔐 Checking Docker permissions..."
DOCKER_CHECK=$(wsl.exe -d "$WSL_DISTRO" bash -c "docker ps 2>&1" || true)
if echo "$DOCKER_CHECK" | grep -q "permission denied"; then
    echo "❌ Docker permission denied. Fixing..."
    echo "   Running: sudo usermod -aG docker \$USER"
    wsl.exe -d "$WSL_DISTRO" bash -c "sudo usermod -aG docker \$USER 2>/dev/null || true" 
    echo ""
    echo "⚠️  Docker permissions updated. You may need to:"
    echo "   1. Log out and log back into WSL: exit, then: wsl -d $WSL_DISTRO"
    echo "   2. Or run with sudo: sudo docker ps"
    echo ""
    echo "For now, will attempt to use sudo for docker commands..."
    echo ""
fi

echo ""

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  HF_TOKEN environment variable not set"
    echo "   Some gated models may fail to download"
    echo "   Set with: export HF_TOKEN='hf_your_token_here'"
fi

# Check if DOCKER_HUB_TOKEN and DOCKER_HUB_USERNAME are set for Docker Hub login
DOCKER_LOGIN_CMD=""
if [ -n "$DOCKER_HUB_TOKEN" ] && [ -n "$DOCKER_HUB_USERNAME" ]; then
    echo "🔑 Docker Hub credentials detected, will log in..."
    DOCKER_LOGIN_CMD="echo '$DOCKER_HUB_TOKEN' | docker login -u '$DOCKER_HUB_USERNAME' --password-stdin && "
    echo "✅ Docker Hub username: $DOCKER_HUB_USERNAME"
elif [ -n "$DOCKER_HUB_TOKEN" ]; then
    echo "⚠️  DOCKER_HUB_TOKEN set but DOCKER_HUB_USERNAME not set"
    echo "   Set both for automatic Docker Hub login:"
    echo "   export DOCKER_HUB_USERNAME='your_username'"
    echo "   export DOCKER_HUB_TOKEN='dckr_pat_xxxxx'"
else
    echo "ℹ️  DOCKER_HUB_TOKEN not set"
    echo "   You must be pre-logged into Docker Hub for push to work:"
    echo "   docker login"
    echo "   Or set credentials:"
    echo "   export DOCKER_HUB_USERNAME='your_username'"
    echo "   export DOCKER_HUB_TOKEN='dckr_pat_xxxxx'"
fi

echo ""
echo "Running build_and_push.py with arguments: $@"
echo ""

# Build properly escaped argument string for the inner bash
if [ -n "$HF_TOKEN" ]; then
    BUILD_ARGS=$(printf '%q ' "--hf-token" "$HF_TOKEN" "$@")
else
    BUILD_ARGS=$(printf '%q ' "$@")
fi

# Run the build script in WSL (with optional Docker login)
wsl.exe -d "$WSL_DISTRO" bash -c "${DOCKER_LOGIN_CMD}cd '$WSL_SCRIPT_PATH' && python3 build_and_push.py $BUILD_ARGS"
BUILD_EXIT_CODE=$?

echo ""
if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo "✅ Build completed successfully"
else
    echo "❌ Build failed with exit code $BUILD_EXIT_CODE"
fi

exit $BUILD_EXIT_CODE
