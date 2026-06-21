#!/bin/bash
# Run build_and_push.py via Git Bash calling WSL Ubuntu distro
# Works with Git Bash on Windows
# Usage: ./run_build_git_bash.sh [build] [push] [options]
# Example: export HF_TOKEN="hf_xxxxx" && ./run_build_git_bash.sh build
# Example: export HF_TOKEN="hf_xxxxx" && ./run_build_git_bash.sh build push
# Example: export HF_TOKEN="hf_xxxxx" && ./run_build_git_bash.sh build push --compact-after-each

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_DISTRO="${WSL_DISTRO:-OpenFork}"
COMPACT_AFTER_EACH=false
PASSTHROUGH_ARGS=()
EXPECT_HF_TOKEN=false

for arg in "$@"; do
    if [ "$EXPECT_HF_TOKEN" = true ]; then
        export HF_TOKEN="$arg"
        EXPECT_HF_TOKEN=false
        continue
    fi

    case "$arg" in
        --hf-token)
            EXPECT_HF_TOKEN=true
            ;;
        --hf-token=*)
            export HF_TOKEN="${arg#--hf-token=}"
            ;;
        --compact-after-each)
            COMPACT_AFTER_EACH=true
            ;;
        *)
            PASSTHROUGH_ARGS+=("$arg")
            ;;
    esac
done

if [ "$EXPECT_HF_TOKEN" = true ]; then
    echo "❌ --hf-token was provided without a value."
    exit 1
fi

has_arg() {
    local expected="$1"
    shift
    for value in "$@"; do
        if [ "$value" = "$expected" ]; then
            return 0
        fi
    done
    return 1
}

add_arg_if_missing() {
    local expected="$1"
    if ! has_arg "$expected" "${PASSTHROUGH_ARGS[@]}"; then
        PASSTHROUGH_ARGS+=("$expected")
    fi
}

strip_delay_args() {
    PASSTHROUGH_ARGS_NO_DELAY=()
    local skip_next=false
    for value in "${PASSTHROUGH_ARGS[@]}"; do
        if [ "$skip_next" = true ]; then
            skip_next=false
            continue
        fi

        case "$value" in
            --delay)
                skip_next=true
                ;;
            --delay=*)
                ;;
            *)
                PASSTHROUGH_ARGS_NO_DELAY+=("$value")
                ;;
        esac
    done
}

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
COMPACT_SCRIPT_UNIX_PATH="$SCRIPT_DIR/../../desktop/scripts/compact-wsl.ps1"
if command -v cygpath &> /dev/null; then
    COMPACT_SCRIPT_WIN_PATH="$(cygpath -w "$COMPACT_SCRIPT_UNIX_PATH")"
else
    COMPACT_SCRIPT_WIN_PATH="D:\\openfork\\desktop\\scripts\\compact-wsl.ps1"
fi
COMPACT_SCRIPT_WIN_PATH="${COMPACT_WSL_SCRIPT_PATH:-$COMPACT_SCRIPT_WIN_PATH}"

echo "📍 Running in Git Bash"
echo "🐧 Target WSL distro: $WSL_DISTRO"
echo "📂 Script path in WSL: $WSL_SCRIPT_PATH"
if [ "$COMPACT_AFTER_EACH" = true ]; then
    echo "🧹 Compact after each image: enabled"
    echo "📂 Compact script: $COMPACT_SCRIPT_WIN_PATH"
fi
echo ""

ensure_docker_service() {
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
}

ensure_docker_service

echo ""

append_wslenv_var() {
    local var_name="$1"
    if [ -n "${!var_name:-}" ]; then
        case ":${WSLENV:-}:" in
            *":${var_name}/u:"*) ;;
            *) export WSLENV="${WSLENV:+$WSLENV:}${var_name}/u" ;;
        esac
    fi
}

append_wslenv_var HF_TOKEN
append_wslenv_var DOCKER_HUB_USERNAME
append_wslenv_var DOCKER_HUB_TOKEN

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
    DOCKER_LOGIN_CMD="printf '%s' \"\$DOCKER_HUB_TOKEN\" | docker login -u \"\$DOCKER_HUB_USERNAME\" --password-stdin && "
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
echo "Running build_and_push.py with arguments: ${PASSTHROUGH_ARGS[*]}"
echo ""

quote_args() {
    if [ "$#" -eq 0 ]; then
        return 0
    fi
    printf '%q ' "$@"
}

run_build_python() {
    local build_args
    build_args=$(quote_args "$@")
    wsl.exe -d "$WSL_DISTRO" bash -c "${DOCKER_LOGIN_CMD}cd '$WSL_SCRIPT_PATH' && python3 build_and_push.py $build_args"
}

list_image_indexes() {
    local list_args
    list_args=$(quote_args "$@")
    wsl.exe -d "$WSL_DISTRO" bash -c "cd '$WSL_SCRIPT_PATH' && python3 build_and_push.py --list-image-indexes $list_args"
}

compact_wsl_disk() {
    echo ""
    echo "🧹 Compacting WSL disk before continuing..."
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$COMPACT_SCRIPT_WIN_PATH" -DistroName "$WSL_DISTRO"
}

if [ "$COMPACT_AFTER_EACH" = true ]; then
    add_arg_if_missing "--cleanup-after-each"
    add_arg_if_missing "--prune-build-cache-after-each"
    add_arg_if_missing "--trim-after-each"
    strip_delay_args

    IMAGE_INDEXES=$(list_image_indexes "${PASSTHROUGH_ARGS[@]}")
    if [ -z "$IMAGE_INDEXES" ]; then
        echo "❌ No images matched the requested filters."
        exit 1
    fi

    BUILD_EXIT_CODE=0
    FIRST_IMAGE=true
    for IMAGE_INDEX in $IMAGE_INDEXES; do
        ensure_docker_service

        if [ "$FIRST_IMAGE" = true ]; then
            CURRENT_ARGS=("${PASSTHROUGH_ARGS[@]}")
            FIRST_IMAGE=false
        else
            CURRENT_ARGS=("${PASSTHROUGH_ARGS_NO_DELAY[@]}")
        fi

        echo ""
        echo "========================================"
        echo "Running image index: $IMAGE_INDEX"
        echo "========================================"
        set +e
        run_build_python --image-index "$IMAGE_INDEX" "${CURRENT_ARGS[@]}"
        BUILD_EXIT_CODE=$?
        set -e
        if [ $BUILD_EXIT_CODE -ne 0 ]; then
            break
        fi

        set +e
        compact_wsl_disk
        BUILD_EXIT_CODE=$?
        set -e
        if [ $BUILD_EXIT_CODE -ne 0 ]; then
            break
        fi
    done
else
    # Run the build script in WSL (with optional Docker login)
    set +e
    run_build_python "${PASSTHROUGH_ARGS[@]}"
    BUILD_EXIT_CODE=$?
    set -e
fi

echo ""
if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo "✅ Build completed successfully"
else
    echo "❌ Build failed with exit code $BUILD_EXIT_CODE"
fi

exit $BUILD_EXIT_CODE
