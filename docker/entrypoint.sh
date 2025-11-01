#!/bin/bash
set -e
echo "[Entrypoint] Starting ComfyUI..."
exec python3 main.py --listen --port 8188