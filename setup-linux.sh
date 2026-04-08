#!/bin/bash
set -e

# NOTE: This script is intended to be run with root privileges (e.g. via pkexec or sudo)

echo "[OpenFork] Starting Native Linux Deployment..."

if ! command -v apt-get &> /dev/null; then
    echo "[OpenFork] This installer currently supports apt-based Linux distributions only."
    exit 1
fi

# Determine the actual username of the user running the app
REAL_USER=${SUDO_USER:-$(logname 2>/dev/null || echo $USER)}
if [ "$REAL_USER" = "root" ]; then
    # Try to find the user via who am i if logname failed or is root
    REAL_USER=$(who am i | awk '{print $1}')
fi

echo "[OpenFork] Setting up Docker and NVIDIA Container Toolkit for user: $REAL_USER"

# Ensure base tooling exists
if ! command -v curl &> /dev/null || ! command -v gpg &> /dev/null; then
    echo "[OpenFork] Installing required system tools..."
    apt-get update
    apt-get install -y curl gnupg ca-certificates
fi

echo "[OpenFork] Checking for Docker..."
if ! command -v docker &> /dev/null; then
    echo "[OpenFork] Installing Docker Engine..."
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sh /tmp/get-docker.sh
    rm /tmp/get-docker.sh
    
    if [ ! -z "$REAL_USER" ]; then
        echo "[OpenFork] Adding $REAL_USER to docker group..."
        usermod -aG docker "$REAL_USER"
    fi
else
    echo "[OpenFork] Docker is already installed."
fi

echo "[OpenFork] Checking for NVIDIA Container Toolkit..."
if ! command -v nvidia-ctk &> /dev/null; then
    echo "[OpenFork] Installing NVIDIA Container Toolkit..."
    if [ ! -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]; then
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    fi
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    apt-get update
    apt-get install -y nvidia-container-toolkit
else
    echo "[OpenFork] NVIDIA Container Toolkit is already installed."
fi

if command -v nvidia-ctk &> /dev/null; then
    echo "[OpenFork] Configuring NVIDIA Container Toolkit for Docker..."
    nvidia-ctk runtime configure --runtime=docker
fi

echo "[OpenFork] Ensuring Docker daemon is enabled and running..."
if command -v systemctl &> /dev/null; then
    systemctl daemon-reload || true
    systemctl enable docker
    systemctl restart docker
else
    service docker restart
fi

echo "[OpenFork] Waiting for Docker daemon..."
for i in $(seq 1 20); do
    if docker info &> /dev/null; then
        echo "[OpenFork] Docker daemon is running."
        break
    fi
    sleep 1
done

if ! docker info &> /dev/null; then
    echo "[OpenFork] Docker daemon did not become ready. Check your Docker service logs."
    exit 1
fi

echo "[OpenFork] Setup Complete!"
echo "[OpenFork] Note: If Docker was just installed, you may need to LOG OUT and LOG BACK IN to apply user permissions."
