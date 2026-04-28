#!/bin/bash
set -e

# NOTE: This script is intended to be run with root privileges (e.g. via pkexec or sudo)

echo "[OpenFork] Starting Native Linux Deployment..."

detect_package_manager() {
    if command -v apt-get >/dev/null 2>&1; then
        echo "apt"
    elif command -v dnf >/dev/null 2>&1; then
        echo "dnf"
    elif command -v yum >/dev/null 2>&1; then
        echo "yum"
    elif command -v zypper >/dev/null 2>&1; then
        echo "zypper"
    else
        echo ""
    fi
}

docker_ready() {
    docker info >/dev/null 2>&1
}

docker_ready_for_user() {
    if [ -z "$REAL_USER" ] || [ "$REAL_USER" = "root" ]; then
        return 1
    fi

    su - "$REAL_USER" -c "docker info >/dev/null 2>&1"
}

install_packages() {
    case "$PACKAGE_MANAGER" in
        apt)
            apt-get update
            apt-get install -y "$@"
            ;;
        dnf)
            dnf install -y "$@"
            ;;
        yum)
            yum install -y "$@"
            ;;
        zypper)
            zypper --non-interactive install "$@"
            ;;
        *)
            echo "[OpenFork] No supported package manager is available for installing: $*"
            exit 1
            ;;
    esac
}

refresh_gpg_command() {
    if command -v gpg >/dev/null 2>&1; then
        GPG_CMD="gpg"
    elif command -v gpg2 >/dev/null 2>&1; then
        GPG_CMD="gpg2"
    else
        GPG_CMD=""
    fi
}

# Determine the actual username of the user running the app
REAL_USER=${SUDO_USER:-$(logname 2>/dev/null || echo "$USER")}
if [ "$REAL_USER" = "root" ]; then
    REAL_USER=$(who am i | awk '{print $1}')
fi

refresh_gpg_command

PACKAGE_MANAGER=""
if ! command -v docker >/dev/null 2>&1 || \
   ! command -v nvidia-ctk >/dev/null 2>&1 || \
   ! command -v curl >/dev/null 2>&1 || \
   [ -z "$GPG_CMD" ]; then
    PACKAGE_MANAGER=$(detect_package_manager)
fi

echo "[OpenFork] Setting up Docker and NVIDIA Container Toolkit for user: $REAL_USER"

if [ -n "$PACKAGE_MANAGER" ]; then
    echo "[OpenFork] Using package manager: $PACKAGE_MANAGER"
elif ! command -v docker >/dev/null 2>&1 || ! command -v nvidia-ctk >/dev/null 2>&1; then
    echo "[OpenFork] Docker or NVIDIA Container Toolkit is missing, and this distro does not expose a supported package manager."
    echo "[OpenFork] Install Docker and NVIDIA Container Toolkit manually, then rerun setup."
    exit 1
fi

if ! command -v curl >/dev/null 2>&1 || [ -z "$GPG_CMD" ]; then
    echo "[OpenFork] Installing required system tools..."
    case "$PACKAGE_MANAGER" in
        apt)
            install_packages curl gnupg ca-certificates
            ;;
        dnf|yum)
            install_packages curl gnupg2 ca-certificates
            ;;
        zypper)
            install_packages curl gpg2 ca-certificates
            ;;
    esac
    refresh_gpg_command
fi

echo "[OpenFork] Checking for Docker..."
if ! command -v docker >/dev/null 2>&1; then
    echo "[OpenFork] Installing Docker Engine..."
    case "$PACKAGE_MANAGER" in
        apt|dnf|yum)
            curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
            sh /tmp/get-docker.sh
            rm /tmp/get-docker.sh
            ;;
        zypper)
            install_packages docker
            ;;
        *)
            echo "[OpenFork] Automatic Docker installation is not supported on this distro."
            exit 1
            ;;
    esac

    if [ -n "$REAL_USER" ] && [ "$REAL_USER" != "root" ]; then
        echo "[OpenFork] Adding $REAL_USER to docker group..."
        groupadd -f docker || true
        usermod -aG docker "$REAL_USER" || true
    fi
else
    echo "[OpenFork] Docker is already installed."
fi

echo "[OpenFork] Checking for NVIDIA Container Toolkit..."
if ! command -v nvidia-ctk >/dev/null 2>&1; then
    echo "[OpenFork] Installing NVIDIA Container Toolkit..."
    case "$PACKAGE_MANAGER" in
        apt)
            if [ ! -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]; then
                curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
                    "$GPG_CMD" --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            fi
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
            apt-get update
            apt-get install -y nvidia-container-toolkit
            ;;
        dnf|yum)
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
                tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null
            install_packages nvidia-container-toolkit
            ;;
        *)
            echo "[OpenFork] Automatic NVIDIA Container Toolkit installation is not supported on this distro yet."
            echo "[OpenFork] Install nvidia-container-toolkit manually, then rerun setup."
            exit 1
            ;;
    esac
else
    echo "[OpenFork] NVIDIA Container Toolkit is already installed."
fi

if command -v nvidia-ctk >/dev/null 2>&1; then
    echo "[OpenFork] Configuring NVIDIA Container Toolkit for Docker..."
    nvidia-ctk runtime configure --runtime=docker
fi

echo "[OpenFork] Ensuring Docker daemon is enabled and running..."
if docker_ready || docker_ready_for_user; then
    echo "[OpenFork] Docker is already accessible."
else
    if command -v systemctl >/dev/null 2>&1; then
        mkdir -p /etc/systemd/system/docker.service.d
        cat > /etc/systemd/system/docker.service.d/openfork-override.conf <<'EOF'
[Service]
# OpenFork manages Docker lifecycle on native Linux installs.
EOF
    fi

    if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files | grep -q '^docker\.service'; then
        systemctl daemon-reload || true
        systemctl enable docker
        systemctl restart docker || systemctl start docker
    elif command -v service >/dev/null 2>&1; then
        service docker restart || service docker start
    else
        echo "[OpenFork] No supported Docker service manager found."
        echo "[OpenFork] Start Docker manually, then rerun setup if Docker stays unavailable."
    fi
fi

echo "[OpenFork] Waiting for Docker daemon..."
for i in $(seq 1 20); do
    if docker_ready || docker_ready_for_user; then
        echo "[OpenFork] Docker daemon is running."
        break
    fi
    sleep 1
done

if ! docker_ready && ! docker_ready_for_user; then
    echo "[OpenFork] Docker daemon did not become ready. Check your Docker service logs."
    exit 1
fi

if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files | grep -q '^fstrim\.timer'; then
    echo "[OpenFork] Enabling periodic filesystem trim..."
    systemctl enable fstrim.timer >/dev/null 2>&1 || true
    systemctl start fstrim.timer >/dev/null 2>&1 || true
fi

# Write a marker file indicating this is an OpenFork-managed system
# This is used by the uninstall script to verify before cleanup
cat > /etc/openfork-managed <<EOF
managed-by=openfork
real-user=${REAL_USER:-unknown}
package-manager=${PACKAGE_MANAGER:-unknown}
EOF
chmod 644 /etc/openfork-managed

echo "[OpenFork] Setup Complete!"
echo "[OpenFork] Note: If Docker was just installed, you may need to LOG OUT and LOG BACK IN to apply user permissions."
