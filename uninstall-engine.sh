#!/bin/bash
#
# OpenFork AI Engine Uninstall
# Removes Docker, NVIDIA Container Toolkit, and OpenFork markers.
# Called automatically by DEB postuninstall or manually by user.
# Mirrors the logic of uninstall-engine.ps1 (Windows equivalent).
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${CYAN}[OpenFork Uninstall]${NC} $1"
}

log_warn() {
    echo -e "${RED}[OpenFork Uninstall WARNING]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OpenFork Uninstall]${NC} $1"
}

# Ensure running as root
if [ "$(id -u)" -ne 0 ]; then
    log_info "Requesting root privileges..."
    exec sudo "$0" "$@"
fi

# Determine the actual username of the user running the app
REAL_USER="${SUDO_USER:-$(logname 2>/dev/null || echo "$USER")}"
if [ "$REAL_USER" = "root" ] || [ -z "$REAL_USER" ]; then
    REAL_USER="$(who am i 2>/dev/null | awk '{print $1}' || echo '')"
fi

if [ -z "$REAL_USER" ]; then
    log_warn "Could not determine the user who installed OpenFork. Skipping docker group cleanup."
    REAL_USER=""
fi

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

remove_packages() {
    local pm="$1"
    shift
    case "$pm" in
        apt)
            apt-get remove -y "$@" 2>/dev/null || true
            apt-get autoremove -y 2>/dev/null || true
            ;;
        dnf)
            dnf remove -y "$@" 2>/dev/null || true
            ;;
        yum)
            yum remove -y "$@" 2>/dev/null || true
            ;;
        zypper)
            zypper --non-interactive remove "$@" 2>/dev/null || true
            ;;
    esac
}

# Safety gate: Check if this is an OpenFork-managed system
test_openfork_managed() {
    if [ -f /etc/openfork-managed ]; then
        return 0
    fi

    # Fallback: Check for OpenFork-specific systemd override.
    if [ -f /etc/systemd/system/docker.service.d/openfork-override.conf ] 2>/dev/null; then
        return 0
    fi

    return 1
}

log_info "Starting OpenFork engine cleanup..."

# Safety check: Only proceed if this is an OpenFork-managed system
if ! test_openfork_managed; then
    log_warn "System does not appear to be OpenFork-managed (no /etc/openfork-managed marker found)."
    log_warn "Skipping cleanup to avoid removing user's own Docker/NVIDIA installation."
    log_info "If you installed OpenFork on this system, you may need to manually remove Docker and NVIDIA Container Toolkit."
    exit 0
fi

PACKAGE_MANAGER=$(detect_package_manager)

if [ -z "$PACKAGE_MANAGER" ]; then
    log_warn "No supported package manager detected. Cannot automatically remove packages."
    log_info "Please manually remove docker and nvidia-container-toolkit packages."
else
    log_info "Using package manager: $PACKAGE_MANAGER"
    
    # Remove NVIDIA Container Toolkit
    log_info "Removing NVIDIA Container Toolkit..."
    case "$PACKAGE_MANAGER" in
        apt)
            # Remove nvidia-container-toolkit
            remove_packages "$PACKAGE_MANAGER" nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container-tools libnvidia-container1
            # Remove NVIDIA apt repo
            rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list
            rm -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            apt-get update >/dev/null 2>&1 || true
            ;;
        dnf|yum)
            remove_packages "$PACKAGE_MANAGER" nvidia-container-toolkit libnvidia-container-tools libnvidia-container1
            rm -f /etc/yum.repos.d/nvidia-container-toolkit.repo
            ;;
        zypper)
            remove_packages "$PACKAGE_MANAGER" nvidia-container-toolkit
            ;;
    esac
    log_success "NVIDIA Container Toolkit removed."
    
    # Remove Docker
    log_info "Removing Docker Engine..."
    case "$PACKAGE_MANAGER" in
        apt)
            remove_packages "$PACKAGE_MANAGER" docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-compose
            # Also try generic 'docker.io' package name (Debian/Ubuntu)
            remove_packages "$PACKAGE_MANAGER" docker.io 2>/dev/null || true
            ;;
        dnf|yum)
            remove_packages "$PACKAGE_MANAGER" docker docker-ce docker-ce-cli docker-engine containerd.io docker-buildx-plugin docker-compose-plugin
            ;;
        zypper)
            remove_packages "$PACKAGE_MANAGER" docker docker-compose
            ;;
    esac
    log_success "Docker removed."
fi

# Revert docker group membership for the original user
if [ -n "$REAL_USER" ] && [ "$REAL_USER" != "root" ]; then
    log_info "Reverting docker group membership for user: $REAL_USER"
    if id "$REAL_USER" >/dev/null 2>&1; then
        gpasswd -d "$REAL_USER" docker 2>/dev/null || true
        if id -nG "$REAL_USER" 2>/dev/null | tr ' ' '\n' | grep -qx "docker"; then
            log_warn "User $REAL_USER is still a member of docker group (membership may be managed elsewhere)."
        else
            log_success "User $REAL_USER removed from docker group."
            log_info "Note: User must log out and log back in for group changes to take effect."
        fi
    fi
fi

# Remove OpenFork-specific configuration files and markers
log_info "Cleaning up OpenFork markers and configuration..."
rm -f /etc/openfork-managed
rm -f /etc/openfork-setup-log

# Clean up Docker systemd overrides if they exist
if [ -d /etc/systemd/system/docker.service.d ]; then
    rm -f /etc/systemd/system/docker.service.d/openfork-override.conf
    rmdir /etc/systemd/system/docker.service.d 2>/dev/null || true
    systemctl daemon-reload 2>/dev/null || true
fi

# Disable fstrim if it was enabled by OpenFork
if command -v systemctl >/dev/null 2>&1; then
    # Note: We keep fstrim enabled since it's beneficial for all users
    # systemctl disable fstrim.timer >/dev/null 2>&1 || true
    # systemctl stop fstrim.timer >/dev/null 2>&1 || true
    true
fi

log_success "OpenFork engine cleanup complete."
log_info "Removing OpenFork-managed Docker data directories..."
rm -rf /var/lib/docker 2>/dev/null || true
rm -rf /var/lib/containerd 2>/dev/null || true
log_info "Docker data cleanup complete."
