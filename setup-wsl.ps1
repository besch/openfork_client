<#
.SYNOPSIS
    Automated OpenFork AI Engine Setup
    Enables WSL, installs Ubuntu, Docker Engine, and NVIDIA Container Toolkit.
#>

param (
    [switch]$InstallOnly,
    [string]$InstallPath
)

$ErrorActionPreference = "Stop"
$VerbosePreference = "Continue"

$progressLog = "C:\Windows\Temp\openfork_install_progress.log"
[System.IO.File]::WriteAllText($progressLog, "", [System.Text.Encoding]::UTF8)

function Write-Log {
    param([string]$Message)
    Write-Host "[OpenFork Setup] $Message" -ForegroundColor Cyan
    $ts = Get-Date -Format "HH:mm:ss"
    Add-Content -Path $progressLog -Value "[$ts] $Message" -Encoding UTF8
}

function Check-IsAdmin {
    $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Check-IsAdmin)) {
    Write-Log "Need Administrator privileges to install WSL features. Please re-run as Administrator."
    Exit 1
}

Write-Log "Checking Windows Subsystem for Linux (WSL) status..."
$wslFeature = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -ErrorAction SilentlyContinue
$vmpFeature = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -ErrorAction SilentlyContinue

$requiresReboot = $false

if ($null -ne $wslFeature -and $wslFeature.State -ne "Enabled") {
    Write-Log "Enabling WSL feature..."
    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -NoRestart
    $requiresReboot = $true
}

if ($null -ne $vmpFeature -and $vmpFeature.State -ne "Enabled") {
    Write-Log "Enabling Virtual Machine Platform feature..."
    Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -NoRestart
    $requiresReboot = $true
}

if ($requiresReboot) {
    Write-Log "WSL features were enabled. A system reboot is required."
    Write-Output "REBOOT_REQUIRED"
    Exit 0
}

Write-Log "Checking for Ubuntu distribution..."
try {
    $dists = (wsl -l -v | Out-String) -replace "\0", ""
    if ($dists -notmatch "Ubuntu") {
        if ($null -ne $InstallPath -and $InstallPath -ne "") {
            Write-Log "Installing Ubuntu to custom path: $InstallPath"
            if (-not (Test-Path $InstallPath)) {
                New-Item -ItemType Directory -Path $InstallPath -Force
            }
            
            $rootfsPath = Join-Path $InstallPath "ubuntu-rootfs.tar.gz"
            $rootfsUrl = "https://cloud-images.ubuntu.com/wsl/releases/24.04/current/ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz"
            
            Write-Log "Downloading Ubuntu rootfs (~130MB)..."
            Invoke-WebRequest -Uri $rootfsUrl -OutFile $rootfsPath -UseBasicParsing
            
            # Clean up any leftover VHDX from a previous failed import
            # (causes ERROR_FILE_EXISTS even when no distro is registered)
            $leftoverVhdx = Join-Path $InstallPath "ext4.vhdx"
            if (Test-Path $leftoverVhdx) {
                Write-Log "Found leftover ext4.vhdx from previous failed install. Removing..."
                Remove-Item $leftoverVhdx -Force
            }
            
            Write-Log "Importing Ubuntu to $InstallPath..."
            wsl --import Ubuntu $InstallPath $rootfsPath --version 2
            
            Remove-Item $rootfsPath -Force
        } else {
            Write-Log "Installing Ubuntu without launch (Default path)..."
            wsl --install -d Ubuntu --no-launch
        }
        
        Write-Log "Waiting for WSL to list Ubuntu..."
        $retry = 0
        while (((wsl -l -v | Out-String) -replace "\0", "") -notmatch "Ubuntu") {
            if ($retry -gt 60) { throw "Timeout waiting for Ubuntu install" }
            Start-Sleep -Seconds 2
            $retry++
        }

        Write-Log "Provisioning default user automatically to bypass interactive prompt..."
        # Use root to bypass initial prompt and silently create a default 'openfork' user
        $provisionScript = @"
if ! id -u openfork > /dev/null 2>&1; then
    useradd -m -s /bin/bash openfork
    echo "openfork:openfork" | chpasswd
    usermod -aG sudo openfork
    echo "openfork ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/openfork
fi
mkdir -p /etc
echo -e "[boot]\nsystemd=true\n[user]\ndefault=openfork" > /etc/wsl.conf
"@
        $provisionScript | wsl -d Ubuntu --user root -e bash -c "cat > /tmp/provision.sh && bash /tmp/provision.sh"
        
        Write-Log "Restarting WSL to apply new default user and systemd setting..."
        wsl --shutdown
        Start-Sleep -Seconds 2
    } else {
        Write-Log "Ubuntu is already installed. Ensuring systemd is enabled..."
        wsl -d Ubuntu --user root -e bash -c "if ! grep -q 'systemd=true' /etc/wsl.conf 2>/dev/null; then mkdir -p /etc && echo -e '[boot]\nsystemd=true' >> /etc/wsl.conf && echo 'SYSTEMD_ENABLED' > /tmp/systemd_changed; fi"
        if (wsl -d Ubuntu -e cat /tmp/systemd_changed 2>$null) {
            Write-Log "Systemd was just enabled. Restarting WSL..."
            wsl --shutdown
            Start-Sleep -Seconds 2
            wsl -d Ubuntu -e rm -f /tmp/systemd_changed
        }
    }
} catch {
    Write-Log "Detailed Error: $($_.Exception.Message)"
    if ($_.ScriptStackTrace) { Write-Log "Stack: $($_.ScriptStackTrace)" }
    Write-Log "Failed to check or install Ubuntu via wsl command. Make sure WSL is fully updated."
    Write-Output "ERROR: Failed to install Ubuntu."
    Exit 1
}

Write-Log "Enabling Sparse VHD for automatic disk space reclamation..."
try {
    # This requires WSL version 2.0.0 or higher.
    wsl --manage Ubuntu --set-sparse true
    Write-Log "Sparse VHD enabled successfully."
} catch {
    Write-Log "Warning: Could not enable sparse VHD. Your Windows version may be too old to support automatic disk reclamation."
}

Write-Log "Ensuring WSL is running and executing setup script..."

$script = @"
#!/bin/bash
set -e

echo "[Linux] Checking for Docker..."
if ! command -v docker &> /dev/null; then
    echo "[Linux] Installing Docker Engine..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sed -i 's/sleep 20/sleep 1/g' get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
else
    echo "[Linux] Docker is already installed."
fi

echo "[Linux] Checking for NVIDIA Container Toolkit..."
if ! command -v nvidia-ctk &> /dev/null; then
    echo "[Linux] Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
else
    echo "[Linux] NVIDIA Container Toolkit is already installed."
fi

echo "[Linux] Configuring Docker to listen on TCP..."
# Create or modify daemon.json to listen on tcp and unix socket
sudo mkdir -p /etc/docker
echo '{"hosts": ["tcp://0.0.0.0:2375", "unix:///var/run/docker.sock"], "tls": false}' | sudo tee /etc/docker/daemon.json

# Override docker.service to not pass -H fd:// which conflicts with daemon.json hosts
sudo mkdir -p /etc/systemd/system/docker.service.d
echo -e "[Service]\nExecStart=\nExecStart=/usr/bin/dockerd" | sudo tee /etc/systemd/system/docker.service.d/override.conf

# Start docker
if command -v systemctl &> /dev/null; then
    sudo systemctl daemon-reload
    sudo systemctl enable docker
    sudo systemctl restart docker
else
    # Fallback to service if systemd somehow isn't active
    sudo service docker restart
fi

# Ensure docker is ready before proceeding
echo "[Linux] Waiting for Docker daemon to be ready..."
sleep 2
for i in {1..15}; do
    if sudo docker info &> /dev/null; then
        echo "[Linux] Docker daemon is running."
        break
    fi
    sleep 1
done

echo "[Linux] OpenFork AI Engine Setup Complete."
"@

# Write the bash script to a Windows temp file to avoid stdin pipe issues when running elevated.
# When PowerShell is launched via Start-Process -Verb RunAs, the stdin pipe is broken,
# so piping to `wsl ... cat >` silently produces an empty file. Using a file on disk is reliable.
Write-Log "Writing setup script to temp file..."
$tempScriptPath = "C:\Windows\Temp\openfork_setup.sh"
[System.IO.File]::WriteAllText($tempScriptPath, $script.Replace("`r`n", "`n"), [System.Text.Encoding]::UTF8)

# Convert Windows path to WSL /mnt/ path (works for any drive letter)
$driveLetter = $tempScriptPath[0].ToString().ToLower()
$wslScriptPath = "/mnt/$driveLetter/" + $tempScriptPath.Substring(3).Replace('\', '/')

Write-Log "Running Docker setup commands inside WSL Ubuntu..."
wsl -d Ubuntu --user root -- bash $wslScriptPath

Remove-Item $tempScriptPath -Force -ErrorAction SilentlyContinue

Write-Log "Setup Complete!"
Write-Output "SUCCESS"
Exit 0
