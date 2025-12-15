"""
ComfyUIManagerClient - Direct API calls to ComfyUI-Manager for node installation.

This is much faster than comfy-cli because it runs inside the already-loaded
ComfyUI process, avoiding Python/torch initialization overhead.
"""

import logging
import time
import requests
from typing import Union
from dataclasses import dataclass


@dataclass
class InstallResult:
    """Result of a node installation operation."""
    success: bool
    package_name: str
    message: str


class ComfyUIManagerClient:
    """Client for ComfyUI-Manager REST API."""
    
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188"):
        self.base_url = comfyui_url.rstrip("/")
        self._manager_available = None
    
    def is_manager_available(self) -> bool:
        """Check if ComfyUI-Manager is installed and responding."""
        if self._manager_available is not None:
            return self._manager_available
        
        try:
            # ComfyUI-Manager exposes /manager/version endpoint
            response = requests.get(
                f"{self.base_url}/manager/version",
                timeout=5
            )
            self._manager_available = response.status_code == 200
            if self._manager_available:
                logging.info(f"ComfyUI-Manager is available: {response.text.strip()}")
            else:
                logging.warning("ComfyUI-Manager endpoint returned non-200")
        except requests.exceptions.RequestException as e:
            logging.warning(f"ComfyUI-Manager not available: {e}")
            self._manager_available = False
        
        return self._manager_available
    
    def get_extension_list(self) -> list[dict]:
        """Get list of available extensions from ComfyUI-Manager."""
        try:
            response = requests.get(
                f"{self.base_url}/manager/extension_list",
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get("custom_nodes", [])
            return []
        except Exception as e:
            logging.error(f"Failed to get extension list: {e}")
            return []
    
    def find_package_for_node(self, class_type: str, node_map: dict) -> Union[str, None]:
        """Find the git URL for a package that provides a given node class_type."""
        return node_map.get(class_type)
    
    def install_by_git_url(self, git_url: str) -> InstallResult:
        """Install a custom node package by its git URL.
        
        ComfyUI-Manager API: POST /manager/install
        Body: {"url": "https://github.com/..."}
        """
        package_name = git_url.rstrip('/').split('/')[-1].replace('.git', '')
        
        if not self.is_manager_available():
            return InstallResult(
                success=False,
                package_name=package_name,
                message="ComfyUI-Manager is not available"
            )
        
        logging.info(f"Installing {package_name} via ComfyUI-Manager API...")
        
        try:
            response = requests.post(
                f"{self.base_url}/manager/install",
                json={"url": git_url},
                timeout=300  # 5 minute timeout for installation
            )
            
            if response.status_code == 200:
                logging.info(f"Successfully installed {package_name}")
                return InstallResult(
                    success=True,
                    package_name=package_name,
                    message="Installed successfully"
                )
            else:
                error_msg = response.text or f"HTTP {response.status_code}"
                logging.error(f"Failed to install {package_name}: {error_msg}")
                return InstallResult(
                    success=False,
                    package_name=package_name,
                    message=error_msg
                )
                
        except requests.exceptions.Timeout:
            return InstallResult(
                success=False,
                package_name=package_name,
                message="Installation timed out after 5 minutes"
            )
        except Exception as e:
            logging.error(f"Error installing {package_name}: {e}")
            return InstallResult(
                success=False,
                package_name=package_name,
                message=str(e)
            )
    
    def reboot_comfyui(self) -> bool:
        """Trigger ComfyUI restart to load newly installed nodes.
        
        ComfyUI-Manager API: POST /manager/reboot
        """
        logging.info("Triggering ComfyUI reboot via Manager API...")
        
        try:
            response = requests.post(
                f"{self.base_url}/manager/reboot",
                timeout=10
            )
            # Reboot endpoint may not respond as server shuts down
            return True
        except requests.exceptions.RequestException:
            # Expected - server is restarting
            return True
    
    def wait_for_ready(self, timeout_seconds: int = 90) -> bool:
        """Wait for ComfyUI to become ready after reboot."""
        logging.info(f"Waiting up to {timeout_seconds}s for ComfyUI to restart...")
        
        start = time.time()
        # Wait a bit for server to actually stop
        time.sleep(3)
        
        while time.time() - start < timeout_seconds:
            try:
                response = requests.get(
                    f"{self.base_url}/object_info",
                    timeout=5
                )
                if response.status_code == 200:
                    elapsed = int(time.time() - start)
                    logging.info(f"ComfyUI is ready after {elapsed}s")
                    # Invalidate cached data
                    self._manager_available = None
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        logging.error(f"ComfyUI did not become ready within {timeout_seconds}s")
        return False
    
    def install_packages(self, git_urls: list[str]) -> tuple[bool, list[InstallResult]]:
        """Install multiple packages and reboot ComfyUI.
        
        Returns:
            Tuple of (all_success, list of results)
        """
        if not git_urls:
            return True, []
        
        results = []
        needs_reboot = False
        
        for git_url in git_urls:
            result = self.install_by_git_url(git_url)
            results.append(result)
            if result.success:
                needs_reboot = True
        
        # Reboot if any packages were installed
        if needs_reboot:
            self.reboot_comfyui()
            if not self.wait_for_ready():
                logging.error("ComfyUI failed to restart after installation")
                return False, results
        
        all_success = all(r.success for r in results)
        return all_success, results


def install_via_git_clone(git_url: str, custom_nodes_dir: str) -> InstallResult:
    """Fallback: Install custom node by cloning git repository directly.
    
    This is a backup method if ComfyUI-Manager is not available.
    """
    import subprocess
    import os
    
    package_name = git_url.rstrip('/').split('/')[-1].replace('.git', '')
    target_dir = os.path.join(custom_nodes_dir, package_name)
    
    if os.path.exists(target_dir):
        logging.info(f"Package {package_name} already exists at {target_dir}")
        return InstallResult(
            success=True,
            package_name=package_name,
            message="Already installed"
        )
    
    logging.info(f"Cloning {git_url} to {target_dir}...")
    
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", git_url, target_dir],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            # Check for requirements.txt and install
            requirements_file = os.path.join(target_dir, "requirements.txt")
            if os.path.exists(requirements_file):
                logging.info(f"Installing requirements for {package_name}...")
                subprocess.run(
                    ["pip", "install", "-r", requirements_file],
                    capture_output=True,
                    timeout=300
                )
            
            return InstallResult(
                success=True,
                package_name=package_name,
                message="Cloned successfully"
            )
        else:
            return InstallResult(
                success=False,
                package_name=package_name,
                message=result.stderr
            )
            
    except subprocess.TimeoutExpired:
        return InstallResult(
            success=False,
            package_name=package_name,
            message="Git clone timed out"
        )
    except FileNotFoundError:
        return InstallResult(
            success=False,
            package_name=package_name,
            message="Git is not installed"
        )
    except Exception as e:
        return InstallResult(
            success=False,
            package_name=package_name,
            message=str(e)
        )
