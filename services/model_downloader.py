"""
ModelDownloader - Downloads AI models for ComfyUI workflows.

Uses ComfyUI-Manager's model-list.json as the source of truth for model URLs.
This approach ensures compatibility with ComfyUI ecosystem updates.
"""

import os
import logging
import json
import time
import requests
from typing import Union
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Model metadata from ComfyUI-Manager registry."""
    name: str
    filename: str
    url: str
    save_path: str  # Relative to ComfyUI/models/
    model_type: str
    size: str = ""
    base: str = ""


@dataclass 
class DownloadResult:
    """Result of a model download operation."""
    success: bool
    filename: str
    message: str
    local_path: str = ""


class ModelDownloader:
    """Downloads models using ComfyUI-Manager's registry."""
    
    MODEL_LIST_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/model-list.json"
    CACHE_DURATION_HOURS = 24
    
    # Map ComfyUI-Manager save_path values to actual directories
    # "default" means use the model type as directory name
    SAVE_PATH_MAP = {
        "default": None,  # Uses model_type lowercase
        "vae_approx": "vae_approx",
        "custom": None,  # Requires special handling
    }
    
    # Map model_type to ComfyUI models subdirectory
    MODEL_TYPE_TO_DIR = {
        "checkpoints": "checkpoints",
        "checkpoint": "checkpoints",
        "lora": "loras",
        "loras": "loras",
        "vae": "vae",
        "clip": "clip",
        "clip_vision": "clip_vision",
        "controlnet": "controlnet",
        "upscale": "upscale_models",
        "upscale_models": "upscale_models",
        "embeddings": "embeddings",
        "hypernetworks": "hypernetworks",
        "unet": "unet",
        "diffusion_models": "diffusion_models",
        "text_encoders": "text_encoders",
        "taesd": "vae_approx",
        "ipadapter": "ipadapter",
        "insightface": "insightface",
        "ultralytics": "ultralytics",
        "mmdets": "mmdets",
        "sams": "sams",
        "onnx": "onnx",
        "facerestore_models": "facerestore_models",
        "facedetection": "facedetection",
        "animatediff_models": "animatediff_models",
        "animatediff_motion_lora": "animatediff_motion_lora",
        "video_models": "video_models",
        "tts": "TTS",
    }
    
    def __init__(self, comfyui_install_dir: str, cache_dir: str = None):
        self.comfyui_install_dir = comfyui_install_dir
        self.models_dir = os.path.join(comfyui_install_dir, "models") if comfyui_install_dir else None
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".openfork_cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self._model_registry: dict[str, ModelInfo] = {}
        self._registry_loaded = False
    
    def _get_cache_path(self) -> str:
        return os.path.join(self.cache_dir, "model-list.json")
    
    def _load_registry(self, force_refresh: bool = False) -> bool:
        """Load model registry from cache or GitHub."""
        if self._registry_loaded and not force_refresh:
            return True
        
        cache_path = self._get_cache_path()
        
        # Check cache first
        if not force_refresh and os.path.exists(cache_path):
            try:
                mtime = os.path.getmtime(cache_path)
                age_hours = (time.time() - mtime) / 3600
                
                if age_hours < self.CACHE_DURATION_HOURS:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self._parse_registry(data)
                    logging.info(f"Loaded {len(self._model_registry)} models from cache ({age_hours:.1f}h old)")
                    return True
            except Exception as e:
                logging.warning(f"Could not load model registry cache: {e}")
        
        # Fetch from GitHub
        try:
            logging.info("Fetching model-list.json from ComfyUI-Manager...")
            response = requests.get(self.MODEL_LIST_URL, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Save to cache
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            
            self._parse_registry(data)
            logging.info(f"Loaded {len(self._model_registry)} models from GitHub")
            return True
            
        except Exception as e:
            logging.error(f"Failed to fetch model registry: {e}")
            
            # Try using stale cache as fallback
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self._parse_registry(data)
                    logging.warning(f"Using stale cache with {len(self._model_registry)} models")
                    return True
                except Exception:
                    pass
            
            return False
    
    def _parse_registry(self, data: dict):
        """Parse model-list.json into lookup dictionary."""
        self._model_registry.clear()
        
        models = data.get("models", [])
        for model in models:
            filename = model.get("filename", "")
            if not filename:
                continue
            
            self._model_registry[filename.lower()] = ModelInfo(
                name=model.get("name", filename),
                filename=filename,
                url=model.get("url", ""),
                save_path=model.get("save_path", "default"),
                model_type=model.get("type", "").lower(),
                size=model.get("size", ""),
                base=model.get("base", "")
            )
        
        self._registry_loaded = True
    
    def get_model_info(self, filename: str) -> Union[ModelInfo, None]:
        """Look up model info by filename."""
        self._load_registry()
        return self._model_registry.get(filename.lower())
    
    def get_save_directory(self, model_info: ModelInfo) -> str:
        """Determine the correct save directory for a model."""
        if not self.models_dir:
            return ""
        
        save_path = model_info.save_path.lower()
        
        # Handle special save_path values
        if save_path in self.SAVE_PATH_MAP:
            mapped = self.SAVE_PATH_MAP[save_path]
            if mapped:
                return os.path.join(self.models_dir, mapped)
        
        # Check if save_path is a custom path (contains /)
        if "/" in save_path or "\\" in save_path:
            return os.path.join(self.models_dir, save_path)
        
        # Map by model type
        model_type = model_info.model_type.lower()
        if model_type in self.MODEL_TYPE_TO_DIR:
            return os.path.join(self.models_dir, self.MODEL_TYPE_TO_DIR[model_type])
        
        # Fallback to save_path as directory name
        if save_path and save_path != "default":
            return os.path.join(self.models_dir, save_path)
        
        # Last resort: use model type
        return os.path.join(self.models_dir, model_type or "unknown")
    
    def is_model_installed(self, filename: str) -> bool:
        """Check if a model file exists in any models subdirectory."""
        if not self.models_dir or not os.path.exists(self.models_dir):
            return False
        
        # Check all subdirectories
        for root, _, files in os.walk(self.models_dir):
            if filename in files:
                return True
        
        return False
    
    def download_model(self, filename: str, progress_callback=None) -> DownloadResult:
        """Download a model by filename."""
        model_info = self.get_model_info(filename)
        
        if not model_info:
            return DownloadResult(
                success=False,
                filename=filename,
                message=f"Model '{filename}' not found in ComfyUI-Manager registry"
            )
        
        if not model_info.url:
            return DownloadResult(
                success=False,
                filename=filename,
                message=f"No download URL for model '{filename}'"
            )
        
        # Check if already installed
        if self.is_model_installed(filename):
            logging.info(f"Model '{filename}' is already installed")
            return DownloadResult(
                success=True,
                filename=filename,
                message="Already installed"
            )
        
        # Determine save location
        save_dir = self.get_save_directory(model_info)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        
        logging.info(f"Downloading model '{filename}' ({model_info.size}) to {save_dir}")
        
        try:
            response = requests.get(model_info.url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress = (downloaded / total_size) * 100
                            progress_callback(filename, progress, downloaded, total_size)
            
            logging.info(f"Successfully downloaded model '{filename}'")
            return DownloadResult(
                success=True,
                filename=filename,
                message="Downloaded successfully",
                local_path=save_path
            )
            
        except Exception as e:
            logging.error(f"Failed to download model '{filename}': {e}")
            # Clean up partial download
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except Exception:
                    pass
            
            return DownloadResult(
                success=False,
                filename=filename,
                message=str(e)
            )
    
    def download_models(self, filenames: list[str], progress_callback=None) -> list[DownloadResult]:
        """Download multiple models."""
        results = []
        for i, filename in enumerate(filenames):
            if progress_callback:
                progress_callback(filename, 0, 0, 0, current=i+1, total=len(filenames))
            
            result = self.download_model(filename, progress_callback)
            results.append(result)
        
        return results
    
    def get_all_installed_models(self) -> dict[str, list[str]]:
        """Scan all model directories and return installed models by type."""
        installed = {}
        
        if not self.models_dir or not os.path.exists(self.models_dir):
            return installed
        
        for subdir in os.listdir(self.models_dir):
            subdir_path = os.path.join(self.models_dir, subdir)
            if os.path.isdir(subdir_path):
                files = []
                for f in os.listdir(subdir_path):
                    f_path = os.path.join(subdir_path, f)
                    if os.path.isfile(f_path) and self._is_model_file(f):
                        files.append(f)
                if files:
                    installed[subdir] = files
        
        return installed
    
    def _is_model_file(self, filename: str) -> bool:
        """Check if a file is a model file based on extension."""
        model_extensions = {
            '.safetensors', '.ckpt', '.pt', '.pth', '.bin',
            '.onnx', '.pkl', '.pickle', '.h5', '.pb'
        }
        ext = os.path.splitext(filename.lower())[1]
        return ext in model_extensions
