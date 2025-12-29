"""
Job Processors Package

This package contains all job processors for the DGN client, organized by output type.
All processors are re-exported from this __init__ for backward compatibility.
"""

from .base import BaseJobProcessor
from .output_handlers import VideoOutputHandler, AudioOutputHandler, ImageOutputHandler
from .comfyui_processor import ComfyUIProcessor

# Video processors
from .video.wan22_text import WAN22TextToVideoJobProcessor
from .video.wan22_image import WAN22ImageToVideoJobProcessor, ImageToVideoFromLastFrameJobProcessor
from .video.hunyuan_text import HunyuanTextToVideoJobProcessor
from .video.hunyuan_image import HunyuanImageToVideoJobProcessor
from .video.ltx_text import LTXTextToVideoJobProcessor
from .video.ltx_image import LTXImageToVideoJobProcessor
from .video.upscaler import VideoUpscalerJobProcessor
from .video.turbodiffusion import TurboDiffusionT2VJobProcessor, TurboDiffusionI2VJobProcessor

# Audio processors
from .audio.stable_audio import StableAudioJobProcessor
from .audio.foley import FoleyJobProcessor
from .audio.vibevoice import VibeVoiceJobProcessor, VibeVoiceMultiCloneJobProcessor
from .audio.diffrhythm_cli import DiffRhythmCLIJobProcessor
from .audio.chatterbox import ChatterboxTTSJobProcessor, ChatterboxVoiceCloneJobProcessor

# Image processors
from .image.text_to_image import TextToImageJobProcessor
from .image.zimage import ZImageTextToImageProcessor, ZImageControlNetProcessor, ZImageInpaintProcessor
from .image.qwen import QwenImageEditProcessor, QwenImageInpaintProcessor, QwenImageT2IProcessor

# Text processors
from .text.text_generation import TextGenerationJobProcessor

__all__ = [
    "BaseJobProcessor",
    "VideoOutputHandler",
    "AudioOutputHandler",
    "ImageOutputHandler",
    "ComfyUIProcessor",
    "WAN22TextToVideoJobProcessor",
    "WAN22ImageToVideoJobProcessor",
    "ImageToVideoFromLastFrameJobProcessor",
    "HunyuanTextToVideoJobProcessor",
    "HunyuanImageToVideoJobProcessor",
    "LTXTextToVideoJobProcessor",
    "LTXImageToVideoJobProcessor",
    "VideoUpscalerJobProcessor",
    "TurboDiffusionT2VJobProcessor",
    "TurboDiffusionI2VJobProcessor",
    "StableAudioJobProcessor",
    "FoleyJobProcessor",
    "VibeVoiceJobProcessor",
    "VibeVoiceMultiCloneJobProcessor",
    "DiffRhythmCLIJobProcessor",
    "TextToImageJobProcessor",
    "ZImageTextToImageProcessor",
    "ZImageControlNetProcessor",
    "ZImageInpaintProcessor",
    "QwenImageEditProcessor",
    "QwenImageInpaintProcessor",
    "QwenImageT2IProcessor",
    "TextGenerationJobProcessor",
    "ChatterboxTTSJobProcessor",
    "ChatterboxVoiceCloneJobProcessor",
]


