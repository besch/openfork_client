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
from .video.ltx2_text import LTX2TextToVideoJobProcessor
from .video.ltx2_image import LTX2ImageToVideoJobProcessor
from .video.ltx23_text import LTX23TextToVideoJobProcessor
from .video.ltx23_image import LTX23ImageToVideoJobProcessor
from .video.ltx23_text_16gb import LTX23TextToVideo16GBJobProcessor
from .video.ltx23_image_16gb import LTX23ImageToVideo16GBJobProcessor
from .video.upscaler import VideoUpscalerJobProcessor
from .video.turbodiffusion import TurboDiffusionT2VJobProcessor, TurboDiffusionI2VJobProcessor
from .video.svi_shot import SVIShotImageToVideoJobProcessor
from .video.svi_film import SVIFilmImageToVideoJobProcessor

# Audio processors
from .audio.stable_audio import StableAudioJobProcessor

from .audio.vibevoice import VibeVoiceJobProcessor, VibeVoiceMultiCloneJobProcessor
from .audio.diffrhythm_cli import DiffRhythmCLIJobProcessor
from .audio.heartmula_cli import HeartMuLaCLIJobProcessor
from .audio.acestep_cli import AceStepCLIJobProcessor
from .audio.chatterbox import ChatterboxTTSJobProcessor, ChatterboxVoiceCloneJobProcessor
from .audio.qwen3_tts import Qwen3TTSJobProcessor, Qwen3VoiceDesignJobProcessor, Qwen3VoiceCloneJobProcessor
from .audio.mmaudio import MMAudioJobProcessor
from .audio.lavasr import LavaSRJobProcessor

# Image processors
from .image.text_to_image import TextToImageJobProcessor
from .image.zimage import ZImageTextToImageProcessor, ZImageControlNetProcessor, ZImageInpaintProcessor
from .image.qwen import QwenImageEditProcessor, QwenImageInpaintProcessor, QwenImageT2IProcessor
from .image.qwen_turbo import QwenImageEditTurboProcessor, QwenImageInpaintTurboProcessor, QwenImageT2ITurboProcessor

# Text processors
from .llm.llm import LLMJobProcessor

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
    "LTX2TextToVideoJobProcessor",
    "LTX2ImageToVideoJobProcessor",
    "LTX23TextToVideoJobProcessor",
    "LTX23ImageToVideoJobProcessor",
    "LTX23TextToVideo16GBJobProcessor",
    "LTX23ImageToVideo16GBJobProcessor",
    "VideoUpscalerJobProcessor",
    "TurboDiffusionT2VJobProcessor",
    "TurboDiffusionI2VJobProcessor",
    "StableAudioJobProcessor",

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
    "LLMJobProcessor",
    "ChatterboxTTSJobProcessor",
    "ChatterboxVoiceCloneJobProcessor",
    "SVIShotImageToVideoJobProcessor",
    "SVIFilmImageToVideoJobProcessor",
    "HeartMuLaCLIJobProcessor",
    "AceStepCLIJobProcessor",
    "Qwen3TTSJobProcessor",
    "Qwen3VoiceDesignJobProcessor",
    "Qwen3VoiceCloneJobProcessor",
    "QwenImageEditTurboProcessor",
    "QwenImageInpaintTurboProcessor",
    "QwenImageT2ITurboProcessor",
    "MMAudioJobProcessor",
    "LavaSRJobProcessor",
]


