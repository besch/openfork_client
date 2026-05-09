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
from .video.wan22_image import (
    WAN22ImageToVideoJobProcessor,
    ImageToVideoFromLastFrameJobProcessor,
)
from .video.hunyuan_text import HunyuanTextToVideoJobProcessor
from .video.hunyuan_image import HunyuanImageToVideoJobProcessor
from .video.ltx23_text import LTX23TextToVideoWan2GPProcessor
from .video.ltx23_image import LTX23ImageToVideoWan2GPProcessor
from .video.ltx23_comfyui_text import LTX23ComfyUITextToVideoProcessor
from .video.ltx23_comfyui_image import LTX23ComfyUIImageToVideoProcessor
from .video.ltx23_comfyui_text_16gb import LTX23ComfyUITextToVideoProcessor16GB
from .video.ltx23_comfyui_image_16gb import LTX23ComfyUIImageToVideoProcessor16GB
from .video.ltx23_comfyui_text_8gb import LTX23ComfyUITextToVideoProcessor8GB
from .video.ltx23_comfyui_image_8gb import LTX23ComfyUIImageToVideoProcessor8GB
from .video.ltx23_comfyui_text_12gb import LTX23ComfyUITextToVideoProcessor12GB
from .video.ltx23_comfyui_image_12gb import LTX23ComfyUIImageToVideoProcessor12GB
from .video.upscaler import VideoUpscalerJobProcessor
from .video.sparkvsr import SparkVSRUpscalerJobProcessor
from .video.turbodiffusion import (
    TurboDiffusionT2VJobProcessor,
    TurboDiffusionI2VJobProcessor,
)
from .video.davinci_magihuman import (
    DaVinciMagiHumanT2VProcessor,
    DaVinciMagiHumanI2VProcessor,
)
from .video.scail import SCAILImageToVideoProcessor
from .video.inspatio_world import InSpatioWorldJobProcessor

# Audio processors
from .audio.stable_audio import StableAudioJobProcessor

from .audio.vibevoice import VibeVoiceJobProcessor, VibeVoiceMultiCloneJobProcessor
from .audio.diffrhythm_cli import DiffRhythmCLIJobProcessor
from .audio.heartmula_cli import HeartMuLaCLIJobProcessor
from .audio.acestep_cli import AceStepCLIJobProcessor
from .audio.chatterbox import (
    ChatterboxTTSJobProcessor,
    ChatterboxVoiceCloneJobProcessor,
)
from .audio.qwen3_tts import (
    Qwen3TTSJobProcessor,
    Qwen3VoiceDesignJobProcessor,
    Qwen3VoiceCloneJobProcessor,
)
from .audio.mmaudio import MMAudioJobProcessor
from .audio.lavasr import LavaSRJobProcessor
from .audio.prismaudio import PrismAudioJobProcessor

# Image processors
from .image.text_to_image import TextToImageJobProcessor
from .image.zimage import (
    ZImageTextToImageProcessor,
    ZImageControlNetProcessor,
    ZImageInpaintProcessor,
)
from .image.qwen import (
    QwenImageEditProcessor,
    QwenImageInpaintProcessor,
    QwenImageT2IProcessor,
)
from .image.qwen_turbo import (
    QwenImageEditTurboProcessor,
    QwenImageInpaintTurboProcessor,
    QwenImageT2ITurboProcessor,
)
from .image.anima import AnimaTextToImageProcessor
from .image.ernie_image import ErnieImageProcessor

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
    "LTX23TextToVideoWan2GPProcessor",
    "LTX23ImageToVideoWan2GPProcessor",
    "LTX23ComfyUITextToVideoProcessor",
    "LTX23ComfyUIImageToVideoProcessor",
    "LTX23ComfyUITextToVideoProcessor16GB",
    "LTX23ComfyUIImageToVideoProcessor16GB",
    "LTX23ComfyUITextToVideoProcessor8GB",
    "LTX23ComfyUIImageToVideoProcessor8GB",
    "LTX23ComfyUITextToVideoProcessor12GB",
    "LTX23ComfyUIImageToVideoProcessor12GB",
    "VideoUpscalerJobProcessor",
    "SparkVSRUpscalerJobProcessor",
    "TurboDiffusionT2VJobProcessor",
    "TurboDiffusionI2VJobProcessor",
    "DaVinciMagiHumanT2VProcessor",
    "DaVinciMagiHumanI2VProcessor",
    "SCAILImageToVideoProcessor",
    "InSpatioWorldJobProcessor",
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
    "HeartMuLaCLIJobProcessor",
    "AceStepCLIJobProcessor",
    "Qwen3TTSJobProcessor",
    "Qwen3VoiceDesignJobProcessor",
    "Qwen3VoiceCloneJobProcessor",
    "QwenImageEditTurboProcessor",
    "QwenImageInpaintTurboProcessor",
    "QwenImageT2ITurboProcessor",
    "AnimaTextToImageProcessor",
    "ErnieImageProcessor",
    "MMAudioJobProcessor",
    "LavaSRJobProcessor",
    "PrismAudioJobProcessor",
]
