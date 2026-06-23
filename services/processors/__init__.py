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
from .video.wan22_wan2gp import (
    WAN22TextToVideoWan2GPProcessor,
    WAN22ImageToVideoWan2GPProcessor,
    ImageToVideoFromLastFrameWan2GPProcessor,
)
from .video.ltx23_text import LTX23TextToVideoWan2GPProcessor
from .video.ltx23_image import LTX23ImageToVideoWan2GPProcessor
from .video.upscaler import VideoUpscalerJobProcessor
from .video.sparkvsr import SparkVSRUpscalerJobProcessor
from .video.turbodiffusion import (
    TurboDiffusionT2VJobProcessor,
    TurboDiffusionI2VJobProcessor,
)
from .video.dreamid_omni import DreamIDOmniImageToVideoProcessor
from .video.davinci_magihuman import (
    DaVinciMagiHumanT2VProcessor,
    DaVinciMagiHumanI2VProcessor,
)
from .video.scail2 import SCAIL2ImageToVideoProcessor
from .video.vista4d import Vista4DVideoToVideoProcessor
from .video.inspatio_world import InSpatioWorldJobProcessor

# Audio processors
from .audio.stable_audio import StableAudioJobProcessor
from .audio.audiox import AudioXJobProcessor

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
from .audio.f5_tts import F5TTSJobProcessor, F5VoiceCloneJobProcessor
from .audio.wavtts import WavTTSJobProcessor, WavTTSVoiceCloneJobProcessor
from .audio.dots_tts import DotsTTSJobProcessor, DotsTTSVoiceCloneJobProcessor
from .audio.scenema_audio import (
    ScenemaAudioTTSProcessor,
    ScenemaAudioVoiceCloneProcessor,
)
from .audio.dramabox import DramaboxTTSProcessor, DramaboxVoiceCloneProcessor
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
    QwenImage2512LoraT2IProcessor,
    QwenImageEditProcessor,
    QwenImageInpaintProcessor,
    QwenImageT2IProcessor,
)
from .image.qwen_turbo import (
    QwenImageEditTurboProcessor,
    QwenImageInpaintTurboProcessor,
    QwenImageT2ITurboProcessor,
)
from .image.flux_kontext import (
    FluxKontextEditProcessor,
    FluxKontextT2IProcessor,
)
from .image.anima import AnimaTextToImageProcessor
from .image.ernie_image import ErnieImageProcessor
from .image.ideogram4 import Ideogram4ImageProcessor
from .image.pid import PiDImageUpscaleProcessor
from .image.telestylev2 import TeleStyleV2Processor

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
    "WAN22TextToVideoWan2GPProcessor",
    "WAN22ImageToVideoWan2GPProcessor",
    "ImageToVideoFromLastFrameWan2GPProcessor",
    "LTX23TextToVideoWan2GPProcessor",
    "LTX23ImageToVideoWan2GPProcessor",
    "VideoUpscalerJobProcessor",
    "SparkVSRUpscalerJobProcessor",
    "TurboDiffusionT2VJobProcessor",
    "TurboDiffusionI2VJobProcessor",
    "DreamIDOmniImageToVideoProcessor",
    "DaVinciMagiHumanT2VProcessor",
    "DaVinciMagiHumanI2VProcessor",
    "SCAIL2ImageToVideoProcessor",
    "Vista4DVideoToVideoProcessor",
    "InSpatioWorldJobProcessor",
    "StableAudioJobProcessor",
    "AudioXJobProcessor",
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
    "QwenImage2512LoraT2IProcessor",
    "LLMJobProcessor",
    "ChatterboxTTSJobProcessor",
    "ChatterboxVoiceCloneJobProcessor",
    "HeartMuLaCLIJobProcessor",
    "AceStepCLIJobProcessor",
    "Qwen3TTSJobProcessor",
    "Qwen3VoiceDesignJobProcessor",
    "Qwen3VoiceCloneJobProcessor",
    "F5TTSJobProcessor",
    "F5VoiceCloneJobProcessor",
    "WavTTSJobProcessor",
    "WavTTSVoiceCloneJobProcessor",
    "DotsTTSJobProcessor",
    "DotsTTSVoiceCloneJobProcessor",
    "ScenemaAudioTTSProcessor",
    "ScenemaAudioVoiceCloneProcessor",
    "DramaboxTTSProcessor",
    "DramaboxVoiceCloneProcessor",
    "QwenImageEditTurboProcessor",
    "QwenImageInpaintTurboProcessor",
    "QwenImageT2ITurboProcessor",
    "FluxKontextEditProcessor",
    "FluxKontextT2IProcessor",
    "AnimaTextToImageProcessor",
    "ErnieImageProcessor",
    "Ideogram4ImageProcessor",
    "PiDImageUpscaleProcessor",
    "TeleStyleV2Processor",
    "MMAudioJobProcessor",
    "LavaSRJobProcessor",
    "PrismAudioJobProcessor",
]
