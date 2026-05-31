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
from .video.scail import SCAILImageToVideoProcessor
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
from .image.pid import PiDImageUpscaleProcessor

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
    "LTX23TextToVideoWan2GPProcessor",
    "LTX23ImageToVideoWan2GPProcessor",
    "VideoUpscalerJobProcessor",
    "SparkVSRUpscalerJobProcessor",
    "TurboDiffusionT2VJobProcessor",
    "TurboDiffusionI2VJobProcessor",
    "DreamIDOmniImageToVideoProcessor",
    "DaVinciMagiHumanT2VProcessor",
    "DaVinciMagiHumanI2VProcessor",
    "SCAILImageToVideoProcessor",
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
    "LLMJobProcessor",
    "ChatterboxTTSJobProcessor",
    "ChatterboxVoiceCloneJobProcessor",
    "HeartMuLaCLIJobProcessor",
    "AceStepCLIJobProcessor",
    "Qwen3TTSJobProcessor",
    "Qwen3VoiceDesignJobProcessor",
    "Qwen3VoiceCloneJobProcessor",
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
    "PiDImageUpscaleProcessor",
    "MMAudioJobProcessor",
    "LavaSRJobProcessor",
    "PrismAudioJobProcessor",
]
