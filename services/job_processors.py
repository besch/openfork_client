"""
Job Processors

BACKWARD COMPATIBILITY LAYER
----------------------------
This module re-exports all processor classes from the new modular structure
for backward compatibility with existing code that imports from here.

New code should import directly from services.processors instead.

.. deprecated::
    This module will be removed in a future version.
    Use ``from services.processors import ...`` instead.
"""

import warnings

warnings.warn(
    "Import from services.processors instead of services.job_processors. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

from services.processors import (
    BaseJobProcessor,
    VideoOutputHandler,
    AudioOutputHandler,
    ImageOutputHandler,
    ComfyUIProcessor,
    WAN22TextToVideoJobProcessor,
    WAN22ImageToVideoJobProcessor,
    ImageToVideoFromLastFrameJobProcessor,
    HunyuanTextToVideoJobProcessor,
    HunyuanImageToVideoJobProcessor,
    LTXTextToVideoJobProcessor,
    LTXImageToVideoJobProcessor,
    LTX2TextToVideoJobProcessor,
    LTX2ImageToVideoJobProcessor,
    VideoUpscalerJobProcessor,
    TurboDiffusionT2VJobProcessor,
    TurboDiffusionI2VJobProcessor,
    StableAudioJobProcessor,
    FoleyJobProcessor,
    VibeVoiceJobProcessor,
    VibeVoiceMultiCloneJobProcessor,
    DiffRhythmCLIJobProcessor,
    ChatterboxTTSJobProcessor,
    ChatterboxVoiceCloneJobProcessor,
    TextToImageJobProcessor,
    ZImageTextToImageProcessor,
    ZImageControlNetProcessor,
    ZImageInpaintProcessor,
    QwenImageEditProcessor,
    QwenImageInpaintProcessor,
    QwenImageT2IProcessor,
    LLMJobProcessor,
    SVIShotImageToVideoJobProcessor,
    SVIFilmImageToVideoJobProcessor,
    HeartMuLaCLIJobProcessor,
    AceStepCLIJobProcessor,
    Qwen3TTSJobProcessor,
    Qwen3VoiceDesignJobProcessor,
    Qwen3VoiceCloneJobProcessor,
    QwenImageEditTurboProcessor,
    QwenImageInpaintTurboProcessor,
    QwenImageT2ITurboProcessor,
)

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
    "LTX2TextToVideoJobProcessor",
    "LTX2ImageToVideoJobProcessor",
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
]


