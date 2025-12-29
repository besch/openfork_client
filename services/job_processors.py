"""
Job Processors

BACKWARD COMPATIBILITY LAYER
----------------------------
This module re-exports all processor classes from the new modular structure
for backward compatibility with existing code that imports from here.

New code should import directly from services.processors instead.
"""

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
    TextGenerationJobProcessor,
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


