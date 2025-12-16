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
    VideoUpscalerJobProcessor,
    TextToVideoLightningJobProcessor,
    ImageToVideoLightningJobProcessor,
    StableAudioJobProcessor,
    FoleyJobProcessor,
    VibeVoiceJobProcessor,
    VibeVoiceMultiCloneJobProcessor,
    DiffRhythmJobProcessor,
    DiffRhythmCLIJobProcessor,
    TextToImageJobProcessor,
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
    "VideoUpscalerJobProcessor",
    "TextToVideoLightningJobProcessor",
    "ImageToVideoLightningJobProcessor",
    "StableAudioJobProcessor",
    "FoleyJobProcessor",
    "VibeVoiceJobProcessor",
    "VibeVoiceMultiCloneJobProcessor",
    "DiffRhythmJobProcessor",
    "DiffRhythmCLIJobProcessor",
    "TextToImageJobProcessor",
    "TextGenerationJobProcessor",
]
