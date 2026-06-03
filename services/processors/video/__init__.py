"""Video processor modules."""

from .wan22_text import WAN22TextToVideoJobProcessor
from .wan22_image import WAN22ImageToVideoJobProcessor, ImageToVideoFromLastFrameJobProcessor
from .wan22_wan2gp import (
    ImageToVideoFromLastFrameWan2GPProcessor,
    WAN22ImageToVideoWan2GPProcessor,
    WAN22TextToVideoWan2GPProcessor,
)
from .upscaler import VideoUpscalerJobProcessor
from .sparkvsr import SparkVSRUpscalerJobProcessor
from .turbodiffusion import TurboDiffusionT2VJobProcessor, TurboDiffusionI2VJobProcessor
from .dreamid_omni import DreamIDOmniImageToVideoProcessor
from .scail import SCAILImageToVideoProcessor
from .vista4d import Vista4DVideoToVideoProcessor

__all__ = [
    "WAN22TextToVideoJobProcessor",
    "WAN22ImageToVideoJobProcessor",
    "ImageToVideoFromLastFrameJobProcessor",
    "WAN22TextToVideoWan2GPProcessor",
    "WAN22ImageToVideoWan2GPProcessor",
    "ImageToVideoFromLastFrameWan2GPProcessor",
    "VideoUpscalerJobProcessor",
    "SparkVSRUpscalerJobProcessor",
    "DreamIDOmniImageToVideoProcessor",
    "SCAILImageToVideoProcessor",
    "Vista4DVideoToVideoProcessor",
    "TurboDiffusionI2VJobProcessor",
]
