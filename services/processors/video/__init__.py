"""Video processor modules."""

from .wan22_text import WAN22TextToVideoJobProcessor
from .wan22_image import WAN22ImageToVideoJobProcessor, ImageToVideoFromLastFrameJobProcessor
from .upscaler import VideoUpscalerJobProcessor
from .sparkvsr import SparkVSRUpscalerJobProcessor
from .turbodiffusion import TurboDiffusionT2VJobProcessor, TurboDiffusionI2VJobProcessor
from .scail import SCAILImageToVideoProcessor
from .vista4d import Vista4DVideoToVideoProcessor

__all__ = [
    "WAN22TextToVideoJobProcessor",
    "WAN22ImageToVideoJobProcessor",
    "ImageToVideoFromLastFrameJobProcessor",
    "VideoUpscalerJobProcessor",
    "SparkVSRUpscalerJobProcessor",
    "SCAILImageToVideoProcessor",
    "Vista4DVideoToVideoProcessor",
    "TurboDiffusionI2VJobProcessor",
]
