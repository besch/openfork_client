"""Video processor modules."""

from .wan22_text import WAN22TextToVideoJobProcessor
from .wan22_image import WAN22ImageToVideoJobProcessor, ImageToVideoFromLastFrameJobProcessor
from .upscaler import VideoUpscalerJobProcessor

__all__ = [
    "WAN22TextToVideoJobProcessor",
    "WAN22ImageToVideoJobProcessor",
    "ImageToVideoFromLastFrameJobProcessor",
    "VideoUpscalerJobProcessor",
]

