"""Video processor modules."""

from .wan22_text import WAN22TextToVideoJobProcessor
from .wan22_image import WAN22ImageToVideoJobProcessor, ImageToVideoFromLastFrameJobProcessor
from .upscaler import VideoUpscalerJobProcessor
from .lightning import TextToVideoLightningJobProcessor, ImageToVideoLightningJobProcessor

__all__ = [
    "WAN22TextToVideoJobProcessor",
    "WAN22ImageToVideoJobProcessor",
    "ImageToVideoFromLastFrameJobProcessor",
    "VideoUpscalerJobProcessor",
    "TextToVideoLightningJobProcessor",
    "ImageToVideoLightningJobProcessor",
]
