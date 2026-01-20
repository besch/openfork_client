"""Video processor modules."""

from .wan22_text import WAN22TextToVideoJobProcessor
from .wan22_image import WAN22ImageToVideoJobProcessor, ImageToVideoFromLastFrameJobProcessor
from .upscaler import VideoUpscalerJobProcessor
from .turbodiffusion import TurboDiffusionT2VJobProcessor, TurboDiffusionI2VJobProcessor
from .svi_shot import SVIShotImageToVideoJobProcessor
from .svi_film import SVIFilmImageToVideoJobProcessor

__all__ = [
    "WAN22TextToVideoJobProcessor",
    "WAN22ImageToVideoJobProcessor",
    "ImageToVideoFromLastFrameJobProcessor",
    "VideoUpscalerJobProcessor",
    "TurboDiffusionT2VJobProcessor",
    "TurboDiffusionI2VJobProcessor",
    "SVIShotImageToVideoJobProcessor",
    "SVIFilmImageToVideoJobProcessor",
]
