"""
Lightning Video Processors

Subclasses of WAN22 processors for lightning-fast video generation.
"""

from .wan22_text import WAN22TextToVideoJobProcessor
from .wan22_image import WAN22ImageToVideoJobProcessor


class TextToVideoLightningJobProcessor(WAN22TextToVideoJobProcessor):
    """Lightning variant of text-to-video processor."""
    workflow_name = "WAN22_LIGHTNING_TEXT_TO_VIDEO.json"


class ImageToVideoLightningJobProcessor(WAN22ImageToVideoJobProcessor):
    """Lightning variant of image-to-video processor."""
    workflow_name = "WAN22_LIGHTNING_IMAGE_TO_VIDEO.json"
