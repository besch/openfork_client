"""
Qwen-Image Turbo Processors
Extends standard Qwen processors to support Turbo mode via specific workflow files.
"""

from services.processors.image.qwen import QwenImageEditProcessor, QwenImageInpaintProcessor, QwenImageT2IProcessor

class QwenImageEditTurboProcessor(QwenImageEditProcessor):
    """Turbo version of QwenImageEditProcessor."""
    
    @property
    def workflow_file(self) -> str:
        return "qwen-image-edit-8gb-turbo_api.json"

class QwenImageInpaintTurboProcessor(QwenImageInpaintProcessor):
    """Turbo version of QwenImageInpaintProcessor."""
    
    @property
    def workflow_file(self) -> str:
        return "qwen-image-inpaint-8gb-turbo_api.json"

class QwenImageT2ITurboProcessor(QwenImageT2IProcessor):
    """Turbo version of QwenImageT2IProcessor."""
    
    @property
    def workflow_file(self) -> str:
        return "qwen-image-t2i-8gb-turbo_api.json"
