from typing import Dict, Any, Optional
from services.processors.base import JobProcessor
from utils.comfyui_workflow_utils import (
    inject_prompt_into_yume_workflow,
    inject_image_into_yume_workflow,
)

class YumeTextToVideoJobProcessor(JobProcessor):
    """
    Processor for Yume 1.5 Text-to-Video content generation jobs.
    """

    async def process(self, job_inputs: Dict[str, Any], workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the job inputs and inject them into the ComfyUI workflow.
        """
        prompt = job_inputs.get("prompt")
        
        # Inject parameters into the workflow
        workflow = inject_prompt_into_yume_workflow(
            workflow,
            prompt=prompt,
            frame_rate=job_inputs.get("frame_rate", 24),
            steps=job_inputs.get("steps", 30),
            cfg=job_inputs.get("cfg_scale", 7.0),
            seed=job_inputs.get("seed", 0),
            width=job_inputs.get("width", 1280),
            height=job_inputs.get("height", 720),
            num_frames=job_inputs.get("num_frames", 49),
        )

        return workflow

class YumeImageToVideoJobProcessor(JobProcessor):
    """
    Processor for Yume 1.5 Image-to-Video content generation jobs.
    """

    async def process(self, job_inputs: Dict[str, Any], workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the job inputs and inject them into the ComfyUI workflow.
        """
        prompt = job_inputs.get("prompt")
        start_image_path = job_inputs.get("start_image")
        
        # Inject parameters into the workflow
        workflow = inject_image_into_yume_workflow(
            workflow,
            image_path=start_image_path,
            prompt=prompt,
            frame_rate=job_inputs.get("frame_rate", 24),
            steps=job_inputs.get("steps", 30),
            cfg=job_inputs.get("cfg_scale", 7.0),
            seed=job_inputs.get("seed", 0),
            width=job_inputs.get("width", 1280),
            height=job_inputs.get("height", 720),
            num_frames=job_inputs.get("num_frames", 49),
        )

        return workflow
