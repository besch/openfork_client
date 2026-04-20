"""
Anima Image Processor

Processor for Anima text-to-image.
"""

import copy
from typing import Optional

from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import ImageOutputHandler


class AnimaTextToImageProcessor(ComfyUIProcessor, ImageOutputHandler):
    """Processor for Anima text-to-image generation."""

    _DIMENSIONS_BY_ASPECT_RATIO = {
        "1:1": (1024, 1024),
        "16:9": (1360, 768),
        "9:16": (768, 1360),
        "4:3": (1152, 864),
        "3:4": (864, 1152),
        "3:2": (1216, 832),
        "2:3": (832, 1216),
        "21:9": (1536, 640),
    }

    def process(self):
        if not self.job:
            self._fail_job(
                "Job object is None for AnimaTextToImageProcessor. Cannot proceed."
            )
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        api_graph = copy.deepcopy(workflow_data["prompt"])

        self._apply_inputs_to_workflow(api_graph, inputs)

        payload = {"prompt": api_graph}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        image_storage_path = self.handle_image_output(outputs)
        if not image_storage_path:
            return

        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=image_storage_path,
            thumbnail_storage_path=image_storage_path,
            prompt=self.positive_prompt,
        )

    @classmethod
    def _get_dimensions(cls, aspect_ratio: Optional[str]) -> tuple[int, int]:
        return cls._DIMENSIONS_BY_ASPECT_RATIO.get(aspect_ratio or "1:1", (1024, 1024))

    def _apply_inputs_to_workflow(self, api_graph, inputs) -> None:
        width, height = self._get_dimensions(inputs.get("aspect_ratio"))

        for node_id, node in api_graph.items():
            class_type = node.get("class_type", "")
            if class_type == "CLIPTextEncode":
                if (
                    "positive" in str(node.get("_meta", {}).get("title", "")).lower()
                    or node_id == "6"
                ):
                    node["inputs"]["text"] = self.positive_prompt
                elif (
                    "negative" in str(node.get("_meta", {}).get("title", "")).lower()
                    or node_id == "7"
                ):
                    node["inputs"]["text"] = inputs.get("negative_prompt", "")

            if class_type == "KSampler":
                if "steps" in inputs:
                    node["inputs"]["steps"] = inputs["steps"]
                if "cfg" in inputs:
                    node["inputs"]["cfg"] = inputs["cfg"]
                if "sampler_name" in inputs:
                    node["inputs"]["sampler_name"] = inputs["sampler_name"]
                if "scheduler" in inputs:
                    node["inputs"]["scheduler"] = inputs["scheduler"]
                if "seed" in inputs:
                    node["inputs"]["seed"] = inputs["seed"]
            elif class_type == "EmptyLatentImage":
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height
