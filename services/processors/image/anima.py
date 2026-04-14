"""
Anima Image Processor

Processor for Anima text-to-image.
"""

from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import ImageOutputHandler


class AnimaTextToImageProcessor(ComfyUIProcessor, ImageOutputHandler):
    """Processor for Anima text-to-image generation."""

    def process(self):
        if not self.job:
            self._fail_job(
                "Job object is None for AnimaTextToImageProcessor. Cannot proceed."
            )
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        api_graph = workflow_data["prompt"]
        inputs = self.job.get("inputs", {})

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
