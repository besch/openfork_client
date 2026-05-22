"""
FLUX.1 Kontext [dev] processors.

The workflows use GGUF quantized model weights for RTX 30/40 compatibility and
low-VRAM operation, while keeping the standard ComfyUI Flux Kontext sampler graph.
"""

import copy
import logging
import random

from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import ImageOutputHandler
from services.processors.image.qwen import QwenImageEditProcessor


class FluxKontextWorkflowMixin:
    """Shared workflow injection helpers for FLUX Kontext image jobs."""

    def _is_8gb_tier(self):
        workflow_type = (self.workflow_type or "").lower()
        service_type = (getattr(self.client, "active_service_type", "") or "").lower()
        return "8gb" in f"{workflow_type} {service_type}"

    def _tier_long_edge(self):
        workflow_type = (self.workflow_type or "").lower()
        service_type = (getattr(self.client, "active_service_type", "") or "").lower()
        model_key = f"{workflow_type} {service_type}"

        if "8gb" in model_key:
            return 512
        if "12gb" in model_key:
            return 896
        return 1024

    @staticmethod
    def _as_int(value, fallback):
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _as_float(value, fallback):
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _round_to_multiple(value, multiple=16):
        return max(multiple, int(round(value / multiple)) * multiple)

    def _requested_long_edge(self, inputs=None):
        values = inputs or self.job.get("inputs", {})
        requested = (
            values.get("max_long_edge")
            or values.get("long_edge")
            or values.get("target_long_edge")
        )
        if requested is None:
            return self._tier_long_edge()
        if self._is_8gb_tier():
            return max(256, min(self._as_int(requested, self._tier_long_edge()), 512))
        return max(384, min(self._as_int(requested, self._tier_long_edge()), self._tier_long_edge()))

    def _get_dimensions(self, aspect_ratio, has_second_image=False, requested_long_edge=None):
        """Return tier-aware dimensions for generated/normalized output."""
        long_edge = (
            self._requested_long_edge({"max_long_edge": requested_long_edge})
            if requested_long_edge is not None
            else self._requested_long_edge()
        )
        ratio_map = {
            "1:1": (long_edge, long_edge),
            "16:9": (long_edge, self._round_to_multiple(long_edge * 9 / 16)),
            "9:16": (self._round_to_multiple(long_edge * 9 / 16), long_edge),
            "4:3": (long_edge, self._round_to_multiple(long_edge * 3 / 4)),
            "3:4": (self._round_to_multiple(long_edge * 3 / 4), long_edge),
            "3:2": (long_edge, self._round_to_multiple(long_edge * 2 / 3)),
            "2:3": (self._round_to_multiple(long_edge * 2 / 3), long_edge),
            "21:9": (long_edge, self._round_to_multiple(long_edge * 9 / 21)),
        }
        return ratio_map.get(aspect_ratio, (long_edge, long_edge))

    def _seed(self, seed):
        return self._as_int(seed, random.randint(0, 2**63 - 1))

    def _advanced_settings(self):
        inputs = self.job.get("inputs", {})
        default_steps = 12 if self._is_8gb_tier() else 20
        return {
            "steps": self._as_int(inputs.get("steps"), default_steps),
            "guidance": self._as_float(
                inputs.get("cfg", inputs.get("cfg_scale")),
                2.5,
            ),
            "sampler_name": inputs.get("sampler_name") or inputs.get("sampler") or "euler",
            "scheduler": inputs.get("scheduler") or "simple",
            "max_shift": inputs.get("flux_max_shift"),
            "base_shift": inputs.get("flux_base_shift"),
        }

    def _apply_common_workflow_inputs(
        self,
        wf,
        prompt,
        width,
        height,
        seed=None,
        denoise_strength=None,
        image_filename=None,
    ):
        settings = self._advanced_settings()
        actual_seed = self._seed(seed)

        for node in wf.values():
            class_type = node.get("class_type")
            inputs = node.get("inputs", {})

            if class_type == "CLIPTextEncode" and inputs.get("text") not in ("", None):
                inputs["text"] = prompt
            elif class_type == "DualCLIPLoader" and self._is_8gb_tier():
                inputs["device"] = "cpu"
            elif class_type == "LoadImage" and image_filename:
                inputs["image"] = image_filename
            elif class_type == "EmptySD3LatentImage":
                inputs["width"] = width
                inputs["height"] = height
            elif class_type == "ModelSamplingFlux":
                inputs["width"] = width
                inputs["height"] = height
                if settings["max_shift"] is not None:
                    inputs["max_shift"] = self._as_float(settings["max_shift"], inputs.get("max_shift", 1.15))
                if settings["base_shift"] is not None:
                    inputs["base_shift"] = self._as_float(settings["base_shift"], inputs.get("base_shift", 0.5))
            elif class_type == "FluxGuidance":
                inputs["guidance"] = settings["guidance"]
            elif class_type == "RandomNoise":
                inputs["noise_seed"] = actual_seed
            elif class_type == "KSamplerSelect":
                inputs["sampler_name"] = settings["sampler_name"]
            elif class_type == "BasicScheduler":
                inputs["steps"] = settings["steps"]
                inputs["scheduler"] = settings["scheduler"]
                if denoise_strength is not None:
                    inputs["denoise"] = self._as_float(denoise_strength, inputs.get("denoise", 1.0))
            elif class_type == "ImageScaleToMaxDimension":
                inputs["largest_size"] = max(width, height)
            elif class_type == "FluxKontextImageScale" and "largest_size" in inputs:
                inputs["largest_size"] = max(width, height)

        logging.info(
            "FLUX Kontext configured - size: %sx%s, steps: %s, guidance: %s, seed: %s",
            width,
            height,
            settings["steps"],
            settings["guidance"],
            actual_seed,
        )

    def _prepare_workflow(self, workflow_data):
        return copy.deepcopy(workflow_data.get("prompt", workflow_data))


class FluxKontextT2IProcessor(FluxKontextWorkflowMixin, ComfyUIProcessor, ImageOutputHandler):
    """Processor for FLUX.1 Kontext [dev] text-to-image generation."""

    def process(self):
        if not self.job:
            self._fail_job("Job object is None for FluxKontextT2IProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        aspect_ratio = inputs.get("aspect_ratio", "1:1")
        requested_long_edge = self._requested_long_edge(inputs)
        width, height = self._get_dimensions(
            aspect_ratio,
            requested_long_edge=requested_long_edge,
        )

        wf_ready = self._prepare_workflow(workflow_data)
        self._apply_common_workflow_inputs(
            wf_ready,
            self.positive_prompt,
            width,
            height,
            seed=inputs.get("seed"),
        )

        outputs = self._trigger_and_get_output({"prompt": wf_ready}, timeout_sec=1200)
        if not outputs:
            return

        image_storage_path = self.handle_image_output(
            outputs,
            target_dimensions=(width, height),
        )
        if not image_storage_path:
            return

        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=image_storage_path,
            thumbnail_storage_path=image_storage_path,
            prompt=self.positive_prompt,
        )


class FluxKontextEditProcessor(FluxKontextWorkflowMixin, QwenImageEditProcessor):
    """Processor for FLUX.1 Kontext [dev] instruction-based single-image editing."""

    def process(self):
        if not self.job:
            self._fail_job("Job object is None for FluxKontextEditProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        aspect_ratio = inputs.get("aspect_ratio", "1:1")
        denoise_strength = inputs.get("denoise_strength", inputs.get("strength", 1.0))
        requested_long_edge = self._requested_long_edge(inputs)
        width, height = self._get_dimensions(
            aspect_ratio,
            requested_long_edge=requested_long_edge,
        )

        source_image_filename = self._get_source_image()
        if not source_image_filename:
            return

        self._copy_image_to_container(source_image_filename)

        wf_ready = self._inject_edit_workflow(
            workflow_data,
            self.positive_prompt,
            source_image_filename,
            denoise_strength,
            aspect_ratio=aspect_ratio,
            seed=inputs.get("seed"),
        )

        outputs = self._trigger_and_get_output({"prompt": wf_ready}, timeout_sec=1200)
        if not outputs:
            return

        image_storage_path = self.handle_image_output(
            outputs,
            target_dimensions=(width, height),
        )
        if not image_storage_path:
            return

        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=image_storage_path,
            thumbnail_storage_path=image_storage_path,
            prompt=self.positive_prompt,
        )

    def _inject_edit_workflow(
        self,
        workflow_data,
        prompt,
        image_filename,
        denoise_strength,
        second_image_filename=None,
        aspect_ratio="1:1",
        seed=None,
    ):
        wf = self._prepare_workflow(workflow_data)
        width, height = self._get_dimensions(aspect_ratio)
        self._apply_common_workflow_inputs(
            wf,
            prompt,
            width,
            height,
            seed=seed,
            denoise_strength=denoise_strength,
            image_filename=image_filename,
        )
        return wf
