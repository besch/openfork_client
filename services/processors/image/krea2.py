"""
Krea 2 image processor.

Krea 2 Turbo is exposed through standard ComfyUI text-to-image workflows using
the Comfy-Org repackaged FP8 Turbo weights or low-VRAM GGUF weights.
"""

import copy
import logging
import random

from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import ImageOutputHandler


KREA2_DEFAULT_LONG_EDGE = 1024
KREA2_MAX_LONG_EDGE = 1536
KREA2_16GB_MAX_LONG_EDGE = 1024
KREA2_8GB_DEFAULT_LONG_EDGE = 768
KREA2_8GB_MAX_LONG_EDGE = 768
KREA2_DEFAULT_STEPS = 8
KREA2_DEFAULT_CFG = 1.0
KREA2_DEFAULT_SHIFT = 1.15


def _coerce_int(value, default, minimum=None, maximum=None):
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        coerced = default
    if minimum is not None:
        coerced = max(minimum, coerced)
    if maximum is not None:
        coerced = min(maximum, coerced)
    return coerced


def _coerce_float(value, default, minimum=None, maximum=None):
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        coerced = default
    if minimum is not None:
        coerced = max(minimum, coerced)
    if maximum is not None:
        coerced = min(maximum, coerced)
    return coerced


def _round_to_multiple(value, multiple=16):
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def clamp_krea2_steps(value):
    return _coerce_int(value, KREA2_DEFAULT_STEPS, minimum=1, maximum=24)


def clamp_krea2_cfg(value):
    return _coerce_float(value, KREA2_DEFAULT_CFG, minimum=0.0, maximum=4.0)


class Krea2TextToImageProcessor(ComfyUIProcessor, ImageOutputHandler):
    """Processor for Krea 2 Turbo text-to-image generation."""

    def process(self):
        if not self.job:
            self._fail_job(
                "Job object is None for Krea2TextToImageProcessor. Cannot proceed."
            )
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {}) or {}
        width, height = self._get_dimensions(
            inputs.get("aspect_ratio", "1:1"),
            requested_long_edge=(
                inputs.get("max_long_edge")
                or inputs.get("long_edge")
                or inputs.get("target_long_edge")
            ),
            max_long_edge=self._max_long_edge_for_tier(),
            default_long_edge=self._default_long_edge_for_tier(),
        )

        wf_ready = self._prepare_workflow(workflow_data)
        self._apply_inputs_to_workflow(
            wf_ready,
            prompt=self.positive_prompt,
            width=width,
            height=height,
            inputs=inputs,
        )

        outputs = self._trigger_and_get_output({"prompt": wf_ready})
        if not outputs:
            return

        image_storage_path = self.handle_image_output(
            outputs,
            target_dimensions=(width, height),
        )
        if not image_storage_path:
            return

        completion_metadata = dict(self.job.get("completion_metadata") or {})
        completion_metadata.update(
            {
                "processor": "Krea2TextToImageProcessor",
                "width": width,
                "height": height,
            }
        )

        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=image_storage_path,
            thumbnail_storage_path=image_storage_path,
            prompt=self.positive_prompt,
            completion_metadata=completion_metadata,
        )

    @staticmethod
    def _prepare_workflow(workflow_data):
        return copy.deepcopy(workflow_data.get("prompt", workflow_data))

    def _tier_key(self):
        service_type = getattr(self.client, "active_service_type", "") or ""
        job_service_type = self.job.get("service_type", "") if self.job else ""
        workflow_type = self.workflow_type or ""
        return f"{workflow_type} {service_type} {job_service_type}".lower()

    def _max_long_edge_for_tier(self):
        tier_key = self._tier_key()
        if "8gb" in tier_key:
            return KREA2_8GB_MAX_LONG_EDGE
        if "16gb" in tier_key:
            return KREA2_16GB_MAX_LONG_EDGE
        return KREA2_MAX_LONG_EDGE

    def _default_long_edge_for_tier(self):
        tier_key = self._tier_key()
        if "8gb" in tier_key:
            return KREA2_8GB_DEFAULT_LONG_EDGE
        return KREA2_DEFAULT_LONG_EDGE

    @classmethod
    def _resolve_long_edge(
        cls,
        requested_long_edge=None,
        max_long_edge=KREA2_MAX_LONG_EDGE,
        default_long_edge=KREA2_DEFAULT_LONG_EDGE,
    ):
        return _coerce_int(
            requested_long_edge,
            min(default_long_edge, max_long_edge),
            minimum=512,
            maximum=max_long_edge,
        )

    @classmethod
    def _get_dimensions(
        cls,
        aspect_ratio,
        requested_long_edge=None,
        max_long_edge=KREA2_MAX_LONG_EDGE,
        default_long_edge=KREA2_DEFAULT_LONG_EDGE,
    ):
        long_edge = cls._resolve_long_edge(
            requested_long_edge,
            max_long_edge=max_long_edge,
            default_long_edge=default_long_edge,
        )
        ratio_map = {
            "1:1": (long_edge, long_edge),
            "16:9": (long_edge, _round_to_multiple(long_edge * 9 / 16)),
            "9:16": (_round_to_multiple(long_edge * 9 / 16), long_edge),
            "4:3": (long_edge, _round_to_multiple(long_edge * 3 / 4)),
            "3:4": (_round_to_multiple(long_edge * 3 / 4), long_edge),
            "3:2": (long_edge, _round_to_multiple(long_edge * 2 / 3)),
            "2:3": (_round_to_multiple(long_edge * 2 / 3), long_edge),
            "21:9": (long_edge, _round_to_multiple(long_edge * 9 / 21)),
        }
        return ratio_map.get(aspect_ratio or "1:1", (long_edge, long_edge))

    def _seed(self, value):
        if value is None:
            return random.randint(0, 2**63 - 1)
        return _coerce_int(value, random.randint(0, 2**63 - 1), minimum=0)

    @staticmethod
    def _is_gguf_workflow(api_graph):
        for node in api_graph.values():
            class_type = node.get("class_type", "")
            node_inputs = node.get("inputs", {})
            model_name = str(
                node_inputs.get("unet_name") or node_inputs.get("model_name") or ""
            ).lower()
            if class_type in ("UnetLoaderGGUF", "UnetLoaderGGUFAdvanced"):
                return True
            if model_name.endswith(".gguf"):
                return True
        return False

    def _apply_inputs_to_workflow(self, api_graph, prompt, width, height, inputs):
        is_gguf_workflow = self._is_gguf_workflow(api_graph)
        steps = clamp_krea2_steps(inputs.get("steps", inputs.get("num_steps")))
        cfg = clamp_krea2_cfg(
            inputs.get("cfg", inputs.get("cfg_scale", inputs.get("guidance_scale")))
        )
        sampler_name = inputs.get("sampler_name") or inputs.get("sampler") or "euler"
        scheduler = inputs.get("scheduler") or "simple"
        seed = self._seed(inputs.get("seed"))
        shift = _coerce_float(
            inputs.get("shift", inputs.get("flow_shift")),
            KREA2_DEFAULT_SHIFT,
            minimum=0.0,
            maximum=5.0,
        )

        for node in api_graph.values():
            class_type = node.get("class_type", "")
            node_inputs = node.get("inputs", {})

            if class_type == "CLIPTextEncode":
                if node_inputs.get("text") not in ("", None):
                    node_inputs["text"] = prompt
                elif inputs.get("negative_prompt"):
                    node_inputs["text"] = inputs.get("negative_prompt")
            elif class_type in ("EmptyLatentImage", "EmptySD3LatentImage"):
                node_inputs["width"] = width
                node_inputs["height"] = height
                node_inputs["batch_size"] = 1
            elif class_type == "ModelSamplingAuraFlow":
                node_inputs["shift"] = shift
            elif class_type == "KSampler":
                node_inputs["seed"] = seed
                node_inputs["steps"] = steps
                node_inputs["cfg"] = cfg
                node_inputs["sampler_name"] = sampler_name
                node_inputs["scheduler"] = scheduler
                node_inputs["denoise"] = 1.0
            elif class_type == "CLIPLoader":
                node_inputs.setdefault("clip_name", "qwen3vl_4b_fp8_scaled.safetensors")
                node_inputs.setdefault("type", "krea2")
                if is_gguf_workflow:
                    node_inputs["device"] = "cpu"
                else:
                    node_inputs.setdefault("device", "default")
            elif class_type == "VAELoader":
                node_inputs.setdefault("vae_name", "qwen_image_vae.safetensors")
            elif class_type == "UNETLoader":
                node_inputs.setdefault("unet_name", "krea2_turbo_fp8_scaled.safetensors")
                node_inputs.setdefault("weight_dtype", "default")
            elif class_type in ("UnetLoaderGGUF", "UnetLoaderGGUFAdvanced"):
                node_inputs.setdefault("unet_name", "krea2_turbo-Q3_K_M.gguf")
                if class_type == "UnetLoaderGGUFAdvanced":
                    node_inputs.setdefault("dequant_dtype", "float16")
                    node_inputs.setdefault("patch_dtype", "float16")
                    node_inputs.setdefault("patch_on_device", False)
            elif class_type == "LoadDiffusionModel":
                node_inputs.setdefault("model_name", "krea2_turbo_fp8_scaled.safetensors")
                node_inputs.setdefault("weight_dtype", "default")

        logging.info(
            "Krea 2 Turbo configured - size: %sx%s, steps: %s, cfg: %s, shift: %s, seed: %s",
            width,
            height,
            steps,
            cfg,
            shift,
            seed,
        )
