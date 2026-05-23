"""
FLUX.1 Kontext [dev] processors.

The workflows use GGUF quantized model weights for RTX 30/40 compatibility and
low-VRAM operation, while keeping the standard ComfyUI Flux Kontext sampler graph.
"""

import copy
import logging
import os
import random
import uuid

from config import SUPABASE_URL
from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import ImageOutputHandler
from services.processors.image.qwen import QwenImageEditProcessor
from PIL import Image, ImageChops, ImageFilter, ImageStat


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
                inputs.setdefault("upscale_method", "lanczos")
            elif class_type == "FluxKontextImageScale" and self._is_8gb_tier():
                # ComfyUI's FluxKontextImageScale snaps 16:9 images to large
                # Kontext-preferred resolutions, which can exceed 8GB VRAM.
                # Keep the reference latent aligned with the requested job size.
                node["class_type"] = "ImageScaleToMaxDimension"
                node["inputs"] = {
                    "image": inputs.get("image"),
                    "upscale_method": "lanczos",
                    "largest_size": max(width, height),
                }
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

    def _compact_flux_prompt(self, prompt, limit=1500):
        text = " ".join(str(prompt or "").split())
        if len(text) <= limit:
            return text

        suffix = (
            " No visible text, letters, numbers, labels, signs, UI, logos, "
            "captions, subtitles, title cards, watermarks, or readable symbols."
        )
        head_limit = max(200, limit - len(suffix) - 3)
        return text[:head_limit].rsplit(" ", 1)[0].rstrip(" .,;:") + "." + suffix


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
            self._compact_flux_prompt(self.positive_prompt),
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

    def _fit_cover(self, image, width, height):
        scale = max(width / image.width, height / image.height)
        resized = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        left = max(0, (resized.width - width) // 2)
        top = max(0, (resized.height - height) // 2)
        return resized.crop((left, top, left + width, top + height))

    def _crop_alpha_bounds(self, image):
        alpha = image.getchannel("A")
        bbox = alpha.getbbox()
        return image.crop(bbox) if bbox else image

    def _remove_flat_matte_background(self, image):
        """Convert common pale character-reference mattes into transparency."""
        image = image.convert("RGBA")
        alpha = image.getchannel("A")
        full_bbox = (0, 0, image.width, image.height)
        if alpha.getbbox() and alpha.getbbox() != full_bbox:
            return self._crop_alpha_bounds(image)

        corner = max(8, min(32, image.width // 12, image.height // 12))
        samples = [
            image.crop((0, 0, corner, corner)).convert("RGB"),
            image.crop((image.width - corner, 0, image.width, corner)).convert("RGB"),
            image.crop((0, image.height - corner, corner, image.height)).convert("RGB"),
            image.crop((image.width - corner, image.height - corner, image.width, image.height)).convert("RGB"),
        ]
        channels = [[], [], []]
        for sample in samples:
            mean = ImageStat.Stat(sample).mean
            for idx in range(3):
                channels[idx].append(mean[idx])
        matte = tuple(int(sum(values) / len(values)) for values in channels)

        matte_image = Image.new("RGB", image.size, matte)
        diff = ImageChops.difference(image.convert("RGB"), matte_image).convert("L")
        mask = diff.point(lambda value: 255 if value > 24 else 0)
        mask = mask.filter(ImageFilter.MedianFilter(3))
        mask = mask.filter(ImageFilter.MinFilter(3))
        mask = mask.filter(ImageFilter.MaxFilter(5))
        if not mask.getbbox():
            return image

        cutout = image.copy()
        cutout.putalpha(mask)
        return self._crop_alpha_bounds(cutout)

    def _normalize_image_input_list(self, value):
        if not value:
            return []
        if isinstance(value, (list, tuple)):
            normalized = []
            for item in value:
                normalized.extend(self._normalize_image_input_list(item))
            return normalized
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            return [part.strip() for part in text.split(",") if part.strip()]
        return [str(value)]

    def _plural_reference_storage_paths(self):
        inputs = self.job.get("inputs", {})
        paths = []
        for key in (
            "reference_image_2_storage_paths",
            "reference_image2_storage_paths",
            "second_reference_image_storage_paths",
            "additional_reference_image_storage_paths",
        ):
            paths.extend(self._normalize_image_input_list(inputs.get(key)))
        return list(dict.fromkeys(paths))

    def _has_additional_source_image_input(self):
        return self._has_second_source_image_input() or bool(
            self._plural_reference_storage_paths()
        )

    def _download_reference_storage_image(self, storage_path):
        bucket = self.job.get("bucket", "projects_public")
        downloaded_path = self.orchestrator_service.download_storage_asset(
            bucket,
            storage_path,
            self.client.input_dir,
        )
        if not downloaded_path:
            source_url = f"{os.environ.get('SUPABASE_URL', self.client.config.get('SUPABASE_URL', SUPABASE_URL))}/storage/v1/object/public/{bucket}/{storage_path}"
            logging.info("Downloading Flux reference image from: %s", source_url)
            downloaded_path = self.orchestrator_service.download_asset_by_url(
                source_url,
                self.client.input_dir,
            )
        if downloaded_path:
            filename = os.path.basename(downloaded_path)
            logging.info("Downloaded Flux reference image: %s", filename)
            return filename
        return None

    def _get_additional_source_images(self):
        filenames = []
        for storage_path in self._plural_reference_storage_paths():
            filename = self._download_reference_storage_image(storage_path)
            if filename:
                filenames.append(filename)

        if not filenames:
            single_filename = self._get_second_source_image()
            if single_filename:
                filenames.append(single_filename)

        deduped = []
        for filename in filenames:
            if filename not in deduped:
                deduped.append(filename)

        if self._has_additional_source_image_input() and not deduped:
            self._fail_job("Failed to load additional edit images for Flux Kontext workflow.")
        return deduped

    def _compose_reference_source(self, scene_filename, reference_filenames, width, height):
        """Create one Flux-compatible source image from scene + identity references.

        The public Flux Kontext edit workflow has a single LoadImage node. For 8GB
        combine/edit jobs, precomposing the scene plate and transparent character
        references gives the model the sources without requiring a multi-input graph.
        """
        if isinstance(reference_filenames, str):
            reference_filenames = [reference_filenames]
        reference_filenames = [filename for filename in reference_filenames if filename]
        if not reference_filenames:
            self._fail_job("No Flux reference images were available to compose.")
            return None

        scene_path = os.path.join(self.client.input_dir, scene_filename)
        try:
            scene = Image.open(scene_path).convert("RGBA")
            references = [
                self._remove_flat_matte_background(
                    Image.open(os.path.join(self.client.input_dir, filename)).convert("RGBA")
                )
                for filename in reference_filenames
            ]

            canvas = self._fit_cover(scene, width, height)
            reference_count = len(references)
            max_reference_width = max(
                1,
                round(width * (0.34 if reference_count == 1 else 0.28 if reference_count == 2 else 0.22)),
            )
            max_reference_height = max(
                1,
                round(height * (0.82 if reference_count == 1 else 0.76)),
            )

            if reference_count == 1:
                try:
                    target_index = int(
                        (self.job.get("completion_metadata") or {}).get("target_index", 0)
                    )
                except (TypeError, ValueError):
                    target_index = 0
                anchors = (0.08, 0.62, 0.36)
                center_positions = [
                    anchors[target_index % len(anchors)] + max_reference_width / max(width, 1) / 2
                ]
            elif reference_count == 2:
                center_positions = [0.32, 0.68]
            elif reference_count == 3:
                center_positions = [0.18, 0.5, 0.82]
            else:
                margin = 0.13
                span = 1 - margin * 2
                center_positions = [
                    margin + span * (index + 0.5) / reference_count
                    for index in range(reference_count)
                ]

            for index, reference in enumerate(references):
                reference = reference.copy()
                reference.thumbnail(
                    (max_reference_width, max_reference_height),
                    Image.Resampling.LANCZOS,
                )
                center_x = center_positions[min(index, len(center_positions) - 1)]
                x = round(width * center_x - reference.width / 2)
                y = height - reference.height - max(8, round(height * 0.04))
                x = max(4, min(x, width - reference.width - 4))

                shadow_alpha = reference.getchannel("A").filter(ImageFilter.GaussianBlur(5))
                shadow = Image.new("RGBA", reference.size, (0, 0, 0, 70))
                shadow.putalpha(shadow_alpha)
                canvas.alpha_composite(shadow, (x + 3, y + 4))
                canvas.alpha_composite(reference, (x, y))

            output_name = f"flux_reference_source_{uuid.uuid4().hex[:8]}.png"
            output_path = os.path.join(self.client.input_dir, output_name)
            canvas.convert("RGB").save(output_path, "PNG", optimize=True)
            logging.info(
                "Created Flux reference source %s from scene %s and references %s at %sx%s",
                output_name,
                scene_filename,
                ", ".join(reference_filenames),
                width,
                height,
            )
            return output_name
        except Exception as exc:
            self._fail_job(f"Failed to compose Flux reference source: {exc}")
            return None

    def process(self):
        if not self.job:
            self._fail_job("Job object is None for FluxKontextEditProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        aspect_ratio = inputs.get("aspect_ratio", "1:1")
        default_denoise_strength = (
            0.78 if self._has_additional_source_image_input() else 0.65
        )
        denoise_strength = inputs.get(
            "denoise_strength",
            inputs.get("strength", default_denoise_strength),
        )
        requested_long_edge = self._requested_long_edge(inputs)
        width, height = self._get_dimensions(
            aspect_ratio,
            requested_long_edge=requested_long_edge,
        )

        source_image_filename = self._get_source_image()
        if not source_image_filename:
            return
        additional_images_requested = self._has_additional_source_image_input()
        additional_image_filenames = self._get_additional_source_images()
        if additional_images_requested and not additional_image_filenames:
            return

        prompt = self._compact_flux_prompt(self.positive_prompt)
        if additional_image_filenames:
            composed_filename = self._compose_reference_source(
                source_image_filename,
                additional_image_filenames,
                width,
                height,
            )
            if not composed_filename:
                return
            source_image_filename = composed_filename
            prompt = (
                "Source combines a wide scene plate and character cutouts. Preserve "
                "the wide source composition, keep every required visible character "
                "full-body and uncropped, blend them into one seamless final frame, "
                "keep each cutout identity separate, repaint the cutouts into the "
                "scene lighting, erase any sticker border, white halo, flat-matte "
                "fringe, pedestal shadow, reference-floor blob, or pasted edge, and "
                "avoid panels or portrait crops. "
                + prompt
            )
            prompt = self._compact_flux_prompt(prompt)

        self._copy_image_to_container(source_image_filename)

        wf_ready = self._inject_edit_workflow(
            workflow_data,
            prompt,
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
