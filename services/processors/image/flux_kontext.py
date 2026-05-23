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
from PIL import Image, ImageFilter, ImageStat


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
        default_steps = 20
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

    def _prune_reference_mask_components(self, mask):
        """Remove detached decoration/shadow blobs while keeping real body parts."""
        mask = mask.convert("L")
        width, height = mask.size
        pixels = mask.load()
        visited = bytearray(width * height)
        components = []

        for start_y in range(height):
            row_offset = start_y * width
            for start_x in range(width):
                start_index = row_offset + start_x
                if visited[start_index] or pixels[start_x, start_y] == 0:
                    continue

                stack = [(start_x, start_y)]
                visited[start_index] = 1
                points = []
                min_x = max_x = start_x
                min_y = max_y = start_y

                while stack:
                    x, y = stack.pop()
                    points.append((x, y))
                    if x < min_x:
                        min_x = x
                    elif x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    elif y > max_y:
                        max_y = y

                    for next_x, next_y in (
                        (x - 1, y),
                        (x + 1, y),
                        (x, y - 1),
                        (x, y + 1),
                    ):
                        if (
                            next_x < 0
                            or next_y < 0
                            or next_x >= width
                            or next_y >= height
                        ):
                            continue
                        next_index = next_y * width + next_x
                        if visited[next_index] or pixels[next_x, next_y] == 0:
                            continue
                        visited[next_index] = 1
                        stack.append((next_x, next_y))

                components.append(
                    {
                        "area": len(points),
                        "bbox": (min_x, min_y, max_x, max_y),
                        "points": points,
                    }
                )

        if len(components) <= 1:
            return mask

        def is_flat_shadow(component):
            min_x, min_y, max_x, max_y = component["bbox"]
            component_width = max_x - min_x + 1
            component_height = max_y - min_y + 1
            return (
                component_width > component_height * 2.8
                and component_height < height * 0.16
                and min_y > height * 0.52
            )

        components.sort(key=lambda item: item["area"], reverse=True)
        main = next(
            (component for component in components if not is_flat_shadow(component)),
            components[0],
        )
        main_min_x, main_min_y, main_max_x, main_max_y = main["bbox"]
        main_area = max(1, main["area"])
        main_margin = max(8, round(max(width, height) * 0.08))

        def is_near_main(component):
            min_x, min_y, max_x, max_y = component["bbox"]
            return not (
                max_x < main_min_x - main_margin
                or min_x > main_max_x + main_margin
                or max_y < main_min_y - main_margin
                or min_y > main_max_y + main_margin
            )

        keep = Image.new("L", mask.size, 0)
        keep_pixels = keep.load()
        kept_count = 0
        removed_count = 0
        for component in components:
            area = component["area"]
            area_ratio = area / main_area
            should_keep = (
                component is main
                or (
                    not is_flat_shadow(component)
                    and (
                        area_ratio >= 0.035
                        or (area_ratio >= 0.025 and is_near_main(component))
                    )
                )
            )
            if should_keep:
                kept_count += 1
                for x, y in component["points"]:
                    keep_pixels[x, y] = 255
            else:
                removed_count += 1

        if removed_count:
            logging.info(
                "Pruned %s detached reference matte components; kept %s component(s).",
                removed_count,
                kept_count,
            )
        return keep

    def _remove_outer_light_fringe(self, image, mask):
        image = image.convert("RGBA")
        rgb = image.convert("RGB")
        mask = mask.convert("L")
        width, height = mask.size
        rgb_pixels = rgb.load()
        mask_pixels = mask.load()
        visited = bytearray(width * height)
        stack = []
        foreground_area = 0

        def is_light_fringe_pixel(x, y):
            r, g, b = rgb_pixels[x, y]
            max_channel = max(r, g, b)
            min_channel = min(r, g, b)
            return max_channel >= 218 and max_channel - min_channel <= 42

        def touches_background(x, y):
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                return True
            return (
                mask_pixels[x - 1, y] == 0
                or mask_pixels[x + 1, y] == 0
                or mask_pixels[x, y - 1] == 0
                or mask_pixels[x, y + 1] == 0
            )

        for y in range(height):
            for x in range(width):
                if mask_pixels[x, y] == 0:
                    continue
                foreground_area += 1
                if is_light_fringe_pixel(x, y) and touches_background(x, y):
                    index = y * width + x
                    visited[index] = 1
                    stack.append((x, y))

        if not stack or foreground_area == 0:
            return mask

        remove_points = []
        while stack:
            x, y = stack.pop()
            remove_points.append((x, y))
            for next_x, next_y in (
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
            ):
                if (
                    next_x < 0
                    or next_y < 0
                    or next_x >= width
                    or next_y >= height
                ):
                    continue
                index = next_y * width + next_x
                if (
                    visited[index]
                    or mask_pixels[next_x, next_y] == 0
                    or not is_light_fringe_pixel(next_x, next_y)
                ):
                    continue
                visited[index] = 1
                stack.append((next_x, next_y))

        if len(remove_points) / foreground_area > 0.35:
            logging.info(
                "Skipped light fringe cleanup because it would remove %.1f%% of the reference mask.",
                (len(remove_points) / foreground_area) * 100,
            )
            return mask

        for x, y in remove_points:
            mask_pixels[x, y] = 0
        if remove_points:
            logging.info(
                "Removed %s outer light fringe pixels from reference mask.",
                len(remove_points),
            )
        return mask

    def _remove_bottom_shadow_fringe(self, image, mask):
        image = image.convert("RGBA")
        rgb = image.convert("RGB")
        mask = mask.convert("L")
        width, height = mask.size
        rgb_pixels = rgb.load()
        mask_pixels = mask.load()
        visited = bytearray(width * height)
        stack = []
        foreground_area = 0
        min_shadow_y = round(height * 0.68)
        shadow_seed_y = height - max(3, round(height * 0.035))

        def is_shadow_pixel(x, y):
            if y < min_shadow_y:
                return False
            r, g, b = rgb_pixels[x, y]
            max_channel = max(r, g, b)
            min_channel = min(r, g, b)
            return 95 <= max_channel <= 248 and max_channel - min_channel <= 72

        for y in range(height):
            for x in range(width):
                if mask_pixels[x, y] == 0:
                    continue
                foreground_area += 1
                if y >= shadow_seed_y and is_shadow_pixel(x, y):
                    index = y * width + x
                    visited[index] = 1
                    stack.append((x, y))

        if not stack or foreground_area == 0:
            return mask

        remove_points = []
        while stack:
            x, y = stack.pop()
            remove_points.append((x, y))
            for next_x, next_y in (
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
            ):
                if (
                    next_x < 0
                    or next_y < 0
                    or next_x >= width
                    or next_y >= height
                ):
                    continue
                index = next_y * width + next_x
                if (
                    visited[index]
                    or mask_pixels[next_x, next_y] == 0
                    or not is_shadow_pixel(next_x, next_y)
                ):
                    continue
                visited[index] = 1
                stack.append((next_x, next_y))

        min_remove_y = min(y for _, y in remove_points)
        max_remove_y = max(y for _, y in remove_points)
        if max_remove_y - min_remove_y + 1 > height * 0.18:
            logging.info(
                "Skipped bottom shadow cleanup because the candidate region is too tall for a floor shadow.",
            )
            return mask

        if len(remove_points) / foreground_area > 0.28:
            logging.info(
                "Skipped bottom shadow cleanup because it would remove %.1f%% of the reference mask.",
                (len(remove_points) / foreground_area) * 100,
            )
            return mask

        for x, y in remove_points:
            mask_pixels[x, y] = 0
        if remove_points:
            logging.info(
                "Removed %s bottom shadow pixels from reference mask.",
                len(remove_points),
            )
        return mask

    def _clean_reference_alpha(self, image, alpha=None):
        image = image.convert("RGBA")
        alpha = alpha or image.getchannel("A")
        mask = alpha.point(lambda value: 255 if value > 96 else 0)
        mask = mask.filter(ImageFilter.MedianFilter(3))
        # Close tiny gaps, then shrink the outside edge slightly. This removes
        # pale matte halos, sticker borders, and soft pedestal shadows before
        # Flux sees the temporary reference placement.
        mask = mask.filter(ImageFilter.MaxFilter(3))
        mask = mask.filter(ImageFilter.MinFilter(5))
        mask = self._prune_reference_mask_components(mask)
        mask = self._remove_outer_light_fringe(image, mask)
        mask = self._remove_bottom_shadow_fringe(image, mask)
        if not mask.getbbox():
            return image

        cutout = image.copy()
        cutout.putalpha(mask)
        cropped = self._crop_alpha_bounds(cutout)
        cropped_mask = self._remove_bottom_shadow_fringe(
            cropped,
            cropped.getchannel("A"),
        )
        cropped.putalpha(cropped_mask)
        return self._crop_alpha_bounds(cropped)

    def _build_connected_matte_foreground_mask(self, image, matte):
        """Keep pixels not connected to the flat edge matte.

        A global color-difference matte can delete pale character fills when the
        character color is close to the background. Flooding from the image edge
        only removes background-colored regions that are physically connected to
        the matte, preserving enclosed light bodies and costume areas.
        """
        rgb = image.convert("RGB")
        width, height = rgb.size
        pixels = rgb.load()
        matte_r, matte_g, matte_b = matte
        threshold = 28

        def is_matte_like(x, y):
            r, g, b = pixels[x, y]
            return (
                abs(r - matte_r) <= threshold
                and abs(g - matte_g) <= threshold
                and abs(b - matte_b) <= threshold
            )

        background = bytearray(width * height)
        stack = []

        def add_edge_pixel(x, y):
            index = y * width + x
            if background[index] or not is_matte_like(x, y):
                return
            background[index] = 1
            stack.append((x, y))

        for x in range(width):
            add_edge_pixel(x, 0)
            add_edge_pixel(x, height - 1)
        for y in range(height):
            add_edge_pixel(0, y)
            add_edge_pixel(width - 1, y)

        while stack:
            x, y = stack.pop()
            for next_x, next_y in (
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
            ):
                if (
                    next_x < 0
                    or next_y < 0
                    or next_x >= width
                    or next_y >= height
                ):
                    continue
                index = next_y * width + next_x
                if background[index] or not is_matte_like(next_x, next_y):
                    continue
                background[index] = 1
                stack.append((next_x, next_y))

        mask = Image.new("L", rgb.size, 0)
        mask_pixels = mask.load()
        for y in range(height):
            row_offset = y * width
            for x in range(width):
                if not background[row_offset + x]:
                    mask_pixels[x, y] = 255
        return mask

    def _remove_flat_matte_background(self, image):
        """Convert common pale character-reference mattes into transparency."""
        image = image.convert("RGBA")
        alpha = image.getchannel("A")
        full_bbox = (0, 0, image.width, image.height)
        if alpha.getbbox() and alpha.getbbox() != full_bbox:
            return self._clean_reference_alpha(image, alpha)

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

        mask = self._build_connected_matte_foreground_mask(image, matte)
        mask = mask.filter(ImageFilter.MedianFilter(3))
        mask = mask.filter(ImageFilter.MaxFilter(3))
        mask = mask.filter(ImageFilter.MinFilter(5))
        mask = self._prune_reference_mask_components(mask)
        mask = self._remove_outer_light_fringe(image, mask)
        mask = self._remove_bottom_shadow_fringe(image, mask)
        if not mask.getbbox():
            return image

        return self._clean_reference_alpha(image, mask)

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
            0.92 if self._has_additional_source_image_input() else 0.65
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
                "Source combines a wide scene plate and temporary character identity "
                "guides. Treat the pasted guide pixels as layout and identity reference "
                "only, not final artwork. Preserve the wide source composition, keep "
                "every required visible character full-body and uncropped, blend them "
                "into one seamless final frame, keep each guide identity separate, "
                "redraw each character natively into the scene lighting and action, "
                "erase any sticker border, white halo, flat-matte fringe, pedestal "
                "shadow, reference-floor blob, or pasted edge, and avoid panels or "
                "portrait crops. "
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
