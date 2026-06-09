"""
Ideogram 4 image processor.

Communicates with the Ideogram 4 REST API (FastAPI server on port 8000) for
text-to-image generation.
"""

import json
import logging
import os
import time
from typing import Any, Optional

import requests

from config import TimeoutConfig
from services.orchestrator_service import TokenExpiredError
from services.processors.base import BaseJobProcessor
from services.processors.output_handlers import ImageOutputHandler
from services.processors.rest_recovery import (
    get_processor_service_type,
    poll_rest_job_with_clean_exit,
    recover_output_from_clean_container_exit,
)


class Ideogram4ImageProcessor(BaseJobProcessor, ImageOutputHandler):
    API_HOST = "127.0.0.1"
    API_PORT = 8000
    POLL_INTERVAL = 3
    API_WAIT_TIMEOUT = int(os.environ.get("IDEOGRAM4_API_WAIT_TIMEOUT", "1800"))
    DEFAULT_SAMPLER_PRESET = os.environ.get(
        "IDEOGRAM4_DEFAULT_SAMPLER_PRESET", "V4_QUALITY_48"
    )
    LOW_VRAM_SAMPLER_PRESET = os.environ.get(
        "IDEOGRAM4_16GB_SAMPLER_PRESET", "V4_DEFAULT_20"
    )
    LOW_VRAM_HARDWARE_MAX_MB = int(
        os.environ.get("IDEOGRAM4_LOW_VRAM_HARDWARE_MAX_MB", "17000")
    )
    MAX_WAIT_TIME = int(
        os.environ.get(
            "IDEOGRAM4_GENERATION_TIMEOUT",
            str(max(TimeoutConfig.WORKFLOW_TIMEOUT, 7200)),
        )
    )

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        api_host = os.environ.get("IDEOGRAM4_API_HOST", self.API_HOST)
        api_port = int(os.environ.get("IDEOGRAM4_API_PORT", self.API_PORT))
        self.api_base_url = f"http://{api_host}:{api_port}"
        self.session = requests.Session()
        self.session.trust_env = False
        self._api_startup_error: Optional[str] = None

    def process(self):
        if not self.job:
            self._fail_job("Job object is None. Cannot proceed.")
            return

        logging.info("Processing Ideogram 4 job %s", self.job_id)
        inputs = self.job.get("inputs") or {}

        if not self._wait_for_api(timeout=self.API_WAIT_TIMEOUT):
            self._fail_job(
                self._api_startup_error
                or "Ideogram 4 API / model did not become ready in time"
            )
            return

        job_id = self._submit_generation(inputs)
        if not job_id:
            self._fail_job("Failed to submit Ideogram 4 generation")
            return

        local_path = None
        try:
            result = self._poll_for_completion(job_id)

            if result.get("status") != "completed":
                error_msg = result.get("error", "Unknown error")
                self._fail_job(f"Ideogram 4 generation failed: {error_msg}")
                return

            local_path = self._download_output(job_id)
            if not local_path:
                self._fail_job("Failed to download Ideogram 4 output")
                return

            self._apply_transparent_background_if_requested(local_path)
            image_storage_path = self.orchestrator_service.upload_image_output(
                local_path, self.job_id
            )

            if image_storage_path:
                completion_metadata = self.job.get("completion_metadata") or {}
                completion_metadata.update(
                    {
                        "processor": "Ideogram4ImageProcessor",
                        "model": "ideogram4",
                        "seed": result.get("seed"),
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
                logging.info("Ideogram 4 job %s completed successfully", self.job_id)
            else:
                self._fail_job("Job completed, but image upload failed")

        except TokenExpiredError:
            raise
        except Exception as exc:
            logging.error(
                "Error processing Ideogram 4 job %s: %s",
                self.job_id,
                exc,
                exc_info=True,
            )
            self._fail_job(f"Error processing job: {exc}")
        finally:
            if local_path and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except OSError:
                    pass

    def _wait_for_api(self, timeout: int = 1800) -> bool:
        start_time = time.monotonic()
        last_log = -31
        logging.info(
            "Waiting for Ideogram 4 API at %s (timeout: %ss)...",
            self.api_base_url,
            timeout,
        )
        self._api_startup_error = None

        while time.monotonic() - start_time < timeout:
            if self.is_cancelled():
                logging.warning("Shutdown requested while waiting for API.")
                return False

            elapsed = int(time.monotonic() - start_time)

            try:
                response = self.session.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    api_status = data.get("status", "unknown")
                    api_error = data.get("error")

                    if api_status == "error":
                        self._api_startup_error = (
                            "Ideogram 4 model failed to load"
                            if not api_error
                            else f"Ideogram 4 model failed to load: {api_error}"
                        )
                        logging.error("%s (after %ss)", self._api_startup_error, elapsed)
                        return False

                    if data.get("model_loaded"):
                        logging.info(
                            "Ideogram 4 API ready after %ss (model: %s)",
                            elapsed,
                            data.get("model_id", "unknown"),
                        )
                        return True

                    if elapsed - last_log >= 30:
                        msg = (
                            "Ideogram 4 API reachable, model loading "
                            f"(status={api_status}, {elapsed}/{timeout}s)"
                        )
                        if api_error:
                            msg += f" | error: {api_error}"
                        logging.info(msg)
                        last_log = elapsed

            except requests.exceptions.RequestException:
                if elapsed - last_log >= 30:
                    logging.info(
                        "Ideogram 4 API not reachable yet (%s/%ss)",
                        elapsed,
                        timeout,
                    )
                    last_log = elapsed

            time.sleep(5)

        logging.error("Ideogram 4 API did not become ready within %ss", timeout)
        return False

    def _submit_generation(self, inputs: dict) -> Optional[str]:
        try:
            seed = inputs.get("seed")
            width, height = self._resolve_dimensions(inputs.get("aspect_ratio"))
            sampler_preset = self._resolve_sampler_preset(inputs)
            prompt, structured_prompt = self._resolve_prompt(inputs)
            payload = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "sampler_preset": sampler_preset,
                "warn_on_caption_issues": not structured_prompt,
            }
            logging.info(
                "Submitting Ideogram 4 generation size=%sx%s preset=%s tier=%s structured=%s",
                width,
                height,
                sampler_preset,
                self._service_type() or "unknown",
                structured_prompt,
            )
            if seed is not None:
                payload["seed"] = int(seed)
            if inputs.get("use_magic_prompt") is not None:
                payload["use_magic_prompt"] = bool(inputs.get("use_magic_prompt"))

            response = self.session.post(
                f"{self.api_base_url}/generate", json=payload, timeout=30
            )
            if response.status_code == 200:
                return response.json().get("job_id")
            logging.error(
                "Ideogram 4 generate failed: %s %s",
                response.status_code,
                response.text,
            )
            return None
        except requests.exceptions.RequestException as exc:
            logging.error("Ideogram 4 generate request error: %s", exc)
            return None

    def _resolve_prompt(self, inputs: dict) -> tuple[str, bool]:
        structured_prompt = (
            inputs.get("ideogram_json_prompt")
            or inputs.get("ideogram_structured_prompt")
            or inputs.get("json_prompt")
        )
        if structured_prompt:
            if isinstance(structured_prompt, str):
                return structured_prompt, True
            return json.dumps(structured_prompt, separators=(",", ":")), True

        layers = inputs.get("ideogram_layers") or inputs.get("ideogramLayers")
        if isinstance(layers, list) and layers:
            return self._build_structured_prompt(inputs, layers), True

        return self.positive_prompt, False

    def _build_structured_prompt(
        self, inputs: dict, layers: list[dict[str, Any]]
    ) -> str:
        elements = []
        for index, layer in enumerate(layers):
            if not isinstance(layer, dict):
                continue
            element = self._convert_layer(layer, index)
            if element:
                elements.append(element)

        if not elements:
            return self.positive_prompt

        prompt = {
            "high_level_description": self.positive_prompt,
            "style_description": {
                "aesthetics": inputs.get("ideogram_aesthetics")
                or inputs.get("style")
                or "polished, coherent, production-ready",
                "lighting": inputs.get("ideogram_lighting")
                or "balanced cinematic lighting",
                "medium": inputs.get("ideogram_medium") or "digital image",
                "art_style": inputs.get("ideogram_art_style")
                or inputs.get("style")
                or "clean contemporary visual design",
            },
            "compositional_deconstruction": {
                "background": inputs.get("ideogram_background")
                or "A coherent background that supports the listed foreground layers.",
                "elements": elements,
            },
        }
        palette = self._normalize_palette(
            inputs.get("ideogram_color_palette") or inputs.get("color_palette")
        )
        if palette:
            prompt["style_description"]["color_palette"] = palette
        return json.dumps(prompt, separators=(",", ":"))

    def _convert_layer(
        self, layer: dict[str, Any], index: int
    ) -> Optional[dict[str, Any]]:
        bbox = self._layer_bbox(layer)
        if not bbox:
            return None

        layer_type = str(layer.get("type") or layer.get("layer_type") or "obj").lower()
        text = str(layer.get("text") or "").strip()
        desc = str(
            layer.get("desc")
            or layer.get("description")
            or (f'readable text "{text}"' if text else f"visual element {index + 1}")
        ).strip()

        element: dict[str, Any] = {
            "type": "text" if layer_type == "text" else "obj",
            "bbox": bbox,
            "desc": desc,
        }
        if text:
            element["text"] = text
        palette = self._normalize_palette(layer.get("palette"))
        if palette:
            element["color_palette"] = palette
        return element

    def _layer_bbox(self, layer: dict[str, Any]) -> Optional[list[int]]:
        bbox = layer.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            values = [self._coerce_float(value) for value in bbox]
            if all(value is not None for value in values):
                nums = [value for value in values if value is not None]
                if max(nums) <= 1:
                    nums = [value * 1000 for value in nums]
                return self._normalize_bbox(nums)

        coords = [
            self._coerce_float(layer.get("x")),
            self._coerce_float(layer.get("y")),
            self._coerce_float(layer.get("w") or layer.get("width")),
            self._coerce_float(layer.get("h") or layer.get("height")),
        ]
        if any(value is None for value in coords):
            return None

        x, y, w, h = [value for value in coords if value is not None]
        if max(x, y, w, h) <= 1:
            x, y, w, h = x * 1000, y * 1000, w * 1000, h * 1000
        return self._normalize_bbox([y, x, y + h, x + w])

    @staticmethod
    def _normalize_bbox(values: list[float]) -> list[int]:
        y_min, x_min, y_max, x_max = [
            max(0, min(1000, int(round(value)))) for value in values
        ]
        if y_max <= y_min:
            y_max = min(1000, y_min + 1)
        if x_max <= x_min:
            x_max = min(1000, x_min + 1)
        return [y_min, x_min, y_max, x_max]

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_palette(value: Any) -> list[str]:
        if isinstance(value, str):
            raw_values = [item.strip() for item in value.split(",")]
        elif isinstance(value, list):
            raw_values = [str(item).strip() for item in value]
        else:
            return []
        return [item for item in raw_values if item][:12]

    def _poll_for_completion(self, api_job_id: str) -> dict:
        return poll_rest_job_with_clean_exit(
            self,
            self.api_base_url,
            api_job_id,
            poll_interval=self.POLL_INTERVAL,
            max_wait_time=self.MAX_WAIT_TIME,
            service_label="Ideogram 4",
            session=self.session,
        )

    def _download_output(self, api_job_id: str) -> Optional[str]:
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            local_path = os.path.join(self.cache_dir, f"{self.job_id}.png")

            response = self.session.get(
                f"{self.api_base_url}/output/{api_job_id}",
                timeout=60,
                stream=True,
            )
            if response.status_code == 200:
                with open(local_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        handle.write(chunk)
                return local_path
            logging.error(
                "Ideogram 4 download failed: %s %s",
                response.status_code,
                response.text,
            )
            return recover_output_from_clean_container_exit(
                self,
                local_path,
                container_output_path=f"/app/output/{api_job_id}.png",
                extensions=(".png", ".jpg", ".jpeg", ".webp"),
                prefer_name=api_job_id,
            )
        except requests.exceptions.RequestException as exc:
            logging.error("Ideogram 4 download error: %s", exc)
            return recover_output_from_clean_container_exit(
                self,
                local_path,
                container_output_path=f"/app/output/{api_job_id}.png",
                extensions=(".png", ".jpg", ".jpeg", ".webp"),
                prefer_name=api_job_id,
            )

    def _service_type(self) -> str:
        service_type = get_processor_service_type(self)
        if service_type:
            return service_type
        return getattr(self.client, "active_service_type", "") or ""

    def _is_16gb_tier(self) -> bool:
        workflow_type = (self.workflow_type or "").lower()
        service_type = self._service_type().lower()
        return "16gb" in f"{workflow_type} {service_type}"

    def _default_sampler_preset(self) -> str:
        if self._is_16gb_tier():
            return self.LOW_VRAM_SAMPLER_PRESET
        return self.DEFAULT_SAMPLER_PRESET

    def _is_low_vram_hardware(self) -> bool:
        available_vram = getattr(self.client, "available_vram", None)
        if not isinstance(available_vram, (int, float)):
            return False
        return int(available_vram) <= self.LOW_VRAM_HARDWARE_MAX_MB

    def _resolve_sampler_preset(self, inputs: dict) -> str:
        requested = inputs.get("sampler_preset")
        if self._is_low_vram_hardware() and (
            not requested or requested == self.DEFAULT_SAMPLER_PRESET
        ):
            if requested == self.DEFAULT_SAMPLER_PRESET:
                logging.info(
                    "Downgrading Ideogram 4 sampler from %s to %s on %sMB VRAM hardware",
                    requested,
                    self.LOW_VRAM_SAMPLER_PRESET,
                    getattr(self.client, "available_vram", "unknown"),
                )
            return self.LOW_VRAM_SAMPLER_PRESET

        return requested or self._default_sampler_preset()

    def _resolve_dimensions(self, aspect_ratio: Optional[str]) -> tuple[int, int]:
        standard_ratios = {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "9:16": (768, 1344),
            "4:3": (1152, 864),
            "3:4": (864, 1152),
            "3:2": (1216, 816),
            "2:3": (816, 1216),
        }
        low_vram_ratios = {
            "1:1": (768, 768),
            "16:9": (1024, 576),
            "9:16": (576, 1024),
            "4:3": (896, 672),
            "3:4": (672, 896),
            "3:2": (912, 608),
            "2:3": (608, 912),
        }
        ratios = low_vram_ratios if self._is_16gb_tier() else standard_ratios
        fallback = (768, 768) if self._is_16gb_tier() else (1024, 1024)
        return ratios.get(aspect_ratio, fallback)
