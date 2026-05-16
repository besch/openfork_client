"""
Text Generation Processor

Uses Ollama for LLM-based text generation.
"""

import os
import time
import logging
import math
import random
import json
import threading
import requests

from services.processors.base import BaseJobProcessor


class LLMJobProcessor(BaseJobProcessor):
    """Processor for LLM-based text generation using Ollama."""

    @staticmethod
    def _base_json_schema(schema_type: str) -> dict:
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": schema_type,
        }

    @classmethod
    def _script_scene_schema(cls, num_scenes) -> dict:
        schema = {
            **cls._base_json_schema("array"),
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Scene title"},
                    "description": {"type": "string", "description": "Visual description"},
                    "aspect_ratio": {"type": "string", "enum": ["16:9", "9:16"]},
                    "camera_movement": {
                        "type": "string",
                        "enum": ["static", "zoom-in", "zoom-out", "pan-left", "pan-right", "dolly-in", "push-in", "pull-back"],
                    },
                    "duration_seconds": {"type": "integer", "minimum": 4, "maximum": 10},
                },
                "required": ["name", "description", "aspect_ratio", "camera_movement", "duration_seconds"],
                "additionalProperties": False,
            },
        }

        if num_scenes is not None:
            schema["minItems"] = num_scenes
            schema["maxItems"] = num_scenes
            logging.info(f"Enforcing exactly {num_scenes} scene(s) via JSON schema constraints")

        return schema

    @classmethod
    def _generation_style_schema(cls) -> dict:
        return {
            **cls._base_json_schema("object"),
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "prompt_prefix": {
                    "type": "string",
                    "description": "A compact reusable visual style prefix for image and video prompts.",
                },
                "category": {
                    "type": "string",
                    "enum": ["animation", "cinematic", "artistic", "photo", "stylized", "custom"],
                },
            },
            "required": ["name", "description", "prompt_prefix", "category"],
            "additionalProperties": False,
        }

    @classmethod
    def _activity_scenarios_schema(cls, expected_items) -> dict:
        scenarios = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "creature_name": {"type": "string"},
                    "activity": {"type": "string"},
                    "scene_prompt": {"type": "string"},
                    "image_edit_prompt": {"type": "string"},
                    "video_prompt": {"type": "string"},
                    "camera_movement": {
                        "type": "string",
                        "enum": ["static", "zoom-in", "zoom-out", "pan-left", "pan-right", "dolly-in", "push-in", "pull-back"],
                    },
                    "duration_seconds": {"type": "integer", "minimum": 4, "maximum": 10},
                },
                "required": [
                    "creature_name",
                    "activity",
                    "scene_prompt",
                    "image_edit_prompt",
                    "video_prompt",
                    "camera_movement",
                    "duration_seconds",
                ],
                "additionalProperties": False,
            },
        }

        if expected_items is not None:
            scenarios["minItems"] = expected_items
            scenarios["maxItems"] = expected_items

        return {
            **cls._base_json_schema("object"),
            "properties": {
                "scenarios": scenarios,
            },
            "required": ["scenarios"],
            "additionalProperties": False,
        }

    @classmethod
    def _music_prompt_schema(cls) -> dict:
        return {
            **cls._base_json_schema("object"),
            "properties": {
                "prompt": {"type": "string"},
                "mood": {"type": "string"},
                "tempo": {"type": "string"},
                "instruments": {"type": "array", "items": {"type": "string"}},
                "duration_seconds": {"type": "integer", "minimum": 5, "maximum": 300},
            },
            "required": ["prompt", "mood", "tempo", "instruments", "duration_seconds"],
            "additionalProperties": False,
        }

    @classmethod
    def _dialogue_scenes_schema(cls, expected_items) -> dict:
        dialogues = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "script_index": {"type": "integer", "minimum": 0},
                    "speaker": {"type": "string"},
                    "line": {"type": "string"},
                    "qwen3_speaker": {
                        "type": "string",
                        "enum": [
                            "Vivian",
                            "Serena",
                            "Uncle_Fu",
                            "Dylan",
                            "Eric",
                            "Ryan",
                            "Aiden",
                            "Ono_Anna",
                            "Sohee",
                        ],
                    },
                    "start_offset_seconds": {"type": "number", "minimum": 0, "maximum": 20},
                },
                "required": [
                    "script_index",
                    "speaker",
                    "line",
                    "qwen3_speaker",
                    "start_offset_seconds",
                ],
                "additionalProperties": False,
            },
        }

        if expected_items is not None:
            dialogues["minItems"] = expected_items
            dialogues["maxItems"] = expected_items

        return {
            **cls._base_json_schema("object"),
            "properties": {
                "dialogues": dialogues,
            },
            "required": ["dialogues"],
            "additionalProperties": False,
        }

    @classmethod
    def _generic_object_schema(cls) -> dict:
        return {
            **cls._base_json_schema("object"),
            "additionalProperties": True,
        }

    @classmethod
    def _json_schema_for_job(cls, job_type: str, num_scenes, expected_items) -> dict:
        if job_type == "script_generation":
            return cls._script_scene_schema(num_scenes)
        if job_type == "generation_style":
            return cls._generation_style_schema()
        if job_type == "activity_scenarios":
            return cls._activity_scenarios_schema(expected_items)
        if job_type == "dialogue_scenes":
            return cls._dialogue_scenes_schema(expected_items)
        if job_type == "music_prompt":
            return cls._music_prompt_schema()
        return cls._generic_object_schema()

    @staticmethod
    def _validate_generated_json(generated_text: str, job_type: str, num_scenes, expected_items):
        parsed_json = json.loads(generated_text)

        if job_type == "script_generation":
            if not isinstance(parsed_json, list):
                raise ValueError(f"Generated JSON is not an array. Got: {type(parsed_json).__name__}")
            actual_count = len(parsed_json)
            if num_scenes is not None and actual_count != num_scenes:
                logging.warning(
                    f"Schema constraint mismatch: expected {num_scenes} scenes but got {actual_count}"
                )
            return parsed_json

        if job_type == "activity_scenarios":
            if not isinstance(parsed_json, dict):
                raise ValueError(f"Generated JSON is not an object. Got: {type(parsed_json).__name__}")
            scenarios = parsed_json.get("scenarios")
            if not isinstance(scenarios, list):
                raise ValueError("Generated activity scenario output is missing a scenarios array.")
            if expected_items is not None and len(scenarios) != expected_items:
                logging.warning(
                    f"Schema constraint mismatch: expected {expected_items} scenarios but got {len(scenarios)}"
                )
            return parsed_json

        if job_type == "dialogue_scenes":
            if not isinstance(parsed_json, dict):
                raise ValueError(f"Generated JSON is not an object. Got: {type(parsed_json).__name__}")
            dialogues = parsed_json.get("dialogues")
            if not isinstance(dialogues, list):
                raise ValueError("Generated dialogue output is missing a dialogues array.")
            if expected_items is not None and len(dialogues) != expected_items:
                logging.warning(
                    f"Schema constraint mismatch: expected {expected_items} dialogue lines but got {len(dialogues)}"
                )
            return parsed_json

        if not isinstance(parsed_json, dict):
            raise ValueError(f"Generated JSON is not an object. Got: {type(parsed_json).__name__}")

        return parsed_json

    @staticmethod
    def _is_transient_ollama_error(exc: Exception) -> bool:
        text = str(exc).lower()
        transient_markers = (
            "remote end closed connection",
            "connection aborted",
            "connection reset",
            "connection refused",
            "temporarily unavailable",
            "timeout",
            "timed out",
        )
        return any(marker in text for marker in transient_markers)

    def process(self):
        if not self.job:
            self._fail_job(f"Job object is None for LLMJobProcessor. Cannot proceed.")
            return

        api_base = os.getenv("LLM_API_BASE", "http://127.0.0.1:11434")

        logging.info(f"Waiting for Ollama service to be ready at {api_base}...")

        if not self._wait_for_ollama(api_base):
            return

        inputs = self.job.get("inputs", {})
        # Default to qwen2.5:3b (blazing fast, excellent at JSON and creative writing)
        model_name = inputs.get("model", "qwen2.5:3b")
            
        system_prompt = inputs.get("system_prompt", "You are a helpful assistant.")
        temperature = inputs.get("temperature", 0.7)
        max_tokens = inputs.get("max_tokens", 16000)

        seed = inputs.get("seed")
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
            logging.info(f"No seed provided, using random seed: {seed}")

        if not self._ensure_model_available(api_base, model_name):
            return

        self._generate_text(api_base, model_name, system_prompt, temperature, max_tokens, seed)

    def _wait_for_ollama(self, api_base: str, timeout: int = 60) -> bool:
        """Wait for Ollama service to be ready."""
        for attempt in range(timeout):
            if self.is_cancelled():
                logging.info("Shutdown event received during Ollama readiness check.")
                return False
            try:
                response = requests.get(f"{api_base}/api/tags", timeout=2)
                response.raise_for_status()
                logging.info(f"Ollama service is ready! (attempt {attempt + 1}/{timeout})")
                return True
            except requests.exceptions.RequestException as e:
                if attempt % 10 == 0:
                    logging.debug(f"Ollama not ready yet (attempt {attempt + 1}/{timeout}): {e}")
                time.sleep(0.5)

        logging.error(f"Ollama service failed to become ready within {timeout} seconds at {api_base}")
        self._fail_job("Ollama service not ready")
        return False

    def _ensure_model_available(self, api_base: str, model_name: str) -> bool:
        """Check if model exists and pull if needed."""
        try:
            logging.info(f"Checking if model {model_name} is available...")
            tags_response = requests.get(f"{api_base}/api/tags", timeout=5)
            tags_response.raise_for_status()
            tags_data = tags_response.json()

            available_models = [m.get("name", "") for m in tags_data.get("models", [])]
            logging.info(f"Available models: {available_models}")

            if model_name not in available_models:
                logging.info(f"Model {model_name} not found. Pulling it now...")
                pull_payload = {"name": model_name, "stream": False}
                pull_response = requests.post(f"{api_base}/api/pull", json=pull_payload, timeout=600)
                pull_response.raise_for_status()
                logging.info(f"Successfully pulled model {model_name}")
            else:
                logging.info(f"Model {model_name} is already available")

            return True
        except Exception as e:
            self._fail_job(f"Failed to verify/pull model {model_name}: {e}")
            return False

    def _generate_text(
        self, api_base: str, model_name: str, system_prompt: str, temperature: float, max_tokens: int, seed: int
    ):
        """Generate text using Ollama with structured output for reliable JSON."""
        try:
            inputs = self.job.get("inputs", {})
            num_scenes = inputs.get("num_scenes")
            job_type = inputs.get("job_type") or ("script_generation" if num_scenes is not None else "general")
            expected_items = inputs.get("expected_items")
            json_schema = self._json_schema_for_job(job_type, num_scenes, expected_items)

            # Dynamically size the context window to fit the requested output.
            # num_ctx must be >= (input tokens + max_tokens).  We add a 2048-token
            # buffer for the system/user prompts and round up to the next power-of-two
            # for efficiency in the attention implementation.
            raw_ctx = max_tokens + 2048
            num_ctx = max(8192, int(2 ** math.ceil(math.log2(raw_ctx))))
            logging.info(
                f"Context window sizing: max_tokens={max_tokens}, num_ctx={num_ctx}, "
                f"job_type={job_type}, num_scenes={num_scenes}, expected_items={expected_items}"
            )

            # Use chat API with format parameter for structured output
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": self.positive_prompt}
                ],
                "stream": False,
                "format": json_schema,
                "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed, "num_ctx": num_ctx},
            }

            logging.info(f"Calling Ollama chat API with structured output at {api_base}/api/chat")
            response = None
            max_attempts = 4
            for attempt in range(1, max_attempts + 1):
                response_holder: list = [None]
                error_holder: list = [None]

                def _do_request():
                    try:
                        response_holder[0] = requests.post(
                            f"{api_base}/api/chat", json=payload, timeout=1200
                        )
                    except Exception as exc:
                        error_holder[0] = exc

                req_thread = threading.Thread(target=_do_request, daemon=True)
                req_thread.start()
                while req_thread.is_alive():
                    req_thread.join(timeout=1.0)
                    if self.is_cancelled():
                        logging.info("Shutdown event received during LLM generation. Aborting.")
                        return

                if error_holder[0]:
                    if (
                        attempt < max_attempts
                        and self._is_transient_ollama_error(error_holder[0])
                    ):
                        delay = min(30, 5 * attempt)
                        logging.warning(
                            "Ollama chat request failed during startup/model load "
                            "(attempt %s/%s): %s. Retrying in %ss.",
                            attempt,
                            max_attempts,
                            error_holder[0],
                            delay,
                        )
                        self.shutdown_event.wait(delay)
                        if self.is_cancelled():
                            logging.info("Shutdown event received during LLM retry delay. Aborting.")
                            return
                        continue
                    raise error_holder[0]

                response = response_holder[0]
                try:
                    response.raise_for_status()
                    break
                except requests.exceptions.RequestException as exc:
                    if attempt < max_attempts and self._is_transient_ollama_error(exc):
                        delay = min(30, 5 * attempt)
                        logging.warning(
                            "Ollama chat returned a transient error "
                            "(attempt %s/%s): %s. Retrying in %ss.",
                            attempt,
                            max_attempts,
                            exc,
                            delay,
                        )
                        self.shutdown_event.wait(delay)
                        if self.is_cancelled():
                            logging.info("Shutdown event received during LLM retry delay. Aborting.")
                            return
                        continue
                    raise

            result = response.json()
            generated_text = result.get("message", {}).get("content", "")

            if not generated_text:
                self._fail_job(f"Ollama returned empty response. Full result: {result}")
                return

            try:
                parsed_json = self._validate_generated_json(
                    generated_text,
                    job_type,
                    num_scenes,
                    expected_items,
                )
            except json.JSONDecodeError as e:
                self._fail_job(f"Generated content is not valid JSON: {e}")
                return
            except ValueError as e:
                self._fail_job(str(e))
                return

            if isinstance(parsed_json, list):
                logging.info(
                    f"Generation complete. Generated {len(parsed_json)} item(s). Length: {len(generated_text)} chars."
                )
            elif job_type == "activity_scenarios":
                logging.info(
                    "Generation complete. Generated %s scenario(s). Length: %s chars.",
                    len(parsed_json.get("scenarios", [])),
                    len(generated_text),
                )
            else:
                logging.info(
                    f"Generation complete for {job_type}. Length: {len(generated_text)} chars."
                )

            output_filename = f"{self.job_id}_{job_type}.txt"
            output_path = os.path.join(self.cache_dir, output_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(generated_text)

            storage_path = self.orchestrator_service.upload_output(output_path, self.job_id, "text/plain")

            if self.is_cancelled():
                logging.info("Shutdown event set after LLM generation completed. Skipping job completion.")
                return

            if storage_path:
                completion_metadata = self.job.get("completion_metadata") or {}
                completion_metadata = dict(completion_metadata)
                completion_metadata.update({"model": model_name, "job_type": job_type})
                self.orchestrator_service.update_job_status(
                    self.job_id,
                    "completed",
                    storage_path=storage_path,
                    completion_metadata=completion_metadata,
                )
            else:
                self._fail_job("Failed to upload generated script.")

            if os.path.exists(output_path):
                os.remove(output_path)

        except Exception as e:
            self._fail_job(f"Text generation failed: {e}")
