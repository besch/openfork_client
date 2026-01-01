"""
Text Generation Processor

Uses Ollama for LLM-based text generation.
"""

import os
import time
import logging
import random
import requests

from services.processors.base import BaseJobProcessor


class TextGenerationJobProcessor(BaseJobProcessor):
    """Processor for LLM-based text generation using Ollama."""

    def process(self):
        if not self.job:
            self._fail_job(f"Job object is None for TextGenerationJobProcessor. Cannot proceed.")
            return

        api_base = os.getenv("LLM_API_BASE", "http://localhost:11434")

        logging.info(f"Waiting for Ollama service to be ready at {api_base}...")
        time.sleep(3)

        if not self._wait_for_ollama(api_base):
            return

        inputs = self.job.get("inputs", {})
        # Default to gemma:2b (better at following JSON output instructions than phi3)
        model_name = inputs.get("model", "gemma:2b")
            
        system_prompt = inputs.get("system_prompt", "You are a helpful assistant.")
        temperature = inputs.get("temperature", 0.7)
        max_tokens = inputs.get("max_tokens", 2000)

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
            if self.shutdown_event.is_set():
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
                time.sleep(1)

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
        """Generate text using Ollama."""
        try:
            full_prompt = f"{system_prompt}\n\nUser: {self.positive_prompt}\n\nAssistant:"

            payload = {
                "model": model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed},
            }

            logging.info(f"Calling Ollama API at {api_base}/api/generate with model {model_name}")
            response = requests.post(f"{api_base}/api/generate", json=payload, timeout=1200)
            response.raise_for_status()

            result = response.json()
            generated_text = result.get("response", "")

            if not generated_text:
                self._fail_job(f"Ollama returned empty response. Full result: {result}")
                return

            logging.info(f"Generation complete. Length: {len(generated_text)} chars.")

            output_filename = f"{self.job_id}_script.txt"
            output_path = os.path.join(self.cache_dir, output_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(generated_text)

            storage_path = self.orchestrator_service.upload_output(output_path, self.job_id, "text/plain")

            if storage_path:
                self.orchestrator_service.update_job_status(
                    self.job_id, "completed", storage_path=storage_path, completion_metadata={"model": model_name}
                )
            else:
                self._fail_job("Failed to upload generated script.")

            if os.path.exists(output_path):
                os.remove(output_path)

        except Exception as e:
            self._fail_job(f"Text generation failed: {e}")
