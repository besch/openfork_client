import requests
import logging
from typing import Union
from services.hardware_profiler import get_hardware_profile

class OrchestratorService:
    def __init__(self, orchestrator_url: str):
        self.orchestrator_url = orchestrator_url

    def update_job_status(self, job_id: str, status: str, output_path: Union[str, None] = None):
        """Update the status of a job, optionally including the output path."""
        try:
            payload = {"status": status}
            if output_path:
                payload["output_path"] = output_path

            response = requests.put(f"{self.orchestrator_url}/api/dgn/job/{job_id}", json=payload)
            if response.status_code == 200:
                logging.info(f"Job {job_id} status updated to {status}")
            else:
                logging.error(f"Error updating job status: {response.text}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not connect to the Orchestrator: {e}")

    def update_provider_status(self, provider_id: str, status: str):
        """Update the status of a provider."""
        try:
            response = requests.put(f"{self.orchestrator_url}/api/dgn/provider-status/{provider_id}", json={"status": status})
            if response.status_code == 200:
                logging.info(f"Provider {provider_id} status updated to {status}")
            else:
                logging.error(f"Error updating provider status: {response.text}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not connect to the Orchestrator: {e}")

    def register_with_orchestrator(self) -> Union[str, None]:
        """Register the client with the orchestrator."""
        hardware_profile = get_hardware_profile()
        logging.info(f"Hardware Profile: {hardware_profile}")

        try:
            response = requests.post(f"{self.orchestrator_url}/api/dgn/register", json=hardware_profile)
            if response.status_code == 200:
                logging.info("Successfully registered with the Orchestrator.")
                return response.json().get('provider_id')
            else:
                logging.error(f"Error registering with the Orchestrator: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not connect to the Orchestrator: {e}")
            return None

    def deregister_from_orchestrator(self, provider_id: str) -> None:
        """Remove provider row when client stops."""
        try:
            response = requests.delete(f"{self.orchestrator_url}/api/dgn/register", params={"providerId": provider_id})
            if response.status_code == 200:
                logging.info("Provider deregistered.")
            else:
                logging.error(f"Error deregistering provider: {response.text}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not connect to the Orchestrator: {e}")