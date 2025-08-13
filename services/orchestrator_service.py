import requests
import logging
from typing import Union
from services.hardware_profiler import get_hardware_profile

import os
class OrchestratorService:
    def __init__(self, orchestrator_url: str):
        self.orchestrator_url = orchestrator_url

    def download_asset(self, asset_id: str, cache_dir: str) -> Union[str, None]:
        """Download a specific asset from the orchestrator and save it to the cache directory."""
        try:
            # Request asset metadata and download URL from the orchestrator
            response = requests.get(f"{self.orchestrator_url}/api/dgn/assets/{asset_id}/download")
            response.raise_for_status()  # Raise an exception for HTTP errors
            asset_info = response.json()

            file_name = asset_info.get('fileName')
            download_url = asset_info.get('downloadUrl')

            if not file_name or not download_url:
                logging.error(f"Invalid asset info received for {asset_id}: {asset_info}")
                return None

            asset_local_path = os.path.join(cache_dir, file_name)

            if not os.path.exists(asset_local_path):
                logging.info(f"Downloading asset: {file_name} from {download_url}")
                # Stream the download to handle large files
                with requests.get(download_url, stream=True) as r:
                    r.raise_for_status()
                    with open(asset_local_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                logging.info(f"Asset {file_name} downloaded to {asset_local_path}")
            else:
                logging.info(f"Asset {file_name} already exists in cache.")
            return asset_local_path
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading asset {asset_id}: {e}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred during asset download for {asset_id}: {e}")
            return None

    def upload_output(self, file_path: str, job_id: str) -> Union[str, None]:
        """Upload the output file to the orchestrator and return the storage path."""
        try:
            with open(file_path, 'rb') as f:
                file_name = os.path.basename(file_path)
                files = {'file': (file_name, f.read(), 'video/mp4')} # Assuming video/mp4 for now
                data = {'jobId': job_id}

                response = requests.post(f"{self.orchestrator_url}/api/dgn/upload-output", files=files, data=data)
                response.raise_for_status()
                
                upload_result = response.json()
                if upload_result.get('success') and upload_result.get('storagePath'):
                    logging.info(f"File {file_name} uploaded successfully to {upload_result['storagePath']}.")
                    return upload_result['storagePath']
                else:
                    logging.error(f"Unexpected response from Orchestrator upload for file {file_name}: {upload_result}")
                    return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not upload file {file_path} to Orchestrator: {e}")
            return None
        except Exception as e:
            logging.error(f"An error occurred during output upload for {file_path}: {e}")
            return None


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