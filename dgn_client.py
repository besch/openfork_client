import time
import requests
import os
from docker_manager import run_container
from comfyui_manager import trigger_workflow, get_workflow_output
from hardware_profiler import get_hardware_profile
from config import ROOT_DIR, SUPABASE_URL, SUPABASE_ANON_KEY, CACHE_DIR
from supabase import create_client, Client
import requests

ORCHESTRATOR_URL = 'http://localhost:3000'

def register_with_orchestrator():
    """Register the client with the orchestrator."""
    hardware_profile = get_hardware_profile()
    print("Hardware Profile:", hardware_profile)

    try:
        response = requests.post(f"{ORCHESTRATOR_URL}/api/dgn/register", json=hardware_profile)
        if response.status_code == 200:
            print("Successfully registered with the Orchestrator.")
            return response.json().get('provider_id')
        else:
            print(f"Error registering with the Orchestrator: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to the Orchestrator: {e}")
        return None

def update_job_status(job_id, status):
    """Update the status of a job."""
    try:
        response = requests.put(f"{ORCHESTRATOR_URL}/api/dgn/jobs/{job_id}", json={"status": status})
        if response.status_code == 200:
            print(f"Job {job_id} status updated to {status}")
        else:
            print(f"Error updating job status: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to the Orchestrator: {e}")

def upload_output(file_path):
    """Upload the output file to the storage service."""
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'video/mp4')}
            response = requests.post(f"{ORCHESTRATOR_URL}/api/dgn/upload", files=files)
            if response.status_code == 200:
                print(f"File {os.path.basename(file_path)} uploaded successfully.")
            else:
                print(f"Error uploading file: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to the Orchestrator: {e}")

def process_workflow_output(outputs):
    """Process the workflow output and upload the generated files."""
    for node_id, node_output in outputs.items():
        if 'filenames' in node_output:
            for filename in node_output['filenames']:
                file_path = os.path.join(ROOT_DIR, 'output', filename)
                if os.path.exists(file_path):
                    upload_output(file_path)
                else:
                    print(f"Output file not found: {file_path}")

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def listen_for_jobs(provider_id):
    """Listen for jobs from the orchestrator."""
    while True:
        try:
            logging.info("Checking for new jobs...")
            response = requests.get(f"{ORCHESTRATOR_URL}/api/dgn/jobs/{provider_id}")
            if response.status_code == 200:
                job = response.json()
                if job:
                    logging.info(f"Received job: {job['id']}")
                    try:
                        update_job_status(job['id'], 'processing')
                        
                        container = run_container()
                        time.sleep(10) # Wait for ComfyUI to start
                        
                        workflow = job.get('workflow')
                        assets = job.get('assets', [])

                        if not workflow:
                            logging.error("No workflow found in job.")
                            update_job_status(job['id'], 'failed')
                            container.stop()
                            continue

                        if not verify_workflow_nodes(workflow):
                            update_job_status(job['id'], 'failed')
                            container.stop()
                            continue

                        if assets:
                            download_assets(assets)

                        prompt_id = trigger_workflow(workflow)
                        if prompt_id:
                            outputs = get_workflow_output(prompt_id)
                            if outputs:
                                process_workflow_output(outputs)
                                update_job_status(job['id'], 'completed')
                            else:
                                logging.error("Workflow failed to produce outputs.")
                                update_job_status(job['id'], 'failed')
                        else:
                            logging.error("Failed to trigger workflow.")
                            update_job_status(job['id'], 'failed')

                        container.stop()
                    except Exception as e:
                        logging.error(f"An error occurred while processing job {job['id']}: {e}")
                        update_job_status(job['id'], 'failed')

                else:
                    logging.info("No new jobs.")
            else:
                logging.error(f"Error checking for jobs: {response.text}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not connect to the Orchestrator: {e}")

        time.sleep(10) # Poll every 10 seconds

def verify_workflow_nodes(workflow):
    """Verify that all nodes in the workflow are on the approved list."""
    APPROVED_NODES = [
        'KSampler',
        'VAELoader',
        'CLIPTextEncode',
        'EmptyLatentImage',
        'LoraLoaderModelOnly',
        'VAEDecode',
        # Add other approved nodes here
    ]

    for node in workflow.get('nodes', []):
        if node.get('type') not in APPROVED_NODES:
            print(f"Security Alert: Workflow contains a non-approved node: {node.get('type')}")
            return False
    return True

def download_assets(assets):
    """Download the assets required by the workflow."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    for asset in assets:
        asset_path = os.path.join(CACHE_DIR, asset)
        if not os.path.exists(asset_path):
            print(f"Downloading asset: {asset}")
            # In a real application, you would download the asset from a storage service.
            # For now, we'll just create an empty file.
            with open(asset_path, 'w') as f:
                f.write("")
    return True

import argparse

def main():
    """Main function to run the DGN client."""
    parser = argparse.ArgumentParser(description="CrowdMovie DGN Client")
    parser.add_argument("--orchestrator-url", default="http://localhost:3000", help="The URL of the orchestrator")
    args = parser.parse_args()

    global ORCHESTRATOR_URL
    ORCHESTRATOR_URL = args.orchestrator_url

    provider_id = register_with_orchestrator()

    if provider_id:
        listen_for_jobs(provider_id)

if __name__ == "__main__":
    main()