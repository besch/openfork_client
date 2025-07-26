import time
import requests
import os
import logging
import argparse

from docker_manager import run_container
from comfyui_manager import trigger_workflow, get_workflow_output
from hardware_profiler import get_hardware_profile
from config import ROOT_DIR, ORCHESTRATOR_URL, SUPABASE_URL, SUPABASE_ANON_KEY, CACHE_DIR
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def upload_output(file_path, job_id):
    """Upload the output file to Supabase Storage."""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        with open(file_path, 'rb') as f:
            file_name = os.path.basename(file_path)
            # Use job_id in the path to organize outputs
            storage_path = f"outputs/{job_id}/{file_name}"
            response = supabase.storage.from_('dgn-assets').upload(storage_path, f.read(), {'content-type': 'video/mp4'})
            if response.status_code == 200:
                logging.info(f"File {file_name} uploaded successfully to {storage_path}.")
            else:
                logging.error(f"Error uploading file {file_name}: {response.text}")
    except Exception as e:
        logging.error(f"Could not upload file {file_path} to Supabase: {e}")

def process_workflow_output(outputs, job_id):
    """Process the workflow output and upload the generated files."""
    for node_id, node_output in outputs.items():
        if 'filenames' in node_output:
            for filename in node_output['filenames']:
                file_path = os.path.join(ROOT_DIR, 'output', filename)
                if os.path.exists(file_path):
                    upload_output(file_path, job_id)
                else:
                    logging.warning(f"Output file not found: {file_path}")

def register_with_orchestrator():
    """Register the client with the orchestrator."""
    hardware_profile = get_hardware_profile()
    logging.info(f"Hardware Profile: {hardware_profile}")

    try:
        response = requests.post(f"{ORCHESTRATOR_URL}/api/dgn/register", json=hardware_profile)
        if response.status_code == 200:
            logging.info("Successfully registered with the Orchestrator.")
            return response.json().get('provider_id')
        else:
            logging.error(f"Error registering with the Orchestrator: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Could not connect to the Orchestrator: {e}")
        return None

def update_job_status(job_id, status):
    """Update the status of a job."""
    try:
        response = requests.put(f"{ORCHESTRATOR_URL}/api/dgn/jobs/{job_id}", json={"status": status})
        if response.status_code == 200:
            logging.info(f"Job {job_id} status updated to {status}")
        else:
            logging.error(f"Error updating job status: {response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Could not connect to the Orchestrator: {e}")

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

                        if not workflow:
                            logging.error("No workflow found in job.")
                            update_job_status(job['id'], 'failed')
                            container.stop()
                            continue

                        if not verify_workflow_nodes(workflow):
                            update_job_status(job['id'], 'failed')
                            container.stop()
                            continue

                        prompt_id = trigger_workflow(workflow)
                        if prompt_id:
                            outputs = get_workflow_output(prompt_id)
                            if outputs:
                                process_workflow_output(outputs, job['id']) # Pass job_id for storage path
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
        'UnetLoaderGGUF',
        'ModelSamplingSD3',
        'LoadImage',
        'PreviewImage',
        'ImageResizeKJv2',
        'VHS_VideoCombine',
        'WanVideoNAG',
        'PathchSageAttentionKJ',
        'ModelPatchTorchSettings',
        'CLIPLoader',
        'WanImageToVideo',
        # All nodes from workflow.json are added here for compatibility.
        # In a production environment, these should be carefully vetted for security.
    ]

    for node in workflow.get('nodes', []):
        if node.get('type') not in APPROVED_NODES:
            logging.error(f"Security Alert: Workflow contains a non-approved node: {node.get('type')}")
            return False
    return True

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
