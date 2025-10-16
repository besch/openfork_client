import argparse
import logging
import multiprocessing
import sys
import threading
import requests

import os

from config import CACHE_DIR, DEV_MODE, ORCHESTRATOR_URL_PROD, ORCHESTRATOR_URL_DEV, DOCKER_IMAGE_MAP
from dgn_client import DGNClient
from services.docker_manager import docker_manager
from utils.shutdown_handler import start_shutdown_server, SHUTDOWN_EVENT
from services.heartbeat_manager import HeartbeatManager
from services.job_listener import JobListener

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

def setup_client(args):
    determined_orchestrator_url = ORCHESTRATOR_URL_DEV if DEV_MODE else ORCHESTRATOR_URL_PROD
    
    logging.info(f"Attempting to connect to orchestrator URL: {determined_orchestrator_url}")
    try:
        response = requests.get(f"{determined_orchestrator_url}/api/dgn/provider-status/health", timeout=5)
        if response.status_code == 200:
            logging.info(f"Successfully connected to orchestrator URL: {determined_orchestrator_url}")
        else:
            logging.warning(f"Orchestrator URL {determined_orchestrator_url} returned status {response.status_code}.")
    except requests.exceptions.RequestException as e:
        logging.warning(f"Could not connect to orchestrator URL {determined_orchestrator_url}: {e}.")

    root_dir = args.root_dir if args.root_dir else os.getcwd()
    logging.info(f"Using root directory: {root_dir}")

    client = DGNClient(
        orchestrator_url=determined_orchestrator_url,
        root_dir=root_dir,
        cache_dir=CACHE_DIR,
        data_dir=args.data_dir,
        access_token=args.access_token,
        refresh_token=args.refresh_token
    )

    client.load_config() # Fetch config from orchestrator

    # Validate service argument against available services from config
    available_services = [s['value'] for s in client.config.get('ui_services', [])]
    if args.service not in available_services:
        logging.error(f"Invalid service '{args.service}'. Available services: {', '.join(s for s in available_services if s != 'auto')}")
        sys.exit(1)

    docker_manager.set_docker_image_map(client.config.get('docker_image_map', {}))

    provider_id = client.orchestrator_service.register_with_orchestrator(service_type=args.service)
    if not provider_id:
        raise RuntimeError("Failed to register with orchestrator. Aborting startup.")
    
    return client, provider_id

def run_client(client, provider_id, service_mode):
    heartbeat_manager = HeartbeatManager(client.orchestrator_service, provider_id, SHUTDOWN_EVENT)
    heartbeat_manager.start()

    job_listener = JobListener(client, provider_id, SHUTDOWN_EVENT)

    print("DGN_CLIENT_RUNNING", flush=True)
    logging.info(f"DGN Client is running in '{service_mode}' mode and listening for jobs.")

    if service_mode == 'auto':
        job_listener.listen_for_jobs_auto()
    else:
        job_listener.listen_for_jobs()

def cleanup(client, provider_id, service_mode):
    logging.info("DGN Client: Initiating shutdown sequence.")
    
    if client and client.current_job:
        job_id = client.current_job.get('id')
        if job_id:
            logging.info(f"A job ({job_id}) was in progress. Attempting to reset its status to 'pending'.")
            try:
                client.orchestrator_service.reset_interrupted_job(job_id)
            except Exception as e:
                logging.error(f"Failed to reset job {job_id}: {e}", exc_info=True)

    if provider_id and client:
        logging.info("DGN Client: Attempting to deregister from orchestrator.")
        try:
            client.orchestrator_service.deregister_from_orchestrator(provider_id)
            logging.info("DGN Client: Successfully deregistered from orchestrator.")
        except Exception as e:
            logging.error(f"DGN Client: Failed to deregister from orchestrator: {e}", exc_info=True)
    
    logging.info("DGN Client: Stopping Docker container(s).")
    try:
        if service_mode != 'auto':
            docker_manager.stop_container(service_type=service_mode)
        elif client and client.active_service_type:
            docker_manager.stop_container(service_type=client.active_service_type)
        else:
            logging.info("DGN Client: No active container to stop.")
    except Exception as e:
        logging.error(f"DGN Client: Failed to stop Docker container: {e}", exc_info=True)

def main():
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description='DGN Client')
    parser.add_argument('--access-token', type=str, required=True, help='Supabase Auth Access Token')
    parser.add_argument('--refresh-token', type=str, required=True, help='Supabase Auth Refresh Token')
    
    service_choices = list(DOCKER_IMAGE_MAP.keys()) + ['AUTO']
    help_string = f'Service to run ({ ", ".join(service_choices)})'
    parser.add_argument('--service', type=str, default='AUTO', help=help_string)

    parser.add_argument('--root-dir', type=str, help='The root directory of the dgn-client.')
    parser.add_argument('--data-dir', type=str, help='The directory for storing user data.')
    args = parser.parse_args()

    if args.service != 'auto':
        docker_manager.run_container(service_type=args.service)
    
    shutdown_thread = threading.Thread(target=start_shutdown_server, daemon=True)
    shutdown_thread.start()

    client = None
    provider_id = None
    try:
        client, provider_id = setup_client(args)

        # If we are running a dedicated service, set it as active on the client.
        if args.service != 'auto':
            client.active_service_type = args.service

        run_client(client, provider_id, args.service)
    except Exception as e:
        logging.error(f"A critical error occurred during client operation: {e}", exc_info=True)
    finally:
        cleanup(client, provider_id, args.service)

    logging.info("Main function completed.")
