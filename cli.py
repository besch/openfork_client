import argparse
import logging
import multiprocessing
import sys
import threading
import requests

import os

from config import Config
from dgn_client import DGNClient
from services.docker_manager import docker_manager
from utils.shutdown_handler import start_shutdown_server, SHUTDOWN_EVENT
from services.heartbeat_manager import HeartbeatManager
from services.job_listener import JobListener

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)], encoding='utf-8')

def setup_client(args):
    determined_orchestrator_url = Config.ORCHESTRATOR_URL_DEV if Config.DEV_MODE else Config.ORCHESTRATOR_URL_PROD
    
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
        data_dir=args.data_dir,
        access_token=args.access_token,
        refresh_token=args.refresh_token,
        accept_policy=args.accept_policy,
        allowed_targets=args.allowed_targets.split(',') if args.allowed_targets else None
    )



    client.provider_id = client.orchestrator_service.register_with_orchestrator()
    if not client.provider_id:
        raise RuntimeError("Failed to register with orchestrator. Aborting startup.")
    
    return client

def run_client(client):
    heartbeat_manager = HeartbeatManager(client.orchestrator_service, client.provider_id, SHUTDOWN_EVENT)
    heartbeat_manager.start()

    job_listener = JobListener(client, SHUTDOWN_EVENT)

    print("DGN_CLIENT_RUNNING", flush=True)
    logging.info(f"DGN Client is running and listening for jobs.")

    job_listener.listen_for_jobs()

def cleanup(client):
    logging.info("DGN Client: Initiating shutdown sequence.")
    
    if client and client.current_job:
        job_id = client.current_job.get('id')
        if job_id:
            logging.info(f"A job ({job_id}) was in progress. Attempting to reset its status to 'pending'.")
            try:
                client.orchestrator_service.reset_interrupted_job(job_id)
            except Exception as e:
                logging.error(f"Failed to reset job {job_id}: {e}", exc_info=True)

    if client and client.provider_id:
        logging.info("DGN Client: Attempting to deregister from orchestrator.")
        try:
            client.orchestrator_service.deregister_from_orchestrator(client.provider_id)
            logging.info("DGN Client: Successfully deregistered from orchestrator.")
        except Exception as e:
            logging.error(f"DGN Client: Failed to deregister from orchestrator: {e}", exc_info=True)
    
    logging.info("DGN Client: Stopping Docker container(s).")
    try:
        docker_manager.stop_container()
    except Exception as e:
        logging.error(f"DGN Client: Failed to stop Docker container: {e}", exc_info=True)

def main():
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description='DGN Client')
    parser.add_argument('--access-token', type=str, required=True, help='Supabase Auth Access Token')
    parser.add_argument('--refresh-token', type=str, required=True, help='Supabase Auth Refresh Token')

    parser.add_argument('--root-dir', type=str, help='The root directory of the dgn-client.')
    parser.add_argument('--data-dir', type=str, help='The directory for storing user data.')
    parser.add_argument('--accept-policy', type=str, default='mine', help='The job acceptance policy (all, mine, project).')
    parser.add_argument('--allowed-targets', type=str, help='For specific_* policies, a comma-separated list of targets (e.g., user/project-slug or user/project-slug:branch-name).')
    args = parser.parse_args()

    shutdown_thread = threading.Thread(target=start_shutdown_server, daemon=True)
    shutdown_thread.start()

    client = None
    try:
        client = setup_client(args)
        run_client(client)
    except Exception as e:
        logging.error(f"A critical error occurred during client operation: {e}", exc_info=True)
    finally:
        cleanup(client)

    logging.info("Main function completed.")
