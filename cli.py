import argparse
import logging
import multiprocessing
import sys
import threading
import requests
import json

import os

from config import DEV_MODE, ORCHESTRATOR_URL_PROD, ORCHESTRATOR_URL_DEV
from dgn_client import DGNClient
from services.docker_manager import docker_manager
from utils.shutdown_handler import start_shutdown_server, SHUTDOWN_EVENT
from services.heartbeat_manager import HeartbeatManager
from services.job_listener import JobListener

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

def listen_for_ipc_commands(client: DGNClient):
    """
    Listens for JSON commands from stdin (sent by the parent Electron process).
    """
    logging.info("IPC listener thread started.")
    for line in sys.stdin:
        if SHUTDOWN_EVENT.is_set():
            break
        try:
            command = json.loads(line)
            cmd_type = command.get("type")
            payload = command.get("payload")

            if cmd_type == "UPDATE_TOKENS":
                logging.info("Received UPDATE_TOKENS command from main process.")
                if payload and "access_token" in payload and "refresh_token" in payload:
                    client.orchestrator_service.update_tokens(
                        payload["access_token"],
                        payload["refresh_token"]
                    )
                else:
                    logging.warning("UPDATE_TOKENS command received with invalid payload.")
            else:
                logging.warning(f"Received unknown IPC command type: {cmd_type}")

        except json.JSONDecodeError:
            logging.warning(f"Could not decode IPC command from stdin: {line.strip()}")
        except Exception as e:
            logging.error(f"Error processing IPC command: {e}")
    logging.info("IPC listener thread stopped.")


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
        data_dir=args.data_dir,
        access_token=args.access_token,
        refresh_token=args.refresh_token,
        accept_policy=args.accept_policy,
        allowed_targets=args.allowed_targets.split(',') if args.allowed_targets else None
    )

    client.load_config() # Fetch config from orchestrator

    # Validate service argument after loading config
    if args.service != 'auto':
        available_services = list(client.docker_image_map.keys())
        if args.service not in available_services:
            logging.error(f"Invalid service '{args.service}'. Available services from config: {', '.join(available_services)}")
            sys.exit(1)

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
    
    parser.add_argument('--service', type=str, default='auto', help='Service to run (e.g., wan22, foley). Default is "auto".')

    parser.add_argument('--root-dir', type=str, help='The root directory of the dgn-client.')
    parser.add_argument('--data-dir', type=str, help='The directory for storing user data.')
    parser.add_argument('--accept-policy', type=str, default='mine', help='The job acceptance policy (all, mine, project).')
    parser.add_argument('--allowed-targets', type=str, help='For specific_* policies, a comma-separated list of targets (e.g., user/project-slug or user/project-slug:branch-name).')
    args = parser.parse_args()


    
    shutdown_thread = threading.Thread(target=start_shutdown_server, daemon=True)
    shutdown_thread.start()

    client = None
    provider_id = None
    try:
        client, provider_id = setup_client(args)

        # Start the IPC listener thread
        ipc_thread = threading.Thread(target=listen_for_ipc_commands, args=(client,), daemon=True)
        ipc_thread.start()

        # If we are running a dedicated service, set it as active on the client.
        if args.service != 'auto':
            client.active_service_type = args.service
            # Start the container now that config is loaded and image map is set
            docker_manager.run_container(service_type=args.service)

        run_client(client, provider_id, args.service)
    except Exception as e:
        logging.error(f"A critical error occurred during client operation: {e}", exc_info=True)
        sys.exit(1)
    finally:
        cleanup(client, provider_id, args.service)

    logging.info("Main function completed.")
