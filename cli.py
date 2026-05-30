import argparse
import logging
import multiprocessing
import re
import sys
import threading
import json
import os

_SERVICE_ID_RE = re.compile(r'^[a-z0-9][a-z0-9_-]{0,79}$', re.IGNORECASE)
_VALID_COMMUNITY_MODES = frozenset(('none', 'trusted_users', 'trusted_projects', 'all'))
_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    re.IGNORECASE,
)


def _sanitize_routing_config(payload: dict) -> dict:
    sanitized = {}

    if "community_mode" in payload or "communityMode" in payload:
        community_mode = payload.get("community_mode", payload.get("communityMode"))
        if community_mode not in _VALID_COMMUNITY_MODES:
            community_mode = "none"
        sanitized["community_mode"] = community_mode

    if (
        "allowed_ids" in payload
        or "allowedIds" in payload
        or "trustedIds" in payload
    ):
        raw_ids = payload.get(
            "allowed_ids",
            payload.get("allowedIds", payload.get("trustedIds", [])),
        )
        if isinstance(raw_ids, str):
            raw_ids = [i.strip() for i in raw_ids.split(",") if i.strip()]
        elif not isinstance(raw_ids, (list, tuple, set)):
            raw_ids = []
        sanitized["allowed_ids"] = [
            i for i in raw_ids if isinstance(i, str) and _UUID_RE.match(i)
        ][:500]

    if "process_own_jobs" in payload or "processOwnJobs" in payload:
        sanitized["process_own_jobs"] = (
            payload.get("process_own_jobs", payload.get("processOwnJobs")) is True
        )

    if "monetize_mode" in payload or "monetizeMode" in payload:
        sanitized["monetize_mode"] = (
            payload.get("monetize_mode", payload.get("monetizeMode")) is True
        )

    return sanitized

def _configure_stdio_encoding() -> None:
    """Keep Electron pipe output UTF-8 on Windows codepages."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_stdio_encoding()

# Force reset logging to ensure our config takes effect regardless of module import order
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    stream=sys.stdout,
    force=True
)
logging.getLogger().setLevel(logging.INFO)

from utils.recent_logs import install_recent_logs_handler

install_recent_logs_handler()

from config import DEV_MODE, ORCHESTRATOR_URL_PROD, ORCHESTRATOR_URL_DEV
from utils.shutdown_handler import start_shutdown_server, SHUTDOWN_EVENT

def listen_for_ipc_commands(client: "DGNClient"):
    """
    Listens for JSON commands from stdin (sent by the parent Electron process).
    """
    logging.info("IPC listener thread started.")
    for line in sys.stdin:
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
                    watcher = getattr(client, "realtime_watcher", None)
                    if watcher:
                        watcher.update_token(payload["access_token"])
                else:
                    logging.warning("UPDATE_TOKENS command received with invalid payload.")
            elif cmd_type == "AUTH_FAILED_PERMANENTLY":
                logging.error("Received AUTH_FAILED_PERMANENTLY command from main process.")
                client.orchestrator_service.mark_auth_failed_permanently()
                SHUTDOWN_EVENT.set()
            elif cmd_type == "UPDATE_ROUTING_CONFIG":
                logging.info("Received UPDATE_ROUTING_CONFIG command from main process.")
                if isinstance(payload, dict):
                    client.apply_routing_config(_sanitize_routing_config(payload))
                else:
                    logging.warning("UPDATE_ROUTING_CONFIG command received with invalid payload.")
            elif cmd_type == "REQUEST_STOP":
                logging.info("Received REQUEST_STOP command from main process.")
                client.stop_requested = True
                download_manager = getattr(client, "download_manager", None)
                if download_manager:
                    download_manager.shutdown()

                if client.current_job:
                    job_id = client.current_job.get("id")
                    if job_id:
                        client.interrupted_job_id = job_id
                        client.interrupted_job_execution_token = client.current_job.get("execution_token")
                        logging.info(
                            f"Stop requested while job {job_id} is in progress. "
                            "It will be reset to 'pending' during shutdown cleanup."
                        )

                SHUTDOWN_EVENT.set()
            elif cmd_type == "CANCEL_DOWNLOAD":
                service_type = payload.get("service_type") if isinstance(payload, dict) else None
                if not isinstance(service_type, str) or not _SERVICE_ID_RE.match(service_type):
                    logging.warning(f"CANCEL_DOWNLOAD received invalid service_type: {service_type!r}")
                else:
                    logging.info(f"Received CANCEL_DOWNLOAD command for {service_type}.")
                    if client.download_manager:
                        client.download_manager.cancel_download(service_type)
            elif cmd_type == "SYNC_CACHED_IMAGES":
                logging.info("Received SYNC_CACHED_IMAGES command from main process.")
                if client.download_manager:
                    client.download_manager.sync_cached_images_with_server()
            elif cmd_type == "UPDATE_STORAGE_CONFIG":
                logging.info("Received UPDATE_STORAGE_CONFIG command from main process.")
                if isinstance(payload, dict) and client.download_manager:
                    client.download_manager.update_cache_limit_gb(
                        payload.get("docker_image_cache_limit_gb")
                    )
                    if hasattr(client, "job_wakeup_event"):
                        client.job_wakeup_event.set()
            elif cmd_type == "SET_COMPACTION_PENDING":
                pending = bool(payload.get("pending")) if isinstance(payload, dict) else False
                client.compaction_pending = pending
                if client.download_manager and hasattr(client.download_manager, "set_compaction_paused"):
                    client.download_manager.set_compaction_paused(pending)
                if not pending and hasattr(client, "job_wakeup_event"):
                    client.job_wakeup_event.set()
                logging.info(f"Set compaction_pending={pending}.")
            else:
                logging.warning(f"Received unknown IPC command type: {cmd_type}")

        except json.JSONDecodeError:
            logging.warning(f"Could not decode IPC command from stdin: {line.strip()}")
        except Exception as e:
            logging.error(f"Error processing IPC command: {e}")
        if SHUTDOWN_EVENT.is_set():
            break
    logging.info("IPC listener thread stopped.")
    if not SHUTDOWN_EVENT.is_set():
        logging.warning("Stdin closed (EOF) in IPC listener thread. Parent process probably exited. Initiating shutdown.")
        SHUTDOWN_EVENT.set()


def setup_client(args):
    import requests
    from dgn_client import DGNClient

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
        dgn_api_key=args.dgn_api_key,
        process_own_jobs=args.process_own_jobs,
        community_mode=args.community_mode,
        allowed_targets=args.allowed_targets.split(',') if args.allowed_targets else None,
        monetize_mode=args.monetize_mode,
    )

    client.load_config() # Fetch config from orchestrator

    # Filter compatible_services based on SELECTED_WORKFLOWS env var
    # This allows cloud deployments to specify which workflows this client should handle
    selected_workflows_env = os.environ.get("SELECTED_WORKFLOWS", "").strip()
    if selected_workflows_env:
        selected_workflows = [w.strip() for w in selected_workflows_env.split(",") if w.strip()]
        if selected_workflows:
            # Get the services required by the selected workflows
            allowed_services = set()
            for wf_id in selected_workflows:
                if wf_id in client.config:
                    service_name = client.config[wf_id].get("service_name")
                    if service_name:
                        allowed_services.add(service_name)
                else:
                    logging.warning(f"Unknown workflow in SELECTED_WORKFLOWS: {wf_id}")
            
            if allowed_services:
                # Intersect with VRAM-compatible services
                original_count = len(client.compatible_services)
                client.compatible_services = client.compatible_services & allowed_services
                logging.info(f"SELECTED_WORKFLOWS filter: {selected_workflows_env}")
                logging.info(f"Filtered services: {original_count} -> {len(client.compatible_services)} ({', '.join(sorted(client.compatible_services))})")
            else:
                logging.warning("No valid services found for SELECTED_WORKFLOWS. Using all compatible services.")

    from config import HEADLESS_MODE

    # Validate service argument after loading config
    if args.service != 'auto':
        available_services = list(client.docker_image_map.keys())
        if args.service not in available_services:
            logging.error(f"Invalid service '{args.service}'. Available services from config: {', '.join(available_services)}")
            sys.exit(1)
        if client.services_config.get(args.service, {}).get("disabled"):
            logging.error(f"Service '{args.service}' is disabled in the DGN configuration.")
            sys.exit(1)
        configured_workflows = [
            workflow_type
            for workflow_type, workflow_config in client.config.items()
            if workflow_config.get("service_name") == args.service
            and not workflow_config.get("disabled")
        ]
        if (
            HEADLESS_MODE
            and configured_workflows
            and args.service not in client.processable_services
        ):
            logging.error(
                "Service '%s' is configured but this DGN client build cannot "
                "load any processor for its workflows: %s. Refusing to register "
                "a headless provider that would fail assigned jobs.",
                args.service,
                ", ".join(configured_workflows) or "(none)",
            )
            sys.exit(1)
        if not HEADLESS_MODE and args.service not in client.compatible_services:
            logging.error(
                f"Service '{args.service}' is not compatible with this client. "
                "It may exceed your GPU/RAM requirements or the OpenFork Docker "
                "image storage limit configured in Docker Management settings."
            )
            sys.exit(1)

    # Scan for cached Docker images before registration (for smart job assignment)
    # This doesn't affect credits - just helps route jobs to providers with cached images
    cached_images = []
    if client.download_manager and client.services_config:
        all_service_types = list(client.compatible_services)
        cached_images = client.download_manager.get_cached_service_types(all_service_types)


    # In headless mode, the Docker container only has ONE service's models installed
    # So we must restrict supported_services to only that service, not all VRAM-compatible ones
    if HEADLESS_MODE and args.service != 'auto':
        # Only claim to support the exact service this container has
        registration_services = [args.service]
        logging.info(f"Headless mode: restricting supported_services to [{args.service}] (Docker image only has this service)")
        
        # Headless clients exist to provide network capacity, not for personal use.
        # Monetize headless clients are a separate paid-job pool and must not be
        # rewritten into the public credit pool.
        if client.monetize_mode:
            if client.community_mode != 'none' or client.process_own_jobs:
                logging.warning(
                    "Headless monetize mode: restricting routing to paid jobs "
                    "(community_mode='none', process_own_jobs=False)"
                )
            client.community_mode = 'none'
            client.process_own_jobs = False
            client.allowed_ids = []
        elif client.community_mode != 'all' or client.process_own_jobs:
            logging.warning(
                f"Headless mode: overriding routing config to community_mode='all', "
                f"process_own_jobs=False (headless clients serve public network only)"
            )
            client.community_mode = 'all'
            client.process_own_jobs = False
            client.allowed_ids = []  # Clear any user-specific filtering
        
        # TIER 0 routing ensured by reporting cached image
        if args.service not in cached_images:
            cached_images.append(args.service)
            logging.info(f"Headless mode: auto-adding {args.service} to cached_images (container is pre-baked)")

    else:
        registration_services = list(client.compatible_services)
    
    registration_result = client.orchestrator_service.register_with_orchestrator(
        service_type=args.service,
        supported_services=registration_services,
        cached_images=cached_images,
        process_own_jobs=client.process_own_jobs,
        community_mode=client.community_mode,
        allowed_ids=client.allowed_ids,
        monetize_mode=client.monetize_mode,
        hardware_profile=client.hardware_profile,
    )
    if not registration_result:
        raise RuntimeError("Failed to register with orchestrator. Aborting startup.")
    
    provider_id = registration_result.get("provider_id")
    user_id = registration_result.get("user_id")
    print(
        json.dumps(
            {
                "type": "PROVIDER_REGISTERED",
                "payload": {"provider_id": provider_id},
            }
        ),
        flush=True,
    )
    
    # For process_own_jobs, ensure allowed_ids contains the user's ID
    if client.process_own_jobs and user_id:
        if user_id not in client.allowed_ids:
            client.allowed_ids.append(user_id)
            logging.info(f"process_own_jobs: Added user_id from registration to allowed_ids")
    
    # Update download manager with provider_id for reporting newly cached images
    if client.download_manager:
        client.download_manager.orchestrator_service = client.orchestrator_service
        client.download_manager.provider_id = provider_id
    
    return client, provider_id

def run_client(client, provider_id, service_mode):
    from services.heartbeat_manager import HeartbeatManager
    from services.job_listener import JobListener

    heartbeat_manager = HeartbeatManager(client.orchestrator_service, provider_id, SHUTDOWN_EVENT, client=client)
    heartbeat_manager.start()

    # Start Realtime job watcher in OAuth (Electron) mode so providers are
    # woken up by Supabase Realtime instead of waiting for the poll interval.
    # Headless/API-key mode has no user JWT and keeps its own poll cycle.
    access_token = getattr(client.orchestrator_service, "access_token", None)
    if access_token and not client.orchestrator_service.use_api_key:
        from services.realtime_job_watcher import RealtimeJobWatcher
        watcher = RealtimeJobWatcher(
            access_token=access_token,
            wakeup_event=client.job_wakeup_event,
            shutdown_event=SHUTDOWN_EVENT,
            client=client,
        )
        client.realtime_watcher = watcher
        watcher.start()
        logging.info("Realtime job watcher started.")

    job_listener = JobListener(client, provider_id, SHUTDOWN_EVENT)

    print("DGN_CLIENT_RUNNING", flush=True)
    logging.info(f"DGN Client is running in '{service_mode}' mode and listening for jobs.")

    if service_mode == 'auto':
        job_listener.listen_for_jobs_auto()
    else:
        job_listener.listen_for_jobs()

def cleanup(client, provider_id, service_mode):
    logging.info("DGN Client: Initiating shutdown sequence.")

    interrupted_job_id = None
    interrupted_job_execution_token = None
    if client:
        if client.current_job:
            interrupted_job_id = client.current_job.get("id")
            interrupted_job_execution_token = client.current_job.get("execution_token")
        elif getattr(client, "interrupted_job_id", None):
            interrupted_job_id = client.interrupted_job_id
            interrupted_job_execution_token = getattr(
                client,
                "interrupted_job_execution_token",
                None,
            )

    if interrupted_job_id:
        should_reset = True
        try:
            job_details = client.orchestrator_service.get_job(interrupted_job_id)
            if isinstance(job_details, dict):
                current_status = job_details.get("status")
                if current_status and current_status != "processing":
                    should_reset = False
                    logging.info(
                        f"Job {interrupted_job_id} is already in terminal/non-processing state "
                        f"('{current_status}'). Skipping reset."
                    )
        except Exception as e:
            logging.warning(
                f"Could not verify current status for job {interrupted_job_id} before reset: {e}"
            )

    if interrupted_job_id and should_reset:
        logging.info(
            f"Job {interrupted_job_id} was interrupted during shutdown. "
            "Attempting to reset its status to 'pending'."
        )
        try:
            client.orchestrator_service.reset_interrupted_job(
                interrupted_job_id,
                execution_token=interrupted_job_execution_token,
                reason="provider_shutdown",
            )
        except Exception as e:
            logging.error(f"Failed to reset job {interrupted_job_id}: {e}", exc_info=True)

    if provider_id and client:
        logging.info("DGN Client: Attempting to deregister from orchestrator.")
        try:
            client.orchestrator_service.deregister_from_orchestrator(provider_id)
            logging.info("DGN Client: Successfully deregistered from orchestrator.")
        except Exception as e:
            logging.error(f"DGN Client: Failed to deregister from orchestrator: {e}", exc_info=True)
    
    # Skip Docker cleanup in headless mode (container isn't managed by us)
    from config import HEADLESS_MODE
    if HEADLESS_MODE:
        logging.info("DGN Client: Headless mode - skipping Docker cleanup.")
    else:
        from services.docker_manager import docker_manager

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
    
    # Authentication options (API key OR OAuth tokens)
    parser.add_argument('--dgn-api-key', type=str, help='DGN API Key for headless mode (alternative to OAuth tokens)')
    parser.add_argument('--access-token', type=str, help='Supabase Auth Access Token')
    parser.add_argument('--refresh-token', type=str, help='Supabase Auth Refresh Token')
    
    parser.add_argument('--service', type=str, default='auto', help='Service to run (e.g., wan22, foley). Default is "auto".')

    parser.add_argument('--root-dir', type=str, help='The root directory of the dgn-client.')
    parser.add_argument('--data-dir', type=str, help='The directory for storing user data.')
    parser.add_argument('--process-own-jobs', action='store_true', default=False, help='Pick up jobs submitted by your own user first (mine-policy jobs).')
    parser.add_argument('--community-mode', type=str, default='all', choices=['none', 'trusted_users', 'trusted_projects', 'all'], help='What community jobs to accept: none, trusted_users, trusted_projects, all.')
    parser.add_argument('--allowed-targets', type=str, help='For trusted_users/trusted_projects modes, a comma-separated list of targets.')
    # This argument is used solely for process identification by the cleanup logic
    parser.add_argument('--process-marker', type=str, help='Unique marker for process identification')
    parser.add_argument('--monetize-mode', action='store_true', default=False, help='Enable monetize mode: poll only for paid monetize jobs and emit MONETIZE_JOB_COMPLETE events.')
    args = parser.parse_args()

    # Validate authentication - need either API key OR OAuth tokens.
    # If tokens are not provided via CLI args (preferred, more secure), wait for
    # an UPDATE_TOKENS message from stdin before proceeding.
    if not args.dgn_api_key and not (args.access_token and args.refresh_token):
        logging.info("No tokens in CLI args — waiting for UPDATE_TOKENS via stdin...")
        try:
            line = sys.stdin.readline()
            if not line:
                logging.error("stdin closed before receiving initial tokens. Aborting.")
                sys.exit(1)
            init_msg = json.loads(line.strip())
            if init_msg.get("type") == "UPDATE_TOKENS":
                payload = init_msg.get("payload", {})
                args.access_token = payload.get("access_token")
                args.refresh_token = payload.get("refresh_token")
                logging.info("Received initial tokens via stdin.")
            else:
                logging.error(f"Expected UPDATE_TOKENS as first stdin message, got: {init_msg.get('type')}. Aborting.")
                sys.exit(1)
        except Exception as e:
            logging.error(f"Failed to read initial tokens from stdin: {e}")
            sys.exit(1)

    if not args.dgn_api_key and not (args.access_token and args.refresh_token):
        parser.error('Either --dgn-api-key OR both --access-token and --refresh-token are required')

    if args.service != 'auto':
        # In headless mode (running inside cloud container), don't manage Docker
        # - container is already running with ComfyUI
        from config import HEADLESS_MODE
        from services.docker_manager import docker_manager
        if not HEADLESS_MODE:
            docker_manager.run_container(service_type=args.service)
            # Start log streaming for dedicated service
            log_thread = threading.Thread(
                target=docker_manager.stream_logs,
                args=(args.service, SHUTDOWN_EVENT),
                daemon=True
            )
            log_thread.start()
        else:
            # Start log tailers for headless mode. Wan2GP backends do not write
            # to the ComfyUI log, so choose the backend log for this service.
            from utils.log_tailer import LogTailer, get_headless_log_paths

            log_paths = get_headless_log_paths(args.service)
            logging.info(
                "Headless mode detected - skipping Docker management. "
                "Backend should be running at 127.0.0.1:8188; tailing %s",
                ", ".join(log_paths),
            )
            for log_path in log_paths:
                tailer = LogTailer(log_path, service_type=args.service)
                log_thread = threading.Thread(
                    target=tailer.tail,
                    args=(SHUTDOWN_EVENT,),
                    daemon=True
                )
                log_thread.start()
    
    shutdown_thread = threading.Thread(target=start_shutdown_server, daemon=True)
    shutdown_thread.start()

    client = None
    provider_id = None
    try:
        client, provider_id = setup_client(args)

        # Start the IPC listener thread only when using OAuth tokens (Electron mode)
        # In headless API key mode, there's no parent process to listen to
        if not args.dgn_api_key:
            ipc_thread = threading.Thread(target=listen_for_ipc_commands, args=(client,), daemon=True)
            ipc_thread.start()

        # If we are running a dedicated service, set it as active on the client.
        if args.service != 'auto':
            client.active_service_type = args.service

        run_client(client, provider_id, args.service)
    except Exception as e:
        logging.error(f"A critical error occurred during client operation: {e}", exc_info=True)
    finally:
        cleanup(client, provider_id, args.service)

    logging.info("Main function completed.")


if __name__ == "__main__":
    main()
