import logging
import json
import threading
from typing import Dict, Any, Optional

from config import HEADLESS_MODE, TimeoutConfig
from services.docker_manager import docker_manager
from exceptions import AuthError, ProviderError
from services.orchestrator_service import TokenExpiredError, ProviderNotFoundError


class JobListener:
    """Listens for and processes DGN jobs from the orchestrator."""
    
    def __init__(self, client, provider_id: str, shutdown_event: threading.Event):
        self.client = client
        self.orchestrator_service = client.orchestrator_service
        self.provider_id = provider_id
        self.shutdown_event = shutdown_event

    def _process_job_safely(self, job: Dict[str, Any]) -> bool:
        """
        Process a job with proper error handling.
        
        Args:
            job: Job dictionary from the orchestrator
            
        Returns:
            True if processing completed (success or handled failure),
            False if a critical auth error occurred that should stop the listener.
        """
        try:
            processor = self.client._get_job_processor(job, self.shutdown_event)
            processor.process()
            return True
        except (TokenExpiredError, AuthError):
            self.orchestrator_service.signal_auth_expired()
            logging.warning(f"Auth expired during processing of job {job.get('id')}.")
            raise  # Re-raise to be caught by outer exception handler
        except Exception as e:
            logging.error(
                f"An error occurred while processing job {job.get('id')}: {e}",
                exc_info=True,
            )
            if job and job.get('id'):
                self.orchestrator_service.update_job_status(job.get('id'), 'failed')
            return True
        finally:
            self.client.current_job = None

    def listen_for_jobs(self) -> None:
        """Listen for jobs from the orchestrator (for dedicated providers)."""
        while not self.shutdown_event.is_set():
            job = None
            try:
                # Acquire the processing lock BEFORE fetching a job to ensure
                # only one job is acquired and processed at a time
                with self.client.processing_lock:
                    logging.info(f"Checking for new jobs for provider {self.provider_id}...")
                    job = self.orchestrator_service.get_next_job(
                        provider_id=self.provider_id,
                        accept_policy=self.client.accept_policy,
                        allowed_ids=self.client.allowed_ids
                    )

                    if job and job.get('id'):
                        self.client.current_job = job
                        logging.info(f"Received job: {job['id']}")
                        
                        # Process the job using the shared helper
                        self._process_job_safely(job)
                        
                        if not self.shutdown_event.is_set():
                            self.orchestrator_service.update_provider_status(self.provider_id, 'available')
                            logging.info("Provider status set to available. Waiting for next job...")
                    else:
                        logging.info("No new jobs.")
            except TokenExpiredError:
                self.orchestrator_service.signal_auth_expired()
                logging.warning("Could not fetch job due to expired token.")
                if self.orchestrator_service.is_auth_failed_permanently():
                    logging.error("Auth permanently failed. Stopping job listener.")
                    break
            except ProviderNotFoundError:
                logging.warning("Provider registration expired. Signaling main process for restart.")
                print(json.dumps({"status": "PROVIDER_EXPIRED"}), flush=True)
                break  # Exit the loop, Electron will restart the client
            except Exception as e:
                logging.error(f"Could not connect to the Orchestrator: {e}")

            if not (job and job.get('id')):
                    # Use faster polling frequency for headless cloud instances
                    poll_interval = TimeoutConfig.HEADLESS_JOB_POLL_INTERVAL if HEADLESS_MODE else TimeoutConfig.JOB_POLL_INTERVAL
                    self.shutdown_event.wait(poll_interval)
        logging.info("Shutdown event received. Exiting job listening loop.")

    def _get_service_type_for_job(self, job: Dict[str, Any]) -> Optional[str]:
        """Get the service type for a job based on its workflow type."""
        workflow_type = job.get('workflow_type', 'image_to_video')
        try:
            return self.client.get_service_type_for_workflow(workflow_type)
        except ValueError:
            # Unknown workflow type, fall back to service_type from job
            return job.get('service_type')

    def _handle_prefetch_suggestions(self, download_manager) -> None:
        """Fetch and apply pre-fetch suggestions from the server.
        
        When idle (no processable jobs), this proactively downloads Docker images
        that the server identifies as high-demand based on network-wide analysis:
        - Jobs pending in queue by service_type
        - Current cache coverage across all providers
        - This provider's capabilities
        
        This improves network efficiency by ensuring images are ready before
        jobs need them. The server considers:
        - Number of pending jobs per service_type
        - Number of providers with that image cached vs downloading
        - Cache deficit (pending jobs - cached providers)
        
        NOTE: This does NOT affect credits in any way. Credits are calculated
        based on actual processing time and VRAM usage when jobs complete.
        Pre-fetching is purely for reducing job wait times.
        """
        if not download_manager:
            return
        
        # Don't fetch suggestions if we already have downloads in progress
        # This prevents queue explosion and respects MAX_CONCURRENT_DOWNLOADS
        if download_manager._active_downloads or download_manager._download_queue:
            return
        
        try:
            suggestions = self.orchestrator_service.get_prefetch_suggestions(self.provider_id)
            
            if suggestions:
                logging.info(f"Received pre-fetch suggestions from server: {suggestions}")
                
                # Start downloads for suggested service types (download manager handles queueing)
                for service_type in suggestions[:2]:  # Limit to 2 to avoid queue buildup
                    if not download_manager.has_image(service_type) and \
                       not download_manager.is_downloading(service_type) and \
                       not download_manager.is_queued(service_type):
                        logging.info(f"Pre-fetching suggested image: {service_type}")
                        download_manager.start_background_download(service_type)
        except Exception as e:
            # Non-critical - just log and continue
            logging.debug(f"Failed to get/apply pre-fetch suggestions: {e}")


    def listen_for_jobs_auto(self) -> None:
        """Listen for jobs and dynamically start/stop containers with image pre-fetching."""
        logging.info("Entering auto job listening loop with Docker image pre-fetching.")
        
        # Get download manager from client (may be None in headless mode)
        download_manager = getattr(self.client, 'download_manager', None)
        
        try:
            while not self.shutdown_event.is_set():
                logging.info("Top of auto-mode loop iteration.")
                job = None
                found_processable_job = False
                
                try:
                    # Acquire the processing lock BEFORE fetching a job
                    with self.client.processing_lock:
                        logging.info("Auto mode: Peeking at available jobs...")
                        
                        # Step 1: Peek at available jobs without reserving
                        available_jobs = self.orchestrator_service.peek_available_jobs(
                            provider_id=self.provider_id,
                            accept_policy=self.client.accept_policy,
                            allowed_ids=self.client.allowed_ids,
                            limit=10
                        )
                        
                        if not available_jobs:
                            logging.info("No available jobs found in peek.")
                        else:
                            logging.info(f"Peeked {len(available_jobs)} available jobs.")
                            
                            # Step 2: Find first job with available Docker image
                            for peeked_job in available_jobs:
                                service_type = self._get_service_type_for_job(peeked_job)
                                if not service_type:
                                    continue
                                
                                # Check if Docker image is available
                                image_available = True
                                if not HEADLESS_MODE and download_manager:
                                    image_available = download_manager.has_image(service_type)
                                
                                if image_available:
                                    # Image is ready - reserve and process this job
                                    logging.info(f"Found job {peeked_job.get('id')} with available image ({service_type}). Reserving...")
                                    
                                    job = self.orchestrator_service.get_next_job(
                                        provider_id=self.provider_id,
                                        accept_policy=self.client.accept_policy,
                                        allowed_ids=self.client.allowed_ids,
                                        job_id=peeked_job.get('id')
                                    )
                                    
                                    if job and job.get('id'):
                                        found_processable_job = True
                                        self.client.current_job = job
                                        job_id = job['id']
                                        logging.info(f"Reserved job: {job_id}")
                                        
                                        workflow_type = job.get('workflow_type', 'image_to_video')
                                        actual_service_type = self.client.get_service_type_for_workflow(workflow_type)
                                        self.client.active_service_type = actual_service_type
                                        logging.info(f"Job requires service '{actual_service_type}'.")
                                        
                                        if not HEADLESS_MODE:
                                            logging.info("Starting container...")
                                            docker_manager.run_container(service_type=actual_service_type)
                                        else:
                                            logging.info("Headless mode - container already running, skipping Docker management.")
                                        
                                        # Check if service uses ComfyUI backend from configuration
                                        service_config = self.client.services_config.get(actual_service_type, {})
                                        uses_comfyui = service_config.get("backend", "comfyui") == "comfyui"
                                        
                                        if uses_comfyui:
                                            if self.client.comfyui_client.wait_for_ready(self.shutdown_event):
                                                # Process the job using shared helper
                                                self._process_job_safely(job)
                                            else:
                                                if not self.shutdown_event.is_set():
                                                    logging.error(f"ComfyUI for service '{actual_service_type}' failed to start. Failing job.")
                                                    self.orchestrator_service.update_job_status(job_id, 'failed')
                                                    self.client.current_job = None
                                        else:
                                            # For text_generation, directly process the job (no ComfyUI needed)
                                            self._process_job_safely(job)

                                        if not self.shutdown_event.is_set():
                                            logging.info(f"Job processing finished.")
                                            if not HEADLESS_MODE:
                                                logging.info(f"Stopping container for service '{actual_service_type}'...")
                                                docker_manager.stop_container(service_type=actual_service_type)
                                            self.client.active_service_type = None
                                            self.orchestrator_service.update_provider_status(self.provider_id, 'available')
                                            logging.info("Provider status set to available. Waiting for next job...")
                                    
                                    break  # Exit the peeked jobs loop after processing one job
                                else:
                                    # Image not available - start background download
                                    if download_manager and not download_manager.is_downloading(service_type) and not download_manager.is_queued(service_type):
                                        logging.info(f"Image for service '{service_type}' not available. Starting background download...")
                                        download_manager.start_background_download(service_type)
                                    elif download_manager and (download_manager.is_downloading(service_type) or download_manager.is_queued(service_type)):
                                        logging.debug(f"Image for service '{service_type}' already downloading/queued.")
                                    # Continue to check next job
                        
                        # Handle pre-fetch suggestions when idle
                        # This proactively downloads images for high-demand workflows
                        # NOTE: This does NOT affect credits - it's purely for network efficiency
                        if not found_processable_job:
                            self._handle_prefetch_suggestions(download_manager)
                        
                        if not found_processable_job and available_jobs:
                            logging.info("All available jobs require images that are still downloading. Waiting...")
                        elif not found_processable_job:
                            logging.info("No new jobs found in this check.")

                            
                except TokenExpiredError:
                    self.orchestrator_service.signal_auth_expired()
                    logging.warning("Could not fetch job due to expired token.")
                    if self.orchestrator_service.is_auth_failed_permanently():
                        logging.error("Auth permanently failed. Stopping job listener.")
                        break
                except ProviderNotFoundError:
                    logging.warning("Provider registration expired. Signaling main process for restart.")
                    print(json.dumps({"status": "PROVIDER_EXPIRED"}), flush=True)
                    break  # Exit the loop, Electron will restart the client
                except Exception as e:
                    logging.error(f"An error occurred in auto job listening loop: {e}", exc_info=True)

                if not found_processable_job:
                    # Use faster polling frequency for headless cloud instances
                    poll_interval = TimeoutConfig.HEADLESS_JOB_POLL_INTERVAL if HEADLESS_MODE else TimeoutConfig.JOB_POLL_INTERVAL
                    self.shutdown_event.wait(poll_interval)
        finally:
            logging.info("Shutdown event received or loop exited. Exiting auto job listening loop.")
            
            # Shutdown the download manager
            if download_manager:
                download_manager.shutdown()
            
            if self.client.active_service_type and not HEADLESS_MODE:
                logging.info(f"Ensuring container for service '{self.client.active_service_type}' is stopped.")
                docker_manager.stop_container(service_type=self.client.active_service_type)
                self.client.active_service_type = None

