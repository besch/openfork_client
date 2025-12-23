import logging
import json
from config import HEADLESS_MODE
from services.docker_manager import docker_manager
from services.orchestrator_service import TokenExpiredError, ProviderNotFoundError

class JobListener:
    def __init__(self, client, provider_id, shutdown_event):
        self.client = client
        self.orchestrator_service = client.orchestrator_service
        self.provider_id = provider_id
        self.shutdown_event = shutdown_event

    def listen_for_jobs(self):
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
                        
                        # Process the job while holding the lock
                        try:
                            processor = self.client._get_job_processor(job, self.shutdown_event)
                            processor.process()
                        except TokenExpiredError:
                            self.orchestrator_service.signal_auth_expired()
                            logging.warning(
                                f"Auth expired during processing of job {job.get('id')}."
                            )
                            raise  # Re-raise to be caught by outer exception handler
                        except Exception as e:
                            logging.error(
                                f"An error occurred while processing job {job.get('id')}: {e}",
                                exc_info=True,
                            )
                            if job and job.get('id'):
                                self.orchestrator_service.update_job_status(job.get('id'), 'failed')
                        finally:
                            self.client.current_job = None
                        
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
                self.shutdown_event.wait(10)
        logging.info("Shutdown event received. Exiting job listening loop.")

    def listen_for_jobs_auto(self):
        """Listen for jobs and dynamically start/stop containers."""
        logging.info("Entering auto job listening loop.")
        try:
            while not self.shutdown_event.is_set():
                logging.info("Top of auto-mode loop iteration.")
                job = None
                try:
                    # Acquire the processing lock BEFORE fetching a job to ensure
                    # only one job is acquired and processed at a time
                    with self.client.processing_lock:
                        logging.info("Auto mode: Checking for new jobs...")
                        job = self.orchestrator_service.get_next_job(
                            provider_id=self.provider_id,
                            accept_policy=self.client.accept_policy,
                            allowed_ids=self.client.allowed_ids
                        )

                        if job and job.get('id'):
                            self.client.current_job = job
                            job_id = job['id']
                            logging.info(f"Received job: {job_id}")

                            workflow_type = job.get('workflow_type', 'image_to_video')
                            service_type = self.client.get_service_type_for_workflow(workflow_type)
                            self.client.active_service_type = service_type
                            logging.info(f"Job requires service '{service_type}'.")
                            if not HEADLESS_MODE:
                                logging.info("Starting container...")
                                docker_manager.run_container(service_type=service_type)
                            else:
                                logging.info("Headless mode - container already running, skipping Docker management.")
                            
                            # Check if service uses ComfyUI backend from configuration
                            service_config = self.client.services_config.get(service_type, {})
                            uses_comfyui = service_config.get("backend", "comfyui") == "comfyui"
                            
                            if uses_comfyui:
                                if self.client.comfyui_client.wait_for_ready(self.shutdown_event):
                                    # Process the job while holding the lock
                                    try:
                                        processor = self.client._get_job_processor(job, self.shutdown_event)
                                        processor.process()
                                    except TokenExpiredError:
                                        self.orchestrator_service.signal_auth_expired()
                                        logging.warning(
                                            f"Auth expired during processing of job {job.get('id')}."
                                        )
                                        raise  # Re-raise to be caught by outer exception handler
                                    except Exception as e:
                                        logging.error(
                                            f"An error occurred while processing job {job.get('id')}: {e}",
                                            exc_info=True,
                                        )
                                        if job and job.get('id'):
                                            self.orchestrator_service.update_job_status(job.get('id'), 'failed')
                                    finally:
                                        self.client.current_job = None
                                else:
                                    if not self.shutdown_event.is_set():
                                        logging.error(f"ComfyUI for service '{service_type}' failed to start. Failing job.")
                                        self.orchestrator_service.update_job_status(job_id, 'failed')
                                        self.client.current_job = None
                            else:
                                # For text_generation, directly process the job (no ComfyUI needed)
                                try:
                                    processor = self.client._get_job_processor(job, self.shutdown_event)
                                    processor.process()
                                except TokenExpiredError:
                                    self.orchestrator_service.signal_auth_expired()
                                    logging.warning(
                                        f"Auth expired during processing of job {job.get('id')}."
                                    )
                                    raise  # Re-raise to be caught by outer exception handler
                                except Exception as e:
                                    logging.error(
                                        f"An error occurred while processing job {job.get('id')}: {e}",
                                        exc_info=True,
                                    )
                                    if job and job.get('id'):
                                        self.orchestrator_service.update_job_status(job.get('id'), 'failed')
                                finally:
                                    self.client.current_job = None

                            if not self.shutdown_event.is_set():
                                logging.info(f"Job processing finished.")
                                if not HEADLESS_MODE:
                                    logging.info(f"Stopping container for service '{service_type}'...")
                                    docker_manager.stop_container(service_type=service_type)
                                self.client.active_service_type = None
                                self.orchestrator_service.update_provider_status(self.provider_id, 'available')
                                logging.info("Provider status set to available. Waiting for next job...")
                        else:
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

                if not (job and job.get('id')):
                    self.shutdown_event.wait(10)
        finally:
            logging.info("Shutdown event received or loop exited. Exiting auto job listening loop.")
            if self.client.active_service_type:
                logging.info(f"Ensuring container for service '{self.client.active_service_type}' is stopped.")
                docker_manager.stop_container(service_type=self.client.active_service_type)
                self.client.active_service_type = None
