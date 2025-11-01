import logging
from services.docker_manager import docker_manager

class JobListener:
    def __init__(self, client, shutdown_event):
        self.client = client
        self.orchestrator_service = client.orchestrator_service
        self.shutdown_event = shutdown_event
        self.container_started = False

    def _ensure_container_running(self):
        """
        Ensure the ComfyUI container is running.
        Starts it if not already running.
        """
        if docker_manager.is_container_running():
            logging.debug("Container is already running")
            return True
        
        logging.info("Starting ComfyUI container...")
        try:
            # Start with no initial dependencies - they'll be installed dynamically per job
            docker_manager.run_container(dependencies={'custom_node_urls': [], 'model_urls': []})
            self.container_started = True
            
            # Wait for container to be ready
            if not self.client.comfyui_client.wait_for_ready(self.shutdown_event, timeout=180):
                logging.error("ComfyUI container failed to become ready")
                return False
            
            logging.info("ComfyUI container is ready")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start container: {e}", exc_info=True)
            return False

    def listen_for_jobs(self):
        """
        Listen for jobs with persistent container management.
        Container stays running across jobs for better performance.
        """
        logging.info("Entering job listening loop.")
        
        # Ensure container is running at startup
        if not self._ensure_container_running():
            logging.error("Failed to start container at startup. Exiting.")
            return
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while not self.shutdown_event.is_set():
            logging.debug("Top of loop iteration.")
            job = None
            
            try:
                # Check if container is still healthy
                if not docker_manager.is_container_running():
                    logging.warning("Container stopped unexpectedly. Restarting...")
                    if not self._ensure_container_running():
                        logging.error("Failed to restart container. Waiting before retry...")
                        self.shutdown_event.wait(30)
                        continue
                
                # Check for new jobs
                logging.debug("Checking for new jobs...")
                job = self.orchestrator_service.get_next_job(
                    provider_id=self.client.provider_id,
                    accept_policy=self.client.accept_policy,
                    allowed_ids=self.client.allowed_ids
                )

                if job and job.get('id'):
                    consecutive_errors = 0  # Reset error counter on successful job fetch
                    self.client.current_job = job
                    job_id = job['id']
                    
                    logging.info(f"Received job: {job_id}")
                    
                    # Update provider status to busy
                    self.orchestrator_service.update_provider_status(
                        self.client.provider_id, 
                        'busy'
                    )
                    
                    # Process the job
                    # The job processor will handle dependency installation internally
                    self.client._process_job(job, self.shutdown_event)
                    
                    # Job completed - update provider status
                    if not self.shutdown_event.is_set():
                        self.orchestrator_service.update_provider_status(
                            self.client.provider_id, 
                            'available'
                        )
                        logging.info("Job processing finished. Ready for next job.")
                        self.client.current_job = None
                        
                else:
                    logging.debug("No new jobs found in this check.")
                    consecutive_errors = 0  # Reset on successful check (even if no jobs)
                    
            except Exception as e:
                consecutive_errors += 1
                logging.error(
                    f"An error occurred in job listening loop (error {consecutive_errors}/{max_consecutive_errors}): {e}", 
                    exc_info=True
                )
                
                # If we hit max consecutive errors, try to recover
                if consecutive_errors >= max_consecutive_errors:
                    logging.error(
                        f"Hit {max_consecutive_errors} consecutive errors. Attempting container restart..."
                    )
                    try:
                        docker_manager.restart_container()
                        if self.client.comfyui_client.wait_for_ready(self.shutdown_event, timeout=180):
                            logging.info("Container restarted successfully after errors")
                            consecutive_errors = 0  # Reset counter after successful recovery
                        else:
                            logging.error("Container failed to become ready after restart")
                    except Exception as restart_error:
                        logging.error(f"Failed to restart container: {restart_error}")
                
                # Mark job as failed if we were processing one
                if job and job.get('id'):
                    try:
                        self.orchestrator_service.update_job_status(job['id'], 'failed')
                    except Exception as update_error:
                        logging.error(f"Failed to update job status: {update_error}")

            # Wait before next poll (only if no job was found)
            if not (job and job.get('id')):
                self.shutdown_event.wait(10)
        
        # Cleanup on shutdown
        logging.info("Shutdown event received. Cleaning up...")
        
        try:
            if self.container_started:
                logging.info("Stopping container...")
                docker_manager.stop_container()
                logging.info("Container stopped")
        except Exception as e:
            logging.error(f"Error stopping container during shutdown: {e}")
        
        logging.info("Exiting job listening loop.")