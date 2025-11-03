import logging
import time
from services.docker_manager import docker_manager
from services.auto_installer import fix_all_custom_node_dependencies

class JobListener:
    def __init__(self, client, shutdown_event):
        self.client = client
        self.orchestrator_service = client.orchestrator_service
        self.shutdown_event = shutdown_event
        self.container_started = False
        self.startup_fix_run = False  # Track if we've run fix on startup

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

    def _run_startup_maintenance(self):
        """
        Run maintenance tasks after container startup to ensure reliability.
        This includes fixing custom node dependencies and refreshing nodes.
        """
        if self.startup_fix_run:
            return True
        
        try:
            logging.info("=" * 60)
            logging.info("Running startup maintenance tasks...")
            logging.info("=" * 60)
            
            # 1. Fix all custom node dependencies
            # This ensures any existing custom nodes have their deps installed
            logging.info("Step 1/3: Fixing custom node dependencies...")
            fix_result = fix_all_custom_node_dependencies()
            
            if fix_result:
                logging.info("[OK] Dependencies fixed successfully")
            else:
                logging.warning("[WARNING] Dependency fix reported issues, but continuing...")
            
            # 2. Wait for nodes to initialize
            logging.info("Step 2/3: Waiting for nodes to initialize...")
            time.sleep(5)
            
            # 3. Refresh node cache
            logging.info("Step 3/3: Refreshing node cache...")
            self.client.comfyui_client.refresh_nodes()
            
            # Verify we can fetch nodes
            nodes = self.client.comfyui_client.get_installed_nodes(use_cache=False)
            logging.info(f"[OK] Startup maintenance complete. {len(nodes)} nodes available.")
            logging.info("=" * 60)
            
            self.startup_fix_run = True
            return True
            
        except Exception as e:
            logging.error(f"Error during startup maintenance: {e}", exc_info=True)
            # Don't fail completely - try to continue
            return False

    def _handle_container_restart(self):
        """
        Handle container restart with proper maintenance.
        """
        logging.warning("Container needs restart. Restarting...")
        
        try:
            # Restart the container
            if not docker_manager.restart_container():
                logging.error("Failed to restart container")
                return False
            
            # Wait for container to be ready
            if not self.client.comfyui_client.wait_for_ready(self.shutdown_event, timeout=180):
                logging.error("Container failed to become ready after restart")
                return False
            
            # Reset startup fix flag so maintenance runs again
            self.startup_fix_run = False
            
            # Run maintenance tasks
            self._run_startup_maintenance()
            
            logging.info("Container restarted and maintenance completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error during container restart: {e}", exc_info=True)
            return False

    def listen_for_jobs(self):
        """
        Listen for jobs with persistent container management.
        Container stays running across jobs for better performance.
        
        IMPROVEMENTS:
        - Run startup maintenance (cm-cli fix) after container starts
        - Better error recovery with container health checks
        - Proper maintenance after restarts
        """
        logging.info("Entering job listening loop.")
        
        # Ensure container is running at startup
        if not self._ensure_container_running():
            logging.error("Failed to start container at startup. Exiting.")
            return
        
        # Run startup maintenance
        self._run_startup_maintenance()
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while not self.shutdown_event.is_set():
            logging.debug("Top of loop iteration.")
            job = None
            
            try:
                # Check if container is still healthy
                if not docker_manager.is_container_running():
                    logging.warning("Container stopped unexpectedly.")
                    
                    if not self._ensure_container_running():
                        logging.error("Failed to restart container. Waiting before retry...")
                        self.shutdown_event.wait(30)
                        continue
                    
                    # Run maintenance after restart
                    self._run_startup_maintenance()
                
                # Also check if ComfyUI API is responding
                if not self.client.comfyui_client.check_health(use_cache=False):
                    logging.warning("ComfyUI API not responding. Container may be unhealthy.")
                    
                    # Try to recover
                    if not self._handle_container_restart():
                        logging.error("Failed to recover container. Waiting before retry...")
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
                    try:
                        self.client._process_job(job, self.shutdown_event)
                    except Exception as job_error:
                        logging.error(f"Error processing job {job_id}: {job_error}", exc_info=True)
                        
                        # Try to mark job as failed
                        try:
                            self.orchestrator_service.update_job_status(job_id, 'failed')
                        except Exception as update_error:
                            logging.error(f"Failed to update job status: {update_error}")
                    
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
                        f"Hit {max_consecutive_errors} consecutive errors. Attempting full recovery..."
                    )
                    
                    if self._handle_container_restart():
                        logging.info("Container recovered successfully after errors")
                        consecutive_errors = 0  # Reset counter after successful recovery
                    else:
                        logging.error("Failed to recover container. Will retry next iteration.")
                
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