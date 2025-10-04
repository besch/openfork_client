import logging
from services.docker_manager import docker_manager

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
                logging.info(f"Checking for new jobs for provider {self.provider_id}...")
                job = self.orchestrator_service.get_next_job(self.provider_id)

                if job and job.get('id'):
                    self.client.current_job = job
                    logging.info(f"Received job: {job['id']}")
                    self.client._process_job(job, self.shutdown_event)
                    
                    if not self.shutdown_event.is_set():
                        self.orchestrator_service.update_provider_status(self.provider_id, 'available')
                        logging.info("Provider status set to available. Waiting for next job...")
                        self.client.current_job = None
                else:
                    logging.info("No new jobs.")
            except Exception as e:
                logging.error(f"Could not connect to the Orchestrator: {e}")

            if not (job and job.get('id')):
                self.shutdown_event.wait(10)
        logging.info("Shutdown event received. Exiting job listening loop.")

    def listen_for_jobs_auto(self):
        """Listen for jobs and dynamically start/stop containers."""
        while not self.shutdown_event.is_set():
            job = None
            try:
                logging.info("Auto mode: Checking for new jobs...")
                job = self.orchestrator_service.get_next_job(self.provider_id)

                if job and job.get('id'):
                    self.client.current_job = job
                    job_id = job['id']
                    logging.info(f"Received job: {job_id}")

                    workflow_type = job.get('workflow_type', 'image_to_video')
                    service_type = self.client.get_service_type_for_workflow(workflow_type)
                    self.client.active_service_type = service_type
                    
                    logging.info(f"Job requires service '{service_type}'. Starting container...")
                    docker_manager.run_container(service_type=service_type)
                    
                    if self.client.comfyui_client.wait_for_ready(self.shutdown_event):
                        self.client._process_job(job, self.shutdown_event)
                    else:
                        if not self.shutdown_event.is_set():
                            logging.error(f"ComfyUI for service '{service_type}' failed to start. Failing job.")
                            self.orchestrator_service.update_job_status(job_id, 'failed')

                    if not self.shutdown_event.is_set():
                        logging.info(f"Job processing finished. Stopping container for service '{service_type}'...")
                        docker_manager.stop_container(service_type=service_type)
                        self.client.active_service_type = None
                        self.orchestrator_service.update_provider_status(self.provider_id, 'available')
                        logging.info("Provider status set to available. Waiting for next job...")
                        self.client.current_job = None
                else:
                    logging.info("No new jobs.")
            except Exception as e:
                logging.error(f"An error occurred in auto job listening loop: {e}", exc_info=True)

            if not (job and job.get('id')):
                self.shutdown_event.wait(10)
        logging.info("Shutdown event received. Exiting auto job listening loop.")
