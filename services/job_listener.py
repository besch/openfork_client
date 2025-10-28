import logging
from services.docker_manager import docker_manager

class JobListener:
    def __init__(self, client, shutdown_event):
        self.client = client
        self.orchestrator_service = client.orchestrator_service
        self.shutdown_event = shutdown_event



    def listen_for_jobs(self):
        """Listen for jobs and dynamically start/stop containers."""
        logging.info("Entering job listening loop.")
        while not self.shutdown_event.is_set():
            logging.info("Top of loop iteration.")
            job = None
            try:
                logging.info("Checking for new jobs...")
                job = self.orchestrator_service.get_next_job(
                    provider_id=self.client.provider_id,
                    accept_policy=self.client.accept_policy,
                    allowed_ids=self.client.allowed_ids
                )

                if job and job.get('id'):
                    self.client.current_job = job
                    job_id = job['id']
                    logging.info(f"Received job: {job_id}")

                    # Fetch the full workflow template to get dependencies
                    workflow_template = self.orchestrator_service.get_workflow_template(job['workflow_template_id'])
                    dependencies = {}
                    if workflow_template:
                        dependencies['custom_node_urls'] = workflow_template.get('custom_node_urls', [])
                        dependencies['model_urls'] = workflow_template.get('model_urls', [])
                    
                    logging.info(f"Starting unified ComfyUI container with dependencies: {dependencies}")
                    docker_manager.run_container(dependencies=dependencies)
                    
                    if self.client.comfyui_client.wait_for_ready(self.shutdown_event):
                        self.client._process_job(job, self.shutdown_event)
                    else:
                        if not self.shutdown_event.is_set():
                            logging.error(f"ComfyUI failed to start. Failing job.")
                            self.orchestrator_service.update_job_status(job_id, 'failed')

                    if not self.shutdown_event.is_set():
                        logging.info(f"Job processing finished. Stopping unified ComfyUI container...")
                        docker_manager.stop_container()
                        self.orchestrator_service.update_provider_status(self.client.provider_id, 'available')
                        logging.info("Provider status set to available. Waiting for next job...")
                        self.client.current_job = None
                else:
                    logging.info("No new jobs found in this check.")
            except Exception as e:
                logging.error(f"An error occurred in job listening loop: {e}", exc_info=True)

            if not (job and job.get('id')):
                self.shutdown_event.wait(10)
        logging.info("Shutdown event received. Exiting job listening loop.")
