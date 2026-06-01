import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import docker

from services.docker_prod_service import DockerProdManager


class DockerProdManagerStopContainerTests(unittest.TestCase):
    def test_stop_container_treats_removal_in_progress_as_benign(self):
        manager = DockerProdManager.__new__(DockerProdManager)
        manager.client = Mock()
        manager._wait_for_container_removal = Mock(return_value=True)

        container = Mock()
        container.id = "a" * 64
        container.remove.side_effect = docker.errors.APIError(
            "409 Client Error",
            response=SimpleNamespace(
                status_code=409,
                url="http://127.0.0.1:2375/v1.54/containers/test",
                reason="Conflict",
            ),
            explanation=(
                f"removal of container {container.id} is already in progress"
            ),
        )
        manager.client.containers.get.return_value = container

        with self.assertLogs(level="INFO") as logs:
            manager.stop_container("flux-kontext-dev-8gb")

        manager._wait_for_container_removal.assert_called_once_with(
            "dgn-client-flux-kontext-dev-8gb",
            container_id=container.id,
            timeout=15.0,
        )
        self.assertTrue(
            any("removal is already in progress" in entry for entry in logs.output)
        )

    def test_large_image_uses_extended_start_timeout(self):
        manager = DockerProdManager.__new__(DockerProdManager)
        manager.services_config = {
            "turbodiffusion-8gb": {
                "disk_required_gb": 160,
            }
        }

        self.assertGreaterEqual(
            manager.get_start_api_timeout("turbodiffusion-8gb"),
            600,
        )

    def test_final_name_conflict_gets_cleanup_retry(self):
        manager = DockerProdManager.__new__(DockerProdManager)
        manager.client = Mock()
        manager.services_config = {"turbodiffusion-8gb": {"disk_required_gb": 160}}
        manager.docker_image_map = {
            "turbodiffusion-8gb": "beschiak/openfork-turbodiffusion:latest"
        }
        manager.pull_image = Mock()
        manager._set_api_timeout = Mock(return_value=None)
        manager._restore_api_timeout = Mock()
        manager._poll_conflicting_container_state = Mock(return_value="exited")
        manager._force_remove_by_name = Mock()
        manager._wait_for_container_removal = Mock(return_value=True)

        conflict_id = "a" * 64
        conflict = docker.errors.APIError(
            "409 Client Error",
            response=SimpleNamespace(
                status_code=409,
                url="http://127.0.0.1:2375/v1.54/containers/create",
                reason="Conflict",
            ),
            explanation=(
                'Conflict. The container name "/dgn-client-turbodiffusion-8gb" '
                f'is already in use by container "{conflict_id}".'
            ),
        )

        manager.client.containers.get.side_effect = docker.errors.NotFound(
            "not found"
        )
        manager.client.containers.list.return_value = []
        manager.client.containers.run.side_effect = [conflict, None]
        manager.client.api.inspect_container.return_value = {
            "State": {"Status": "exited"}
        }

        with patch.dict("os.environ", {"OPENFORK_DOCKER_START_MAX_ATTEMPTS": "1"}):
            manager.run_container("turbodiffusion-8gb")

        self.assertEqual(manager.client.containers.run.call_count, 2)
        manager._force_remove_by_name.assert_called_once()


if __name__ == "__main__":
    unittest.main()
