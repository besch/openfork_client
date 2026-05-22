import unittest
from types import SimpleNamespace
from unittest.mock import Mock

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


if __name__ == "__main__":
    unittest.main()
