import unittest
import zlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import docker

from services import docker_prod_service
from services.docker_prod_service import DockerProdManager


class DockerProdManagerStopContainerTests(unittest.TestCase):
    def test_windows_host_candidates_ignore_inherited_docker_desktop_host(self):
        manager = DockerProdManager.__new__(DockerProdManager)

        with (
            patch.object(docker_prod_service.os, "name", "nt"),
            patch.dict(
                "os.environ",
                {"DOCKER_HOST": "npipe:////./pipe/dockerDesktopLinuxEngine"},
                clear=True,
            ),
        ):
            hosts = manager._build_docker_host_candidates()

        self.assertEqual(hosts[0], "tcp://127.0.0.1:2375")
        self.assertIn("tcp://localhost:2375", hosts)
        self.assertNotIn("npipe:////./pipe/dockerDesktopLinuxEngine", hosts)

    def test_windows_init_does_not_fallback_to_from_env_when_docker_host_is_inherited(self):
        with (
            patch.object(docker_prod_service.os, "name", "nt"),
            patch.dict(
                "os.environ",
                {"DOCKER_HOST": "npipe:////./pipe/dockerDesktopLinuxEngine"},
                clear=True,
            ),
            patch.object(
                DockerProdManager, "_connect_to_docker_hosts", return_value=False
            ),
            patch(
                "services.docker_prod_service.should_use_api_file_copy",
                return_value=False,
            ),
            patch("services.docker_prod_service.docker.from_env") as from_env,
        ):
            manager = DockerProdManager()

        from_env.assert_not_called()
        self.assertIsNone(manager.client)

    def test_pull_decompression_error_is_transient(self):
        manager = DockerProdManager.__new__(DockerProdManager)

        self.assertTrue(
            manager._is_transient_transport_error(
                zlib.error("Error -3 while decompressing data: incorrect header check")
            )
        )

    def test_pull_image_retries_decompression_error_with_cli_fallback(self):
        manager = DockerProdManager.__new__(DockerProdManager)
        manager.client = Mock()
        manager.client.images.get.side_effect = [
            docker.errors.ImageNotFound("missing"),
            SimpleNamespace(tags=["beschiak/openfork-acestep-8gb:latest"]),
        ]
        manager._get_pull_platform = Mock(return_value="linux/amd64")
        manager._refresh_client_connection = Mock(return_value=True)
        manager._restart_docker_in_wsl = Mock()
        manager._run_docker_cli_pull = Mock()

        with patch(
            "services.docker_progress_logger.stream_pull_with_progress",
            side_effect=zlib.error(
                "Error -3 while decompressing data: incorrect header check"
            ),
        ):
            manager.pull_image("beschiak/openfork-acestep-8gb:latest")

        manager._refresh_client_connection.assert_called_once()
        manager._restart_docker_in_wsl.assert_not_called()
        manager._run_docker_cli_pull.assert_called_once_with(
            "beschiak/openfork-acestep-8gb:latest",
            "linux/amd64",
            shutdown_event=None,
        )

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

    def test_ltx23_low_vram_container_uses_offload_runtime_flags(self):
        manager = DockerProdManager.__new__(DockerProdManager)
        manager.client = Mock()
        manager.services_config = {"ltx23-video-12gb": {"port": 8188}}
        manager.docker_image_map = {
            "ltx23-video-12gb": "beschiak/openfork-ltx23-wan2gp-12gb-hdr:latest"
        }
        manager.pull_image = Mock()
        manager._set_api_timeout = Mock(return_value=None)
        manager._restore_api_timeout = Mock()

        manager.client.containers.get.side_effect = docker.errors.NotFound("missing")
        manager.client.containers.list.return_value = []
        manager.client.containers.run.return_value = None

        with (
            patch("platform.system", return_value="Linux"),
            patch("services.hardware_profiler.get_available_vram", return_value=12000),
        ):
            manager.run_container("ltx23-video-12gb")

        run_kwargs = manager.client.containers.run.call_args.kwargs
        self.assertEqual(run_kwargs["ipc_mode"], "host")
        self.assertEqual(run_kwargs["shm_size"], "16g")
        self.assertEqual(len(run_kwargs["ulimits"]), 1)
        self.assertEqual(run_kwargs["ulimits"][0].name, "memlock")
        self.assertEqual(run_kwargs["ulimits"][0].get("Soft"), -1)
        self.assertEqual(run_kwargs["ulimits"][0].get("Hard"), -1)


if __name__ == "__main__":
    unittest.main()
