import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import docker

if not hasattr(docker, "errors"):
    class _DockerImageNotFound(Exception):
        pass

    class _DockerNotFound(Exception):
        pass

    class _DockerApiError(Exception):
        def __init__(self, *args, response=None, **kwargs):
            super().__init__(*args)
            self.response = response

    docker.errors = SimpleNamespace(
        ImageNotFound=_DockerImageNotFound,
        NotFound=_DockerNotFound,
        APIError=_DockerApiError,
    )

from services.docker_download_manager import DockerDownloadManager
from services.job_listener import JobListener


class FakeImageStore:
    def __init__(self, image_names):
        self.present = set(image_names)
        self.removed = []

    def get(self, image_name):
        if image_name not in self.present:
            raise docker.errors.ImageNotFound(f"{image_name} not found")
        return SimpleNamespace(tags=[image_name])

    def remove(self, image, force=False):
        if image not in self.present:
            raise docker.errors.ImageNotFound(f"{image} not found")
        self.present.remove(image)
        self.removed.append((image, force))


class FakeContainer:
    def __init__(self, status="exited"):
        self.status = status
        self.removed = False

    def remove(self, force=False):
        self.removed = True


class FakeContainersAPI:
    def __init__(self, running_service_types=None):
        running = running_service_types or []
        self.running_names = {f"dgn-client-{service_type}" for service_type in running}
        self.ancestor_containers = {}

    def get(self, container_name):
        if container_name in self.running_names:
            return SimpleNamespace(status="running")
        raise docker.errors.NotFound("container not found")

    def list(self, all=False, filters=None):
        filters = filters or {}
        ancestor = filters.get("ancestor")
        return list(self.ancestor_containers.get(ancestor, []))


class FakeDockerClient:
    def __init__(self, image_names, running_service_types=None):
        self.images = FakeImageStore(image_names)
        self.containers = FakeContainersAPI(running_service_types)


class FakeDockerManager:
    def __init__(self, service_types, cached_service_types=None, running_service_types=None):
        self.services_config = {service_type: {} for service_type in service_types}
        self.docker_image_map = {
            service_type: f"beschiak/openfork-{service_type}:latest"
            for service_type in service_types
        }
        cached_service_types = cached_service_types or []
        cached_image_names = [
            self.docker_image_map[service_type] for service_type in cached_service_types
        ]
        self.client = FakeDockerClient(
            cached_image_names, running_service_types=running_service_types
        )

    def get_image_name(self, service_type):
        return self.docker_image_map[service_type]

    def get_container_name(self, service_type):
        return f"dgn-client-{service_type}"


class FakeDownloadManager:
    def __init__(self):
        self._active_downloads = set()
        self._download_queue = []
        self.started = []

    def has_image(self, service_type):
        return False

    def is_downloading(self, service_type):
        return False

    def is_queued(self, service_type):
        return False

    def start_background_download(self, service_type):
        self.started.append(service_type)


class DockerCachePolicyTests(unittest.TestCase):
    def test_evicts_prefetched_untouched_image_before_used_images(self):
        docker_manager = FakeDockerManager(
            service_types=["wan22", "foley", "hunyuan"],
            cached_service_types=["wan22", "foley", "hunyuan"],
        )
        orchestrator_service = Mock()
        manager = DockerDownloadManager(
            docker_manager,
            orchestrator_service=orchestrator_service,
            provider_id="provider-1",
            max_cached_images=3,
        )

        manager._last_job_times["foley"] = 100.0
        manager._last_job_times["hunyuan"] = 200.0

        evicted = manager._evict_lru_image()

        self.assertEqual(evicted, "wan22")
        self.assertEqual(
            docker_manager.client.images.removed,
            [(docker_manager.get_image_name("wan22"), True)],
        )
        _, kwargs = orchestrator_service.report_cached_images.call_args
        self.assertEqual(kwargs["provider_id"], "provider-1")
        self.assertEqual(kwargs["cached_images"], ["foley", "hunyuan"])
        self.assertEqual(kwargs["mode"], "replace")

    def test_evicts_oldest_used_image_when_all_cached_images_were_used(self):
        docker_manager = FakeDockerManager(
            service_types=["wan22", "foley", "hunyuan"],
            cached_service_types=["wan22", "foley", "hunyuan"],
        )
        manager = DockerDownloadManager(docker_manager, max_cached_images=3)

        manager._last_job_times["wan22"] = 300.0
        manager._last_job_times["foley"] = 100.0
        manager._last_job_times["hunyuan"] = 200.0

        evicted = manager._evict_lru_image()

        self.assertEqual(evicted, "foley")
        self.assertNotIn(
            docker_manager.get_image_name("foley"),
            docker_manager.client.images.present,
        )


class PrefetchPolicyTests(unittest.TestCase):
    def test_private_policies_skip_global_prefetch_suggestions(self):
        for policy in ("mine", "project", "users"):
            with self.subTest(policy=policy):
                orchestrator_service = Mock()
                client = SimpleNamespace(
                    orchestrator_service=orchestrator_service,
                    accept_policy=policy,
                    monetize_mode=False,
                )
                listener = JobListener(
                    client, provider_id="provider-1", shutdown_event=threading.Event()
                )
                download_manager = FakeDownloadManager()

                listener._handle_prefetch_suggestions(download_manager)

                orchestrator_service.get_prefetch_suggestions.assert_not_called()
                self.assertEqual(download_manager.started, [])

    def test_monetize_policy_can_prefetch_top_two_suggestions(self):
        orchestrator_service = Mock()
        orchestrator_service.get_prefetch_suggestions.return_value = [
            "wan22",
            "foley",
            "hunyuan",
        ]
        client = SimpleNamespace(
            orchestrator_service=orchestrator_service,
            accept_policy="monetize",
            monetize_mode=True,
        )
        listener = JobListener(
            client, provider_id="provider-1", shutdown_event=threading.Event()
        )
        download_manager = FakeDownloadManager()

        listener._handle_prefetch_suggestions(download_manager)

        orchestrator_service.get_prefetch_suggestions.assert_called_once_with("provider-1")
        self.assertEqual(download_manager.started, ["wan22", "foley"])


if __name__ == "__main__":
    unittest.main()
