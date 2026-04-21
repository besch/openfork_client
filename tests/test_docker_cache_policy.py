import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import docker
import requests

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

from services.docker_download_manager import (
    DockerDownloadManager,
    ImageAvailability,
)
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


class FlakyImageStore:
    def __init__(self, image_name):
        self.image_name = image_name
        self.calls = 0

    def get(self, image_name):
        self.calls += 1
        if self.calls == 1:
            raise requests.exceptions.ConnectionError("docker temporarily unavailable")
        if image_name != self.image_name:
            raise docker.errors.ImageNotFound(f"{image_name} not found")
        return SimpleNamespace(tags=[image_name])


class BrokenImageStore:
    def get(self, image_name):
        raise requests.exceptions.ConnectionError("docker unavailable")


class ReconnectingDockerManager(FakeDockerManager):
    def __init__(self, service_type):
        super().__init__([service_type], cached_service_types=[])
        self.service_type = service_type
        self.refresh_calls = 0
        image_name = self.get_image_name(service_type)
        self.client = SimpleNamespace(
            images=FlakyImageStore(image_name),
            containers=FakeContainersAPI(),
        )

    def _is_transient_transport_error(self, exc):
        return isinstance(exc, requests.exceptions.ConnectionError)

    def _refresh_client_connection(self):
        self.refresh_calls += 1
        self.client = FakeDockerClient([self.get_image_name(self.service_type)])
        return True


class UnavailableDockerManager(FakeDockerManager):
    def __init__(self, service_type):
        super().__init__([service_type], cached_service_types=[])
        self.client = SimpleNamespace(
            images=BrokenImageStore(),
            containers=FakeContainersAPI(),
        )
        self.refresh_calls = 0

    def _is_transient_transport_error(self, exc):
        return isinstance(exc, requests.exceptions.ConnectionError)

    def _refresh_client_connection(self):
        self.refresh_calls += 1
        return False


class FakeDownloadManager:
    def __init__(self, availability=ImageAvailability.MISSING):
        self._active_downloads = set()
        self._download_queue = []
        self.started = []
        self.availability = availability

    def has_image(self, service_type):
        return self.availability == ImageAvailability.AVAILABLE

    def get_image_availability(self, service_type):
        return self.availability

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

    def test_transient_docker_error_is_not_treated_as_missing_image(self):
        docker_manager = ReconnectingDockerManager("ltx23-video-8gb")
        manager = DockerDownloadManager(docker_manager)

        self.assertEqual(
            manager.get_image_availability("ltx23-video-8gb"),
            ImageAvailability.AVAILABLE,
        )
        self.assertTrue(manager.has_image("ltx23-video-8gb"))
        self.assertEqual(docker_manager.refresh_calls, 1)

    def test_unknown_image_availability_does_not_start_background_download(self):
        docker_manager = UnavailableDockerManager("ltx23-video-8gb")
        manager = DockerDownloadManager(docker_manager)

        self.assertEqual(
            manager.get_image_availability("ltx23-video-8gb"),
            ImageAvailability.UNKNOWN,
        )
        self.assertFalse(manager.start_background_download("ltx23-video-8gb"))
        self.assertEqual(docker_manager.refresh_calls, 2)
        self.assertEqual(manager.get_all_statuses(), {})


class PrefetchPolicyTests(unittest.TestCase):
    def test_private_policies_skip_global_prefetch_suggestions(self):
        policy_to_community_mode = {
            "mine": "none",
            "project": "trusted_projects",
            "users": "trusted_users",
        }
        for policy, community_mode in policy_to_community_mode.items():
            with self.subTest(policy=policy):
                orchestrator_service = Mock()
                client = SimpleNamespace(
                    orchestrator_service=orchestrator_service,
                    accept_policy=policy,
                    community_mode=community_mode,
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

    def test_prefetch_skips_unknown_image_availability(self):
        orchestrator_service = Mock()
        orchestrator_service.get_prefetch_suggestions.return_value = ["wan22"]
        client = SimpleNamespace(
            orchestrator_service=orchestrator_service,
            accept_policy="monetize",
            monetize_mode=True,
        )
        listener = JobListener(
            client, provider_id="provider-1", shutdown_event=threading.Event()
        )
        download_manager = FakeDownloadManager(
            availability=ImageAvailability.UNKNOWN
        )

        listener._handle_prefetch_suggestions(download_manager)

        orchestrator_service.get_prefetch_suggestions.assert_called_once_with("provider-1")
        self.assertEqual(download_manager.started, [])


if __name__ == "__main__":
    unittest.main()
