import contextlib
import io
import json
import os
import threading
import time
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

import services.docker_download_manager as docker_download_manager_module
from services.docker_download_manager import (
    DockerDownloadManager,
    DownloadStatus,
    ImageAvailability,
)
from services.docker_progress_logger import DockerPullProgressLogger
from services.job_listener import JobListener
from services.realtime_job_watcher import RealtimeJobWatcher
from dgn_client import DGNClient


class FakeImageStore:
    def __init__(self, image_names, image_sizes=None):
        self.present = set(image_names)
        self.image_sizes = image_sizes or {}
        self.removed = []

    def get(self, image_name):
        if image_name not in self.present:
            raise docker.errors.ImageNotFound(f"{image_name} not found")
        return SimpleNamespace(
            tags=[image_name],
            attrs={"Size": self.image_sizes.get(image_name, 1024 ** 3)},
        )

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
    def __init__(self, image_names, running_service_types=None, image_sizes=None):
        self.images = FakeImageStore(image_names, image_sizes=image_sizes)
        self.containers = FakeContainersAPI(running_service_types)


class FakeDockerManager:
    def __init__(self, service_types, cached_service_types=None, running_service_types=None):
        self.services_config = {
            service_type: {"disk_required_gb": 1}
            for service_type in service_types
        }
        self.docker_image_map = {
            service_type: f"beschiak/openfork-{service_type}:latest"
            for service_type in service_types
        }
        cached_service_types = cached_service_types or []
        cached_image_names = [
            self.docker_image_map[service_type] for service_type in cached_service_types
        ]
        image_sizes = {
            self.docker_image_map[service_type]: 1024 ** 3
            for service_type in service_types
        }
        self.client = FakeDockerClient(
            cached_image_names,
            running_service_types=running_service_types,
            image_sizes=image_sizes,
        )

    def get_image_name(self, service_type):
        return self.docker_image_map[service_type]

    def get_container_name(self, service_type):
        return f"dgn-client-{service_type}"


class PullRecordingDockerManager(FakeDockerManager):
    def __init__(self, service_types, cached_service_types=None, running_service_types=None):
        super().__init__(
            service_types,
            cached_service_types=cached_service_types,
            running_service_types=running_service_types,
        )
        self.pull_calls = []

    def pull_image(
        self,
        image_name,
        shutdown_event=None,
        service_type=None,
        emit_pull_complete=True,
    ):
        self.pull_calls.append(
            {
                "image_name": image_name,
                "shutdown_event": shutdown_event,
                "service_type": service_type,
                "emit_pull_complete": emit_pull_complete,
            }
        )
        self.client.images.present.add(image_name)


class NonRegisteringPullDockerManager(PullRecordingDockerManager):
    def pull_image(
        self,
        image_name,
        shutdown_event=None,
        service_type=None,
        emit_pull_complete=True,
    ):
        self.pull_calls.append(
            {
                "image_name": image_name,
                "shutdown_event": shutdown_event,
                "service_type": service_type,
                "emit_pull_complete": emit_pull_complete,
            }
        )


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
        self.accept_policies = []
        self.availability = availability

    def has_image(self, service_type):
        return self.availability == ImageAvailability.AVAILABLE

    def get_image_availability(self, service_type):
        return self.availability

    def is_downloading(self, service_type):
        return False

    def is_queued(self, service_type):
        return False

    def start_background_download(self, service_type, accept_policy=None):
        self.started.append(service_type)
        self.accept_policies.append(accept_policy)


class DockerCachePolicyTests(unittest.TestCase):
    def test_cache_limit_uses_gb_budget(self):
        docker_manager = FakeDockerManager(["wan22"])
        manager = DockerDownloadManager(docker_manager, cache_limit_gb=250)

        self.assertEqual(manager.cache_limit_bytes, 250 * 1024 ** 3)
        self.assertTrue(manager.service_fits_cache_budget("wan22"))

        docker_manager.services_config["wan22"]["disk_required_gb"] = 300
        self.assertFalse(manager.service_fits_cache_budget("wan22"))

    def test_cache_limit_uses_configured_footprint_when_docker_underreports_size(self):
        docker_manager = PullRecordingDockerManager(
            ["zimage-turbo-8gb", "llm"],
            cached_service_types=["zimage-turbo-8gb"],
        )
        docker_manager.services_config["zimage-turbo-8gb"]["disk_required_gb"] = 120
        docker_manager.services_config["llm"]["disk_required_gb"] = 20
        zimage_name = docker_manager.get_image_name("zimage-turbo-8gb")
        docker_manager.client.images.image_sizes[zimage_name] = 30 * 1024 ** 3
        manager = DockerDownloadManager(docker_manager, cache_limit_gb=50)

        self.assertEqual(
            manager._get_cached_image_size_bytes("zimage-turbo-8gb"),
            120 * 1024 ** 3,
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertTrue(manager.start_background_download("llm"))

        self.assertEqual(
            docker_manager.client.images.removed,
            [(zimage_name, True)],
        )
        self.assertEqual(manager._download_queue, ["llm"])
        self.assertEqual(docker_manager.pull_calls, [])

        events = [
            json.loads(line)
            for line in stdout.getvalue().splitlines()
            if line.startswith("{")
        ]
        eviction = next(
            event for event in events if event.get("type") == "IMAGE_EVICTED"
        )
        self.assertEqual(eviction["payload"]["service_type"], "zimage-turbo-8gb")
        self.assertEqual(eviction["payload"]["freed_bytes"], 120 * 1024 ** 3)

    def test_cache_usage_deduplicates_shared_docker_image_aliases(self):
        docker_manager = FakeDockerManager(
            ["turbodiffusion", "turbodiffusion-8gb", "llm"],
            cached_service_types=["turbodiffusion-8gb", "llm"],
        )
        shared_image = docker_manager.get_image_name("turbodiffusion-8gb")
        docker_manager.docker_image_map["turbodiffusion"] = shared_image
        docker_manager.services_config["turbodiffusion"]["disk_required_gb"] = 160
        docker_manager.services_config["turbodiffusion-8gb"]["disk_required_gb"] = 160
        docker_manager.services_config["llm"]["disk_required_gb"] = 10
        docker_manager.client.images.image_sizes[shared_image] = 160 * 1024 ** 3

        manager = DockerDownloadManager(docker_manager, cache_limit_gb=250)

        self.assertEqual(
            manager._get_cache_usage_bytes(),
            170 * 1024 ** 3,
        )

    def test_fresh_download_protects_shared_image_alias_from_eviction(self):
        docker_manager = FakeDockerManager(
            ["turbodiffusion", "turbodiffusion-8gb", "old"],
            cached_service_types=["turbodiffusion-8gb", "old"],
        )
        shared_image = docker_manager.get_image_name("turbodiffusion-8gb")
        old_image = docker_manager.get_image_name("old")
        docker_manager.docker_image_map["turbodiffusion"] = shared_image
        docker_manager.services_config["turbodiffusion"]["disk_required_gb"] = 160
        docker_manager.services_config["turbodiffusion-8gb"]["disk_required_gb"] = 160
        docker_manager.services_config["old"]["disk_required_gb"] = 80
        docker_manager.client.images.image_sizes[shared_image] = 160 * 1024 ** 3
        docker_manager.client.images.image_sizes[old_image] = 80 * 1024 ** 3
        manager = DockerDownloadManager(docker_manager, cache_limit_gb=200)
        manager._recently_downloaded["turbodiffusion-8gb"] = time.time()

        evicted = manager._evict_until_within_cache_limit(reason="storage_limit")

        self.assertEqual(evicted, 1)
        self.assertEqual(docker_manager.client.images.removed, [(old_image, True)])
        self.assertIn(shared_image, docker_manager.client.images.present)

    def test_job_completion_releases_fresh_image_eviction_shield(self):
        docker_manager = PullRecordingDockerManager(
            ["qwen3-tts", "zimage-turbo-8gb"],
            cached_service_types=["qwen3-tts"],
        )
        docker_manager.services_config["qwen3-tts"]["disk_required_gb"] = 100
        docker_manager.services_config["zimage-turbo-8gb"]["disk_required_gb"] = 120
        qwen3_name = docker_manager.get_image_name("qwen3-tts")
        manager = DockerDownloadManager(docker_manager, cache_limit_gb=120)
        manager._recently_downloaded["qwen3-tts"] = time.time()

        manager.notify_job_complete("qwen3-tts")

        self.assertNotIn("qwen3-tts", manager._recently_downloaded)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertTrue(manager.start_background_download("zimage-turbo-8gb"))

        self.assertEqual(docker_manager.client.images.removed, [(qwen3_name, True)])
        self.assertEqual(manager._download_queue, ["zimage-turbo-8gb"])

    def test_compaction_pause_blocks_new_background_downloads(self):
        docker_manager = PullRecordingDockerManager(["wan22"])
        manager = DockerDownloadManager(docker_manager)

        manager.set_compaction_paused(True)

        self.assertFalse(manager.start_background_download("wan22"))
        self.assertEqual(manager._download_queue, [])
        self.assertEqual(docker_manager.pull_calls, [])

    def test_compaction_pause_freezes_existing_queue(self):
        docker_manager = PullRecordingDockerManager(["first", "second"])
        manager = DockerDownloadManager(docker_manager)
        manager._download_queue.append("second")

        manager.set_compaction_paused(True)

        self.assertFalse(manager.start_next_queued_download())
        self.assertEqual(manager._download_queue, ["second"])
        self.assertEqual(docker_manager.pull_calls, [])

    def test_compaction_pause_cancels_active_download_and_requeues_on_resume(self):
        docker_manager = PullRecordingDockerManager(["wan22"])
        manager = DockerDownloadManager(docker_manager)
        cancel_event = threading.Event()
        with manager._lock:
            manager._active_downloads.add("wan22")
            manager._active_download_policies["wan22"] = "mine"
            manager._cancellation_events["wan22"] = cancel_event
            manager._download_status["wan22"] = DownloadStatus.DOWNLOADING

        manager.set_compaction_paused(True)

        self.assertTrue(cancel_event.is_set())
        self.assertIn("wan22", manager._paused_downloads)
        self.assertEqual(manager._download_queue_policies["wan22"], "mine")

        with manager._lock:
            manager._active_downloads.clear()
            manager._cancellation_events.clear()
            manager._job_active = True

        manager.set_compaction_paused(False)

        self.assertEqual(manager._download_queue, ["wan22"])
        self.assertEqual(manager._download_queue_policies["wan22"], "mine")
        self.assertEqual(docker_manager.pull_calls, [])

    def test_storage_limit_eviction_pauses_desktop_download_before_pull(self):
        old_kind = os.environ.get("OPENFORK_CLIENT_KIND")
        os.environ["OPENFORK_CLIENT_KIND"] = "desktop"
        try:
            docker_manager = PullRecordingDockerManager(
                ["old", "incoming"],
                cached_service_types=["old"],
            )
            docker_manager.services_config["old"]["disk_required_gb"] = 80
            docker_manager.services_config["incoming"]["disk_required_gb"] = 80
            manager = DockerDownloadManager(docker_manager, cache_limit_gb=120)

            self.assertTrue(manager.start_background_download("incoming"))

            self.assertTrue(manager._compaction_paused)
            self.assertEqual(manager._download_queue, ["incoming"])
            self.assertEqual(docker_manager.pull_calls, [])
        finally:
            if old_kind is None:
                os.environ.pop("OPENFORK_CLIENT_KIND", None)
            else:
                os.environ["OPENFORK_CLIENT_KIND"] = old_kind

    def test_storage_limit_eviction_freezes_queued_desktop_download(self):
        old_kind = os.environ.get("OPENFORK_CLIENT_KIND")
        os.environ["OPENFORK_CLIENT_KIND"] = "desktop"
        try:
            docker_manager = PullRecordingDockerManager(
                ["old", "incoming"],
                cached_service_types=["old"],
            )
            docker_manager.services_config["old"]["disk_required_gb"] = 80
            docker_manager.services_config["incoming"]["disk_required_gb"] = 80
            manager = DockerDownloadManager(docker_manager, cache_limit_gb=120)
            with manager._lock:
                manager._download_queue.append("incoming")
                manager._download_status["incoming"] = DownloadStatus.PENDING

            self.assertFalse(manager.start_next_queued_download(reason="test"))

            self.assertTrue(manager._compaction_paused)
            self.assertEqual(manager._download_queue, ["incoming"])
            self.assertEqual(docker_manager.pull_calls, [])
        finally:
            if old_kind is None:
                os.environ.pop("OPENFORK_CLIENT_KIND", None)
            else:
                os.environ["OPENFORK_CLIENT_KIND"] = old_kind

    def test_storage_limit_eviction_pause_expires_without_desktop_claim(self):
        old_kind = os.environ.get("OPENFORK_CLIENT_KIND")
        os.environ["OPENFORK_CLIENT_KIND"] = "desktop"
        try:
            docker_manager = PullRecordingDockerManager(
                ["old", "incoming"],
                cached_service_types=["old"],
            )
            docker_manager.services_config["old"]["disk_required_gb"] = 80
            docker_manager.services_config["incoming"]["disk_required_gb"] = 80
            manager = DockerDownloadManager(docker_manager, cache_limit_gb=120)

            self.assertTrue(manager.start_background_download("incoming"))
            self.assertTrue(manager._compaction_paused)

            with manager._lock:
                manager._compaction_pause_deadline = time.time() - 1
                manager._post_download_hold_until = time.time() - 1

            self.assertTrue(manager.start_next_queued_download(reason="test"))
            self.assertFalse(manager._compaction_paused)
            self.assertEqual(manager._download_queue, [])
        finally:
            if old_kind is None:
                os.environ.pop("OPENFORK_CLIENT_KIND", None)
            else:
                os.environ["OPENFORK_CLIENT_KIND"] = old_kind

    def test_set_job_inactive_does_not_start_queue_before_listener_checks_jobs(self):
        docker_manager = PullRecordingDockerManager(["first", "second"])
        docker_manager.services_config["second"] = {"disk_required_gb": 0.001}
        manager = DockerDownloadManager(docker_manager)
        with manager._lock:
            manager._download_queue.append("second")
            manager._download_status["second"] = DownloadStatus.PENDING

        manager.set_job_active(False)

        self.assertEqual(manager._download_queue, ["second"])
        self.assertFalse(manager.is_downloading("second"))
        self.assertEqual(docker_manager.pull_calls, [])

        self.assertTrue(manager.start_next_queued_download(reason="listener-idle"))
        for _ in range(100):
            if manager.get_download_status("second") == DownloadStatus.COMPLETED:
                break
            time.sleep(0.01)

        self.assertEqual(docker_manager.pull_calls[0]["service_type"], "second")
        self.assertEqual(
            manager.get_download_status("second"),
            DownloadStatus.COMPLETED,
        )

    def test_apply_routing_config_updates_allowed_ids_and_cache_cap(self):
        client = DGNClient.__new__(DGNClient)
        client.process_own_jobs = True
        client.community_mode = "none"
        client.monetize_mode = False
        client.allowed_ids = ["owner-id"]
        client.download_manager = SimpleNamespace(
            update_routing_config=Mock(),
        )
        client.job_wakeup_event = threading.Event()

        client.apply_routing_config(
            {
                "process_own_jobs": True,
                "community_mode": "all",
                "allowed_ids": [],
                "monetize_mode": False,
            }
        )

        self.assertEqual(client.community_mode, "all")
        self.assertEqual(client.allowed_ids, [])
        client.download_manager.update_routing_config.assert_called_once_with(
            community_mode="all",
            monetize_mode=False,
        )
        self.assertTrue(client.job_wakeup_event.is_set())

    def test_apply_routing_config_does_not_interrupt_active_job(self):
        active_job = {"id": "job-active", "execution_token": "token-active"}
        orchestrator_service = Mock()
        client = DGNClient.__new__(DGNClient)
        client.process_own_jobs = True
        client.community_mode = "none"
        client.monetize_mode = False
        client.allowed_ids = ["owner-id"]
        client.current_job = active_job
        client.interrupted_job_id = None
        client.interrupted_job_execution_token = None
        client.stop_requested = False
        client.active_service_type = "wan22"
        client.download_manager = SimpleNamespace(
            update_routing_config=Mock(),
        )
        client.job_wakeup_event = threading.Event()
        client.orchestrator_service = orchestrator_service

        client.apply_routing_config(
            {
                "process_own_jobs": True,
                "community_mode": "all",
                "allowed_ids": [],
                "monetize_mode": False,
            }
        )

        self.assertIs(client.current_job, active_job)
        self.assertEqual(client.active_service_type, "wan22")
        self.assertIsNone(client.interrupted_job_id)
        self.assertIsNone(client.interrupted_job_execution_token)
        self.assertFalse(client.stop_requested)
        orchestrator_service.reset_interrupted_job.assert_not_called()
        orchestrator_service.update_job_status.assert_not_called()
        orchestrator_service.update_provider_status.assert_not_called()

    def test_evicts_prefetched_untouched_image_before_used_images(self):
        docker_manager = FakeDockerManager(
            service_types=["wan22", "foley", "ltx23"],
            cached_service_types=["wan22", "foley", "ltx23"],
        )
        orchestrator_service = Mock()
        manager = DockerDownloadManager(
            docker_manager,
            orchestrator_service=orchestrator_service,
            provider_id="provider-1",
        )

        manager._last_job_times["foley"] = 100.0
        manager._last_job_times["ltx23"] = 200.0

        evicted = manager._evict_lru_image()

        self.assertEqual(evicted, "wan22")
        self.assertEqual(
            docker_manager.client.images.removed,
            [(docker_manager.get_image_name("wan22"), True)],
        )
        _, kwargs = orchestrator_service.report_cached_images.call_args
        self.assertEqual(kwargs["provider_id"], "provider-1")
        self.assertEqual(kwargs["cached_images"], ["foley", "ltx23"])
        self.assertEqual(kwargs["mode"], "replace")

    def test_evicts_oldest_used_image_when_all_cached_images_were_used(self):
        docker_manager = FakeDockerManager(
            service_types=["wan22", "foley", "ltx23"],
            cached_service_types=["wan22", "foley", "ltx23"],
        )
        manager = DockerDownloadManager(docker_manager)

        manager._last_job_times["wan22"] = 300.0
        manager._last_job_times["foley"] = 100.0
        manager._last_job_times["ltx23"] = 200.0

        evicted = manager._evict_lru_image()

        self.assertEqual(evicted, "foley")
        self.assertNotIn(
            docker_manager.get_image_name("foley"),
            docker_manager.client.images.present,
        )

    def test_idle_cleanup_evicts_multiple_stale_images_per_pass(self):
        docker_manager = FakeDockerManager(
            service_types=["wan22", "foley", "ltx23"],
            cached_service_types=["wan22", "foley", "ltx23"],
        )
        manager = DockerDownloadManager(
            docker_manager,
            community_mode="all",
            monetize_mode=False,
        )

        old = time.time() - (3 * 60 * 60)
        recent = time.time()
        manager._last_cache_activity_times["wan22"] = old
        manager._last_cache_activity_times["foley"] = old
        manager._last_cache_activity_times["ltx23"] = recent

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            evicted = manager._evict_idle_images(tier="healthy")

        self.assertEqual(evicted, 2)
        self.assertNotIn(
            docker_manager.get_image_name("wan22"),
            docker_manager.client.images.present,
        )
        self.assertNotIn(
            docker_manager.get_image_name("foley"),
            docker_manager.client.images.present,
        )
        self.assertIn(
            docker_manager.get_image_name("ltx23"),
            docker_manager.client.images.present,
        )

    def test_idle_cleanup_does_not_immediately_evict_unknown_legacy_cache(self):
        docker_manager = FakeDockerManager(
            service_types=["wan22", "foley"],
            cached_service_types=["wan22", "foley"],
        )
        manager = DockerDownloadManager(
            docker_manager,
            community_mode="all",
            monetize_mode=False,
        )

        evicted = manager._evict_idle_images(tier="healthy")

        self.assertEqual(evicted, 0)
        self.assertEqual(
            docker_manager.client.images.present,
            {
                docker_manager.get_image_name("wan22"),
                docker_manager.get_image_name("foley"),
            },
        )
        self.assertEqual(
            set(manager._last_cache_activity_times),
            {"wan22", "foley"},
        )

    def test_private_mine_policy_disables_idle_cleanup_when_healthy(self):
        docker_manager = FakeDockerManager(
            service_types=["wan22", "foley"],
            cached_service_types=["wan22", "foley"],
        )
        manager = DockerDownloadManager(
            docker_manager,
            community_mode="none",
            monetize_mode=False,
        )
        old = time.time() - (24 * 60 * 60)
        manager._last_cache_activity_times["wan22"] = old
        manager._last_cache_activity_times["foley"] = old

        evicted = manager._evict_idle_images(tier="healthy")

        self.assertEqual(evicted, 0)
        self.assertEqual(
            docker_manager.client.images.present,
            {
                docker_manager.get_image_name("wan22"),
                docker_manager.get_image_name("foley"),
            },
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

    def test_download_state_is_emitted_before_image_eviction(self):
        docker_manager = PullRecordingDockerManager(
            service_types=["wan22", "foley", "ltx23", "stream-diffvsr-upscaler"],
            cached_service_types=["wan22", "foley", "ltx23"],
        )
        docker_manager.services_config["stream-diffvsr-upscaler"] = {
            "disk_required_gb": 0.001,
        }
        manager = DockerDownloadManager(docker_manager, cache_limit_gb=3)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertTrue(manager.start_background_download("stream-diffvsr-upscaler"))
            for _ in range(100):
                if (
                    manager.get_download_status("stream-diffvsr-upscaler")
                    == DownloadStatus.COMPLETED
                ):
                    break
                time.sleep(0.01)

        events = [
            json.loads(line)
            for line in stdout.getvalue().splitlines()
            if line.startswith("{")
        ]
        first_download_state = next(
            i
            for i, event in enumerate(events)
            if event.get("type") == "DOCKER_DOWNLOAD_STATE"
            and event.get("payload", {}).get("status") == "starting"
        )
        first_eviction = next(
            i for i, event in enumerate(events) if event.get("type") == "IMAGE_EVICTED"
        )

        self.assertLess(first_download_state, first_eviction)

    def test_cancelled_worker_exits_before_touching_docker_pull(self):
        docker_manager = PullRecordingDockerManager(
            service_types=["stream-diffvsr-upscaler"],
        )
        manager = DockerDownloadManager(docker_manager)
        cancel_event = threading.Event()
        cancel_event.set()

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            manager._download_worker("stream-diffvsr-upscaler", cancel_event)

        events = [
            json.loads(line)
            for line in stdout.getvalue().splitlines()
            if line.startswith("{")
        ]

        self.assertEqual(docker_manager.pull_calls, [])
        self.assertTrue(
            any(
                event.get("type") == "DOCKER_DOWNLOAD_STATE"
                and event.get("payload", {}).get("status") == "cancelled"
                for event in events
            )
        )

    def test_completed_download_wakes_listener_before_chaining_queue(self):
        wakeup_event = threading.Event()
        docker_manager = PullRecordingDockerManager(
            service_types=["first", "second"],
            cached_service_types=["first"],
        )
        docker_manager.services_config["second"] = {"disk_required_gb": 0.001}
        manager = DockerDownloadManager(
            docker_manager,
            wakeup_event=wakeup_event,
        )
        manager._signal_wakeup_after_settle = wakeup_event.set

        with manager._lock:
            manager._active_downloads.add("first")
            manager._download_status["first"] = DownloadStatus.COMPLETED
            manager._download_queue.append("second")
            manager._download_status["second"] = DownloadStatus.PENDING

        manager._finish_download("first")

        self.assertTrue(wakeup_event.wait(1))
        self.assertEqual(manager._download_queue, ["second"])
        self.assertFalse(manager.is_downloading("second"))
        self.assertEqual(docker_manager.pull_calls, [])

    def test_post_download_hold_blocks_listener_queue_start_until_expired(self):
        wakeup_event = threading.Event()
        docker_manager = PullRecordingDockerManager(
            service_types=["first", "second"],
            cached_service_types=["first"],
        )
        docker_manager.services_config["second"] = {"disk_required_gb": 0.001}
        manager = DockerDownloadManager(
            docker_manager,
            wakeup_event=wakeup_event,
        )
        manager._signal_wakeup_after_settle = wakeup_event.set

        with manager._lock:
            manager._active_downloads.add("first")
            manager._download_status["first"] = DownloadStatus.COMPLETED
            manager._download_queue.append("second")
            manager._download_status["second"] = DownloadStatus.PENDING

        manager._finish_download("first")

        self.assertTrue(manager.is_post_download_hold_active())
        self.assertFalse(manager.start_next_queued_download(reason="test"))
        self.assertEqual(docker_manager.pull_calls, [])

        with manager._lock:
            manager._post_download_hold_until = time.time() - 1

        self.assertTrue(manager.start_next_queued_download(reason="test"))
        for _ in range(100):
            if manager.get_download_status("second") == DownloadStatus.COMPLETED:
                break
            time.sleep(0.01)

        self.assertEqual(docker_manager.pull_calls[0]["service_type"], "second")
        self.assertEqual(
            manager.get_download_status("second"),
            DownloadStatus.COMPLETED,
        )

    def test_start_background_download_queues_during_post_download_hold(self):
        docker_manager = PullRecordingDockerManager(service_types=["first", "second"])
        docker_manager.services_config["second"] = {"disk_required_gb": 0.001}
        manager = DockerDownloadManager(docker_manager)

        with manager._lock:
            manager._post_download_hold_until = time.time() + 30

        self.assertTrue(manager.start_background_download("second"))
        self.assertEqual(manager._download_queue, ["second"])
        self.assertEqual(docker_manager.pull_calls, [])

        with manager._lock:
            manager._post_download_hold_until = time.time() - 1

        self.assertTrue(manager.start_next_queued_download(reason="test"))
        for _ in range(100):
            if manager.get_download_status("second") == DownloadStatus.COMPLETED:
                break
            time.sleep(0.01)

        self.assertEqual(docker_manager.pull_calls[0]["service_type"], "second")
        self.assertEqual(
            manager.get_download_status("second"),
            DownloadStatus.COMPLETED,
        )

    def test_listener_can_resume_queue_when_no_job_became_processable(self):
        docker_manager = PullRecordingDockerManager(
            service_types=["first", "second"],
            cached_service_types=["first"],
        )
        docker_manager.services_config["second"] = {"disk_required_gb": 0.001}
        manager = DockerDownloadManager(docker_manager)
        with manager._lock:
            manager._download_queue.append("second")
            manager._download_status["second"] = DownloadStatus.PENDING

        self.assertTrue(manager.start_next_queued_download(reason="test"))
        for _ in range(100):
            if manager.get_download_status("second") == DownloadStatus.COMPLETED:
                break
            time.sleep(0.01)

        self.assertEqual(docker_manager.pull_calls[0]["service_type"], "second")
        self.assertEqual(
            manager.get_download_status("second"),
            DownloadStatus.COMPLETED,
        )

    def test_download_worker_fails_if_pull_does_not_register_image(self):
        docker_manager = NonRegisteringPullDockerManager(service_types=["wan22"])
        manager = DockerDownloadManager(docker_manager)
        cancel_event = threading.Event()

        old_timeout = docker_download_manager_module.POST_PULL_VERIFY_TIMEOUT_SECS
        old_interval = docker_download_manager_module.POST_PULL_VERIFY_INTERVAL_SECS
        docker_download_manager_module.POST_PULL_VERIFY_TIMEOUT_SECS = 0.02
        docker_download_manager_module.POST_PULL_VERIFY_INTERVAL_SECS = 0.01
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                manager._download_worker("wan22", cancel_event)
        finally:
            docker_download_manager_module.POST_PULL_VERIFY_TIMEOUT_SECS = old_timeout
            docker_download_manager_module.POST_PULL_VERIFY_INTERVAL_SECS = old_interval

        self.assertEqual(docker_manager.pull_calls[0]["service_type"], "wan22")
        self.assertFalse(docker_manager.pull_calls[0]["emit_pull_complete"])
        self.assertEqual(
            manager.get_download_status("wan22"),
            DownloadStatus.FAILED,
        )
        self.assertFalse(manager.is_post_download_hold_active())
        events = [
            json.loads(line)
            for line in stdout.getvalue().splitlines()
            if line.startswith("{")
        ]
        self.assertFalse(
            any(event.get("type") == "DOCKER_PULL_COMPLETE" for event in events)
        )

    def test_verified_download_emits_pull_complete_after_registration(self):
        docker_manager = PullRecordingDockerManager(service_types=["wan22"])
        manager = DockerDownloadManager(docker_manager)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            manager._download_worker("wan22", threading.Event())

        events = [
            json.loads(line)
            for line in stdout.getvalue().splitlines()
            if line.startswith("{")
        ]
        event_types = [event.get("type") for event in events]

        self.assertIn("DOCKER_PULL_COMPLETE", event_types)
        self.assertIn("DOCKER_DOWNLOAD_STATE", event_types)
        self.assertEqual(
            manager.get_download_status("wan22"),
            DownloadStatus.COMPLETED,
        )
        self.assertFalse(docker_manager.pull_calls[0]["emit_pull_complete"])

    def test_failed_download_holds_queue_for_listener(self):
        wakeup_event = threading.Event()
        docker_manager = NonRegisteringPullDockerManager(
            service_types=["first", "second"]
        )
        docker_manager.services_config["second"] = {"disk_required_gb": 0.001}
        manager = DockerDownloadManager(
            docker_manager,
            wakeup_event=wakeup_event,
        )
        cancel_event = threading.Event()

        with manager._lock:
            manager._download_queue.append("second")
            manager._download_status["second"] = DownloadStatus.PENDING

        old_timeout = docker_download_manager_module.POST_PULL_VERIFY_TIMEOUT_SECS
        old_interval = docker_download_manager_module.POST_PULL_VERIFY_INTERVAL_SECS
        docker_download_manager_module.POST_PULL_VERIFY_TIMEOUT_SECS = 0.02
        docker_download_manager_module.POST_PULL_VERIFY_INTERVAL_SECS = 0.01
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                manager._download_worker("first", cancel_event)
        finally:
            docker_download_manager_module.POST_PULL_VERIFY_TIMEOUT_SECS = old_timeout
            docker_download_manager_module.POST_PULL_VERIFY_INTERVAL_SECS = old_interval

        self.assertTrue(wakeup_event.is_set())
        self.assertEqual(manager._download_queue, ["second"])
        self.assertFalse(manager.is_downloading("second"))
        self.assertEqual(
            [call["service_type"] for call in docker_manager.pull_calls],
            ["first"],
        )

    def test_pull_progress_does_not_report_processing_for_unknown_layers(self):
        logger = DockerPullProgressLogger(
            "beschiak/openfork-test:latest",
            throttle_interval=0,
            service_type="test",
        )

        logger.parse_progress_event(
            {"id": "a", "status": "Pulling fs layer", "progressDetail": {}}
        )
        logger.parse_progress_event(
            {"id": "b", "status": "Pulling fs layer", "progressDetail": {}}
        )
        logger.parse_progress_event(
            {"id": "a", "status": "Already exists", "progressDetail": {}}
        )

        self.assertLess(logger.calculate_overall_progress(), 100)
        self.assertNotEqual(logger.get_current_status(), "Processing")

    def test_pull_progress_does_not_jump_from_cached_layers_before_bytes(self):
        logger = DockerPullProgressLogger(
            "beschiak/openfork-test:latest",
            throttle_interval=0,
            service_type="test",
        )

        for layer_id in ("a", "b", "c", "d"):
            logger.parse_progress_event(
                {"id": layer_id, "status": "Pulling fs layer", "progressDetail": {}}
            )
        logger.parse_progress_event(
            {"id": "a", "status": "Already exists", "progressDetail": {}}
        )

        self.assertLessEqual(logger.calculate_overall_progress(), 5)

        logger.parse_progress_event(
            {
                "id": "b",
                "status": "Downloading",
                "progressDetail": {"current": 10, "total": 1000},
            }
        )
        self.assertLessEqual(logger.calculate_overall_progress(), 5)

        logger.parse_progress_event(
            {
                "id": "b",
                "status": "Downloading",
                "progressDetail": {"current": 300, "total": 1000},
            }
        )
        self.assertGreater(logger.calculate_overall_progress(), 5)

    def test_pull_progress_advances_on_layer_phase_when_byte_totals_are_partial(self):
        logger = DockerPullProgressLogger(
            "beschiak/openfork-test:latest",
            throttle_interval=0,
            service_type="test",
        )

        for layer_id in ("a", "b", "c", "d"):
            logger.parse_progress_event(
                {"id": layer_id, "status": "Pulling fs layer", "progressDetail": {}}
            )
        logger.parse_progress_event(
            {"id": "a", "status": "Already exists", "progressDetail": {}}
        )
        logger.parse_progress_event(
            {
                "id": "b",
                "status": "Downloading",
                "progressDetail": {"current": 300, "total": 1000},
            }
        )
        before = logger.calculate_overall_progress()

        logger.parse_progress_event(
            {"id": "c", "status": "Download complete", "progressDetail": {}}
        )

        self.assertGreater(logger.calculate_overall_progress(), before)

    def test_pull_progress_slowly_advances_while_active_progress_is_flat(self):
        logger = DockerPullProgressLogger(
            "beschiak/openfork-test:latest",
            throttle_interval=0.5,
            service_type="test",
        )
        logger.parse_progress_event(
            {
                "id": "a",
                "status": "Downloading",
                "progressDetail": {"current": 10, "total": 1000},
            }
        )
        logger.last_progress = 22
        logger.last_emit_time = time.time() - 10

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            logger.emit_progress()

        events = [
            json.loads(line)
            for line in stdout.getvalue().splitlines()
            if line.startswith("{")
        ]
        self.assertEqual(events[-1]["payload"]["progress"], 23)

    def test_pull_progress_creeps_during_finalizing_phase(self):
        logger = DockerPullProgressLogger(
            "beschiak/openfork-test:latest",
            throttle_interval=0.5,
            service_type="test",
        )
        for layer_id in ("a", "b"):
            logger.parse_progress_event(
                {"id": layer_id, "status": "Pulling fs layer", "progressDetail": {}}
            )
        logger.parse_progress_event(
            {
                "id": "a",
                "status": "Downloading",
                "progressDetail": {"current": 1000, "total": 1000},
            }
        )
        logger.parse_progress_event(
            {"id": "a", "status": "Download complete", "progressDetail": {}}
        )
        logger.last_progress = 90
        logger.last_emit_time = time.time() - 10

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            logger.emit_progress()

        events = [
            json.loads(line)
            for line in stdout.getvalue().splitlines()
            if line.startswith("{")
        ]
        self.assertEqual(events[-1]["payload"]["progress"], 91)
        self.assertEqual(events[-1]["payload"]["status"], "Finalizing")


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
            "ltx23",
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

    def test_public_peek_download_gate_uses_server_coverage(self):
        orchestrator_service = Mock()
        orchestrator_service.get_prefetch_suggestions.return_value = ["wan22"]
        client = SimpleNamespace(orchestrator_service=orchestrator_service)
        listener = JobListener(
            client, provider_id="provider-1", shutdown_event=threading.Event()
        )

        gate = listener._get_download_gate_for_jobs(
            [({"id": "job-1"}, "all")]
        )

        self.assertEqual(gate, {"wan22"})
        orchestrator_service.get_prefetch_suggestions.assert_called_once_with(
            "provider-1",
            limit=20,
            return_none_on_error=True,
        )
        self.assertTrue(
            listener._should_download_missing_image("wan22", "all", gate)
        )
        self.assertFalse(
            listener._should_download_missing_image("foley", "all", gate)
        )

    def test_private_peek_downloads_bypass_global_coverage_gate(self):
        orchestrator_service = Mock()
        client = SimpleNamespace(orchestrator_service=orchestrator_service)
        listener = JobListener(
            client, provider_id="provider-1", shutdown_event=threading.Event()
        )

        gate = listener._get_download_gate_for_jobs(
            [({"id": "job-1"}, "mine")]
        )

        orchestrator_service.get_prefetch_suggestions.assert_not_called()
        self.assertIsNone(gate)
        self.assertTrue(
            listener._should_download_missing_image("wan22", "mine", set())
        )

    def test_download_gate_fails_open_when_server_unavailable(self):
        orchestrator_service = Mock()
        orchestrator_service.get_prefetch_suggestions.return_value = None
        client = SimpleNamespace(orchestrator_service=orchestrator_service)
        listener = JobListener(
            client, provider_id="provider-1", shutdown_event=threading.Event()
        )

        gate = listener._get_download_gate_for_jobs(
            [({"id": "job-1"}, "all")]
        )

        self.assertIsNone(gate)
        self.assertTrue(
            listener._should_download_missing_image("wan22", "all", gate)
        )

    def test_download_manager_respects_rejected_download_claim(self):
        docker_manager = PullRecordingDockerManager(service_types=["wan22"])
        docker_manager.services_config["wan22"] = {"disk_required_gb": 0.001}
        orchestrator_service = Mock()
        orchestrator_service.report_download_state.return_value = False
        manager = DockerDownloadManager(
            docker_manager,
            orchestrator_service=orchestrator_service,
            provider_id="provider-1",
        )

        self.assertFalse(
            manager.start_background_download("wan22", accept_policy="all")
        )

        orchestrator_service.report_download_state.assert_called_once_with(
            provider_id="provider-1",
            service_type="wan22",
            action="start",
            accept_policy="all",
            return_none_on_error=True,
        )
        self.assertEqual(docker_manager.pull_calls, [])
        self.assertEqual(manager.get_all_statuses(), {})

    def test_download_manager_passes_private_policy_to_download_claim(self):
        docker_manager = PullRecordingDockerManager(service_types=["wan22"])
        docker_manager.services_config["wan22"] = {"disk_required_gb": 0.001}
        orchestrator_service = Mock()
        orchestrator_service.report_download_state.return_value = True
        manager = DockerDownloadManager(
            docker_manager,
            orchestrator_service=orchestrator_service,
            provider_id="provider-1",
        )

        self.assertTrue(
            manager.start_background_download("wan22", accept_policy="mine")
        )
        for _ in range(100):
            if manager.get_download_status("wan22") == DownloadStatus.COMPLETED:
                break
            time.sleep(0.01)

        first_call = orchestrator_service.report_download_state.call_args_list[0]
        self.assertEqual(
            first_call.kwargs,
            {
                "provider_id": "provider-1",
                "service_type": "wan22",
                "action": "start",
                "accept_policy": "mine",
                "return_none_on_error": True,
            },
        )
        self.assertEqual(docker_manager.pull_calls[0]["service_type"], "wan22")


class RealtimeNotificationFilterTests(unittest.TestCase):
    def _watcher_for(self, **client_attrs):
        client = SimpleNamespace(**client_attrs)
        return RealtimeJobWatcher(
            access_token="token",
            wakeup_event=threading.Event(),
            shutdown_event=threading.Event(),
            client=client,
        )

    def test_realtime_notification_skips_incompatible_service(self):
        watcher = self._watcher_for(
            compatible_services={"wan22-8gb"},
            community_mode="all",
            process_own_jobs=False,
            monetize_mode=False,
        )

        self.assertFalse(
            watcher._should_wake_for_notification(
                {
                    "service_type": "ltx23-video-24gb",
                    "accept_policy": "all",
                    "monetize_job": False,
                }
            )
        )

    def test_realtime_notification_wakes_for_matching_public_service(self):
        watcher = self._watcher_for(
            compatible_services={"wan22-8gb"},
            community_mode="all",
            process_own_jobs=False,
            monetize_mode=False,
        )

        self.assertTrue(
            watcher._should_wake_for_notification(
                {
                    "service_type": "wan22-8gb",
                    "accept_policy": "all",
                    "monetize_job": False,
                }
            )
        )

    def test_realtime_notification_wakes_for_legacy_rows_without_service_type(self):
        watcher = self._watcher_for(
            compatible_services={"wan22-8gb"},
            community_mode="all",
            process_own_jobs=False,
            monetize_mode=False,
        )

        self.assertTrue(
            watcher._should_wake_for_notification(
                {
                    "accept_policy": "all",
                    "monetize_job": False,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
