import contextlib
import io
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cli import SHUTDOWN_EVENT, cleanup, listen_for_ipc_commands
from exceptions import InfrastructureError
from services.comfyui_service import ComfyUIClient
from services.job_listener import JobListener
from services.orchestrator_service import OrchestratorService


class StopResetBehaviorTests(unittest.TestCase):
    def tearDown(self):
        SHUTDOWN_EVENT.clear()

    def test_request_stop_marks_active_job_for_reset(self):
        client = SimpleNamespace(
            current_job={"id": "job-123", "execution_token": "token-123"},
            interrupted_job_id=None,
            interrupted_job_execution_token=None,
            stop_requested=False,
            orchestrator_service=Mock(),
            download_manager=None,
        )

        original_stdin = sys.stdin
        sys.stdin = io.StringIO('{"type":"REQUEST_STOP"}\n')
        try:
            listen_for_ipc_commands(client)
        finally:
            sys.stdin = original_stdin

        self.assertTrue(client.stop_requested)
        self.assertEqual(client.interrupted_job_id, "job-123")
        self.assertEqual(client.interrupted_job_execution_token, "token-123")
        self.assertTrue(SHUTDOWN_EVENT.is_set())

    def test_request_stop_is_processed_even_if_shutdown_was_already_signaled(self):
        client = SimpleNamespace(
            current_job={"id": "job-race", "execution_token": "token-race"},
            interrupted_job_id=None,
            interrupted_job_execution_token=None,
            stop_requested=False,
            orchestrator_service=Mock(),
            download_manager=None,
        )

        original_stdin = sys.stdin
        sys.stdin = io.StringIO('{"type":"REQUEST_STOP"}\n')
        SHUTDOWN_EVENT.set()
        try:
            listen_for_ipc_commands(client)
        finally:
            sys.stdin = original_stdin

        self.assertTrue(client.stop_requested)
        self.assertEqual(client.interrupted_job_id, "job-race")
        self.assertEqual(client.interrupted_job_execution_token, "token-race")

    def test_update_routing_config_ipc_hot_reloads_client_policy(self):
        client = SimpleNamespace(
            current_job=None,
            interrupted_job_id=None,
            interrupted_job_execution_token=None,
            stop_requested=False,
            orchestrator_service=Mock(),
            download_manager=None,
            apply_routing_config=Mock(),
        )

        original_stdin = sys.stdin
        sys.stdin = io.StringIO(
            '{"type":"UPDATE_ROUTING_CONFIG","payload":{"community_mode":"all","allowed_ids":[]}}\n'
        )
        try:
            listen_for_ipc_commands(client)
        finally:
            sys.stdin = original_stdin

        client.apply_routing_config.assert_called_once_with(
            {"community_mode": "all", "allowed_ids": []}
        )

    def test_cleanup_resets_interrupted_job_even_after_current_job_is_cleared(self):
        orchestrator_service = Mock()
        orchestrator_service.get_job.return_value = {"status": "processing"}
        client = SimpleNamespace(
            current_job=None,
            interrupted_job_id="job-456",
            interrupted_job_execution_token="token-456",
            stop_requested=True,
            orchestrator_service=orchestrator_service,
            active_service_type=None,
        )

        cleanup(client, provider_id=None, service_mode="auto")

        orchestrator_service.reset_interrupted_job.assert_called_once_with(
            "job-456",
            execution_token="token-456",
            reason="provider_shutdown",
        )

    def test_job_listener_skips_job_complete_when_stop_interrupts_processing(self):
        shutdown_event = threading.Event()

        class FakeProcessor:
            def process(self_inner):
                shutdown_event.set()

        client = SimpleNamespace(
            stop_requested=True,
            interrupted_job_id=None,
            interrupted_job_execution_token=None,
            current_job={"id": "job-789", "execution_token": "token-789"},
            orchestrator_service=Mock(),
            services_config={},
            available_vram=0,
            get_service_type_for_workflow=lambda workflow_type: "wan22",
            _get_job_processor=lambda job, event: FakeProcessor(),
        )
        client.orchestrator_service.get_job.return_value = {"status": "processing"}

        listener = JobListener(client, provider_id="provider-1", shutdown_event=shutdown_event)
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = listener._process_job_safely({"id": "job-789"})

        self.assertTrue(result)
        self.assertEqual(client.interrupted_job_id, "job-789")
        self.assertIsNone(client.current_job)
        self.assertNotIn("JOB_COMPLETE", output.getvalue())

    def test_job_listener_defers_reset_when_parent_pipe_closes_without_stop_request(self):
        shutdown_event = threading.Event()

        class FakeProcessor:
            def process(self_inner):
                shutdown_event.set()

        orchestrator_service = Mock()
        orchestrator_service.get_job.return_value = {"status": "processing"}
        client = SimpleNamespace(
            stop_requested=False,
            interrupted_job_id=None,
            interrupted_job_execution_token=None,
            current_job={
                "id": "job-parent-crash",
                "execution_token": "token-parent-crash",
            },
            orchestrator_service=orchestrator_service,
            services_config={},
            available_vram=0,
            get_service_type_for_workflow=lambda workflow_type: "llm",
            _get_job_processor=lambda job, event: FakeProcessor(),
            download_manager=None,
        )

        listener = JobListener(
            client,
            provider_id="provider-1",
            shutdown_event=shutdown_event,
        )
        listener._monitor_job_cancellation = Mock()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = listener._process_job_safely(
                {
                    "id": "job-parent-crash",
                    "workflow_type": "llm",
                    "execution_token": "token-parent-crash",
                }
            )

        self.assertTrue(result)
        self.assertEqual(client.interrupted_job_id, "job-parent-crash")
        self.assertEqual(client.interrupted_job_execution_token, "token-parent-crash")
        self.assertIsNone(client.current_job)
        orchestrator_service.update_job_status.assert_not_called()
        self.assertNotIn("JOB_COMPLETE", output.getvalue())
        self.assertNotIn("JOB_FAILED", output.getvalue())

    def test_job_listener_emits_job_cleared_when_job_is_cancelled_elsewhere(self):
        shutdown_event = threading.Event()

        class FakeProcessor:
            def process(self_inner):
                return None

        client = SimpleNamespace(
            stop_requested=False,
            interrupted_job_id=None,
            interrupted_job_execution_token=None,
            current_job={"id": "job-321", "execution_token": "token-321"},
            orchestrator_service=Mock(),
            services_config={},
            available_vram=0,
            get_service_type_for_workflow=lambda workflow_type: "wan22",
            _get_job_processor=lambda job, event: FakeProcessor(),
            download_manager=None,
        )
        client.orchestrator_service.get_job.return_value = {"status": "cancelled"}

        listener = JobListener(client, provider_id="provider-1", shutdown_event=shutdown_event)
        listener._monitor_job_cancellation = Mock()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = listener._process_job_safely(
                {"id": "job-321", "workflow_type": "wan_video"}
            )

        self.assertTrue(result)
        self.assertIsNone(client.current_job)
        self.assertIn("JOB_CLEARED", output.getvalue())
        self.assertIn("cancelled", output.getvalue())
        self.assertNotIn("JOB_COMPLETE", output.getvalue())
        self.assertNotIn("JOB_FAILED", output.getvalue())

    def test_job_listener_emits_job_cleared_when_job_lease_is_lost(self):
        shutdown_event = threading.Event()

        class FakeProcessor:
            def process(self_inner):
                return None

        client = SimpleNamespace(
            stop_requested=False,
            interrupted_job_id=None,
            interrupted_job_execution_token=None,
            current_job={"id": "job-lease-lost", "execution_token": "token-lease-lost"},
            orchestrator_service=Mock(),
            services_config={},
            available_vram=0,
            get_service_type_for_workflow=lambda workflow_type: "wan22",
            _get_job_processor=lambda job, event: FakeProcessor(),
            download_manager=None,
        )
        client.orchestrator_service.get_job.return_value = {"status": "lease_lost"}

        listener = JobListener(client, provider_id="provider-1", shutdown_event=shutdown_event)
        listener._monitor_job_cancellation = Mock()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = listener._process_job_safely(
                {"id": "job-lease-lost", "workflow_type": "wan_video"}
            )

        self.assertTrue(result)
        self.assertIsNone(client.current_job)
        self.assertIn("JOB_CLEARED", output.getvalue())
        self.assertIn("lease_lost", output.getvalue())
        self.assertNotIn("JOB_COMPLETE", output.getvalue())
        self.assertNotIn("JOB_FAILED", output.getvalue())

    def test_job_listener_emits_job_cleared_when_terminal_job_throws(self):
        shutdown_event = threading.Event()

        class FailingProcessor:
            def process(self_inner):
                raise RuntimeError("processor interrupted after remote cancellation")

        client = SimpleNamespace(
            stop_requested=False,
            interrupted_job_id=None,
            interrupted_job_execution_token=None,
            current_job={"id": "job-654", "execution_token": "token-654"},
            orchestrator_service=Mock(),
            services_config={},
            available_vram=0,
            get_service_type_for_workflow=lambda workflow_type: "wan22",
            _get_job_processor=lambda job, event: FailingProcessor(),
            download_manager=None,
        )
        client.orchestrator_service.get_job.return_value = {"status": "deleted"}

        listener = JobListener(client, provider_id="provider-1", shutdown_event=shutdown_event)
        listener._monitor_job_cancellation = Mock()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = listener._process_job_safely(
                {"id": "job-654", "workflow_type": "wan_video"}
            )

        self.assertTrue(result)
        self.assertIsNone(client.current_job)
        self.assertIn("JOB_CLEARED", output.getvalue())
        self.assertIn("deleted", output.getvalue())
        self.assertNotIn("JOB_FAILED", output.getvalue())

    def test_job_listener_skips_completion_after_infrastructure_interrupt(self):
        shutdown_event = threading.Event()

        client = SimpleNamespace(
            stop_requested=False,
            interrupted_job_id=None,
            interrupted_job_execution_token=None,
            current_job={"id": "job-oom", "execution_token": "token-oom"},
            orchestrator_service=Mock(),
            services_config={},
            available_vram=0,
            get_service_type_for_workflow=lambda workflow_type: "ltx23-video-8gb",
            download_manager=None,
        )

        class InterruptedProcessor:
            def process(self_inner):
                client.interrupted_job_id = "job-oom"
                client.interrupted_job_execution_token = "token-oom"

            def close(self_inner):
                return None

        client._get_job_processor = lambda job, event: InterruptedProcessor()
        client.orchestrator_service.get_job.return_value = {"status": "pending"}

        listener = JobListener(client, provider_id="provider-1", shutdown_event=shutdown_event)
        listener._monitor_job_cancellation = Mock()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = listener._process_job_safely(
                {"id": "job-oom", "workflow_type": "ltx23-text-to-video-8gb"}
            )

        self.assertTrue(result)
        self.assertIsNone(client.current_job)
        self.assertNotIn("JOB_COMPLETE", output.getvalue())
        self.assertNotIn("JOB_FAILED", output.getvalue())

    def test_job_listener_requeues_infrastructure_error_from_processor(self):
        shutdown_event = threading.Event()

        class OomProcessor:
            def process(self_inner):
                raise InfrastructureError("Wan2GP CUDA out of memory")

            def close(self_inner):
                return None

        orchestrator_service = Mock()
        client = SimpleNamespace(
            stop_requested=False,
            interrupted_job_id=None,
            interrupted_job_execution_token=None,
            current_job={"id": "job-gpu-oom", "execution_token": "token-gpu-oom"},
            active_service_type=None,
            orchestrator_service=orchestrator_service,
            services_config={},
            available_vram=0,
            get_service_type_for_workflow=lambda workflow_type: "ltx23-video-8gb",
            _get_job_processor=lambda job, event: OomProcessor(),
            download_manager=None,
        )

        listener = JobListener(client, provider_id="provider-1", shutdown_event=shutdown_event)
        listener._monitor_job_cancellation = Mock()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = listener._process_job_safely(
                {
                    "id": "job-gpu-oom",
                    "workflow_type": "ltx23-text-to-video-8gb",
                    "execution_token": "token-gpu-oom",
                }
            )

        self.assertTrue(result)
        self.assertEqual(client.interrupted_job_id, "job-gpu-oom")
        orchestrator_service.reset_interrupted_job.assert_called_once_with(
            "job-gpu-oom",
            execution_token="token-gpu-oom",
            reason="provider_gpu_oom",
        )
        orchestrator_service.update_job_status.assert_not_called()
        self.assertIn("JOB_FAILED", output.getvalue())
        self.assertIn("job requeued", output.getvalue())

    def test_service_startup_failure_skips_failed_status_after_container_requeue(self):
        shutdown_event = threading.Event()
        orchestrator_service = Mock()
        client = SimpleNamespace(
            interrupted_job_id="job-startup-oom",
            interrupted_job_execution_token="token-startup-oom",
            current_job={"id": "job-startup-oom"},
            active_service_type=None,
            orchestrator_service=orchestrator_service,
            get_service_type_for_workflow=lambda workflow_type: "wan22",
        )
        listener = JobListener(client, provider_id="provider-1", shutdown_event=shutdown_event)
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            listener._handle_service_startup_failure(
                {
                    "id": "job-startup-oom",
                    "workflow_type": "wan_video",
                    "execution_token": "token-startup-oom",
                },
                "wan22",
            )

        orchestrator_service.update_job_status.assert_not_called()
        self.assertNotIn("JOB_FAILED", output.getvalue())

    def test_service_startup_failure_requeues_without_container_crash(self):
        shutdown_event = threading.Event()
        orchestrator_service = Mock()
        client = SimpleNamespace(
            interrupted_job_id=None,
            interrupted_job_execution_token=None,
            current_job={
                "id": "job-startup-timeout",
                "execution_token": "token-startup-timeout",
            },
            active_service_type=None,
            orchestrator_service=orchestrator_service,
            get_service_type_for_workflow=lambda workflow_type: "wan22",
        )
        listener = JobListener(client, provider_id="provider-1", shutdown_event=shutdown_event)
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            listener._handle_service_startup_failure(
                {
                    "id": "job-startup-timeout",
                    "workflow_type": "wan_video",
                    "execution_token": "token-startup-timeout",
                },
                "wan22",
            )

        orchestrator_service.reset_interrupted_job.assert_called_once_with(
            "job-startup-timeout",
            execution_token="token-startup-timeout",
            reason="provider_service_startup_failed",
        )
        orchestrator_service.update_job_status.assert_not_called()
        self.assertEqual(client.interrupted_job_id, "job-startup-timeout")
        self.assertIsNone(client.current_job)
        self.assertIn("JOB_FAILED", output.getvalue())
        self.assertIn("job requeued", output.getvalue())

    def test_duplicate_infrastructure_requeue_is_ignored_for_same_execution_token(self):
        shutdown_event = threading.Event()
        orchestrator_service = Mock()
        client = SimpleNamespace(
            interrupted_job_id=None,
            interrupted_job_execution_token=None,
            current_job={
                "id": "job-startup-race",
                "execution_token": "token-startup-race",
            },
            active_service_type=None,
            orchestrator_service=orchestrator_service,
            get_service_type_for_workflow=lambda workflow_type: "wan22",
        )
        listener = JobListener(client, provider_id="provider-1", shutdown_event=shutdown_event)
        job = {
            "id": "job-startup-race",
            "workflow_type": "wan_video",
            "execution_token": "token-startup-race",
        }
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            listener._requeue_job_after_infrastructure_error(
                job,
                InfrastructureError("Container exited unexpectedly with code 255"),
                reason="provider_container_crash",
            )
            listener._handle_service_startup_failure(job, "wan22")

        orchestrator_service.reset_interrupted_job.assert_called_once_with(
            "job-startup-race",
            execution_token="token-startup-race",
            reason="provider_container_crash",
        )
        self.assertEqual(output.getvalue().count("JOB_FAILED"), 1)
        self.assertEqual(client.interrupted_job_id, "job-startup-race")
        self.assertEqual(client.interrupted_job_execution_token, "token-startup-race")

    def test_retried_job_with_new_execution_token_clears_stale_infrastructure_marker(self):
        shutdown_event = threading.Event()

        class SuccessfulProcessor:
            def process(self_inner):
                return None

            def close(self_inner):
                return None

        orchestrator_service = Mock()
        orchestrator_service.get_job.return_value = {"status": "completed"}
        client = SimpleNamespace(
            stop_requested=False,
            interrupted_job_id="job-retry",
            interrupted_job_execution_token="old-token",
            current_job={
                "id": "job-retry",
                "execution_token": "new-token",
            },
            active_service_type=None,
            orchestrator_service=orchestrator_service,
            services_config={},
            available_vram=0,
            get_service_type_for_workflow=lambda workflow_type: "wan22",
            _get_job_processor=lambda job, event: SuccessfulProcessor(),
            download_manager=None,
        )

        listener = JobListener(client, provider_id="provider-1", shutdown_event=shutdown_event)
        listener._monitor_job_cancellation = Mock()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = listener._process_job_safely(
                {
                    "id": "job-retry",
                    "workflow_type": "wan_video",
                    "execution_token": "new-token",
                }
            )

        self.assertTrue(result)
        self.assertIsNone(client.interrupted_job_id)
        self.assertIsNone(client.interrupted_job_execution_token)
        self.assertIn("JOB_COMPLETE", output.getvalue())
        orchestrator_service.reset_interrupted_job.assert_not_called()

    def test_processor_completed_remote_status_still_emits_completion_events(self):
        shutdown_event = threading.Event()

        class SuccessfulProcessor:
            def process(self_inner):
                return None

            def close(self_inner):
                return None

        orchestrator_service = Mock()
        orchestrator_service.get_job.return_value = {"status": "completed"}
        client = SimpleNamespace(
            stop_requested=False,
            interrupted_job_id=None,
            interrupted_job_execution_token=None,
            current_job={
                "id": "job-completed-remotely",
                "execution_token": "token-completed-remotely",
            },
            active_service_type=None,
            orchestrator_service=orchestrator_service,
            services_config={},
            available_vram=0,
            get_service_type_for_workflow=lambda workflow_type: "zimage-turbo-8gb",
            _get_job_processor=lambda job, event: SuccessfulProcessor(),
            download_manager=None,
        )

        listener = JobListener(client, provider_id="provider-1", shutdown_event=shutdown_event)
        listener._monitor_job_cancellation = Mock()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = listener._process_job_safely(
                {
                    "id": "job-completed-remotely",
                    "workflow_type": "zimage-turbo-8gb-text-to-image",
                    "execution_token": "token-completed-remotely",
                    "monetize_job": True,
                }
            )

        self.assertTrue(result)
        self.assertIn("JOB_COMPLETE", output.getvalue())
        self.assertIn("MONETIZE_JOB_COMPLETE", output.getvalue())
        self.assertNotIn("JOB_CLEARED", output.getvalue())
        orchestrator_service.clear_active_job.assert_called_once_with(
            "job-completed-remotely"
        )

    def test_job_listener_requeues_generic_provider_local_runtime_error(self):
        shutdown_event = threading.Event()

        class UnreachableComfyProcessor:
            def process(self_inner):
                raise RuntimeError(
                    "Cannot reach ComfyUI at http://127.0.0.1:8188 (/prompt): "
                    "Connection refused"
                )

            def close(self_inner):
                return None

        orchestrator_service = Mock()
        client = SimpleNamespace(
            stop_requested=False,
            interrupted_job_id=None,
            interrupted_job_execution_token=None,
            current_job={
                "id": "job-comfy-unreachable",
                "execution_token": "token-comfy-unreachable",
            },
            active_service_type=None,
            orchestrator_service=orchestrator_service,
            services_config={},
            available_vram=0,
            get_service_type_for_workflow=lambda workflow_type: "wan22",
            _get_job_processor=lambda job, event: UnreachableComfyProcessor(),
            download_manager=None,
        )

        listener = JobListener(client, provider_id="provider-1", shutdown_event=shutdown_event)
        listener._monitor_job_cancellation = Mock()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = listener._process_job_safely(
                {
                    "id": "job-comfy-unreachable",
                    "workflow_type": "wan_video",
                    "execution_token": "token-comfy-unreachable",
                }
            )

        self.assertTrue(result)
        self.assertEqual(client.interrupted_job_id, "job-comfy-unreachable")
        orchestrator_service.reset_interrupted_job.assert_called_once_with(
            "job-comfy-unreachable",
            execution_token="token-comfy-unreachable",
            reason="provider_service_startup_failed",
        )
        orchestrator_service.update_job_status.assert_not_called()
        self.assertIn("JOB_FAILED", output.getvalue())
        self.assertIn("job requeued", output.getvalue())

    def test_cancellation_monitor_ignores_unverified_job_status(self):
        shutdown_event = threading.Event()
        orchestrator_service = Mock()

        def get_job_once(_job_id):
            shutdown_event.set()
            return None

        orchestrator_service.get_job.side_effect = get_job_once
        client = SimpleNamespace(
            orchestrator_service=orchestrator_service,
            active_service_type="dreamid-omni-24gb",
        )
        processor = SimpleNamespace(comfyui_client=Mock())
        listener = JobListener(
            client,
            provider_id="provider-1",
            shutdown_event=shutdown_event,
        )

        with patch("services.job_listener.docker_manager") as docker_manager_mock:
            listener._monitor_job_cancellation("job-transient-api-error", processor)

            self.assertTrue(shutdown_event.wait(1))
            processor.comfyui_client.interrupt_workflow.assert_not_called()
            docker_manager_mock.stop_container.assert_not_called()
            orchestrator_service.update_job_status.assert_not_called()

    def test_remote_cancel_cleanup_interrupts_before_container_stop(self):
        shutdown_event = threading.Event()
        orchestrator_service = Mock()
        client = SimpleNamespace(
            orchestrator_service=orchestrator_service,
            active_service_type="dreamid-omni-24gb",
        )
        processor = SimpleNamespace(comfyui_client=Mock())
        events = []
        processor.comfyui_client.interrupt_workflow.side_effect = (
            lambda *args, **kwargs: events.append("interrupt")
        )

        listener = JobListener(
            client,
            provider_id="provider-1",
            shutdown_event=shutdown_event,
        )

        with patch("services.job_listener.docker_manager") as docker_manager_mock:
            docker_manager_mock.stop_container.side_effect = (
                lambda service_type: events.append(f"stop:{service_type}")
            )

            listener._cleanup_remote_cancelled_job(
                "job-cancelled",
                processor,
                "cancelled",
            )

        self.assertEqual(events, ["interrupt", "stop:dreamid-omni-24gb"])
        orchestrator_service.clear_active_job.assert_called_once_with("job-cancelled")
        orchestrator_service.update_job_status.assert_not_called()

    def test_remote_cancel_cleanup_is_idempotent(self):
        shutdown_event = threading.Event()
        orchestrator_service = Mock()
        client = SimpleNamespace(
            orchestrator_service=orchestrator_service,
            active_service_type="qwen-8gb",
        )
        processor = SimpleNamespace(comfyui_client=Mock())

        listener = JobListener(
            client,
            provider_id="provider-1",
            shutdown_event=shutdown_event,
        )

        with patch("services.job_listener.docker_manager") as docker_manager_mock:
            listener._cleanup_remote_cancelled_job(
                "job-cancelled",
                processor,
                "cancelled",
            )
            listener._cleanup_remote_cancelled_job(
                "job-cancelled",
                processor,
                "cancelled",
            )

        processor.comfyui_client.interrupt_workflow.assert_called_once()
        docker_manager_mock.stop_container.assert_called_once_with(
            service_type="qwen-8gb"
        )
        orchestrator_service.clear_active_job.assert_called_once_with("job-cancelled")

    def test_flux_kontext_8gb_runtime_config_uses_conservative_memory_path(self):
        args, env = JobListener._flux_kontext_runtime_config(
            "flux-kontext-dev-8gb"
        )

        self.assertIn("--cpu-vae", args)
        self.assertIn("--fp16-unet", args)
        self.assertIn("--disable-dynamic-vram", args)
        self.assertIn("--disable-async-offload", args)
        self.assertIn("--disable-pinned-memory", args)
        self.assertNotIn("--fp32-vae", args)
        self.assertNotIn("--force-fp16", args)
        self.assertEqual(env["PYTORCH_JIT"], "0")

    def test_flux_kontext_12gb_runtime_config_keeps_pinned_memory_off(self):
        args, _env = JobListener._flux_kontext_runtime_config(
            "flux-kontext-dev-12gb"
        )

        self.assertIn("--fp16-vae", args)
        self.assertIn("--disable-pinned-memory", args)
        self.assertNotIn("--disable-dynamic-vram", args)

    def test_comfyui_wait_for_ready_aborts_on_container_crash_event(self):
        shutdown_event = threading.Event()
        abort_event = threading.Event()
        abort_event.set()
        client = ComfyUIClient("ws://127.0.0.1:65535/ws")

        self.assertFalse(
            client.wait_for_ready(
                shutdown_event,
                timeout=30,
                abort_event=abort_event,
            )
        )

    def test_comfyui_output_wait_interrupts_when_job_lease_is_lost(self):
        client = ComfyUIClient("ws://127.0.0.1:8188/ws?clientId={}")
        orchestrator_service = Mock()
        orchestrator_service.get_job.return_value = {"status": "lease_lost"}
        fake_ws = Mock()
        fake_thread = Mock()
        time_values = iter([0, 1, 6, 6])

        with patch("services.comfyui_service.websocket.WebSocket", return_value=fake_ws), patch(
            "services.comfyui_service.threading.Thread",
            return_value=fake_thread,
        ), patch("services.comfyui_service.logging.info"), patch(
            "services.comfyui_service.logging.warning"
        ), patch("services.comfyui_service.logging.error"), patch(
            "services.comfyui_service.logging.debug"
        ), patch.object(client, "interrupt_workflow") as interrupt_mock, patch(
            "services.comfyui_service.time.time",
            side_effect=lambda: next(time_values, 6),
        ):
            result = client.get_workflow_output(
                "prompt-123",
                "job-lease-lost",
                orchestrator_service,
                timeout_sec=60,
            )

        self.assertEqual(result, "interrupted")
        orchestrator_service.get_job.assert_called_once_with("job-lease-lost")
        interrupt_mock.assert_called_once_with(quiet_if_unreachable=True)
        fake_ws.close.assert_called_once()
        fake_thread.start.assert_called_once()
        fake_thread.join.assert_called_once_with(timeout=5)

    def test_comfyui_output_wait_quietly_interrupts_on_abort_event(self):
        client = ComfyUIClient("ws://127.0.0.1:8188/ws?clientId={}")
        orchestrator_service = Mock()
        abort_event = threading.Event()
        abort_event.set()
        fake_ws = Mock()
        fake_thread = Mock()

        with patch("services.comfyui_service.websocket.WebSocket", return_value=fake_ws), patch(
            "services.comfyui_service.threading.Thread",
            return_value=fake_thread,
        ), patch("services.comfyui_service.logging.info"), patch(
            "services.comfyui_service.logging.warning"
        ), patch("services.comfyui_service.logging.error"), patch(
            "services.comfyui_service.logging.debug"
        ), patch.object(client, "interrupt_workflow") as interrupt_mock:
            result = client.get_workflow_output(
                "prompt-123",
                "job-container-crash",
                orchestrator_service,
                abort_event=abort_event,
                timeout_sec=60,
            )

        self.assertEqual(result, "interrupted")
        interrupt_mock.assert_called_once_with(quiet_if_unreachable=True)
        fake_ws.close.assert_called_once()
        fake_thread.start.assert_called_once()
        fake_thread.join.assert_called_once_with(timeout=5)

    def test_orchestrator_service_clears_active_job_when_job_lease_is_lost(self):
        service = OrchestratorService("http://example.test")
        service._active_job_id = "job-lease-lost"
        service._active_execution_token = "token-lease-lost"
        response = Mock(status_code=403)
        response.content = b'{"error":"Forbidden"}'
        service._make_request = Mock(return_value=response)

        result = service.get_job("job-lease-lost")

        self.assertEqual(result, {"status": "lease_lost"})
        self.assertIsNone(service.get_active_execution_token())

    def test_oom_text_is_requeueable_infrastructure_error(self):
        listener = JobListener(
            SimpleNamespace(orchestrator_service=Mock()),
            provider_id="provider-1",
            shutdown_event=threading.Event(),
        )

        self.assertTrue(
            listener._is_requeueable_infrastructure_error(
                RuntimeError("container killed by OOM")
            )
        )

    def test_docker_api_context_cancel_is_requeueable_infrastructure_error(self):
        listener = JobListener(
            SimpleNamespace(orchestrator_service=Mock()),
            provider_id="provider-1",
            shutdown_event=threading.Event(),
        )

        exc = RuntimeError(
            "500 Server Error for "
            "http://127.0.0.1:2375/v1.54/containers/create"
            "?name=dgn-client-qwen3-tts: Internal Server Error "
            '("Canceled: grpc: the client connection is closing: '
            'context canceled")'
        )

        self.assertTrue(listener._is_requeueable_infrastructure_error(exc))
        self.assertEqual(
            listener._get_infrastructure_recovery_reason(exc),
            "provider_service_startup_failed",
        )


if __name__ == "__main__":
    unittest.main()
