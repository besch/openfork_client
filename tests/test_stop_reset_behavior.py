import contextlib
import io
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from cli import SHUTDOWN_EVENT, cleanup, listen_for_ipc_commands
from services.job_listener import JobListener


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


if __name__ == "__main__":
    unittest.main()
