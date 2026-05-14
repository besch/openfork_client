import threading
import unittest
from types import SimpleNamespace

from services.job_listener import JobListener


class FakeOrchestrator:
    def __init__(self, client=None, shutdown_event=None):
        self.client = client
        self.shutdown_event = shutdown_event
        self.get_next_job_calls = []
        self.peek_available_jobs_calls = []

    def get_next_job(self, **kwargs):
        self.get_next_job_calls.append(kwargs)
        return None

    def peek_available_jobs(self, **kwargs):
        self.peek_available_jobs_calls.append(kwargs)
        return []

    def get_prefetch_suggestions(self, *args, **kwargs):
        return []


def make_client(orchestrator, **overrides):
    values = {
        "orchestrator_service": orchestrator,
        "compaction_pending": False,
        "monetize_mode": False,
        "process_own_jobs": True,
        "community_mode": "all",
        "allowed_ids": [],
        "processing_lock": threading.Lock(),
        "job_wakeup_event": threading.Event(),
        "download_manager": None,
        "services_config": {"svc": {}},
        "active_service_type": None,
        "current_job": None,
        "get_service_type_for_workflow": lambda workflow_type: "svc",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class CompactionPauseTests(unittest.TestCase):
    def test_fetch_next_job_skips_all_reservation_when_compaction_pending(self):
        orchestrator = FakeOrchestrator()
        client = make_client(
            orchestrator,
            compaction_pending=True,
            monetize_mode=True,
            process_own_jobs=True,
            community_mode="all",
        )
        listener = JobListener(client, "provider-1", threading.Event())

        self.assertIsNone(listener._fetch_next_job_with_priority())
        self.assertEqual(orchestrator.get_next_job_calls, [])

    def test_fetch_next_job_stops_between_priority_tracks_if_compaction_arrives(self):
        class FlipAfterFirstCallOrchestrator(FakeOrchestrator):
            def get_next_job(self, **kwargs):
                self.get_next_job_calls.append(kwargs)
                self.client.compaction_pending = True
                return None

        orchestrator = FlipAfterFirstCallOrchestrator()
        client = make_client(
            orchestrator,
            monetize_mode=True,
            process_own_jobs=True,
            community_mode="all",
        )
        orchestrator.client = client
        listener = JobListener(client, "provider-1", threading.Event())

        self.assertIsNone(listener._fetch_next_job_with_priority())
        self.assertEqual(len(orchestrator.get_next_job_calls), 1)
        self.assertEqual(
            orchestrator.get_next_job_calls[0]["accept_policy"],
            "monetize",
        )

    def test_auto_mode_does_not_reserve_peeked_job_if_compaction_arrives(self):
        shutdown_event = threading.Event()

        class FlipDuringPeekOrchestrator(FakeOrchestrator):
            def peek_available_jobs(self, **kwargs):
                self.peek_available_jobs_calls.append(kwargs)
                self.client.compaction_pending = True
                shutdown_event.set()
                return [{"id": "job-1", "workflow_type": "wf"}]

            def get_next_job(self, **kwargs):
                self.get_next_job_calls.append(kwargs)
                return {"id": "job-1", "workflow_type": "wf"}

        orchestrator = FlipDuringPeekOrchestrator(shutdown_event=shutdown_event)
        client = make_client(
            orchestrator,
            process_own_jobs=True,
            community_mode="none",
        )
        orchestrator.client = client
        listener = JobListener(client, "provider-1", shutdown_event)

        listener.listen_for_jobs_auto()

        self.assertEqual(len(orchestrator.peek_available_jobs_calls), 1)
        self.assertEqual(orchestrator.get_next_job_calls, [])


if __name__ == "__main__":
    unittest.main()
