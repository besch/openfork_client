"""
Supabase Realtime subscriber that wakes the job listener whenever a new
dgn_job_notifications row arrives (inserted by a Postgres trigger the moment
a dgn_jobs row transitions to status='pending').

This replaces the fixed-interval poll cycle with an event-driven wake-up:
providers react in milliseconds instead of waiting up to JOB_POLL_INTERVAL
seconds.  The existing backoff + job_wakeup_event mechanism in JobListener
is unchanged — Realtime just sets the event earlier.

Only active in OAuth (Electron) mode.  Headless API-key clients keep their
existing 2-second poll because Supabase Realtime requires a user JWT for RLS.
Falls back to polling transparently on any connection failure.

Authentication:
- Uses SUPABASE_PUBLISHABLE_KEY (sb_publishable_...) or SUPABASE_ANON_KEY for
  the WebSocket URL's apikey parameter (required by Supabase Realtime).
- Uses access_token from the user's OAuth session in the join payload for RLS.
"""

import asyncio
import json
import logging
import random
import threading
from typing import Optional

from config import SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, SUPABASE_ANON_KEY


_RECONNECT_BASE = 2.0   # initial back-off (seconds)
_RECONNECT_CAP = 60.0   # maximum back-off (seconds)
_HEARTBEAT_INTERVAL = 25  # Supabase server expects a heartbeat < 30 s


class RealtimeJobWatcher:
    """
    Opens a Supabase Realtime WebSocket and subscribes to INSERT events on
    public.dgn_job_notifications.  On each event it calls
    ``wakeup_event.set()`` so the JobListener breaks out of its back-off
    sleep and polls for the new job immediately.

    Thread-safety
    -------------
    * ``start()`` / ``update_token()`` are safe to call from any thread.
    * The WebSocket runs on a dedicated asyncio event loop in a daemon thread.
    """

    def __init__(
        self,
        access_token: str,
        wakeup_event: threading.Event,
        shutdown_event: threading.Event,
    ) -> None:
        self._token = access_token
        self._token_lock = threading.Lock()
        self._wakeup = wakeup_event
        self._shutdown = shutdown_event
        self._connected = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Set from the async loop when a token update arrives from the main thread.
        self._token_changed: Optional[asyncio.Event] = None

    # ── Public API ───────────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn the background watcher thread.  Call once after registration."""
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="realtime-job-watcher",
        )
        self._thread.start()

    def update_token(self, new_token: str) -> None:
        """
        Thread-safe: update the JWT (called when Electron refreshes auth).
        The live WebSocket session picks up the new token without reconnecting.
        """
        with self._token_lock:
            self._token = new_token
        if self._token_changed is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._token_changed.set)

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Thread entry point ───────────────────────────────────────────────────

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._reconnect_loop())
        except Exception as exc:
            logging.warning(f"Realtime watcher thread exited unexpectedly: {exc}")
        finally:
            self._connected = False
            self._loop = None
            loop.close()
            logging.info("Realtime watcher thread stopped.")

    # ── Reconnect loop ───────────────────────────────────────────────────────

    async def _reconnect_loop(self) -> None:
        self._token_changed = asyncio.Event()
        delay = _RECONNECT_BASE
        while not self._shutdown.is_set():
            try:
                await self._subscribe()
                delay = _RECONNECT_BASE  # reset on clean disconnect
            except Exception as exc:
                self._connected = False
                if self._shutdown.is_set():
                    break
                jitter = random.uniform(0.0, delay * 0.2)
                logging.warning(
                    f"Realtime: disconnected ({exc}). "
                    f"Reconnecting in {delay:.0f}s…"
                )
                await asyncio.sleep(delay + jitter)
                delay = min(delay * 1.5, _RECONNECT_CAP)

    # ── WebSocket session ────────────────────────────────────────────────────

    async def _subscribe(self) -> None:
        try:
            import websockets  # already in requirements.txt
        except ImportError:
            logging.warning(
                "websockets package unavailable — Realtime watcher disabled. "
                "Job listener will fall back to polling."
            )
            # Park here until shutdown so the reconnect loop doesn't spin.
            while not self._shutdown.is_set():
                await asyncio.sleep(5)
            return

        api_key = SUPABASE_PUBLISHABLE_KEY
        if not api_key:
            logging.warning(
                "No SUPABASE_PUBLISHABLE_KEY or SUPABASE_ANON_KEY configured. "
                "Realtime watcher disabled. Set one of these env vars."
            )
            while not self._shutdown.is_set():
                await asyncio.sleep(5)
            return

        ws_url = (
            SUPABASE_URL
            .replace("https://", "wss://")
            .replace("http://", "ws://")
            + f"/realtime/v1/websocket?apikey={api_key}&vsn=1.0.0"
        )
        token = self._token_snapshot()
        topic = "realtime:dgn-job-watcher"
        join_ref = "1"
        _ref_counter = 0

        def _next_ref() -> str:
            nonlocal _ref_counter
            _ref_counter += 1
            return str(_ref_counter)

        async with websockets.connect(
            ws_url,
            ping_interval=None,  # we send Phoenix heartbeats manually
            close_timeout=5,
        ) as ws:
            # ── Join channel ─────────────────────────────────────────────
            await ws.send(json.dumps({
                "event": "phx_join",
                "topic": topic,
                "payload": {
                    "config": {
                        "broadcast": {"self": False},
                        "presence": {"key": ""},
                        "postgres_changes": [
                            {
                                "event": "INSERT",
                                "schema": "public",
                                "table": "dgn_job_notifications",
                            },
                        ],
                    },
                    "access_token": token,
                },
                "ref": _next_ref(),
                "join_ref": join_ref,
            }))

            last_heartbeat = asyncio.get_event_loop().time()

            while not self._shutdown.is_set():
                # ── Token refresh ─────────────────────────────────────
                if self._token_changed is not None and self._token_changed.is_set():
                    self._token_changed.clear()
                    new_token = self._token_snapshot()
                    await ws.send(json.dumps({
                        "event": "access_token",
                        "topic": topic,
                        "payload": {"access_token": new_token},
                        "ref": _next_ref(),
                        "join_ref": join_ref,
                    }))
                    logging.info("Realtime: access_token refreshed on live channel.")

                # ── Phoenix heartbeat ─────────────────────────────────
                now = asyncio.get_event_loop().time()
                if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                    await ws.send(json.dumps({
                        "event": "heartbeat",
                        "topic": "phoenix",
                        "payload": {},
                        "ref": _next_ref(),
                    }))
                    last_heartbeat = now

                # ── Receive ───────────────────────────────────────────
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event = msg.get("event")

                if event == "phx_reply":
                    resp = msg.get("payload", {})
                    if resp.get("status") == "ok" and not self._connected:
                        self._connected = True
                        logging.info(
                            "Realtime: subscribed to dgn_job_notifications — "
                            "event-driven job wake-ups active."
                        )
                        # Do one immediate poll to catch jobs submitted while
                        # we were connecting.
                        self._wakeup.set()
                    elif resp.get("status") != "ok":
                        err = resp.get("response", resp)
                        logging.warning(f"Realtime: channel join rejected: {err}")
                        return  # triggers reconnect

                elif event == "postgres_changes":
                    data = msg.get("payload", {}).get("data", {})
                    logging.debug(
                        f"Realtime: INSERT on dgn_job_notifications "
                        f"(policy={data.get('record', {}).get('accept_policy')}) "
                        "→ waking job listener"
                    )
                    self._wakeup.set()

                elif event in ("phx_error", "phx_close"):
                    self._connected = False
                    logging.info(f"Realtime: received {event}. Reconnecting…")
                    return  # triggers reconnect

        self._connected = False

    def _token_snapshot(self) -> str:
        with self._token_lock:
            return self._token
