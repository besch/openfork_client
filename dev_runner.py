import os
import sys
import time
import subprocess
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("Missing dependency: watchdog. Install with: pip install watchdog")
    sys.exit(1)


ROOT = Path(__file__).resolve().parent
WATCH_DIRS = [ROOT]  # watch the whole dgn-client directory
WATCH_EXTS = {".py"}  # only restart on Python file changes
ENTRYPOINT = str(ROOT / "dgn_client.py")
ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:3000")


class RestartOnChangeHandler(FileSystemEventHandler):
    def __init__(self, restart_callback):
        super().__init__()
        self.restart_callback = restart_callback

    def on_any_event(self, event):
        # Ignore directory events and temporary files
        if event.is_directory:
            return
        # Only react to python files
        if not any(event.src_path.endswith(ext) for ext in WATCH_EXTS):
            return
        # Skip changes from editor swap/temp files
        basename = os.path.basename(event.src_path)
        if basename.startswith(".") or basename.endswith("~") or basename.endswith(".swp"):
            return
        print(f"[dev-runner] Change detected: {event.event_type} -> {event.src_path}")
        self.restart_callback()


class ClientProcess:
    def __init__(self, entrypoint: str, orchestrator_url: str):
        self.entrypoint = entrypoint
        self.orchestrator_url = orchestrator_url
        self.proc = None
        self.provider_id = None  # capture provider_id from child stdout

    def _parse_provider_id(self, line: str):
        # dgn_client.py logs: "Successfully registered with the Orchestrator."
        # and we can follow with a log like "provider_id=<uuid>" if present.
        # Fallback: extract uuid in JSON line if printed.
        line = line.strip()
        if "provider_id" in line:
            # naive parse
            parts = line.replace(",", " ").split()
            for p in parts:
                if p.startswith("provider_id"):
                    _, _, val = p.partition("=")
                    if val:
                        self.provider_id = val
        # also capture lines that look like UUIDs after a marker
        # no-op if not found

    def start(self):
        cmd = [sys.executable, self.entrypoint, "--orchestrator-url", self.orchestrator_url]
        print(f"[dev-runner] Starting: {' '.join(cmd)}")
        # Use a new process group so we can terminate tree on Windows
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        # Capture stdout to detect provider_id so we can deregister on abrupt restart
        self.proc = subprocess.Popen(
            cmd,
            creationflags=creationflags,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def _drain_and_capture(self):
        # Read a chunk of output and try to detect provider_id
        if not self.proc or not self.proc.stdout:
            return
        try:
            # non-blocking-ish read: read up to 50 lines quickly
            for _ in range(50):
                if self.proc.poll() is not None:
                    break
                line = self.proc.stdout.readline()
                if not line:
                    break
                print(line, end="")  # mirror child's log to console
                self._parse_provider_id(line)
        except Exception:
            pass

    def _best_effort_deregister(self):
        # If we captured provider_id, call orchestrator DELETE directly
        if not self.provider_id:
            return
        url = os.environ.get("ORCHESTRATOR_URL", self.orchestrator_url)
        try:
            import urllib.request
            import urllib.parse

            qs = urllib.parse.urlencode({"providerId": self.provider_id})
            full = f"{url}/api/dgn/register?{qs}"
            req = urllib.request.Request(full, method="DELETE")
            with urllib.request.urlopen(req, timeout=3) as resp:
                print(f"[dev-runner] Deregister response: {resp.status}")
        except Exception as e:
            print(f"[dev-runner] Best-effort deregister failed: {e}")

    def stop(self):
        # drain some logs to capture provider_id before stopping
        self._drain_and_capture()

        if self.proc and self.proc.poll() is None:
            print("[dev-runner] Stopping client...")
            try:
                if os.name == "nt":
                    # On Windows send CTRL_BREAK_EVENT to the process group
                    self.proc.send_signal(subprocess.signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                    time.sleep(0.5)
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("[dev-runner] Forcing kill...")
                    self.proc.kill()
            except Exception as e:
                print(f"[dev-runner] Error while stopping process: {e}")

        # If the process was killed before it could run finally {deregister}, do a best-effort delete here.
        self._best_effort_deregister()
        self.proc = None
        self.provider_id = None

    def restart(self):
        self.stop()
        self.start()


def main():
    client = ClientProcess(ENTRYPOINT, ORCHESTRATOR_URL)
    client.start()

    event_handler = RestartOnChangeHandler(client.restart)
    observer = Observer()
    for d in WATCH_DIRS:
        print(f"[dev-runner] Watching {d}")
        observer.schedule(event_handler, str(d), recursive=True)
    observer.start()

    try:
        while True:
            # Continuously drain a bit of output to capture provider_id
            client._drain_and_capture()

            # If child exited (e.g., due to crash), restart it
            if client.proc and client.proc.poll() is not None:
                print("[dev-runner] Client exited; restarting...")
                client.restart()
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[dev-runner] Shutting down...")
    finally:
        observer.stop()
        observer.join()
        client.stop()


if __name__ == "__main__":
    main()
