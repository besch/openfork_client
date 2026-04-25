"""
Wan2GP HTTP server (port 8188) — wraps shared.api for the OpenFork DGN client.

Endpoints:
  GET  /health          → {"status": "ok"} once the Wan2GP session is ready
  POST /generate        → run one generation task (blocks until done)
                          returns {"files": ["<basename>", ...]}
  GET  /output/<name>   → download a generated output file
"""

import base64
import io
import logging
import os
import shlex
import sys
import threading
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

WAN2GP_ROOT = os.environ.get("WAN2GP_ROOT", "/opt/wan2gp")
WAN2GP_OUTPUT = os.environ.get("WAN2GP_OUTPUT", "/opt/wan2gp/outputs")

os.makedirs(WAN2GP_OUTPUT, exist_ok=True)
if WAN2GP_ROOT not in sys.path:
    sys.path.insert(0, WAN2GP_ROOT)


def _wan2gp_cli_args() -> tuple[str, ...]:
    raw_args = os.environ.get("WAN2GP_CLI_ARGS", "").strip()
    if not raw_args:
        return ()
    try:
        parsed = tuple(shlex.split(raw_args))
        logging.info("Using Wan2GP CLI args: %s", " ".join(parsed))
        return parsed
    except ValueError as exc:
        logging.warning("Ignoring invalid WAN2GP_CLI_ARGS=%r: %s", raw_args, exc)
        return ()


# ── Wan2GP session init (blocking — server starts only after model is loaded) ─
logging.info("Initialising Wan2GP session (this may take several minutes)...")
from shared.api import init  # noqa: E402

_session = init(
    root=Path(WAN2GP_ROOT),
    output_dir=Path(WAN2GP_OUTPUT),
    cli_args=_wan2gp_cli_args(),
    console_output=True,
)
logging.info("Wan2GP session ready.")

_gen_lock = threading.Lock()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI()


class GenerateRequest(BaseModel):
    settings: Dict[str, Any]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    """Run a Wan2GP generation task synchronously."""
    from PIL import Image

    settings = dict(req.settings)

    # Decode PIL Image fields encoded as data-URIs by the client
    for key in ("image_start", "image_end"):
        val = settings.get(key)
        if isinstance(val, str) and val.startswith("data:image"):
            _, b64 = val.split(",", 1)
            img_bytes = base64.b64decode(b64)
            settings[key] = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    with _gen_lock:
        job = _session.submit_task(settings)
        result = job.result()

    if not result or not result.success:
        errors = [
            f"{e.stage}: {e.message}" for e in (result.errors if result else [])
        ]
        raise HTTPException(status_code=500, detail={"errors": errors})

    basenames = [Path(str(f)).name for f in result.generated_files]
    return {"files": basenames}


@app.get("/output/{filename}")
def download_output(filename: str):
    """Serve a generated output file."""
    safe_name = Path(filename).name  # strip any path traversal
    file_path = os.path.join(WAN2GP_OUTPUT, safe_name)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        file_path, media_type="application/octet-stream", filename=safe_name
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8188)
