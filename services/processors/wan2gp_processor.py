"""
Wan2GP Processor Base

Wan2GP is used as a Python library (not an HTTP server). The shared.api
module from the Wan2GP installation is imported directly at runtime.

A single WanGPSession is kept alive across jobs so the loaded model is
reused without reloading on each request.
"""

import os
import sys
import logging
import threading
from pathlib import Path
from typing import Optional, Tuple

from config import THUMBNAIL_WIDTH
from services.processors.base import BaseJobProcessor
from utils.media_utils import generate_thumbnail, get_video_duration

WAN2GP_ROOT = os.environ.get("WAN2GP_ROOT", "/opt/wan2gp")
WAN2GP_OUTPUT = os.environ.get("WAN2GP_OUTPUT", "/opt/wan2gp/outputs")

# Aspect ratio → "WIDTHxHEIGHT" strings required by Wan2GP
_ASPECT_RESOLUTIONS = {
    "16:9": "1280x720",
    "9:16": "720x1280",
    "1:1":  "1024x1024",
    "4:3":  "1024x768",
    "3:4":  "768x1024",
    "21:9": "1280x544",
    "2:1":  "1280x640",
}

_session = None
_session_lock = threading.Lock()


def _get_session():
    """Return the module-level WanGPSession, initialising it on first call."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                # LTX-2.3 uses the FP8 Gemma 3 12B text encoder which requires:
                #   1. CC >= 8.9 (Ada Lovelace / Hopper) for FP8 tensor-core support
                #   2. The GPU's SM version must be in this PyTorch build's arch list.
                #      Newer GPUs (e.g. RTX 5060 Ti SM 12.0 / Blackwell) crash with
                #      "no kernel image available" if PyTorch was compiled for SM <= 9.0.
                import torch as _torch_cc
                if _torch_cc.cuda.is_available():
                    _cc_major, _cc_minor = _torch_cc.cuda.get_device_capability()
                    _gpu_name = _torch_cc.cuda.get_device_name(0)
                    # NOTE: We intentionally skip get_arch_list() here.
                    # cu128 wheels use PTX intermediate code for forward-compat
                    # architectures (e.g. sm_89 / RTX 40xx), so sm_89 may not
                    # appear in the SASS arch list even though the GPU and
                    # PyTorch build are fully compatible.
                    # The only meaningful gate for LTX-2.3 is CC >= 8.9 (FP8).
                    if _cc_major < 8 or (_cc_major == 8 and _cc_minor < 9):
                        raise RuntimeError(
                            f"LTX-2.3 requires CUDA compute capability 8.9+ "
                            f"(this GPU: {_gpu_name}, CC {_cc_major}.{_cc_minor}). "
                            f"Supported: RTX 4090/4080/4070 Ti Super/4070/4060 Ti (Ada Lovelace), "
                            f"L40S, H100, H200. "
                            f"FP8 Gemma 3 encoder cannot run on CC {_cc_major}.{_cc_minor}."
                        )
                    logging.info(f"GPU '{_gpu_name}' CC {_cc_major}.{_cc_minor} — FP8 OK for LTX-2.3")

                # PyTorch < 2.4 compat: torch.nn.Buffer was added in 2.4 and is
                # used by mmgp (Memory Management for the GPU Poor).  The shim
                # returns the tensor unchanged — nn.Buffer's only role is to mark
                # a tensor as a non-parameter buffer, which is safe to skip here.
                import torch as _torch
                if not hasattr(_torch.nn, "Buffer"):
                    _torch.nn.Buffer = lambda data=None, persistent=True: data

                # PyTorch < 2.2 compat: torch.unique not implemented for BFloat16.
                # rope.py has multiple call sites.  Replace ALL of them with a compat
                # wrapper appended to the file (appended so torch is in scope; replacing
                # call sites first avoids touching the helper itself).
                # When return_inverse/return_counts are True, torch.unique returns a
                # tuple — only cast the first element (unique values) back to the
                # original dtype; leave index tensors as-is.
                _rope_py = os.path.join(
                    WAN2GP_ROOT, "models", "ltx2", "ltx_core",
                    "model", "transformer", "rope.py",
                )
                _NEW_HELPER = (
                    "\n\n# BFloat16 compat shim — OpenFork DGN client v2\n"
                    "def _uniq_bf16_compat(t, **kw):\n"
                    "    _t = t.float() if t.dtype == torch.bfloat16 else t\n"
                    "    result = torch.unique(_t, **kw)\n"
                    "    if isinstance(result, tuple):\n"
                    "        return (result[0].to(t.dtype),) + result[1:]\n"
                    "    return result.to(t.dtype)\n"
                )
                if os.path.isfile(_rope_py):
                    with open(_rope_py) as _f:
                        _src = _f.read()
                    if "_uniq_bf16_compat(" not in _src:
                        # First time: replace all call sites and append helper
                        _src = _src.replace("torch.unique(", "_uniq_bf16_compat(")
                        _src += _NEW_HELPER
                        with open(_rope_py, "w") as _f:
                            _f.write(_src)
                        logging.info("Patched Wan2GP rope.py: wrapped all torch.unique calls for BFloat16")
                    elif "OpenFork DGN client v2" not in _src:
                        # Call sites already replaced but helper is the old broken version
                        # (missing tuple handling) — strip old helper and append fixed one
                        _old_marker = "\n\n# BFloat16 compatibility shim injected by OpenFork DGN client\n"
                        if _old_marker in _src:
                            _src = _src[: _src.find(_old_marker)]
                        _src += _NEW_HELPER
                        with open(_rope_py, "w") as _f:
                            _f.write(_src)
                        logging.info("Updated Wan2GP rope.py BFloat16 patch: fixed tuple return handling")

                if WAN2GP_ROOT not in sys.path:
                    sys.path.insert(0, WAN2GP_ROOT)
                from shared.api import init  # noqa: PLC0415 — deferred import
                os.makedirs(WAN2GP_OUTPUT, exist_ok=True)
                _session = init(
                    root=Path(WAN2GP_ROOT),
                    output_dir=Path(WAN2GP_OUTPUT),
                    console_output=True,
                )
                logging.info(f"Wan2GP session initialised (root={WAN2GP_ROOT})")
    return _session


class Wan2GPProcessor(BaseJobProcessor):
    """Base class for processors that use the Wan2GP Python API."""

    MAX_GENERATION_SECONDS = 7200  # 2 hours

    @staticmethod
    def aspect_to_resolution(aspect_ratio: str) -> str:
        return _ASPECT_RESOLUTIONS.get(aspect_ratio, "1280x720")

    def _run_task(self, settings: dict) -> list:
        """Submit a generation task to Wan2GP and block until complete.

        Returns a list of absolute output file paths, or an empty list on
        failure / shutdown.
        """
        session = _get_session()
        job = session.submit_task(settings)

        result_holder = [None]
        exc_holder = [None]

        def _worker():
            try:
                result_holder[0] = job.result()
            except Exception as exc:
                exc_holder[0] = exc

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        elapsed = 0
        while t.is_alive():
            if self.shutdown_event.is_set():
                logging.warning(
                    f"Shutdown requested during Wan2GP generation for job {self.job_id}"
                )
                return []
            t.join(timeout=2)
            elapsed += 2
            if elapsed > self.MAX_GENERATION_SECONDS:
                logging.error(f"Wan2GP generation timed out for job {self.job_id}")
                return []

        if exc_holder[0]:
            logging.error(
                f"Wan2GP raised an exception for job {self.job_id}: {exc_holder[0]}",
                exc_info=exc_holder[0],
            )
            return []

        result = result_holder[0]
        if not result or not result.success:
            for err in result.errors if result else []:
                logging.error(f"Wan2GP error [{err.stage}]: {err.message}")
            return []

        return list(result.generated_files)

    def _handle_video_output(
        self, file_path: str
    ) -> Optional[Tuple[str, Optional[str], float]]:
        """Upload a locally generated video file.

        Returns (storage_path, thumbnail_storage_path, duration) or None.
        The source file is deleted after a successful upload.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            video_storage_path = self.orchestrator_service.upload_output(
                file_path, self.job_id, "video/mp4"
            )
            if not video_storage_path:
                logging.error(f"Video upload failed for job {self.job_id}")
                return None

            thumbnail_storage_path = None
            thumb_local = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
            if generate_thumbnail(file_path, thumb_local, width=THUMBNAIL_WIDTH):
                thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(
                    thumb_local, self.job_id
                )
                if os.path.exists(thumb_local):
                    os.remove(thumb_local)

            duration = get_video_duration(file_path)
            return video_storage_path, thumbnail_storage_path, duration

        except Exception as e:
            logging.error(
                f"Failed to handle video output for job {self.job_id}: {e}",
                exc_info=True,
            )
            return None
        finally:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
