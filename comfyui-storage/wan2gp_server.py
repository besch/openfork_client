"""
Wan2GP HTTP server (port 8188) — wraps shared.api for the OpenFork DGN client.

Endpoints:
  GET  /health          → {"status": "ok"} once the Wan2GP session is ready
  POST /generate        → run one generation task (blocks until done)
                          returns {"files": ["<basename>", ...]}
  GET  /output/<name>   → download a generated output file
"""

import base64
import faulthandler
import io
import logging
import os
import shlex
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
faulthandler.enable(all_threads=True)

WAN2GP_ROOT = os.environ.get("WAN2GP_ROOT", "/opt/wan2gp")
WAN2GP_OUTPUT = os.environ.get("WAN2GP_OUTPUT", "/opt/wan2gp/outputs")
WAN2GP_EXIT_AFTER_JOB = os.environ.get("WAN2GP_EXIT_AFTER_JOB", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
WAN2GP_EXIT_DELAY_SECONDS = float(os.environ.get("WAN2GP_EXIT_DELAY_SECONDS", "1"))

_HDR_LORA_PATH = os.path.join(WAN2GP_ROOT, "ckpts", "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors")
_HDR_SCENE_EMB_PATH = os.path.join(WAN2GP_ROOT, "ckpts", "ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors")

os.makedirs(WAN2GP_OUTPUT, exist_ok=True)
if WAN2GP_ROOT not in sys.path:
    sys.path.insert(0, WAN2GP_ROOT)

_gen_lock = threading.Lock()
_exit_lock = threading.Lock()
_exit_scheduled = False


def _schedule_process_exit(reason: str) -> None:
    """Ask the start_cloud supervisor for a fresh process after this job."""
    if not WAN2GP_EXIT_AFTER_JOB:
        return

    global _exit_scheduled
    with _exit_lock:
        if _exit_scheduled:
            return
        _exit_scheduled = True

    delay = max(WAN2GP_EXIT_DELAY_SECONDS, 0.0)
    logging.info(
        "Scheduling Wan2GP process recycle in %.1fs after %s.",
        delay,
        reason,
    )

    def _exit_after_delay() -> None:
        time.sleep(delay)
        logging.info("Exiting Wan2GP process so the supervisor can restart it.")
        os._exit(0)

    threading.Thread(
        target=_exit_after_delay,
        name="wan2gp-process-recycle",
        daemon=True,
    ).start()


def _prefer_host_libcuda() -> None:
    """Avoid NVIDIA forward-compat libcuda on GeForce cloud hosts."""
    host_libcuda = Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1")
    if not host_libcuda.exists():
        return

    disabled_compat = False
    conf_dir = Path("/etc/ld.so.conf.d")
    for conf in conf_dir.glob("*.conf"):
        try:
            text = conf.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "/usr/local/cuda" not in text or "/compat" not in text:
            continue
        try:
            conf.rename(conf.with_name(f"{conf.name}.disabled.{os.getpid()}"))
            disabled_compat = True
        except OSError as exc:
            logging.warning("Could not disable CUDA compat config %s: %s", conf, exc)

    if disabled_compat:
        subprocess.run(["ldconfig"], check=False)
        logging.info(
            "Disabled CUDA forward-compat libcuda path; using host driver libcuda."
        )


_prefer_host_libcuda()


def _ensure_libcuda_linker_name() -> None:
    """Expose libcuda.so for Triton/native builds in WSL GPU containers."""
    try:
        output = subprocess.check_output(
            ["ldconfig", "-p"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        logging.debug("Could not query ldconfig for libcuda: %s", exc)
        return

    driver_path = None
    for line in output.splitlines():
        if "libcuda.so.1" in line and "=>" in line:
            driver_path = Path(line.split("=>", 1)[1].strip())
            break

    if not driver_path:
        return

    linker_path = driver_path.with_name("libcuda.so")
    if linker_path.exists():
        return

    try:
        os.symlink(driver_path.name, linker_path)
        logging.info("Created libcuda linker symlink at %s.", linker_path)
    except FileExistsError:
        return
    except Exception as exc:
        logging.warning("Could not create libcuda linker symlink: %s", exc)


_ensure_libcuda_linker_name()


def _log_huggingface_lookups() -> None:
    """Log offline Hugging Face file lookups so missing baked files are visible."""
    try:
        import huggingface_hub
    except Exception as exc:
        logging.debug("Could not install Hugging Face lookup logging: %s", exc)
        return

    original_hf_hub_download = getattr(huggingface_hub, "hf_hub_download", None)
    if callable(original_hf_hub_download):

        def _hf_hub_download_with_logging(*args, **kwargs):
            repo_id = kwargs.get("repo_id") or (args[0] if args else None)
            filename = kwargs.get("filename") or (args[1] if len(args) > 1 else None)
            logging.info(
                "HF lookup: hf_hub_download repo=%r filename=%r subfolder=%r local_dir=%r",
                repo_id,
                filename,
                kwargs.get("subfolder"),
                kwargs.get("local_dir"),
            )
            return original_hf_hub_download(*args, **kwargs)

        huggingface_hub.hf_hub_download = _hf_hub_download_with_logging

    original_snapshot_download = getattr(huggingface_hub, "snapshot_download", None)
    if callable(original_snapshot_download):

        def _snapshot_download_with_logging(*args, **kwargs):
            repo_id = kwargs.get("repo_id") or (args[0] if args else None)
            logging.info(
                "HF lookup: snapshot_download repo=%r allow_patterns=%r local_dir=%r",
                repo_id,
                kwargs.get("allow_patterns"),
                kwargs.get("local_dir"),
            )
            return original_snapshot_download(*args, **kwargs)

        huggingface_hub.snapshot_download = _snapshot_download_with_logging


_log_huggingface_lookups()


def _has_wan22_transformer_variant(variant: str) -> bool:
    ckpt_dir = Path(WAN2GP_ROOT) / "ckpts"
    model_pairs = (
        (
            f"wan2.2_text2video_14B_high_{variant}.safetensors",
            f"wan2.2_text2video_14B_low_{variant}.safetensors",
        ),
        (
            f"wan2.2_image2video_14B_high_{variant}.safetensors",
            f"wan2.2_image2video_14B_low_{variant}.safetensors",
        ),
    )
    return any(all((ckpt_dir / name).is_file() for name in pair) for pair in model_pairs)


def _patch_wan22_vae_selection() -> None:
    """Make upstream Wan2GP use the baked Wan 2.2 VAE for 48-channel Wan models.

    Wan 2.2 T2V 14B keeps a 16-channel transformer input and must use the
    Wan2.1 VAE. The Wan2.2 VAE has 48 latent channels and only matches the
    newer 48-channel families.
    """
    target = Path(WAN2GP_ROOT) / "models" / "wan" / "any2video.py"
    old = (
        '        elif model_def.get("wan_5B_class", False):\n'
        "            self.vae_stride = (4, 16, 16)\n"
        '            vae_checkpoint = "Wan2.2_VAE.safetensors"\n'
        "            vae = Wan2_2_VAE\n"
    )
    broad = (
        '        elif model_def.get("wan_5B_class", False) or base_model_type in ["t2v_2_2", "i2v_2_2"]:\n'
        '            if model_def.get("wan_5B_class", False):\n'
        "                self.vae_stride = (4, 16, 16)\n"
        '            vae_checkpoint = "Wan2.2_VAE.safetensors"\n'
        "            vae = Wan2_2_VAE\n"
    )
    new = (
        '        elif model_def.get("wan_5B_class", False) or base_model_type in ["ti2v_2_2"]:\n'
        '            if model_def.get("wan_5B_class", False):\n'
        "                self.vae_stride = (4, 16, 16)\n"
        '            vae_checkpoint = "Wan2.2_VAE.safetensors"\n'
        "            vae = Wan2_2_VAE\n"
    )
    try:
        text = target.read_text(encoding="utf-8")
        if new in text:
            return
        if broad in text:
            target.write_text(text.replace(broad, new), encoding="utf-8")
            logging.info("Repaired Wan2GP Wan 2.2 VAE selection.")
            return
        if old not in text:
            logging.warning("Could not find Wan2GP VAE selection block to patch.")
            return
        target.write_text(text.replace(old, new), encoding="utf-8")
        logging.info("Patched Wan2GP Wan 2.2 VAE selection.")
    except Exception as exc:
        logging.warning("Could not patch Wan2GP Wan 2.2 VAE selection: %s", exc)


def _wan2gp_cli_args() -> tuple[str, ...]:
    raw_args = os.environ.get("WAN2GP_CLI_ARGS", "").strip()
    if not raw_args:
        parsed: tuple[str, ...] = ()
    else:
        try:
            parsed = tuple(shlex.split(raw_args))
        except ValueError as exc:
            logging.warning("Ignoring invalid WAN2GP_CLI_ARGS=%r: %s", raw_args, exc)
            parsed = ()

    has_dtype_flag = any(arg in {"--fp16", "--bf16"} for arg in parsed)
    if (
        not has_dtype_flag
        and _has_wan22_transformer_variant("quanto_mfp16_int8")
        and not _has_wan22_transformer_variant("quanto_mbf16_int8")
    ):
        parsed = (*parsed, "--fp16")
        logging.info(
            "Added --fp16 because the local Wan 2.2 transformers are mfp16."
        )
    elif (
        not has_dtype_flag
        and _has_wan22_transformer_variant("mbf16")
        and not _has_wan22_transformer_variant("quanto_mbf16_int8")
    ):
        parsed = (*parsed, "--bf16")
        logging.info(
            "Added --bf16 because the local Wan 2.2 transformers are mbf16."
        )

    if not parsed:
        return ()

    try:
        logging.info("Using Wan2GP CLI args: %s", " ".join(parsed))
        return parsed
    except Exception as exc:
        logging.warning("Ignoring Wan2GP CLI args %r: %s", parsed, exc)
    return ()


# ── Wan2GP session init (blocking — server starts only after model is loaded) ─
logging.info("Initialising Wan2GP session (this may take several minutes)...")
_patch_wan22_vae_selection()


def _init_session_sync():
    from shared.api import init

    try:
        return init(
            root=Path(WAN2GP_ROOT),
            output_dir=Path(WAN2GP_OUTPUT),
            cli_args=_wan2gp_cli_args(),
            console_output=True,
        )
    except BaseException:
        logging.error(
            "Wan2GP session initialization failed:\n%s",
            traceback.format_exc(),
        )
        raise


# Taichi/SCAIL binds parts of its runtime to the process main thread. Initialise
# Wan2GP on the main thread and keep generation there; using FastAPI's sync
# threadpool or a custom executor lets the first SCAIL request succeed but leaves
# later requests failing with Taichi main_thread_id_/FieldsBuilder errors.
_session = _init_session_sync()
logging.info("Wan2GP session ready.")


def _empty_download_def(*_args, **_kwargs) -> dict[str, Any]:
    return {
        "repoId": "",
        "sourceFolderList": [],
        "fileList": [],
        "targetFolderList": [],
    }


def _skip_eager_shared_asset_downloads() -> None:
    """Skip WanGP helper-asset downloads that are not needed for basic T2V/I2V.

    WanGP calls query_core_shared_model_files() before loading every main model.
    The OpenFork Wan2GP image is intentionally offline at runtime and bakes the
    Wan 2.2 transformer/VAE/T5 files only; the eager helper bundle includes
    pose/depth/audio assets that basic T2V/I2V does not use.
    """
    enabled = os.environ.get("WAN2GP_SKIP_EAGER_SHARED_DOWNLOADS", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return

    try:
        import wgp

        wgp.query_core_shared_model_files = _empty_download_def
        wgp.query_matanyone_download_def = _empty_download_def
        patched_handlers = set()
        for handler in getattr(wgp, "model_types_handlers", {}).values():
            if handler in patched_handlers or not hasattr(handler, "query_model_files"):
                continue
            patched_handlers.add(handler)
            original_query_model_files = handler.query_model_files

            def _query_model_files(
                compute_list,
                base_model_type,
                model_def=None,
                _original_query_model_files=original_query_model_files,
            ):
                if str(base_model_type or "") in {"t2v_2_2", "i2v_2_2"}:
                    logging.info(
                        "Skipping Wan2GP Wan 2.2 family helper-asset downloads for %s.",
                        base_model_type,
                    )
                    return []
                return _original_query_model_files(
                    compute_list,
                    base_model_type,
                    model_def,
                )

            handler.query_model_files = staticmethod(_query_model_files)
        logging.info("Skipping Wan2GP eager shared helper-asset downloads.")
    except Exception as exc:
        logging.warning(
            "Could not disable Wan2GP eager shared helper-asset downloads: %s",
            exc,
        )


_skip_eager_shared_asset_downloads()

# Verify HDR IC-LoRA is present (if built into the image)
if not os.path.isfile(os.path.normpath(_HDR_LORA_PATH)):
    logging.warning(
        "HDR IC-LoRA not found at %s — HDR jobs will fail. "
        "Rebuild the image to include Lightricks/LTX-2.3-22b-IC-LoRA-HDR.",
        _HDR_LORA_PATH,
    )

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI()


class GenerateRequest(BaseModel):
    settings: Dict[str, Any]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Run a Wan2GP generation task on the process main thread.

    This intentionally blocks the event loop while generation runs. Wan2GP is a
    single-job backend here, and preserving Taichi's main-thread affinity is more
    important than serving concurrent health checks during generation.
    """
    basenames = _generate_sync(dict(req.settings))
    return {"files": basenames}


def _generate_sync(raw_settings: Dict[str, Any]) -> list[str]:
    """Run Wan2GP on the process main thread for every request.

    SCAIL/Taichi keeps process-main-thread state after the first generation.
    FastAPI sync routes and custom executors run on worker threads, which lets
    the first SCAIL job succeed and the next one fail with Taichi main-thread
    errors.
    """
    from PIL import Image

    settings = dict(raw_settings)

    # Decode PIL Image fields encoded as data-URIs by the client
    for key in ("image_start", "image_end"):
        val = settings.get(key)
        if isinstance(val, str) and val.startswith("data:image"):
            _, b64 = val.split(",", 1)
            img_bytes = base64.b64decode(b64)
            settings[key] = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Inject HDR IC-LoRA if requested
    if settings.get("hdr"):
        settings["lora_filename"] = _HDR_LORA_PATH
        settings["lora_scene_emb_filename"] = _HDR_SCENE_EMB_PATH
        settings["lora_weight"] = settings.get("lora_weight", 1.0)
        logging.info(
            "HDR IC-LoRA enabled: %s (weight=%.2f)",
            _HDR_LORA_PATH,
            settings["lora_weight"],
        )

    with _gen_lock:
        job = _session.submit_task(settings)
        result = job.result()

    if not result or not result.success:
        errors = [
            f"{e.stage}: {e.message}" for e in (result.errors if result else [])
        ]
        _schedule_process_exit("failed generation")
        raise HTTPException(status_code=500, detail={"errors": errors})

    return [Path(str(f)).name for f in result.generated_files]


@app.get("/output/{filename}")
def download_output(filename: str, background_tasks: BackgroundTasks):
    """Serve a generated output file."""
    safe_name = Path(filename).name  # strip any path traversal
    file_path = os.path.join(WAN2GP_OUTPUT, safe_name)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    if WAN2GP_EXIT_AFTER_JOB:
        background_tasks.add_task(
            _schedule_process_exit,
            f"serving generated output {safe_name}",
        )
    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=safe_name,
        background=background_tasks,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8188)
