import logging
import os


def build_wan2gp_environment(service_type: str) -> dict:
    """Return environment overrides for Wan2GP container startup."""
    env = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "CUDA_MODULE_LOADING": "LAZY",
        "PYTORCH_CUDA_ALLOC_CONF": (
            "expandable_segments:True,"
            "max_split_size_mb:128,"
            "garbage_collection_threshold:0.8"
        ),
        "MALLOC_ARENA_MAX": "2",
    }

    lowered_service_type = (service_type or "").lower()
    if "wan22-wan2gp" in lowered_service_type:
        if "8gb" in lowered_service_type:
            env["WAN2GP_CLI_ARGS"] = (
                "--profile 5 --attention sdpa "
                "--preload 0 "
                "--perc-reserved-mem-max 0.20 "
                "--vram-safety-coefficient 0.35"
            )
        elif "10gb" in lowered_service_type:
            env["WAN2GP_CLI_ARGS"] = (
                "--profile 4.5 --attention sdpa "
                "--perc-reserved-mem-max 0.35 "
                "--vram-safety-coefficient 0.60"
            )
        elif "12gb" in lowered_service_type:
            env["WAN2GP_CLI_ARGS"] = (
                "--profile 4.5 --attention sdpa "
                "--perc-reserved-mem-max 0.35 "
                "--vram-safety-coefficient 0.65"
            )
        elif "24gb" in lowered_service_type:
            env["WAN2GP_CLI_ARGS"] = (
                "--profile 4 --attention sdpa "
                "--perc-reserved-mem-max 0.55 "
                "--vram-safety-coefficient 0.80"
            )
        else:
            env["WAN2GP_CLI_ARGS"] = (
                "--profile 4.5 --attention sdpa "
                "--perc-reserved-mem-max 0.45 "
                "--vram-safety-coefficient 0.70"
            )
    elif "davinci" in lowered_service_type:
        if "16gb" in lowered_service_type:
            env["WAN2GP_CLI_ARGS"] = (
                "--profile 4.5 --attention sdpa "
                "--perc-reserved-mem-max 0.45 "
                "--vram-safety-coefficient 0.7"
            )
        elif "32gb" in lowered_service_type:
            env["WAN2GP_CLI_ARGS"] = (
                "--profile 4 --attention sdpa "
                "--perc-reserved-mem-max 0.55 "
                "--vram-safety-coefficient 0.80"
            )
        else:
            env["WAN2GP_CLI_ARGS"] = (
                "--profile 4.5 --attention sdpa "
                "--perc-reserved-mem-max 0.45 "
                "--vram-safety-coefficient 0.70"
            )
    elif "ltx23" in lowered_service_type:
        if "8gb" in lowered_service_type:
            env["WAN2GP_CLI_ARGS"] = (
                "--profile 4.5 --attention sdpa "
                "--perc-reserved-mem-max 0.45 "
                "--vram-safety-coefficient 0.70"
            )
        elif "12gb" in lowered_service_type:
            env["WAN2GP_CLI_ARGS"] = (
                "--profile 4.5 --attention sdpa "
                "--perc-reserved-mem-max 0.55 "
                "--vram-safety-coefficient 0.70"
            )
        elif "32gb" in lowered_service_type:
            env["WAN2GP_CLI_ARGS"] = (
                "--profile 4 --attention sdpa "
                "--perc-reserved-mem-max 0.55 "
                "--vram-safety-coefficient 0.80"
            )
        else:
            env["WAN2GP_CLI_ARGS"] = (
                "--profile 4.5 --attention sdpa "
                "--perc-reserved-mem-max 0.45 "
                "--vram-safety-coefficient 0.70"
            )
    elif "8gb" in lowered_service_type:
        env["WAN2GP_CLI_ARGS"] = (
            "--profile 4.5 --attention sdpa "
            "--perc-reserved-mem-max 0.45"
        )
    else:
        env["WAN2GP_CLI_ARGS"] = "--profile 4 --attention sdpa"

    return env


def get_wan2gp_pre_start_copies(root_dir: str) -> list:
    server_src = os.path.join(root_dir, "comfyui-storage", "wan2gp_server.py")
    if os.path.isfile(server_src):
        return [(server_src, "/opt/wan2gp/wan2gp_server.py")]

    logging.warning(
        "wan2gp_server.py not found at %s; using baked-in version.",
        server_src,
    )
    return []
