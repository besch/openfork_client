"""Shared LTX-2.3 Wan2GP settings."""

MODEL_TYPE_Q8 = "ltx2_22B_distilled_gguf_q8_0"
MODEL_TYPE_Q6 = "ltx2_22B_distilled_gguf_q6_k"
MODEL_TYPE_Q4 = "ltx2_22B_distilled_gguf_q4_k_m"

_RUNTIME_LIMITS = {
    "8gb": {
        "duration_default": 2.0,
        "duration_max": 2.0,
        "steps_default": 6,
        "steps_max": 8,
    },
    "12gb": {
        "duration_default": 3.0,
        "duration_max": 4.0,
        "steps_default": 8,
        "steps_max": 8,
    },
    "16gb": {
        "duration_default": 4.0,
        "duration_max": 5.0,
        "steps_default": 8,
        "steps_max": 12,
    },
    "32gb": {
        "duration_default": 7.0,
        "duration_max": 10.0,
        "steps_default": 8,
        "steps_max": 16,
    },
    "default": {
        "duration_default": 5.0,
        "duration_max": 7.0,
        "steps_default": 8,
        "steps_max": 16,
    },
}


def get_ltx23_model_type(service_type: str) -> str:
    """Return the Wan2GP preset that matches the selected LTX-2.3 tier."""
    service_type = (service_type or "").lower()
    if "8gb" in service_type:
        return MODEL_TYPE_Q4
    if "12gb" in service_type:
        return MODEL_TYPE_Q6
    return MODEL_TYPE_Q8


def get_ltx23_runtime_limits(service_type: str) -> dict:
    service_type = (service_type or "").lower()
    for tier in ("8gb", "12gb", "16gb", "32gb"):
        if tier in service_type:
            return dict(_RUNTIME_LIMITS[tier])
    return dict(_RUNTIME_LIMITS["default"])


def clamp_ltx23_duration(requested_duration, service_type: str) -> float:
    limits = get_ltx23_runtime_limits(service_type)
    try:
        duration = float(requested_duration)
    except (TypeError, ValueError):
        duration = float(limits["duration_default"])
    return max(1.0, min(duration, float(limits["duration_max"])))


def clamp_ltx23_steps(requested_steps, service_type: str) -> int:
    limits = get_ltx23_runtime_limits(service_type)
    try:
        steps = int(requested_steps)
    except (TypeError, ValueError):
        steps = int(limits["steps_default"])
    return max(1, min(steps, int(limits["steps_max"])))
