"""Shared LTX-2.3 Wan2GP settings."""

from typing import Optional

MODEL_TYPE_Q8 = "ltx2_22B_distilled_gguf_q8_0"
MODEL_TYPE_Q6 = "ltx2_22B_distilled_gguf_q6_k"
MODEL_TYPE_Q4 = "ltx2_22B_distilled_gguf_q4_k_m"
MODEL_TYPE_DISTILLED_11 = "ltx2_22B_distilled_1_1"

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
    "24gb": {
        "duration_default": 4.0,
        "duration_max": 7.0,
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
    if "24gb" in service_type or "32gb" in service_type:
        return MODEL_TYPE_DISTILLED_11
    return MODEL_TYPE_Q8


def get_ltx23_runtime_limits(service_type: str) -> dict:
    service_type = (service_type or "").lower()
    for tier in ("8gb", "12gb", "16gb", "24gb", "32gb"):
        if tier in service_type:
            return dict(_RUNTIME_LIMITS[tier])
    return dict(_RUNTIME_LIMITS["default"])


def should_use_ltx23_hdr(inputs: dict, service_type: str) -> bool:
    """Enable the baked HDR IC-LoRA on HDR-capable LTX tiers unless disabled."""
    if "hdr" in inputs:
        value = inputs.get("hdr")
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off"}
        return bool(value)
    service_type = (service_type or "").lower()
    return any(tier in service_type for tier in ("12gb", "16gb", "24gb", "32gb"))


def _clean_prompt_fragment(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def ltx23_audio_prompt(inputs: dict) -> Optional[str]:
    """Return explicit audio direction to append to the LTX prompt."""
    for key in ("silent", "no_audio", "disable_audio"):
        value = inputs.get(key)
        if isinstance(value, str):
            value = value.strip().lower() in {"1", "true", "yes", "on"}
        if value:
            return "Silent video, no audio track."

    explicit_audio = _clean_prompt_fragment(inputs.get("audio_prompt"))
    if explicit_audio:
        return explicit_audio

    parts = []
    sound_fx = _clean_prompt_fragment(inputs.get("sound_fx_prompt"))
    if sound_fx:
        parts.append(f"Sound design: {sound_fx}")

    music = _clean_prompt_fragment(inputs.get("music_prompt"))
    if music:
        parts.append(f"Music: {music}")

    if parts:
        return " ".join(parts)
    return None


def build_ltx23_prompt(positive_prompt: str, inputs: dict) -> tuple[str, Optional[str]]:
    """Append explicit audio direction when provided."""
    prompt = (positive_prompt or "").strip()
    audio_prompt = ltx23_audio_prompt(inputs)
    if not audio_prompt:
        return prompt, None

    lowered_audio = audio_prompt.lower()
    labelled_audio = (
        audio_prompt
        if lowered_audio.startswith(("audio:", "soundtrack:", "sound design:", "music:"))
        else f"Audio: {audio_prompt}"
    )

    if not prompt:
        return labelled_audio, audio_prompt
    if lowered_audio.startswith("silent video"):
        return f"{labelled_audio}\n\n{prompt}", audio_prompt
    return f"{prompt}\n\n{labelled_audio}", audio_prompt


def ltx23_lora_weight(inputs: dict) -> float:
    try:
        return float(inputs.get("lora_weight", 0.9))
    except (TypeError, ValueError):
        return 0.9


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
