"""Shared LTX-2.3 Wan2GP settings."""

MODEL_TYPE_Q8 = "ltx2_22B_distilled_gguf_q8_0"
MODEL_TYPE_Q4 = "ltx2_22B_distilled_gguf_q4_k_m"


def get_ltx23_model_type(service_type: str) -> str:
    """Return the Wan2GP preset that matches the selected LTX-2.3 tier."""
    if "8gb" in (service_type or "").lower():
        return MODEL_TYPE_Q4
    return MODEL_TYPE_Q8
