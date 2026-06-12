"""
Shared WAN 2.2 processor helpers.
"""

import logging


WAN22_DISTILLED_CFG = 1
WAN22_DISTILLED_STEPS = 6


def normalize_classic_wan22_sampling(inputs: dict, job_id: str) -> tuple[int, int]:
    """Classic ComfyUI WAN22 uses the distilled LightX2V preset."""
    requested_cfg = inputs.get("cfg_scale")
    requested_steps = inputs.get("steps")

    if requested_cfg not in (None, WAN22_DISTILLED_CFG):
        logging.warning(
            "WAN22 job %s requested cfg_scale=%s; forcing distilled cfg_scale=%s.",
            job_id,
            requested_cfg,
            WAN22_DISTILLED_CFG,
        )
    if requested_steps not in (None, WAN22_DISTILLED_STEPS):
        logging.warning(
            "WAN22 job %s requested steps=%s; forcing distilled steps=%s.",
            job_id,
            requested_steps,
            WAN22_DISTILLED_STEPS,
        )

    return WAN22_DISTILLED_CFG, WAN22_DISTILLED_STEPS
