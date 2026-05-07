"""
Disk Pressure Tier Detection

Translates current free disk space into a categorical tier (healthy, pressure,
critical) and computes effective idle timeouts based on that tier. The
DockerDownloadManager consults these helpers to evict images even when no new
pull is in flight.
"""

import logging
from typing import Optional

from config import (
    DISK_PRESSURE_HEALTHY_GB,
    DISK_PRESSURE_CRITICAL_GB,
    POLICY_IDLE_TIMEOUT_MINUTES,
)

try:
    from services.disk_space_utils import (
        get_available_disk_space,
        get_docker_storage_path,
    )
except ImportError:  # support relative-import callers
    from .disk_space_utils import (
        get_available_disk_space,
        get_docker_storage_path,
    )


HEALTHY = "healthy"
PRESSURE = "pressure"
CRITICAL = "critical"


def get_free_disk_gb(path: Optional[str] = None) -> float:
    """Return free space at the Docker storage path in GB."""
    target = path or get_docker_storage_path()
    return get_available_disk_space(target) / (1024 ** 3)


def get_disk_pressure_tier(path: Optional[str] = None) -> str:
    """Classify current disk state. Returns one of HEALTHY, PRESSURE, CRITICAL."""
    free_gb = get_free_disk_gb(path)
    if free_gb <= DISK_PRESSURE_CRITICAL_GB:
        return CRITICAL
    if free_gb <= DISK_PRESSURE_HEALTHY_GB:
        return PRESSURE
    return HEALTHY


def policy_key_for(community_mode: str, monetize_mode: bool) -> str:
    """Map (community_mode, monetize_mode) to the canonical policy key used by
    POLICY_IDLE_TIMEOUT_MINUTES.
    """
    if monetize_mode:
        return "monetize"
    return {
        "all": "all",
        "trusted_projects": "project",
        "trusted_users": "users",
        "none": "mine",
    }.get(community_mode or "none", "mine")


def get_effective_idle_minutes(policy: str, tier: str) -> Optional[int]:
    """Return the idle timeout (in minutes) the Electron notifier should report
    at this tier. None means idle-eviction is suppressed.
    """
    base = POLICY_IDLE_TIMEOUT_MINUTES.get(policy)

    if tier == HEALTHY:
        return base  # may be None (mine)

    # Under pressure even mine gets an idle timeout. Use a generous default for mine.
    if base is None:
        base = 240  # 4 hours — gives mine breathing room before considering itself idle

    if tier == PRESSURE:
        return max(15, int(base * 0.5))
    return max(10, int(base * 0.25))  # CRITICAL


def describe_tier_for_log(tier: str, free_gb: float) -> str:
    return f"tier={tier} free={free_gb:.1f}GB healthy>{DISK_PRESSURE_HEALTHY_GB} critical<={DISK_PRESSURE_CRITICAL_GB}"


def log_tier_transition(prev_tier: Optional[str], new_tier: str, free_gb: float) -> None:
    """Helper for the download manager to log tier changes only on transition."""
    if prev_tier == new_tier:
        return
    if prev_tier is None:
        logging.info(f"Disk pressure: {describe_tier_for_log(new_tier, free_gb)}")
    else:
        logging.info(
            f"Disk pressure changed: {prev_tier} → {new_tier} ({describe_tier_for_log(new_tier, free_gb)})"
        )
