"""
Disk Pressure Tier Detection

Translates current free disk space into a categorical tier (healthy, pressure,
critical) and computes effective per-policy caps and idle timeouts based on
that tier. The DockerDownloadManager consults these helpers to evict images
even when no new pull is in flight, and to override `mine` policy's normally
uncapped behavior when free space drops below the configured thresholds.
"""

import logging
from typing import Optional

from config import (
    DISK_PRESSURE_HEALTHY_GB,
    DISK_PRESSURE_CRITICAL_GB,
    MINE_POLICY_PRESSURE_CAP,
    POLICY_MAX_CACHED_IMAGES,
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
    POLICY_MAX_CACHED_IMAGES and POLICY_IDLE_TIMEOUT_MINUTES.
    """
    if monetize_mode:
        return "monetize"
    return {
        "all": "all",
        "trusted_projects": "project",
        "trusted_users": "users",
        "none": "mine",
    }.get(community_mode or "none", "mine")


def get_effective_cap(policy: str, tier: str) -> Optional[int]:
    """Return the maximum number of cached images allowed under the given tier.

    Returns None for "mine" + healthy (preserves the existing hands-off contract
    for the user's own private workloads on a roomy disk). Every other combination
    yields a finite cap so eviction can make forward progress.
    """
    base_cap = POLICY_MAX_CACHED_IMAGES.get(policy)

    if tier == HEALTHY:
        return base_cap  # may be None for mine

    # Pressure / Critical: even mine gets a cap
    if base_cap is None:
        base_cap = MINE_POLICY_PRESSURE_CAP

    if tier == PRESSURE:
        effective = max(2, int(base_cap * 0.6))
    else:  # CRITICAL
        effective = max(1, int(base_cap * 0.6 * 0.6))

    return effective


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
