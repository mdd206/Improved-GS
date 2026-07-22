"""Soft density--scale scheduling and correction helpers.

The original FDS-GS method hard-resets an absolute scale component from a
K-nearest-neighbour density estimate.  ImprovedGS already gives LAS explicit
control over each child's anisotropic shape, so this experiment instead applies
a bounded scalar correction in log-scale space.  Adding the same scalar to all
three axes preserves the axis ratios produced by LAS.
"""
from __future__ import annotations

import math
from typing import Any

import torch


def resolve_soft_density_scale_window(opt: Any) -> tuple[int, int]:
    """Resolve the active iteration window, aligning it with C2F by default."""
    configured_start = int(getattr(opt, "soft_ds_start_iter", -1))
    if configured_start >= 0:
        start_iter = configured_start
    elif bool(getattr(opt, "coarse_to_fine", False)):
        start_iter = int(getattr(opt, "coarse_to_fine_full_iter", 5_000))
    else:
        start_iter = int(getattr(opt, "densify_from_iter", 500))

    configured_end = int(getattr(opt, "soft_ds_until_iter", -1))
    end_iter = configured_end if configured_end >= 0 else int(getattr(opt, "densify_until_iter", 15_000))
    return start_iter, end_iter


def validate_soft_density_scale_options(opt: Any) -> None:
    """Reject schedules or correction strengths that would be unsafe or inert."""
    if not bool(getattr(opt, "soft_density_scale", False)):
        return

    start_iter, end_iter = resolve_soft_density_scale_window(opt)
    interval = int(getattr(opt, "soft_ds_interval", 500))
    strength = float(getattr(opt, "soft_ds_strength", 0.1))
    max_scale_ratio = float(getattr(opt, "soft_ds_max_scale_ratio", 2.0))
    target_multiplier = float(getattr(opt, "soft_ds_target_multiplier", 1.0))
    max_points = int(getattr(opt, "soft_ds_max_points", 500_000))
    min_spacing = float(getattr(opt, "soft_ds_min_spacing", 1e-7))
    sampling_seed = int(getattr(opt, "soft_ds_seed", 34_007))

    if start_iter < 0:
        raise ValueError("soft density-scale start iteration must be >= 0.")
    if end_iter < start_iter:
        raise ValueError("soft density-scale end iteration must be >= its start iteration.")
    if interval < 1:
        raise ValueError("soft_ds_interval must be >= 1.")
    if not (0.0 < strength <= 1.0):
        raise ValueError("soft_ds_strength must be in (0, 1].")
    if max_scale_ratio <= 1.0:
        raise ValueError("soft_ds_max_scale_ratio must be > 1.")
    if target_multiplier <= 0.0:
        raise ValueError("soft_ds_target_multiplier must be > 0.")
    if max_points != 0 and max_points < 4:
        raise ValueError("soft_ds_max_points must be 0 (all points) or >= 4.")
    if min_spacing <= 0.0:
        raise ValueError("soft_ds_min_spacing must be > 0.")
    if sampling_seed < 0:
        raise ValueError("soft_ds_seed must be >= 0.")


def should_run_soft_density_scale(opt: Any, iteration: int) -> bool:
    """Return whether this iteration is a scheduled density--scale update."""
    if not bool(getattr(opt, "soft_density_scale", False)):
        return False
    start_iter, end_iter = resolve_soft_density_scale_window(opt)
    interval = int(getattr(opt, "soft_ds_interval", 500))
    return start_iter <= int(iteration) <= end_iter and (int(iteration) - start_iter) % interval == 0


def compute_soft_density_scale_delta(
    log_absolute_scale: torch.Tensor,
    local_spacing: torch.Tensor,
    strength: float,
    max_scale_ratio: float,
    target_multiplier: float = 1.0,
    min_spacing: float = 1e-7,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute a bounded log-scale correction from local point spacing.

    A robust scene-specific calibration is estimated from the median ratio
    between current absolute scale and local spacing.  This removes arbitrary
    scene units and makes the update constrain the *relative* density--scale
    relation instead of globally shrinking or expanding the whole model.
    """
    if log_absolute_scale.ndim != 1 or local_spacing.ndim != 1:
        raise ValueError("log_absolute_scale and local_spacing must be one-dimensional.")
    if log_absolute_scale.shape != local_spacing.shape:
        raise ValueError("log_absolute_scale and local_spacing must have the same shape.")
    if log_absolute_scale.numel() == 0:
        empty = torch.zeros_like(log_absolute_scale)
        return empty, {
            "median_scale_spacing_ratio": 1.0,
            "mean_abs_log_delta": 0.0,
            "clipped_fraction": 0.0,
        }

    safe_spacing = local_spacing.detach().clamp_min(float(min_spacing))
    current_log_scale = log_absolute_scale.detach()
    log_spacing = torch.log(safe_spacing)
    finite_mask = torch.isfinite(current_log_scale) & torch.isfinite(log_spacing)
    if not torch.any(finite_mask):
        empty = torch.zeros_like(log_absolute_scale)
        return empty, {
            "median_scale_spacing_ratio": 1.0,
            "mean_abs_log_delta": 0.0,
            "clipped_fraction": 0.0,
        }

    log_calibration = torch.median(current_log_scale[finite_mask] - log_spacing[finite_mask])
    log_target = log_spacing + log_calibration + math.log(float(target_multiplier))
    raw_correction = log_target - current_log_scale
    max_log_correction = math.log(float(max_scale_ratio))
    bounded_correction = torch.clamp(raw_correction, -max_log_correction, max_log_correction)
    bounded_correction = torch.where(finite_mask, bounded_correction, torch.zeros_like(bounded_correction))
    delta = float(strength) * bounded_correction

    finite_raw = raw_correction[finite_mask]
    clipped_fraction = torch.mean((torch.abs(finite_raw) > max_log_correction).to(torch.float32))
    statistics = {
        "median_scale_spacing_ratio": float(torch.exp(log_calibration).item()),
        "mean_abs_log_delta": float(torch.mean(torch.abs(delta[finite_mask])).item()),
        "clipped_fraction": float(clipped_fraction.item()),
    }
    return delta, statistics
