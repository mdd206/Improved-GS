"""Ham tinh regularization dua tren so camera ho tro tung Gaussian."""
from __future__ import annotations

import torch


def should_apply_view_support_regularization(
    iteration: int,
    densify_from: int,
    densify_until: int,
    densification_interval: int,
) -> bool:
    """Chi bat regularization o cuoi moi cua so visibility hop le."""
    interval = max(int(densification_interval), 1)
    return int(densify_from) < int(iteration) < int(densify_until) and int(iteration) % interval == 0


def compute_view_support_weights(
    support_counts: torch.Tensor,
    min_views: float,
    min_ratio: float,
) -> torch.Tensor:
    """Tra ve trong so lon cho Gaussian duoc rat it camera nhin thay."""
    flat_support = support_counts.detach().reshape(-1).to(torch.float32)
    if flat_support.numel() == 0:
        return flat_support
    reference_support = flat_support.max()
    support_threshold = torch.maximum(
        reference_support * float(min_ratio),
        reference_support.new_tensor(float(min_views)),
    )
    return (1.0 - flat_support / support_threshold.clamp_min(1e-6)).clamp(0.0, 1.0)


def compute_view_support_penalties(
    opacity: torch.Tensor,
    scaling: torch.Tensor,
    support_counts: torch.Tensor,
    min_views: float,
    min_ratio: float,
    scene_extent: float,
    max_scale_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tinh opacity penalty va scale penalty cho Gaussian thieu support."""
    flat_opacity = opacity.reshape(-1)
    if flat_opacity.numel() == 0:
        zero = opacity.sum() * 0.0
        return zero, zero, support_counts.detach().reshape(-1).to(torch.float32)
    if scaling.ndim != 2 or scaling.shape[0] != flat_opacity.shape[0]:
        raise ValueError("scaling and opacity must contain the same number of Gaussians.")

    support_weights = compute_view_support_weights(support_counts, min_views, min_ratio)
    if support_weights.shape[0] != flat_opacity.shape[0]:
        raise ValueError("support_counts and opacity must contain the same number of Gaussians.")

    weight_sum = support_weights.sum().clamp_min(1.0)
    opacity_penalty = torch.sum(support_weights * flat_opacity) / weight_sum

    safe_extent = max(float(scene_extent), 1e-8)
    relative_max_scale = scaling.max(dim=1).values / safe_extent
    scale_excess = torch.relu(relative_max_scale - float(max_scale_ratio))
    scale_penalty = torch.sum(support_weights * scale_excess) / weight_sum
    return opacity_penalty, scale_penalty, support_weights
