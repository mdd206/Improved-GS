"""Periodic soft density--scale updates for the ImprovedGS experiment."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from utils.density_scale import compute_soft_density_scale_delta, should_run_soft_density_scale

if TYPE_CHECKING:
    from scene.gaussian_model import GaussianModel as GaussianModel3DGS
    from scene.training_context import TrainingContext


DistanceFunction = Callable[[torch.Tensor], torch.Tensor]


def _load_simple_knn_distance() -> DistanceFunction:
    """Load the existing CUDA neighbour-distance kernel only when enabled."""
    try:
        from simple_knn._C import distCUDA2
    except ImportError as error:
        raise RuntimeError(
            "soft density-scale requires the existing simple-knn CUDA extension; "
            "install submodules/simple-knn before training."
        ) from error
    return distCUDA2


def _sample_gaussian_indices(
    point_count: int,
    max_points: int,
    device: torch.device,
    sampling_seed: int,
) -> torch.Tensor:
    """Sample without replacement using an RNG isolated from training."""
    if max_points <= 0 or point_count <= max_points:
        return torch.arange(point_count, dtype=torch.long, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(sampling_seed))
    return torch.randperm(point_count, generator=generator, device=device)[:max_points]


@torch.no_grad()
def run_soft_density_scale_update(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    iteration: int,
    distance_function: DistanceFunction | None = None,
) -> dict[str, float | int] | None:
    """Move sampled Gaussian scales softly toward their local-spacing target.

    The update runs after densification and pruning, so the neighbour estimate
    always sees the final structure for this iteration.  It changes only the
    absolute scale: every sampled Gaussian receives the same log-space delta on
    all three axes, leaving LAS anisotropy untouched.
    """
    opt = context.opt
    if not bool(context.method_config.get("soft_density_scale", False)):
        return None
    if not should_run_soft_density_scale(opt, iteration):
        return None

    all_xyz = gaussians.get_xyz.detach()
    point_count = int(all_xyz.shape[0])
    if point_count < 4:
        return None

    max_points = int(getattr(opt, "soft_ds_max_points", 500_000))
    sampling_seed = int(getattr(opt, "soft_ds_seed", 34_007)) + int(iteration)
    finite_xyz_mask = torch.isfinite(all_xyz).all(dim=1)
    if bool(torch.all(finite_xyz_mask)):
        sampled_indices = _sample_gaussian_indices(
            point_count,
            max_points,
            all_xyz.device,
            sampling_seed,
        )
    else:
        eligible_indices = torch.nonzero(finite_xyz_mask, as_tuple=False).squeeze(1)
        eligible_count = int(eligible_indices.numel())
        if eligible_count < 4:
            return None
        relative_indices = _sample_gaussian_indices(
            eligible_count,
            max_points,
            all_xyz.device,
            sampling_seed,
        )
        sampled_indices = eligible_indices.index_select(0, relative_indices)
    sampled_xyz = all_xyz.index_select(0, sampled_indices).contiguous()
    distance_function = distance_function or _load_simple_knn_distance()
    squared_spacing = distance_function(sampled_xyz)
    if squared_spacing.ndim != 1 or squared_spacing.shape[0] != sampled_indices.shape[0]:
        raise RuntimeError("simple-knn returned an unexpected density-distance shape.")

    min_spacing = float(getattr(opt, "soft_ds_min_spacing", 1e-7))
    local_spacing = torch.sqrt(squared_spacing.detach().clamp_min(min_spacing * min_spacing))
    sampled_log_scales = gaussians._scaling.detach().index_select(0, sampled_indices)
    log_absolute_scale = sampled_log_scales.mean(dim=1)
    delta, statistics = compute_soft_density_scale_delta(
        log_absolute_scale,
        local_spacing,
        strength=float(getattr(opt, "soft_ds_strength", 0.1)),
        max_scale_ratio=float(getattr(opt, "soft_ds_max_scale_ratio", 2.0)),
        target_multiplier=float(getattr(opt, "soft_ds_target_multiplier", 1.0)),
        min_spacing=min_spacing,
    )

    # index_add_ updates the optimizer-owned Parameter in place.  Repeating the
    # scalar delta on all axes preserves sx:sy:sz exactly for every Gaussian.
    repeated_delta = delta[:, None].expand(-1, 3).contiguous()
    gaussians._scaling.index_add_(0, sampled_indices, repeated_delta)
    result: dict[str, float | int] = {
        "iteration": int(iteration),
        "sampled_gaussians": int(sampled_indices.numel()),
        "total_gaussians": point_count,
        **statistics,
    }
    context.runtime_state["soft_density_scale_last_update"] = result
    print(
        "Soft density-scale: iteration={iteration}, adjusted={sampled_gaussians}/{total_gaussians}, "
        "median_ratio={median_scale_spacing_ratio:.6g}, mean_abs_log_delta={mean_abs_log_delta:.6g}, "
        "clipped={clipped_fraction:.2%}".format(**result)
    )
    return result
