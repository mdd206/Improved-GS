"""
Regularization helpers used during loss construction.

The optimization stage computes the image loss first, then calls this module to
add method-specific penalties. MCMC uses fixed opacity and scale penalties, while
GNS adds an opacity penalty only during its regularized pruning window.
"""
from typing import Any

import torch

from scene.gaussian_model import GaussianModel as GaussianModel3DGS
from scene.methods.pruning_methods import resolve_regularized_prune_target_budget
from scene.training_context import TrainingContext
from utils.view_support_regularization import (
    compute_view_support_penalties,
    should_apply_view_support_regularization,
)


def apply_mcmc_regularization(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    loss: torch.Tensor,
) -> torch.Tensor:
    """
        Add MCMC's fixed opacity and scale penalties to the current loss.
    """
    opt = context.opt
    return (
        loss
        + float(opt.mcmc_opacity_reg) * torch.abs(gaussians.get_opacity).mean()
        + float(opt.mcmc_scale_reg) * torch.abs(gaussians.get_scaling).mean()
    )


def compute_gns_pre_l1_regularization(gaussians: GaussianModel3DGS) -> torch.Tensor:
    """
        Compute GNS pre-activation opacity regularization.

        GNS regularizes raw opacity logits instead of activated opacity values,
        which makes the pruning pressure act before the sigmoid.
    """
    if gaussians._opacity.numel() == 0:
        return gaussians._opacity.sum() * 0.0
    return torch.mean(gaussians._opacity.flatten())


def apply_gns_opacity_regularization(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    iteration: int,
    loss: torch.Tensor,
) -> torch.Tensor:
    """
        Add GNS opacity regularization during the pruning window.

        The strength is adjusted every few iterations by comparing the current
        budget boundary opacity with a simple target schedule.
    """
    opt = context.opt
    state = context.runtime_state
    if bool(state.get("gns_finished", False)):
        return loss

    reg_start = int(opt.reg_prune_from_iter)
    reg_end = int(opt.reg_prune_until_iter)
    if not (reg_start <= int(iteration) < reg_end):
        return loss

    opacity_flat = gaussians.get_opacity.flatten()
    final_budget = resolve_regularized_prune_target_budget(gaussians, opt, state)
    if final_budget is not None and opacity_flat.numel() > final_budget:
        kth_index = opacity_flat.shape[0] - int(final_budget) + 1
        if state["opacity_min"] is None:
            state["opacity_min"] = torch.kthvalue(opacity_flat, kth_index).values.item()
        elif iteration % 10 == 0:
            remaining_ratio = 1.0 - (iteration - reg_start) / max(float(reg_end - reg_start - 100), 1.0)
            opacity_goal = max(remaining_ratio * float(state["opacity_min"]), 0.0)
            current_value = torch.kthvalue(opacity_flat, kth_index).values.item()
            if current_value < opacity_goal * 0.9:
                opt.gns_opacity_reg *= 0.95
            elif current_value > opacity_goal * 1.1:
                opt.gns_opacity_reg *= 1.05
    return loss + float(opt.gns_opacity_reg) * compute_gns_pre_l1_regularization(gaussians)


def apply_view_support_regularization(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    iteration: int,
    loss: torch.Tensor,
) -> torch.Tensor:
    """Giam opacity va scale lon cua Gaussian co it view support."""
    if not bool(context.method_config.get("use_view_support_regularization", False)):
        return loss

    opt = context.opt
    densify_from = int(opt.densify_from_iter)
    densify_until = int(opt.densify_until_iter)
    if not should_apply_view_support_regularization(
        iteration,
        densify_from,
        densify_until,
        int(opt.densification_interval),
    ):
        return loss
    if gaussians.denom.numel() == 0:
        return loss
    if float(gaussians.denom.detach().max().item()) <= 0.0:
        return loss

    scene = context.scene
    if scene is None:
        raise ValueError("Training context is missing scene for view-support regularization.")
    opacity_penalty, scale_penalty, support_weights = compute_view_support_penalties(
        gaussians.get_opacity,
        gaussians.get_scaling,
        gaussians.denom,
        float(opt.view_support_min_views),
        float(opt.view_support_min_ratio),
        float(scene.cameras_extent),
        float(opt.view_support_max_scale_ratio),
    )
    if int(iteration) % 1_000 == 0:
        unsupported_ratio = float((support_weights > 0).to(torch.float32).mean().item())
        print(
            "View-support regularization: iteration={} unsupported={:.2%} opacity={:.6f} scale={:.6f}".format(
                iteration,
                unsupported_ratio,
                float(opacity_penalty.detach().item()),
                float(scale_penalty.detach().item()),
            )
        )
    return (
        loss
        + float(opt.view_support_opacity_reg) * opacity_penalty
        + float(opt.view_support_scale_reg) * scale_penalty
    )


def apply_regularization_method(
    context: TrainingContext,
    gaussians: GaussianModel3DGS,
    iteration: int,
    loss: torch.Tensor,
) -> torch.Tensor:
    """
        Select the regularization rule for the active training method.
    """
    method = str(context.method_config["training_method"]).lower()
    if method == "mcmc":
        return apply_mcmc_regularization(context, gaussians, loss)
    if method == "gns":
        return apply_gns_opacity_regularization(context, gaussians, iteration, loss)
    if method == "improvedgs":
        return apply_view_support_regularization(context, gaussians, iteration, loss)
    return loss
