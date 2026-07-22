"""C2F-aligned frequency regularization used by the FreGS-lite experiment.

The implementation keeps the useful signal from FreGS while limiting Kaggle
cost: it transforms luminance with ``rfft2`` instead of applying a full complex
FFT to every RGB channel.  Log-amplitude and wrapped phase discrepancies are
introduced progressively from low to high frequencies and are disabled after
the densification window.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import torch


def resolve_fregs_window(opt: Any) -> tuple[int, int]:
    """Resolve the frequency-loss window from explicit or densification values."""
    configured_start = int(getattr(opt, "fregs_start_iter", -1))
    start_iter = configured_start if configured_start >= 0 else int(getattr(opt, "densify_from_iter", 500))
    configured_end = int(getattr(opt, "fregs_until_iter", -1))
    end_iter = configured_end if configured_end >= 0 else int(getattr(opt, "densify_until_iter", 15_000))
    return start_iter, end_iter


def validate_fregs_lite_options(opt: Any) -> None:
    """Validate the schedule and weights before allocating training tensors."""
    if not bool(getattr(opt, "fregs_lite", False)):
        return

    start_iter, end_iter = resolve_fregs_window(opt)
    interval = int(getattr(opt, "fregs_interval", 1))
    weight = float(getattr(opt, "fregs_weight", 0.01))
    phase_weight = float(getattr(opt, "fregs_phase_weight", 0.1))
    detail_weight = float(getattr(opt, "fregs_detail_weight", 1.0))
    low_radius = float(getattr(opt, "fregs_low_radius", 0.15))
    middle_radius = float(getattr(opt, "fregs_middle_radius", 0.5))
    epsilon = float(getattr(opt, "fregs_epsilon", 1e-6))

    if start_iter < 0:
        raise ValueError("FreGS-lite start iteration must be >= 0.")
    if end_iter < start_iter:
        raise ValueError("FreGS-lite end iteration must be >= its start iteration.")
    if interval < 1:
        raise ValueError("fregs_interval must be >= 1.")
    if weight <= 0.0:
        raise ValueError("fregs_weight must be > 0.")
    if phase_weight < 0.0:
        raise ValueError("fregs_phase_weight must be >= 0.")
    if detail_weight < 0.0:
        raise ValueError("fregs_detail_weight must be >= 0.")
    if not (0.0 < low_radius < middle_radius < 1.0):
        raise ValueError("FreGS-lite radii must satisfy 0 < low < middle < 1.")
    if epsilon <= 0.0:
        raise ValueError("fregs_epsilon must be > 0.")


def should_run_fregs_lite(opt: Any, iteration: int) -> bool:
    """Return whether this iteration receives frequency-domain supervision."""
    if not bool(getattr(opt, "fregs_lite", False)):
        return False
    start_iter, end_iter = resolve_fregs_window(opt)
    interval = int(getattr(opt, "fregs_interval", 1))
    return start_iter <= int(iteration) <= end_iter and (int(iteration) - start_iter) % interval == 0


def _linear_progress(iteration: int, start: int, end: int) -> float:
    """Return a clamped interpolation factor and handle zero-length stages."""
    if end <= start:
        return 1.0 if iteration >= end else 0.0
    return min(max((int(iteration) - int(start)) / float(end - start), 0.0), 1.0)


def resolve_fregs_max_radius(opt: Any, iteration: int) -> float:
    """Expand the active band in lockstep with the C2F resolution schedule."""
    start_iter, end_iter = resolve_fregs_window(opt)
    low_radius = float(getattr(opt, "fregs_low_radius", 0.15))
    middle_radius = float(getattr(opt, "fregs_middle_radius", 0.5))

    if not bool(getattr(opt, "coarse_to_fine", False)):
        progress = _linear_progress(iteration, start_iter, end_iter)
        return low_radius + progress * (1.0 - low_radius)

    middle_iter = int(getattr(opt, "coarse_to_fine_middle_iter", 2_000))
    full_iter = int(getattr(opt, "coarse_to_fine_full_iter", 5_000))
    if int(iteration) < middle_iter:
        return low_radius
    if int(iteration) < full_iter:
        progress = _linear_progress(iteration, middle_iter, full_iter)
        return low_radius + progress * (middle_radius - low_radius)
    progress = _linear_progress(iteration, full_iter, end_iter)
    return middle_radius + progress * (1.0 - middle_radius)


def _device_key(device: torch.device) -> tuple[str, int | None]:
    """Convert a torch device into hashable cache components."""
    return device.type, device.index


@lru_cache(maxsize=24)
def _cached_frequency_geometry(
    height: int,
    width: int,
    device_type: str,
    device_index: int | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache radial coordinates and a normalized 2-D Hann window per resolution."""
    device = torch.device(device_type, device_index) if device_index is not None else torch.device(device_type)
    vertical_frequency = torch.fft.fftfreq(height, device=device, dtype=dtype)
    horizontal_frequency = torch.fft.rfftfreq(width, device=device, dtype=dtype)
    radius = torch.sqrt(vertical_frequency[:, None].square() + horizontal_frequency[None, :].square())
    radius = radius / math.sqrt(0.5**2 + 0.5**2)

    vertical_window = torch.hann_window(height, periodic=False, device=device, dtype=dtype)
    horizontal_window = torch.hann_window(width, periodic=False, device=device, dtype=dtype)
    window = vertical_window[:, None] * horizontal_window[None, :]
    return radius, window.unsqueeze(0)


def _to_luminance(image: torch.Tensor) -> torch.Tensor:
    """Convert a CHW RGB tensor to one luminance channel."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("FreGS-lite expects an RGB tensor with shape [3, H, W].")
    coefficients = image.new_tensor((0.299, 0.587, 0.114)).view(3, 1, 1)
    return torch.sum(image * coefficients, dim=0, keepdim=True)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Compute a stable mean over a frequency mask and optional confidence weights."""
    multiplier = mask.to(values.dtype)
    if weights is not None:
        multiplier = multiplier * weights
    return torch.sum(values * multiplier) / torch.sum(multiplier).clamp_min(1.0)


def _combine_frequency_bands(
    values: torch.Tensor,
    low_mask: torch.Tensor,
    detail_mask: torch.Tensor | None,
    detail_weight: float,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keep the loss scale stable while adding the progressive detail band."""
    low_value = _masked_mean(values, low_mask, confidence)
    if detail_mask is None or detail_weight <= 0.0:
        return low_value
    detail_value = _masked_mean(values, detail_mask, confidence)
    return (low_value + float(detail_weight) * detail_value) / (1.0 + float(detail_weight))


def compute_fregs_lite_loss(
    rendered_image: torch.Tensor,
    target_image: torch.Tensor,
    valid_mask: torch.Tensor | None,
    opt: Any,
    iteration: int,
) -> dict[str, torch.Tensor | float]:
    """Compute progressive log-amplitude and phase losses for one rendered view."""
    zero = rendered_image.sum() * 0.0
    if not should_run_fregs_lite(opt, iteration):
        return {"loss": zero, "amplitude_loss": zero, "phase_loss": zero, "max_radius": 0.0}
    if rendered_image.shape != target_image.shape:
        raise ValueError("Rendered and target images must have identical shapes for FreGS-lite.")

    rendered_luminance = _to_luminance(rendered_image.float())
    target_luminance = _to_luminance(target_image.float())
    height, width = rendered_luminance.shape[-2:]
    device_type, device_index = _device_key(rendered_luminance.device)
    radius, hann_window = _cached_frequency_geometry(
        height,
        width,
        device_type,
        device_index,
        rendered_luminance.dtype,
    )

    if valid_mask is None:
        spatial_weight = torch.ones_like(rendered_luminance)
    else:
        if valid_mask.ndim == 2:
            valid_mask = valid_mask.unsqueeze(0)
        if valid_mask.shape != rendered_luminance.shape:
            raise ValueError("FreGS-lite valid_mask must have shape [1, H, W].")
        spatial_weight = valid_mask.to(device=rendered_luminance.device, dtype=rendered_luminance.dtype).clamp(0.0, 1.0)

    if bool(getattr(opt, "fregs_hann_window", True)):
        spatial_weight = spatial_weight * hann_window
    epsilon = float(getattr(opt, "fregs_epsilon", 1e-6))
    rms_weight = torch.sqrt(torch.mean(spatial_weight.square())).clamp_min(epsilon)
    rendered_luminance = rendered_luminance * spatial_weight / rms_weight
    target_luminance = target_luminance * spatial_weight / rms_weight

    rendered_spectrum = torch.fft.rfft2(rendered_luminance, norm="ortho")
    target_spectrum = torch.fft.rfft2(target_luminance, norm="ortho")
    rendered_amplitude = torch.abs(rendered_spectrum)
    target_amplitude = torch.abs(target_spectrum)
    amplitude_distance = torch.abs(torch.log1p(rendered_amplitude) - torch.log1p(target_amplitude))

    cross_spectrum = rendered_spectrum * torch.conj(target_spectrum)
    amplitude_product = rendered_amplitude * target_amplitude
    phase_cosine = torch.where(
        amplitude_product > epsilon,
        cross_spectrum.real / amplitude_product.clamp_min(epsilon),
        torch.ones_like(amplitude_product),
    )
    phase_distance = 1.0 - phase_cosine.clamp(-1.0, 1.0)
    target_amplitude_mean = target_amplitude.mean().detach()
    phase_confidence = target_amplitude / (target_amplitude + target_amplitude_mean + epsilon)

    low_radius = float(getattr(opt, "fregs_low_radius", 0.15))
    max_radius = resolve_fregs_max_radius(opt, iteration)
    low_mask = radius <= low_radius
    detail_mask = None
    if max_radius > low_radius:
        detail_mask = (radius > low_radius) & (radius <= max_radius)
    detail_progress = min(max((max_radius - low_radius) / (1.0 - low_radius), 0.0), 1.0)
    detail_weight = float(getattr(opt, "fregs_detail_weight", 1.0)) * detail_progress
    amplitude_loss = _combine_frequency_bands(
        amplitude_distance,
        low_mask,
        detail_mask,
        detail_weight,
    )
    phase_loss = _combine_frequency_bands(
        phase_distance,
        low_mask,
        detail_mask,
        detail_weight,
        phase_confidence,
    )

    total_loss = float(getattr(opt, "fregs_weight", 0.01)) * (
        amplitude_loss + float(getattr(opt, "fregs_phase_weight", 0.1)) * phase_loss
    )
    return {
        "loss": total_loss,
        "amplitude_loss": amplitude_loss,
        "phase_loss": phase_loss,
        "max_radius": max_radius,
    }
