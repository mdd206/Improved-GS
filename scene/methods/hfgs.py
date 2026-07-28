"""
HF-GS edge-weighted supervision and scale-aware refinement helpers.

This module implements Sections 3.3.1 and 3.3.2 of HF-GS while leaving the
native ImprovedGS EAS/LAS/RAP/MU components unchanged.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

from third_party.pidinet import build_pidinet


SOBEL_X = torch.tensor(
    [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
    dtype=torch.float32,
).unsqueeze(0)
SOBEL_Y = torch.tensor(
    [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
    dtype=torch.float32,
).unsqueeze(0)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def normalize_prior_map(value_tensor: torch.Tensor) -> torch.Tensor:
    """Min-max normalize a prior map to [0, 1], including constant-map safety."""
    value = torch.nan_to_num(
        value_tensor.detach().float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if value.numel() == 0:
        return value
    min_value = value.amin()
    span = value.amax() - min_value
    if float(span.item()) <= 0.0:
        return torch.zeros_like(value)
    return (value - min_value) / span


def compute_sobel_prior(image: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Compute the normalized Sobel photometric-gradient prior from an RGB image."""
    rgb = image[:3].detach().float().unsqueeze(0)
    grayscale = (
        0.299 * rgb[:, 0:1]
        + 0.587 * rgb[:, 1:2]
        + 0.114 * rgb[:, 2:3]
    )
    sobel_x = SOBEL_X.to(device=grayscale.device, dtype=grayscale.dtype)
    sobel_y = SOBEL_Y.to(device=grayscale.device, dtype=grayscale.dtype)
    padded_grayscale = F.pad(grayscale, (1, 1, 1, 1), mode="replicate")
    gradient_x = F.conv2d(padded_grayscale, sobel_x)
    gradient_y = F.conv2d(padded_grayscale, sobel_y)
    magnitude = torch.sqrt(gradient_x.square() + gradient_y.square() + float(epsilon))
    return normalize_prior_map(magnitude).squeeze(0).squeeze(0)


def weighted_l1_loss(
    image: torch.Tensor,
    target: torch.Tensor,
    weight_map: torch.Tensor | None,
) -> torch.Tensor:
    """
    Compute HF-GS weighted L1 with the same channel/pixel mean as standard 3DGS.

    The paper averages ``W * |I-I_gt|``; it does not divide by ``sum(W)``.
    """
    absolute_error = torch.abs(image - target)
    if weight_map is None:
        return absolute_error.mean()
    if weight_map.ndim == 2:
        weight_map = weight_map.unsqueeze(0)
    return (absolute_error * weight_map).mean()


def _valid_prior_mask(camera: Any, device: torch.device) -> torch.Tensor:
    """Exclude invalid alpha pixels and their one-pixel undistortion border."""
    alpha_mask = getattr(camera, "alpha_mask", None)
    if alpha_mask is None:
        height, width = camera.original_image.shape[-2:]
        return torch.ones((height, width), device=device, dtype=torch.float32)
    invalid = (alpha_mask[:1].to(device=device) < 0.5).float()
    invalid_border = F.max_pool2d(invalid.unsqueeze(0), kernel_size=3, stride=1, padding=1)
    return 1.0 - invalid_border.squeeze(0).squeeze(0)


def _resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    """Resolve a checkpoint relative to the repository root."""
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        repository_root = Path(__file__).resolve().parents[2]
        checkpoint_path = repository_root / checkpoint_path
    return checkpoint_path.resolve()


def load_pidinet(checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    """Load the official table-5 PiDiNet checkpoint for inference."""
    checkpoint_path = _resolve_checkpoint_path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "HF-GS PiDiNet checkpoint was not found: {}".format(checkpoint_path)
        )
    model = build_pidinet()
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint_data.get("state_dict", checkpoint_data)
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def _infer_pidinet_map(
    image: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Infer PiDiNet's fused structural edge map without view normalization."""
    input_image = image[:3].detach().float().unsqueeze(0).to(device)
    mean = IMAGENET_MEAN.to(device=device)
    std = IMAGENET_STD.to(device=device)
    normalized_input = (input_image - mean) / std
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            structural_map = model(normalized_input)[-1]
    return structural_map.float().squeeze(0).squeeze(0)


def compute_pidinet_prior(
    image: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Infer and normalize PiDiNet's fused structural edge map."""
    return normalize_prior_map(_infer_pidinet_map(image, model, device))


def compute_pidinet_prior_tiled(
    image: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    tile_size: int = 1024,
    halo: int = 192,
) -> torch.Tensor:
    """
    Infer PiDiNet in aligned overlapping tiles after a full-frame CUDA OOM.

    The 192-pixel halo exceeds the full model's approximate receptive-field
    radius, and all internal tile starts are aligned to PiDiNet's 8x stride.
    """
    if tile_size <= 2 * halo:
        raise ValueError("hf_pidinet_tile_size must be larger than twice the tile halo.")
    if halo % 8 != 0 or (tile_size - 2 * halo) % 8 != 0:
        raise ValueError("PiDiNet tile halo and core size must be divisible by 8.")

    image = image[:3].detach().float()
    height, width = image.shape[-2:]
    core_size = tile_size - 2 * halo
    structural_map = torch.empty((height, width), device=device, dtype=torch.float32)
    for output_y0 in range(0, height, core_size):
        output_y1 = min(output_y0 + core_size, height)
        input_y0 = max(output_y0 - halo, 0)
        input_y1 = min(output_y1 + halo, height)
        for output_x0 in range(0, width, core_size):
            output_x1 = min(output_x0 + core_size, width)
            input_x0 = max(output_x0 - halo, 0)
            input_x1 = min(output_x1 + halo, width)
            tile = image[:, input_y0:input_y1, input_x0:input_x1]
            tile_prior = _infer_pidinet_map(tile, model, device)
            crop_y0 = output_y0 - input_y0
            crop_y1 = crop_y0 + output_y1 - output_y0
            crop_x0 = output_x0 - input_x0
            crop_x1 = crop_x0 + output_x1 - output_x0
            structural_map[output_y0:output_y1, output_x0:output_x1] = tile_prior[
                crop_y0:crop_y1,
                crop_x0:crop_x1,
            ]
    return normalize_prior_map(structural_map)


def build_hfgs_weight_maps(
    train_cameras: list[Any],
    opt: Any,
) -> tuple[dict[int, torch.Tensor], dict[str, float]]:
    """
    Precompute calibrated HF-GS supervision weights for all training views.

    Priors are kept as CPU float16 tensors while their global means are
    calibrated, then replaced in-place by one CPU float16 weight map per view.
    """
    if not train_cameras:
        return {}, {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epsilon = float(getattr(opt, "hf_edge_epsilon", 1e-6))
    model = load_pidinet(
        getattr(opt, "hf_pidinet_checkpoint", "third_party/pidinet/table5_pidinet.pth"),
        device,
    )
    cached_priors: list[tuple[int, torch.Tensor, torch.Tensor] | None] = []
    gradient_sum = 0.0
    structural_sum = 0.0
    valid_count = 0.0
    started_at = time.perf_counter()
    camera_count = len(train_cameras)
    force_tiled_pidinet = False
    tile_size = int(getattr(opt, "hf_pidinet_tile_size", 1024))
    tile_halo = int(getattr(opt, "hf_pidinet_tile_halo", 192))

    with torch.inference_mode():
        for camera_index, camera in enumerate(train_cameras, start=1):
            image = camera.original_image
            device_image = image.to(device)
            gradient_prior = compute_sobel_prior(device_image, epsilon)
            if force_tiled_pidinet:
                structural_prior = compute_pidinet_prior_tiled(
                    device_image,
                    model,
                    device,
                    tile_size,
                    tile_halo,
                )
            else:
                try:
                    structural_prior = compute_pidinet_prior(device_image, model, device)
                except RuntimeError as error:
                    if device.type != "cuda" or "out of memory" not in str(error).lower():
                        raise
                    torch.cuda.empty_cache()
                    force_tiled_pidinet = True
                    print(
                        "HF-GS PiDiNet full-frame OOM on {}; retrying with {}px tiles.".format(
                            getattr(camera, "image_name", camera.uid),
                            tile_size,
                        ),
                        flush=True,
                    )
                    structural_prior = compute_pidinet_prior_tiled(
                        device_image,
                        model,
                        device,
                        tile_size,
                        tile_halo,
                    )
            valid_mask = _valid_prior_mask(camera, device)
            gradient_prior = gradient_prior * valid_mask
            structural_prior = structural_prior * valid_mask
            gradient_sum += float(gradient_prior.sum().item())
            structural_sum += float(structural_prior.sum().item())
            valid_count += float(valid_mask.sum().item())
            cached_priors.append(
                (
                    int(camera.uid),
                    gradient_prior.to(device="cpu", dtype=torch.float16),
                    structural_prior.to(device="cpu", dtype=torch.float16),
                )
            )
            if camera_index == 1 or camera_index % 10 == 0 or camera_index == camera_count:
                print(
                    "HF-GS edge prior {}/{} ({:.1f}s)".format(
                        camera_index,
                        camera_count,
                        time.perf_counter() - started_at,
                    ),
                    flush=True,
                )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    denominator = max(valid_count, 1.0)
    gradient_mean = gradient_sum / denominator
    structural_mean = structural_sum / denominator
    alpha_p = float(getattr(opt, "hf_edge_alpha_p_ref", 0.12)) / (gradient_mean + epsilon)
    alpha_g = float(getattr(opt, "hf_edge_alpha_g_ref", 0.09)) / (structural_mean + epsilon)

    weight_maps: dict[int, torch.Tensor] = {}
    for prior_index, cached_prior in enumerate(cached_priors):
        if cached_prior is None:
            continue
        camera_uid, gradient_prior, structural_prior = cached_prior
        weight_maps[camera_uid] = (
            1.0
            + alpha_p * gradient_prior.float()
            + alpha_g * structural_prior.float()
        ).to(dtype=torch.float16)
        cached_priors[prior_index] = None

    stats = {
        "gradient_mean": gradient_mean,
        "structural_mean": structural_mean,
        "alpha_p": alpha_p,
        "alpha_g": alpha_g,
    }
    print(
        "HF-GS edge priors: {} views, G_mean={:.6f}, E_mean={:.6f}, "
        "alpha_p={:.6f}, alpha_g={:.6f}".format(
            len(weight_maps),
            gradient_mean,
            structural_mean,
            alpha_p,
            alpha_g,
        )
    )
    return weight_maps, stats


def compute_scale_reference(scales: torch.Tensor, quantile: float = 0.75) -> torch.Tensor:
    """Return the lower bound of the largest quartile of maximum-axis scales."""
    if scales.ndim != 2 or scales.shape[-1] != 3:
        raise ValueError("scales must have shape [N, 3].")
    if scales.shape[0] == 0:
        return scales.new_tensor(0.0)
    maximum_axis_scale = scales.detach().amax(dim=1)
    return torch.quantile(maximum_axis_scale.float(), float(quantile)).to(scales.device)


def compute_scale_aware_thresholds(
    scales: torch.Tensor,
    base_threshold: float,
    scale_reference: torch.Tensor | float,
    eta: float = 0.2,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute the per-Gaussian HF-GS densification threshold ``tau_i``."""
    maximum_axis_scale = scales.detach().amax(dim=1)
    reference = torch.as_tensor(
        scale_reference,
        device=maximum_axis_scale.device,
        dtype=maximum_axis_scale.dtype,
    )
    excess = torch.clamp(maximum_axis_scale / (reference + float(epsilon)) - 1.0, min=0.0)
    scale_factor = 1.0 - float(eta) * excess
    return float(base_threshold) * scale_factor


def compute_scale_contraction_ratios(
    scales: torch.Tensor,
    scale_reference: torch.Tensor | float,
    gamma: float = 0.005,
    minimum_ratio: float = 0.70,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute the isotropic periodic scale-contraction ratio for each Gaussian."""
    maximum_axis_scale = scales.detach().amax(dim=1)
    reference = torch.as_tensor(
        scale_reference,
        device=maximum_axis_scale.device,
        dtype=maximum_axis_scale.dtype,
    )
    excess = maximum_axis_scale / (reference + float(epsilon)) - 1.0
    large_mask = excess > 0.0
    ratios = torch.ones_like(maximum_axis_scale)
    contracted = torch.exp(-float(gamma) * excess[large_mask])
    ratios[large_mask] = torch.clamp(contracted, min=float(minimum_ratio), max=1.0)
    return ratios
