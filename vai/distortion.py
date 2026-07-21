"""Warp SIMPLE_RADIAL dung cho anh render VAI."""
from __future__ import annotations

import torch
import torch.nn.functional as functional


def redistort_image(
    image: torch.Tensor,
    focal: float,
    cx: float,
    cy: float,
    radial_k: float,
    num_iters: int = 15,
    interpolation: str = "bicubic",
) -> torch.Tensor:
    """Warp anh undistort ve camera SIMPLE_RADIAL bang Newton iteration."""
    if interpolation not in {"bilinear", "bicubic"}:
        raise ValueError("redistort_interpolation phai la bilinear hoac bicubic")
    _, height, width = image.shape
    device = image.device
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xd = (xs - float(cx)) / float(focal)
    yd = (ys - float(cy)) / float(focal)
    rd = torch.sqrt(xd * xd + yd * yd)

    ru = rd.clone()
    for _ in range(int(num_iters)):
        value = float(radial_k) * ru**3 + ru - rd
        derivative = 3.0 * float(radial_k) * ru**2 + 1.0
        derivative = torch.where(
            derivative.abs() < 1e-12,
            torch.full_like(derivative, 1e-12),
            derivative,
        )
        ru = ru - value / derivative

    scale = torch.where(rd > 1e-12, ru / rd, torch.ones_like(rd))
    source_x = xd * scale * float(focal) + float(cx)
    source_y = yd * scale * float(focal) + float(cy)
    grid_x = source_x * (2.0 / max(width - 1, 1)) - 1.0
    grid_y = source_y * (2.0 / max(height - 1, 1)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    return functional.grid_sample(
        image.unsqueeze(0),
        grid,
        mode=interpolation,
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)


def redistort_and_crop(
    image: torch.Tensor,
    focal: float,
    render_cx: float,
    render_cy: float,
    radial_k: float,
    target_cx: float,
    target_cy: float,
    target_width: int,
    target_height: int,
    num_iters: int = 15,
    interpolation: str = "bicubic",
) -> torch.Tensor:
    """Redistort canvas mo rong va crop ve dung khung anh trong CSV."""
    distorted = redistort_image(
        image,
        focal,
        render_cx,
        render_cy,
        radial_k,
        num_iters,
        interpolation,
    )
    _, canvas_height, canvas_width = distorted.shape
    offset_x = int(round(float(render_cx) - float(target_cx)))
    offset_y = int(round(float(render_cy) - float(target_cy)))
    x0 = max(offset_x, 0)
    y0 = max(offset_y, 0)
    x1 = min(offset_x + int(target_width), canvas_width)
    y1 = min(offset_y + int(target_height), canvas_height)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(
            "Crop VAI nam ngoai canvas: offset=({}, {}) canvas={}x{} target={}x{}".format(
                offset_x,
                offset_y,
                canvas_width,
                canvas_height,
                target_width,
                target_height,
            )
        )

    cropped = distorted[:, y0:y1, x0:x1]
    if cropped.shape[1:] != (int(target_height), int(target_width)):
        pad_top = y0 - offset_y
        pad_left = x0 - offset_x
        pad_bottom = int(target_height) - cropped.shape[1] - pad_top
        pad_right = int(target_width) - cropped.shape[2] - pad_left
        cropped = functional.pad(
            cropped,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        )
    return cropped
