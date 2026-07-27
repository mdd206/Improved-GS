"""Warp camera PINHOLE da undistort ve camera SIMPLE_RADIAL goc."""
from __future__ import annotations

import torch
import torch.nn.functional as functional


def redistort_image(
    image: torch.Tensor,
    source_fx: float,
    source_fy: float,
    source_cx: float,
    source_cy: float,
    target_focal: float,
    target_cx: float,
    target_cy: float,
    target_width: int,
    target_height: int,
    radial_k: float,
    num_iters: int = 15,
    interpolation: str = "bicubic",
) -> torch.Tensor:
    """Lay mau canvas PINHOLE tren luoi pixel cua camera SIMPLE_RADIAL goc.

    ``source_*`` la intrinsics cua canvas PINHOLE do COLMAP image_undistorter
    tao ra. ``target_*`` la intrinsics va kich thuoc camera SIMPLE_RADIAL goc.
    Hai camera co the co focal, principal point va kich thuoc khac nhau.
    """
    if interpolation not in {"bilinear", "bicubic"}:
        raise ValueError("redistort_interpolation phai la bilinear hoac bicubic")
    if image.ndim != 3:
        raise ValueError(
            f"Anh redistort phai co dang [C,H,W], nhan duoc {tuple(image.shape)}"
        )
    if min(float(source_fx), float(source_fy), float(target_focal)) <= 0.0:
        raise ValueError("Focal length phai lon hon 0")
    if int(target_width) <= 0 or int(target_height) <= 0:
        raise ValueError("Kich thuoc camera dich phai lon hon 0")

    _, source_height, source_width = image.shape
    device = image.device
    ys, xs = torch.meshgrid(
        torch.arange(int(target_height), device=device, dtype=torch.float32),
        torch.arange(int(target_width), device=device, dtype=torch.float32),
        indexing="ij",
    )

    # COLMAP dat tam pixel dau tien tai (0.5, 0.5), con grid_sample danh chi
    # so tam pixel tu 0. Vi vay cong 0.5 khi unproject va tru lai 0.5 khi
    # chuyen toa do camera PINHOLE sang chi so tensor nguon.
    xd = (xs + 0.5 - float(target_cx)) / float(target_focal)
    yd = (ys + 0.5 - float(target_cy)) / float(target_focal)
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
    undistorted_x = xd * scale
    undistorted_y = yd * scale
    source_x = undistorted_x * float(source_fx) + float(source_cx) - 0.5
    source_y = undistorted_y * float(source_fy) + float(source_cy) - 0.5
    grid_x = source_x * (2.0 / max(source_width - 1, 1)) - 1.0
    grid_y = source_y * (2.0 / max(source_height - 1, 1)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    return functional.grid_sample(
        image.unsqueeze(0),
        grid,
        mode=interpolation,
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)
