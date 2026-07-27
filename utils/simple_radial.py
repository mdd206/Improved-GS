"""Reference SIMPLE_RADIAL projection used to validate the CUDA rasterizer."""
from __future__ import annotations

import torch


def project_simple_radial(
    points_camera: torch.Tensor,
    focal_x: float,
    focal_y: float,
    cx: float,
    cy: float,
    radial_k: float,
) -> torch.Tensor:
    """Project camera-space points to zero-based tensor pixel coordinates."""
    if points_camera.shape[-1] != 3:
        raise ValueError("points_camera phai co shape [..., 3]")
    x, y, z = points_camera.unbind(dim=-1)
    u = x / z
    v = y / z
    radius_sq = u * u + v * v
    scale = 1.0 + float(radial_k) * radius_sq
    pixel_x = float(focal_x) * scale * u + float(cx) - 0.5
    pixel_y = float(focal_y) * scale * v + float(cy) - 0.5
    return torch.stack((pixel_x, pixel_y), dim=-1)


def simple_radial_projection_jacobian(
    points_camera: torch.Tensor,
    focal_x: float,
    focal_y: float,
    radial_k: float,
) -> torch.Tensor:
    """Return d(pixel_x,pixel_y)/d(X,Y,Z) for SIMPLE_RADIAL projection."""
    if points_camera.shape[-1] != 3:
        raise ValueError("points_camera phai co shape [..., 3]")
    x, y, z = points_camera.unbind(dim=-1)
    u = x / z
    v = y / z
    radius_sq = u * u + v * v
    k = float(radial_k)
    inv_z = z.reciprocal()

    common = 1.0 + k * radius_sq
    dxx = float(focal_x) * inv_z * (common + 2.0 * k * u * u)
    dxy = float(focal_x) * inv_z * (2.0 * k * u * v)
    dxz = -float(focal_x) * inv_z * u * (1.0 + 3.0 * k * radius_sq)
    dyx = float(focal_y) * inv_z * (2.0 * k * u * v)
    dyy = float(focal_y) * inv_z * (common + 2.0 * k * v * v)
    dyz = -float(focal_y) * inv_z * v * (1.0 + 3.0 * k * radius_sq)

    row_x = torch.stack((dxx, dxy, dxz), dim=-1)
    row_y = torch.stack((dyx, dyy, dyz), dim=-1)
    return torch.stack((row_x, row_y), dim=-2)


def simple_radial_projection_hessians(
    points_camera: torch.Tensor,
    focal_x: float,
    focal_y: float,
    radial_k: float,
) -> torch.Tensor:
    """Return one 3x3 camera-space Hessian for each projected pixel axis."""
    if points_camera.shape[-1] != 3:
        raise ValueError("points_camera phai co shape [..., 3]")
    x, y, z = points_camera.unbind(dim=-1)
    u = x / z
    v = y / z
    u2 = u * u
    v2 = v * v
    uv = u * v
    radius_sq = u2 + v2
    k = float(radial_k)
    inv_z_sq = z.reciprocal().square()
    scale_x = float(focal_x) * inv_z_sq
    scale_y = float(focal_y) * inv_z_sq

    hxxx = scale_x * 6.0 * k * u
    hxxy = scale_x * 2.0 * k * v
    hxyy = scale_x * 2.0 * k * u
    hxxz = -scale_x * (1.0 + 9.0 * k * u2 + 3.0 * k * v2)
    hxyz = -scale_x * 6.0 * k * uv
    hxzz = scale_x * (2.0 * u + 12.0 * k * u * radius_sq)

    hyxx = scale_y * 2.0 * k * v
    hyxy = scale_y * 2.0 * k * u
    hyyy = scale_y * 6.0 * k * v
    hyxz = -scale_y * 6.0 * k * uv
    hyyz = -scale_y * (1.0 + 3.0 * k * u2 + 9.0 * k * v2)
    hyzz = scale_y * (2.0 * v + 12.0 * k * v * radius_sq)

    row_x0 = torch.stack((hxxx, hxxy, hxxz), dim=-1)
    row_x1 = torch.stack((hxxy, hxyy, hxyz), dim=-1)
    row_x2 = torch.stack((hxxz, hxyz, hxzz), dim=-1)
    row_y0 = torch.stack((hyxx, hyxy, hyxz), dim=-1)
    row_y1 = torch.stack((hyxy, hyyy, hyyz), dim=-1)
    row_y2 = torch.stack((hyxz, hyyz, hyzz), dim=-1)
    hessian_x = torch.stack((row_x0, row_x1, row_x2), dim=-2)
    hessian_y = torch.stack((row_y0, row_y1, row_y2), dim=-2)
    return torch.stack((hessian_x, hessian_y), dim=-3)
