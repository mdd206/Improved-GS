"""Hau xu ly va luu anh render VAI."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image


def _gaussian_kernel(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Tao kernel Gaussian 2D voi ban kinh ba sigma."""
    if sigma <= 0:
        raise ValueError("sharpen_sigma phai lon hon 0")
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coordinates**2) / (2.0 * float(sigma) ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.expand(3, 1, -1, -1).contiguous()


def sharpen_image(image: torch.Tensor, amount: float = 1.0, sigma: float = 0.6) -> torch.Tensor:
    """Lam net anh RGB bang unsharp mask va giu gia tri trong khoang 0 den 1."""
    if amount < 0:
        raise ValueError("sharpen_amount khong duoc am")
    if sigma <= 0:
        raise ValueError("sharpen_sigma phai lon hon 0")
    if amount == 0:
        return image.clamp(0.0, 1.0)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Anh sharpen phai co shape [3, H, W]")

    kernel = _gaussian_kernel(float(sigma), image.device, image.dtype)
    radius = kernel.shape[-1] // 2
    padded = functional.pad(
        image.unsqueeze(0),
        (radius, radius, radius, radius),
        mode="reflect",
    )
    blurred = functional.conv2d(padded, kernel, groups=3).squeeze(0)
    return (image + float(amount) * (image - blurred)).clamp(0.0, 1.0)


def save_render_image(
    image: torch.Tensor,
    output_path: str | Path,
    jpeg_quality: int = 95,
    jpeg_subsampling: int = 2,
) -> None:
    """Luu JPEG voi quality/subsampling ro rang hoac PNG lossless."""
    if not 1 <= int(jpeg_quality) <= 100:
        raise ValueError("jpeg_quality phai nam trong [1, 100]")
    if int(jpeg_subsampling) not in {0, 1, 2}:
        raise ValueError("jpeg_subsampling phai la 0, 1 hoac 2")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = (
        image.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    pil_image = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        pil_image.save(
            output_path,
            format="JPEG",
            quality=int(jpeg_quality),
            subsampling=int(jpeg_subsampling),
        )
    else:
        pil_image.save(output_path, format="PNG")
