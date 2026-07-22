"""Lich do phan giai coarse-to-fine cho qua trinh train."""
from __future__ import annotations

from typing import Any


COARSE_RESOLUTION_SCALE = 4.0
MIDDLE_RESOLUTION_SCALE = 2.0
FULL_RESOLUTION_SCALE = 1.0


def validate_coarse_to_fine_schedule(opt: Any) -> None:
    """Kiem tra thu tu cac moc iteration khi coarse-to-fine duoc bat."""
    if not bool(getattr(opt, "coarse_to_fine", False)):
        return

    middle_iter = int(getattr(opt, "coarse_to_fine_middle_iter", 2_000))
    full_iter = int(getattr(opt, "coarse_to_fine_full_iter", 5_000))
    if middle_iter < 1:
        raise ValueError("coarse_to_fine_middle_iter must be at least 1.")
    if full_iter <= middle_iter:
        raise ValueError("coarse_to_fine_full_iter must be greater than coarse_to_fine_middle_iter.")


def build_training_resolution_scales(opt: Any) -> list[float]:
    """Tra ve cac muc do phan giai can nap cho mot lan train."""
    validate_coarse_to_fine_schedule(opt)
    if not bool(getattr(opt, "coarse_to_fine", False)):
        return [FULL_RESOLUTION_SCALE]
    return [COARSE_RESOLUTION_SCALE, MIDDLE_RESOLUTION_SCALE, FULL_RESOLUTION_SCALE]


def resolve_training_resolution_scale(iteration: int, opt: Any) -> float:
    """Chon he so downsample tuong ung voi iteration hien tai."""
    validate_coarse_to_fine_schedule(opt)
    if not bool(getattr(opt, "coarse_to_fine", False)):
        return FULL_RESOLUTION_SCALE

    if int(iteration) < int(opt.coarse_to_fine_middle_iter):
        return COARSE_RESOLUTION_SCALE
    if int(iteration) < int(opt.coarse_to_fine_full_iter):
        return MIDDLE_RESOLUTION_SCALE
    return FULL_RESOLUTION_SCALE
