"""Train-view ranking for independent test-pose fine-tuning."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np

from utils.pose_aware_sampling import CameraPose, pose_from_training_camera


@dataclass(frozen=True)
class RankedTrainingView:
    """One train camera and its similarity to a target test pose."""

    rank: int
    train_index: int
    uid: int
    image_name: str
    distance: float
    cosine: float
    score: float


@dataclass(frozen=True)
class TestPoseViewSelection:
    """Deterministic top-K selection plus the scale used by its score."""

    sigma: float
    sigma_multiplier: float
    requested_top_k: int
    train_count: int
    views: tuple[RankedTrainingView, ...]

    @property
    def selected_indices(self) -> tuple[int, ...]:
        return tuple(view.train_index for view in self.views)

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula": "exp(-d^2 / (2 * (sigma_multiplier * sigma)^2)) * max(0, cos_theta)^2",
            "sigma": float(self.sigma),
            "sigma_definition": "median nearest-neighbor spacing between train camera centers",
            "sigma_multiplier": float(self.sigma_multiplier),
            "requested_top_k": int(self.requested_top_k),
            "selected_top_k": len(self.views),
            "train_count": int(self.train_count),
            "views": [asdict(view) for view in self.views],
        }


def pose_output_directory_name(pose_index: int, image_name: str) -> str:
    """Create a stable, filesystem-safe model folder for one test pose."""
    stem = Path(str(image_name)).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    if safe_stem == "":
        safe_stem = "pose"
    return "{:04d}_{}".format(int(pose_index), safe_stem)


def validate_finetune_schedule(
    fine_tune_steps: int,
    split_from_step: int,
    split_until_step: int,
    top_k: int,
    sigma_multiplier: float,
) -> None:
    """Validate the local schedule before allocating GPU memory."""
    if int(fine_tune_steps) < 1:
        raise ValueError("fine_tune_steps must be at least 1.")
    if int(split_from_step) < 0:
        raise ValueError("split_from_step must be non-negative.")
    if int(split_until_step) < int(split_from_step):
        raise ValueError("split_until_step must be >= split_from_step.")
    if int(split_until_step) > int(fine_tune_steps):
        raise ValueError("split_until_step must not exceed fine_tune_steps.")
    if int(top_k) < 1:
        raise ValueError("top_k must be at least 1.")
    if float(sigma_multiplier) <= 0.0:
        raise ValueError("sigma_multiplier must be greater than 0.")


def configure_local_improvedgs_options(
    opt: Any,
    fine_tune_steps: int,
    split_from_step: int,
    split_until_step: int,
) -> Any:
    """Return an isolated local schedule for a trained ImprovedGS PLY."""
    validate_finetune_schedule(
        fine_tune_steps,
        split_from_step,
        split_until_step,
        top_k=1,
        sigma_multiplier=1.0,
    )
    local_opt = copy.copy(opt)
    local_opt.training_method = "improvedgs"
    local_opt.coarse_to_fine = False
    local_opt.pose_aware_sampling = False
    # The existing dispatcher skips the terminal schedule iteration. A one-step
    # sentinel therefore produces exactly `fine_tune_steps` optimizer updates.
    local_opt.iterations = int(fine_tune_steps) + 1
    local_opt.position_lr_max_steps = int(fine_tune_steps)
    # Dam bao 3.000 local step deu la optimizer update; MU chi bat dau sau stage nay.
    local_opt.mu_start_iter = int(fine_tune_steps) + 1
    local_opt.mu_second_start_iter = int(fine_tune_steps) + 2
    # Densification uses strict bounds: from < step < until.
    local_opt.densify_from_iter = max(int(split_from_step) - 1, 0)
    local_opt.densify_until_iter = int(split_until_step) + 1
    # A 30k PLY is already near its target size, unlike a sparse initialization.
    local_opt.budget_warmup_until_offset = int(split_until_step) + 1
    return local_opt


def median_nearby_camera_spacing(centers: np.ndarray) -> float:
    """Return the median nearest-neighbor distance between train cameras."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("Camera centers must have shape [N, 3].")
    if centers.shape[0] == 0:
        raise ValueError("At least one train camera is required.")
    if centers.shape[0] == 1:
        return 1.0

    pairwise_distances = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :],
        axis=2,
    )
    np.fill_diagonal(pairwise_distances, np.inf)
    spacing = float(np.median(np.min(pairwise_distances, axis=1)))
    return max(spacing, 1e-8)


def select_top_training_views(
    train_cameras: list[Any],
    test_pose: CameraPose,
    top_k: int = 25,
    sigma_multiplier: float = 3.0,
) -> TestPoseViewSelection:
    """Score every train camera with the requested pose formula and keep top-K."""
    if not train_cameras:
        raise ValueError("Test-pose fine-tuning requires at least one train camera.")
    if int(top_k) < 1:
        raise ValueError("top_k must be at least 1.")
    if float(sigma_multiplier) <= 0.0:
        raise ValueError("sigma_multiplier must be greater than 0.")

    train_poses = [pose_from_training_camera(camera) for camera in train_cameras]
    train_centers = np.stack([pose.center for pose in train_poses])
    train_forwards = np.stack([pose.forward for pose in train_poses])
    sigma = median_nearby_camera_spacing(train_centers)

    test_center = np.asarray(test_pose.center, dtype=np.float64).reshape(3)
    test_forward = np.asarray(test_pose.forward, dtype=np.float64).reshape(3)
    test_forward_norm = float(np.linalg.norm(test_forward))
    if test_forward_norm <= 0.0:
        raise ValueError("Test camera forward vector must have non-zero length.")
    test_forward = test_forward / test_forward_norm
    distances = np.linalg.norm(train_centers - test_center[None, :], axis=1)
    cosines = np.clip(train_forwards @ test_forward, -1.0, 1.0)
    distance_scale = float(sigma_multiplier) * sigma
    distance_weights = np.exp(
        -(distances * distances) / (2.0 * distance_scale * distance_scale)
    )
    direction_weights = np.maximum(cosines, 0.0) ** 2
    scores = distance_weights * direction_weights

    # Stable sort makes equal-score ties reproducible in the original camera order.
    ranked_indices = np.argsort(-scores, kind="stable")
    selected_count = min(int(top_k), len(train_cameras))
    ranked_views: list[RankedTrainingView] = []
    for rank, train_index_value in enumerate(ranked_indices[:selected_count], start=1):
        train_index = int(train_index_value)
        camera = train_cameras[train_index]
        ranked_views.append(
            RankedTrainingView(
                rank=rank,
                train_index=train_index,
                uid=int(getattr(camera, "uid", train_index)),
                image_name=str(getattr(camera, "image_name", train_index)),
                distance=float(distances[train_index]),
                cosine=float(cosines[train_index]),
                score=float(scores[train_index]),
            )
        )

    return TestPoseViewSelection(
        sigma=sigma,
        sigma_multiplier=float(sigma_multiplier),
        requested_top_k=int(top_k),
        train_count=len(train_cameras),
        views=tuple(ranked_views),
    )
