"""Tao camera pool uu tien cac train view gan test pose thieu coverage."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraPose:
    center: np.ndarray
    forward: np.ndarray


@dataclass(frozen=True)
class PoseSamplingPlan:
    repeat_counts: dict[int, int]
    train_count: int
    test_count: int
    extra_count: int
    pool_size: int
    median_train_spacing: float
    max_test_gap: float


def _normalize_vector(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        raise ValueError("Camera direction must have non-zero length.")
    return value / norm


def quaternion_to_rotation(qvec: np.ndarray) -> np.ndarray:
    """Doi quaternion COLMAP wxyz thanh ma tran world-to-camera."""
    qvec = np.asarray(qvec, dtype=np.float64).reshape(4)
    qvec = _normalize_vector(qvec)
    w, x, y, z = qvec
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_from_csv_row(row: dict[str, str]) -> CameraPose:
    """Doc center va huong nhin tu mot dong test_poses.csv."""
    rotation_world_to_camera = quaternion_to_rotation(
        np.array([row["qw"], row["qx"], row["qy"], row["qz"]], dtype=np.float64)
    )
    translation = np.array([row["tx"], row["ty"], row["tz"]], dtype=np.float64)
    rotation_camera_to_world = rotation_world_to_camera.T
    center = -rotation_camera_to_world @ translation
    forward = rotation_camera_to_world[:, 2]
    return CameraPose(center=center, forward=_normalize_vector(forward))


def pose_from_training_camera(camera: Any) -> CameraPose:
    """Lay center va huong nhin tu Camera da duoc Scene nap."""
    camera_center = camera.camera_center
    if hasattr(camera_center, "detach"):
        camera_center = camera_center.detach().cpu().numpy()
    center = np.asarray(camera_center, dtype=np.float64).reshape(3)
    rotation_camera_to_world = np.asarray(camera.R, dtype=np.float64).reshape(3, 3)
    return CameraPose(center=center, forward=_normalize_vector(rotation_camera_to_world[:, 2]))


def resolve_test_pose_path(source_path: str | Path, configured_path: str = "") -> Path:
    """Tim test_poses.csv trong scene hoac tai duong dan duoc cau hinh."""
    source_path = Path(source_path)
    if str(configured_path).strip():
        pose_path = Path(configured_path)
        if not pose_path.is_absolute():
            pose_path = source_path / pose_path
    else:
        pose_path = source_path / "test" / "test_poses.csv"
    if not pose_path.is_file():
        raise FileNotFoundError("Khong tim thay test pose de sampling: {}".format(pose_path))
    return pose_path


def load_test_camera_poses(csv_path: str | Path) -> list[CameraPose]:
    """Doc toan bo extrinsics test, khong doc anh hoac ground truth."""
    required_columns = {"qw", "qx", "qy", "qz", "tx", "ty", "tz"}
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required_columns - set(reader.fieldnames or []))
        if missing:
            raise ValueError("test_poses.csv thieu cot: {}".format(", ".join(missing)))
        poses = [pose_from_csv_row(row) for row in reader]
    if not poses:
        raise ValueError("test_poses.csv khong co pose: {}".format(csv_path))
    return poses


def _median_train_spacing(centers: np.ndarray) -> float:
    if centers.shape[0] < 2:
        return 1.0
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    spacing = float(np.median(np.min(distances, axis=1)))
    return max(spacing, 1e-8)


def build_pose_sampling_plan(
    train_cameras: list[Any],
    test_poses: list[CameraPose],
    neighbor_count: int = 3,
    extra_fraction: float = 0.25,
    max_repeat: int = 2,
    angle_weight: float = 0.25,
) -> PoseSamplingPlan:
    """Chon train camera duoc lap them dua tren khoang cach va goc test pose."""
    if not train_cameras:
        raise ValueError("Pose-aware sampling requires at least one training camera.")
    if not test_poses:
        raise ValueError("Pose-aware sampling requires at least one test pose.")
    if int(neighbor_count) < 1:
        raise ValueError("pose_aware_k must be at least 1.")
    if not (0.0 <= float(extra_fraction) <= 1.0):
        raise ValueError("pose_aware_extra_fraction must be in [0, 1].")
    if int(max_repeat) < 1:
        raise ValueError("pose_aware_max_repeat must be at least 1.")
    if float(angle_weight) < 0.0:
        raise ValueError("pose_aware_angle_weight must be non-negative.")

    train_poses = [pose_from_training_camera(camera) for camera in train_cameras]
    train_centers = np.stack([pose.center for pose in train_poses])
    train_forwards = np.stack([pose.forward for pose in train_poses])
    test_centers = np.stack([pose.center for pose in test_poses])
    test_forwards = np.stack([pose.forward for pose in test_poses])
    median_spacing = _median_train_spacing(train_centers)

    position_distances = np.linalg.norm(test_centers[:, None, :] - train_centers[None, :, :], axis=2)
    normalized_distances = position_distances / median_spacing
    direction_dots = np.clip(test_forwards @ train_forwards.T, -1.0, 1.0)
    angle_distances = np.arccos(direction_dots) / np.deg2rad(30.0)
    combined_cost = normalized_distances + float(angle_weight) * angle_distances

    k = min(int(neighbor_count), len(train_cameras))
    importance = np.zeros((len(train_cameras),), dtype=np.float64)
    nearest_gaps = np.min(normalized_distances, axis=1)
    for test_index in range(len(test_poses)):
        nearest_indices = np.argsort(combined_cost[test_index], kind="stable")[:k]
        difficulty = float(np.clip(nearest_gaps[test_index], 0.25, 5.0))
        for rank, train_index in enumerate(nearest_indices):
            importance[train_index] += difficulty / float(rank + 1)

    repeat_counts = {
        int(getattr(camera, "uid", index)): 1
        for index, camera in enumerate(train_cameras)
    }
    max_extra = len(train_cameras) * (int(max_repeat) - 1)
    requested_extra = int(round(len(train_cameras) * float(extra_fraction)))
    remaining_extra = min(requested_extra, max_extra)
    ranking = np.argsort(-importance, kind="stable")
    while remaining_extra > 0:
        allocated = 0
        for train_index in ranking:
            if importance[train_index] <= 0.0 or remaining_extra <= 0:
                break
            uid = int(getattr(train_cameras[int(train_index)], "uid", int(train_index)))
            if repeat_counts[uid] >= int(max_repeat):
                continue
            repeat_counts[uid] += 1
            remaining_extra -= 1
            allocated += 1
        if allocated == 0:
            break

    extra_count = sum(count - 1 for count in repeat_counts.values())
    return PoseSamplingPlan(
        repeat_counts=repeat_counts,
        train_count=len(train_cameras),
        test_count=len(test_poses),
        extra_count=extra_count,
        pool_size=len(train_cameras) + extra_count,
        median_train_spacing=median_spacing,
        max_test_gap=float(nearest_gaps.max()),
    )


def build_repeated_camera_pool(
    cameras: list[Any],
    repeat_counts: dict[int, int] | None = None,
) -> list[Any]:
    """Tao pool ma moi camera co it nhat mot ban sao."""
    repeat_counts = repeat_counts or {}
    pool: list[Any] = []
    for index, camera in enumerate(cameras):
        uid = int(getattr(camera, "uid", index))
        pool.extend([camera] * max(int(repeat_counts.get(uid, 1)), 1))
    return pool
