"""Theo doi va khoi phuc tien do fine-tune tung test pose."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from vai.common import save_json


RESUME_IDENTITY_FIELDS = (
    "scene_name",
    "source_path",
    "base_model_path",
    "base_iteration",
    "fine_tune_steps",
    "split_from_step",
    "split_until_step",
    "top_k",
    "sigma_multiplier",
    "pose_start_index",
    "pose_end_index_exclusive",
    "output_extension",
)


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Doc mot JSON object da ghi tu lan chay truoc."""
    path = Path(path)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("JSON phai la object: {}".format(path))
    return payload


def validate_resume_manifest(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> None:
    """Chan resume nham checkpoint, pose range hoac cau hinh fine-tune."""
    mismatches = [
        "{}: old={!r}, new={!r}".format(
            field,
            previous.get(field),
            current.get(field),
        )
        for field in RESUME_IDENTITY_FIELDS
        if previous.get(field) != current.get(field)
    ]
    if mismatches:
        raise ValueError(
            "Khong the resume vi cau hinh da thay doi:\n{}".format(
                "\n".join(mismatches)
            )
        )


def build_progress_payload(
    manifest: dict[str, Any],
    status: str,
    current_pose: dict[str, Any] | None,
) -> dict[str, Any]:
    """Tao file tien do ngan gon, de doc ngay ca khi job bi ngat."""
    completed_poses = [
        {
            "pose_index": int(pose["pose_index"]),
            "pose_number_in_scene": int(pose["pose_index"]) + 1,
            "test_image_name": pose["test_image_name"],
            "render_path": pose["render_path"],
            "training_seconds": float(pose["training_seconds"]),
        }
        for pose in sorted(
            manifest.get("poses", []),
            key=lambda item: int(item["pose_index"]),
        )
    ]
    return {
        "format_version": 1,
        "status": status,
        "scene_name": manifest["scene_name"],
        "base_iteration": int(manifest["base_iteration"]),
        "fine_tune_steps": int(manifest["fine_tune_steps"]),
        "render_dir": manifest["render_dir"],
        "total_pose_count": int(manifest["test_pose_count"]),
        "source_test_pose_count": int(manifest["source_test_pose_count"]),
        "completed_pose_count": len(completed_poses),
        "completed_pose_indices": [
            pose["pose_index"] for pose in completed_poses
        ],
        "completed_images": [
            pose["test_image_name"] for pose in completed_poses
        ],
        "current_pose": current_pose,
        "completed_poses": completed_poses,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_progress(
    progress_paths: list[Path],
    manifest: dict[str, Any],
    status: str,
    current_pose: dict[str, Any] | None = None,
) -> None:
    """Cap nhat dong thoi file tien do trong model root va PNG root."""
    payload = build_progress_payload(manifest, status, current_pose)
    for progress_path in progress_paths:
        save_json(progress_path, payload)


def valid_completed_pose_result(
    pose_result: dict[str, Any],
    row: dict[str, str],
    pose_index: int,
    render_path: str | Path,
    fine_tune_steps: int,
    save_pose_models: bool,
) -> bool:
    """Chi resume pose khi manifest, PNG va model tuy chon deu day du."""
    render_path = Path(render_path)
    if int(pose_result.get("pose_index", -1)) != int(pose_index):
        return False
    if pose_result.get("test_image_name") != row["image_name"]:
        return False
    if int(pose_result.get("optimizer_updates", -1)) != int(fine_tune_steps):
        return False
    if not render_path.is_file():
        return False
    expected_size = (
        int(float(row["width"])),
        int(float(row["height"])),
    )
    try:
        with Image.open(render_path) as image:
            if image.size != expected_size:
                return False
            image.verify()
    except (OSError, ValueError):
        return False
    if save_pose_models:
        point_cloud_path = str(pose_result.get("point_cloud_path", "")).strip()
        if not point_cloud_path or not Path(point_cloud_path).is_file():
            return False
    return True
