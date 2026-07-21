"""Kiem tra va dong goi submission VAI theo test_poses.csv."""
from __future__ import annotations

import zipfile
from pathlib import Path

from PIL import Image

from vai.common import output_name_for_pose, read_pose_rows


def discover_pose_files(
    phase_dir: str | Path,
    set_name: str,
    subset: list[str] | None = None,
) -> list[tuple[str, Path]]:
    """Tim test_poses.csv cua cac scene trong public hoac private set."""
    set_root = Path(phase_dir) / set_name
    if not set_root.is_dir():
        raise FileNotFoundError(f"Khong tim thay VAI set: {set_root}")
    requested = set(subset or [])
    scenes = []
    for scene_dir in sorted(path for path in set_root.iterdir() if path.is_dir()):
        pose_path = scene_dir / "test" / "test_poses.csv"
        if pose_path.is_file() and (not requested or scene_dir.name in requested):
            scenes.append((scene_dir.name, pose_path))
    found = {name for name, _ in scenes}
    missing = sorted(requested - found)
    if missing:
        raise ValueError("Khong tim thay test pose cho scene: {}".format(", ".join(missing)))
    if not scenes:
        raise ValueError(f"Khong co scene VAI de dong goi trong {set_root}")
    return scenes


def validate_submission_scene(
    submission_root: str | Path,
    scene_name: str,
    pose_path: str | Path,
    output_extension: str = "csv",
    allow_extra: bool = False,
) -> list[Path]:
    """Kiem tra ten, so luong va kich thuoc anh cua mot scene."""
    scene_dir = Path(submission_root) / scene_name
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Thieu thu muc render scene: {scene_dir}")
    pose_rows = read_pose_rows(pose_path)
    expected_paths: list[Path] = []
    expected_names: set[str] = set()
    for row in pose_rows:
        output_name = output_name_for_pose(row["image_name"], output_extension)
        if output_name in expected_names:
            raise ValueError(f"Trung ten output trong CSV: {output_name}")
        expected_names.add(output_name)
        output_path = scene_dir / output_name
        if not output_path.is_file():
            raise FileNotFoundError(f"Thieu anh submission: {output_path}")
        expected_size = (int(float(row["width"])), int(float(row["height"])))
        with Image.open(output_path) as image:
            actual_size = image.size
        if actual_size != expected_size:
            raise ValueError(
                f"Sai kich thuoc {output_path}: got={actual_size} expected={expected_size}"
            )
        expected_paths.append(output_path)

    if not allow_extra:
        actual_names = {path.name for path in scene_dir.iterdir() if path.is_file()}
        extra_names = sorted(actual_names - expected_names)
        if extra_names:
            raise ValueError(f"Scene {scene_name} co file thua: {extra_names[:5]}")
    return expected_paths


def package_submission(
    phase_dir: str | Path,
    set_name: str,
    submission_root: str | Path,
    zip_path: str | Path,
    subset: list[str] | None = None,
    output_extension: str = "csv",
    allow_extra: bool = False,
) -> dict[str, int]:
    """Validate tat ca scene va chi ghi dung file mong doi vao ZIP."""
    scene_files: dict[str, list[Path]] = {}
    for scene_name, pose_path in discover_pose_files(phase_dir, set_name, subset):
        scene_files[scene_name] = validate_submission_scene(
            submission_root,
            scene_name,
            pose_path,
            output_extension,
            allow_extra,
        )

    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for scene_name, paths in scene_files.items():
            for path in paths:
                archive.write(path, f"{scene_name}/{path.name}")
    return {scene_name: len(paths) for scene_name, paths in scene_files.items()}
