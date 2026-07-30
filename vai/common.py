"""Cac ham dung chung cho preprocess, render, evaluate va package VAI."""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from vai import VAI_METADATA_FILENAME


REQUIRED_POSE_COLUMNS = {
    "image_name",
    "qw",
    "qx",
    "qy",
    "qz",
    "tx",
    "ty",
    "tz",
    "fx",
    "fy",
    "cx",
    "cy",
    "width",
    "height",
}


def read_pose_rows(csv_path: str | Path) -> list[dict[str, str]]:
    """Doc va kiem tra danh sach pose test cua mot scene."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Khong tim thay test pose: {csv_path}")
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_POSE_COLUMNS - columns)
        if missing_columns:
            raise ValueError(
                "test_poses.csv thieu cot: {}".format(", ".join(missing_columns))
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"test_poses.csv khong co pose: {csv_path}")
    return rows


def slice_pose_rows(
    pose_rows: list[dict[str, str]],
    start_index: int = 0,
    pose_count: int = -1,
) -> list[dict[str, str]]:
    """Lay mot doan pose lien tiep, giu nguyen thu tu trong CSV."""
    start_index = int(start_index)
    pose_count = int(pose_count)
    if start_index < 0:
        raise ValueError("pose_start_index phai khong am")
    if pose_count == 0 or pose_count < -1:
        raise ValueError("pose_count phai la -1 hoac so nguyen duong")
    if start_index >= len(pose_rows):
        raise ValueError(
            "pose_start_index={} nam ngoai {} pose".format(
                start_index,
                len(pose_rows),
            )
        )
    end_index = len(pose_rows) if pose_count == -1 else start_index + pose_count
    selected_rows = pose_rows[start_index:min(end_index, len(pose_rows))]
    if not selected_rows:
        raise ValueError("Khong co test pose nao trong khoang da chon")
    return selected_rows


def normalize_output_extension(value: str) -> str:
    """Chuan hoa lua chon duoi anh thanh '.png' hoac 'csv'."""
    normalized = str(value).strip().lower()
    if normalized in {"csv", "original", "keep"}:
        return "csv"
    if not normalized.startswith("."):
        normalized = "." + normalized
    if normalized != ".png":
        raise ValueError("VAI chi ho tro output_extension=png hoac csv")
    return normalized


def output_name_for_pose(image_name: str, output_extension: str) -> str:
    """Tao ten file render tu ten anh trong CSV."""
    source_name = Path(image_name).name
    extension = normalize_output_extension(output_extension)
    if extension == "csv":
        return source_name
    return str(Path(source_name).with_suffix(extension))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Ghi JSON nguyen tu de khong hong file neu tien trinh bi ngat."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".{}.".format(path.name),
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_vai_metadata(scene_path: str | Path) -> dict[str, Any]:
    """Doc metadata distortion da tao trong buoc preprocess."""
    metadata_path = Path(scene_path) / VAI_METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Khong tim thay {VAI_METADATA_FILENAME} trong scene da preprocess: {scene_path}"
        )
    with open(metadata_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("format_version", 0)) != 1:
        raise ValueError(f"Phien ban VAI metadata khong duoc ho tro: {metadata_path}")
    return payload


def output_camera_from_metadata(
    metadata: dict[str, Any],
) -> dict[str, float | int]:
    """Doc camera output goc va quy pinhole ve radial_k bang 0."""
    camera = metadata.get("original_camera", {})
    params = camera.get("params", [])
    camera_model = camera.get("model")
    if camera_model == "SIMPLE_RADIAL" and len(params) == 4:
        focal, cx, cy, radial_k = [float(value) for value in params]
    elif camera_model == "SIMPLE_PINHOLE" and len(params) == 3:
        focal, cx, cy = [float(value) for value in params]
        radial_k = 0.0
    else:
        raise ValueError(
            "VAI metadata khong chua camera SIMPLE_RADIAL/SIMPLE_PINHOLE hop le"
        )
    return {
        "focal": focal,
        "cx": cx,
        "cy": cy,
        "radial_k": radial_k,
        "width": int(camera["width"]),
        "height": int(camera["height"]),
    }


def camera_to_dict(camera: Any) -> dict[str, Any]:
    """Chuyen camera COLMAP thanh JSON metadata gon nhe."""
    return {
        "id": int(camera.id),
        "model": str(camera.model),
        "width": int(camera.width),
        "height": int(camera.height),
        "params": [float(value) for value in camera.params],
    }
