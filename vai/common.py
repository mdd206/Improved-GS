"""Cac ham dung chung cho preprocess, render, evaluate va package VAI."""
from __future__ import annotations

import csv
import json
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
    """Ghi JSON UTF-8 voi thu muc cha duoc tao san."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


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


def camera_to_dict(camera: Any) -> dict[str, Any]:
    """Chuyen camera COLMAP thanh JSON metadata gon nhe."""
    return {
        "id": int(camera.id),
        "model": str(camera.model),
        "width": int(camera.width),
        "height": int(camera.height),
        "params": [float(value) for value in camera.params],
    }
