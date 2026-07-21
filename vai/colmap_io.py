"""Ham doc ghi COLMAP toi thieu cho preprocess VAI."""
from __future__ import annotations

import collections
import struct
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple(
    "Image",
    ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"],
)
CAMERA_MODELS = {
    CameraModel(0, "SIMPLE_PINHOLE", 3),
    CameraModel(1, "PINHOLE", 4),
    CameraModel(2, "SIMPLE_RADIAL", 4),
    CameraModel(3, "RADIAL", 5),
    CameraModel(4, "OPENCV", 8),
    CameraModel(5, "OPENCV_FISHEYE", 8),
    CameraModel(6, "FULL_OPENCV", 12),
    CameraModel(7, "FOV", 5),
    CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    CameraModel(9, "RADIAL_FISHEYE", 5),
    CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODEL_IDS = {model.model_id: model for model in CAMERA_MODELS}
CAMERA_MODEL_NAMES = {model.model_name: model for model in CAMERA_MODELS}


def _read_bytes(
    handle: BinaryIO,
    num_bytes: int,
    format_text: str,
) -> tuple[Any, ...]:
    """Doc va giai nen mot nhom byte little-endian."""
    data = handle.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("File COLMAP bi cat ngan")
    return struct.unpack("<" + format_text, data)


def _write_bytes(handle: Any, values: Any, format_text: str) -> None:
    """Dong goi va ghi mot nhom gia tri theo dinh dang nhi phan COLMAP."""
    if isinstance(values, (list, tuple)):
        data = struct.pack("<" + format_text, *values)
    else:
        data = struct.pack("<" + format_text, values)
    handle.write(data)


def write_extrinsics_binary(images: dict[int, Any], path: str | Path) -> None:
    """Ghi cac ban ghi Image vao images.bin ma khong thay doi pose va track."""
    with open(path, "wb") as handle:
        _write_bytes(handle, len(images), "Q")
        for image_id in sorted(images):
            image = images[image_id]
            properties = [
                int(image.id),
                *[float(value) for value in image.qvec],
                *[float(value) for value in image.tvec],
                int(image.camera_id),
            ]
            _write_bytes(handle, properties, "idddddddi")
            handle.write(image.name.encode("utf-8") + b"\x00")
            _write_bytes(handle, len(image.point3D_ids), "Q")
            for xy, point3d_id in zip(image.xys, image.point3D_ids):
                _write_bytes(
                    handle,
                    [float(xy[0]), float(xy[1]), int(point3d_id)],
                    "ddq",
                )


def read_intrinsics_binary(path: str | Path) -> dict[int, Camera]:
    """Doc cameras.bin ma khong import toan bo package scene."""
    cameras: dict[int, Camera] = {}
    with open(path, "rb") as handle:
        camera_count = _read_bytes(handle, 8, "Q")[0]
        for _ in range(camera_count):
            camera_id, model_id, width, height = _read_bytes(handle, 24, "iiQQ")
            if model_id not in CAMERA_MODEL_IDS:
                raise ValueError(f"COLMAP camera model id khong duoc ho tro: {model_id}")
            model = CAMERA_MODEL_IDS[model_id]
            params = _read_bytes(handle, 8 * model.num_params, "d" * model.num_params)
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model.model_name,
                width=width,
                height=height,
                params=np.asarray(params, dtype=np.float64),
            )
    return cameras


def write_intrinsics_binary(cameras: dict[int, Camera], path: str | Path) -> None:
    """Ghi cameras.bin de ho tro kiem thu va chuyen doi model nho."""
    with open(path, "wb") as handle:
        _write_bytes(handle, len(cameras), "Q")
        for camera_id in sorted(cameras):
            camera = cameras[camera_id]
            if camera.model not in CAMERA_MODEL_NAMES:
                raise ValueError(f"COLMAP camera model khong duoc ho tro: {camera.model}")
            model = CAMERA_MODEL_NAMES[camera.model]
            _write_bytes(
                handle,
                [int(camera.id), int(model.model_id), int(camera.width), int(camera.height)],
                "iiQQ",
            )
            for value in camera.params:
                _write_bytes(handle, float(value), "d")


def read_extrinsics_binary(path: str | Path) -> dict[int, Image]:
    """Doc images.bin ma khong can plyfile hoac CUDA."""
    images: dict[int, Image] = {}
    with open(path, "rb") as handle:
        image_count = _read_bytes(handle, 8, "Q")[0]
        for _ in range(image_count):
            properties = _read_bytes(handle, 64, "idddddddi")
            image_id = int(properties[0])
            qvec = np.asarray(properties[1:5], dtype=np.float64)
            tvec = np.asarray(properties[5:8], dtype=np.float64)
            camera_id = int(properties[8])
            name_bytes = bytearray()
            while True:
                current = _read_bytes(handle, 1, "c")[0]
                if current == b"\x00":
                    break
                name_bytes.extend(current)
            point_count = _read_bytes(handle, 8, "Q")[0]
            values = _read_bytes(handle, 24 * point_count, "ddq" * point_count)
            xys = np.column_stack(
                [
                    np.asarray(values[0::3], dtype=np.float64),
                    np.asarray(values[1::3], dtype=np.float64),
                ]
            )
            point3d_ids = np.asarray(values[2::3], dtype=np.int64)
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name_bytes.decode("utf-8"),
                xys=xys,
                point3D_ids=point3d_ids,
            )
    return images
