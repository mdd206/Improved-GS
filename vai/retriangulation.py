"""Tai tam giac hoa sparse point voi pose COLMAP duoc giu co dinh."""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from vai.colmap_io import (
    CAMERA_MODEL_NAMES,
    Camera,
    Point3D,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3d_binary,
    write_point_cloud_ply,
)


def colmap_environment() -> dict[str, str]:
    """Tao environment Qt headless cho COLMAP tren Kaggle."""
    environment = os.environ.copy()
    if not environment.get("DISPLAY"):
        environment["QT_QPA_PLATFORM"] = "offscreen"
    return environment


def _run_colmap(command: list[str], stage: str) -> None:
    """Chay mot buoc COLMAP va gom loi thanh thong bao de doc."""
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=colmap_environment(),
        )
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(f"COLMAP {stage} that bai:\n{details}") from error


def _database_image_rows(database_path: Path) -> list[tuple[int, str, int]]:
    """Doc image id, ten va camera id do feature extractor tao."""
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
        ).fetchall()
    return [(int(image_id), str(name), int(camera_id)) for image_id, name, camera_id in rows]


def _write_fixed_pose_text_model(
    model_path: Path,
    camera: Camera,
    source_images: dict[int, Any],
    database_images: list[tuple[int, str, int]],
) -> tuple[dict[str, Any], set[int]]:
    """Tao model text rong point nhung giu nguyen pose va intrinsic goc."""
    if not database_images:
        raise ValueError("COLMAP feature extractor khong tao image nao")
    database_camera_ids = {row[2] for row in database_images}
    if len(database_camera_ids) != 1:
        raise ValueError(
            f"P1 yeu cau mot camera trong database, nhan duoc {len(database_camera_ids)}"
        )
    database_camera_id = next(iter(database_camera_ids))
    source_by_name = {image.name: image for image in source_images.values()}
    source_by_stem: dict[str, Any] = {}
    for image in source_images.values():
        stem = Path(image.name).stem
        if stem in source_by_stem:
            raise ValueError(f"Trung stem image trong sparse model: {stem}")
        source_by_stem[stem] = image

    matched_source_images: dict[str, Any] = {}
    missing: list[str] = []
    for _, name, _ in database_images:
        source_image = source_by_name.get(name) or source_by_stem.get(Path(name).stem)
        if source_image is None:
            missing.append(name)
        else:
            matched_source_images[name] = source_image
    if missing:
        raise ValueError(
            "Anh feature khong co pose co dinh trong images.bin: {}".format(missing[:5])
        )

    model_path.mkdir(parents=True, exist_ok=True)
    camera_params = " ".join(f"{float(value):.17g}" for value in camera.params)
    (model_path / "cameras.txt").write_text(
        (
            "# Camera list with one line of data per camera:\n"
            "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
            f"{database_camera_id} {camera.model} {camera.width} {camera.height} "
            f"{camera_params}\n"
        ),
        encoding="utf-8",
    )
    image_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for database_image_id, name, _ in database_images:
        source_image = matched_source_images[name]
        pose = [*source_image.qvec, *source_image.tvec]
        pose_text = " ".join(f"{float(value):.17g}" for value in pose)
        image_lines.extend(
            [
                f"{database_image_id} {pose_text} {database_camera_id} {name}",
                "",
            ]
        )
    (model_path / "images.txt").write_text(
        "\n".join(image_lines) + "\n",
        encoding="utf-8",
    )
    (model_path / "points3D.txt").write_text("", encoding="utf-8")
    train_source_ids = {
        int(matched_source_images[name].id)
        for _, name, _ in database_images
    }
    return matched_source_images, train_source_ids


def _pose_delta(
    output_images: dict[int, Any],
    source_by_database_name: dict[str, Any],
) -> float:
    """Do sai lech lon nhat cua pose, chap nhan quaternion doi dau."""
    if len(output_images) != len(source_by_database_name):
        raise ValueError(
            "So image sau triangulate khong khop: output={} input={}".format(
                len(output_images), len(source_by_database_name)
            )
        )
    max_delta = 0.0
    for image in output_images.values():
        if image.name not in source_by_database_name:
            raise ValueError(f"COLMAP triangulate sinh image la: {image.name}")
        source = source_by_database_name[image.name]
        quaternion_delta = min(
            float(np.max(np.abs(image.qvec - source.qvec))),
            float(np.max(np.abs(image.qvec + source.qvec))),
        )
        translation_delta = float(np.max(np.abs(image.tvec - source.tvec)))
        max_delta = max(max_delta, quaternion_delta, translation_delta)
    return max_delta


def _candidate_score(
    index: int,
    original_count: int,
    supports: np.ndarray,
    errors: np.ndarray,
) -> tuple[int, int, float, int]:
    """Xep original manh truoc point moi, point moi truoc original yeu."""
    if index < original_count:
        priority = 3 if int(supports[index]) >= 2 else 1
    else:
        priority = 2
    error = float(errors[index])
    finite_error = error if np.isfinite(error) else float("inf")
    return priority, int(supports[index]), -finite_error, -index


def merge_sparse_points(
    original_points: dict[int, Point3D],
    triangulated_points: dict[int, Point3D],
    train_source_image_ids: set[int],
    max_reprojection_error: float = 2.5,
    min_track_length: int = 2,
    voxel_divisor: float = 6_000.0,
    max_points: int = 600_000,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Loc point moi va hop nhat theo voxel ma khong mat vung original yeu."""
    if not original_points:
        raise ValueError("P1 can point cloud goc de xac dinh scale va giu coverage")
    if float(max_reprojection_error) <= 0.0:
        raise ValueError("retriangulation_max_reproj_error phai duong")
    if int(min_track_length) < 2:
        raise ValueError("retriangulation_min_track_length phai it nhat 2")
    if float(voxel_divisor) <= 0.0:
        raise ValueError("retriangulation_voxel_divisor phai duong")

    original = [original_points[point_id] for point_id in sorted(original_points)]
    accepted_new = [
        triangulated_points[point_id]
        for point_id in sorted(triangulated_points)
        if len(triangulated_points[point_id].image_ids) >= int(min_track_length)
        and float(triangulated_points[point_id].error) <= float(max_reprojection_error)
    ]
    original_xyz = np.stack([point.xyz for point in original]).astype(np.float64)
    original_rgb = np.stack([point.rgb for point in original]).astype(np.uint8)
    original_supports = np.asarray(
        [
            sum(int(image_id) in train_source_image_ids for image_id in point.image_ids)
            for point in original
        ],
        dtype=np.int32,
    )
    original_errors = np.asarray([point.error for point in original], dtype=np.float64)

    if accepted_new:
        new_xyz = np.stack([point.xyz for point in accepted_new]).astype(np.float64)
        new_rgb = np.stack([point.rgb for point in accepted_new]).astype(np.uint8)
        new_supports = np.asarray(
            [len(point.image_ids) for point in accepted_new],
            dtype=np.int32,
        )
        new_errors = np.asarray([point.error for point in accepted_new], dtype=np.float64)
        xyz = np.concatenate([original_xyz, new_xyz], axis=0)
        rgb = np.concatenate([original_rgb, new_rgb], axis=0)
        supports = np.concatenate([original_supports, new_supports], axis=0)
        errors = np.concatenate([original_errors, new_errors], axis=0)
    else:
        xyz = original_xyz
        rgb = original_rgb
        supports = original_supports
        errors = original_errors

    robust_min = np.quantile(original_xyz, 0.01, axis=0)
    robust_max = np.quantile(original_xyz, 0.99, axis=0)
    robust_diagonal = float(np.linalg.norm(robust_max - robust_min))
    if robust_diagonal <= 0.0:
        robust_diagonal = float(np.linalg.norm(np.ptp(original_xyz, axis=0)))
    if robust_diagonal <= 0.0:
        raise ValueError("Point cloud goc khong co kich thuoc hinh hoc")
    voxel_size = robust_diagonal / float(voxel_divisor)
    voxel_indices = np.floor((xyz - robust_min) / voxel_size).astype(np.int64)

    original_count = len(original)
    selected: dict[tuple[int, int, int], int] = {}
    original_voxels: set[tuple[int, int, int]] = set()
    for index, voxel in enumerate(voxel_indices):
        key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
        if index < original_count:
            original_voxels.add(key)
        current = selected.get(key)
        if current is None or _candidate_score(
            index, original_count, supports, errors
        ) > _candidate_score(current, original_count, supports, errors):
            selected[key] = index

    mandatory_keys = sorted(original_voxels)
    added_keys = sorted(set(selected) - original_voxels)
    if int(max_points) > 0:
        if len(mandatory_keys) > int(max_points):
            raise ValueError(
                "retriangulation_max_points nho hon so voxel original: "
                f"{max_points} < {len(mandatory_keys)}"
            )
        available_new = int(max_points) - len(mandatory_keys)
        added_keys = sorted(
            added_keys,
            key=lambda key: _candidate_score(
                selected[key], original_count, supports, errors
            ),
            reverse=True,
        )[:available_new]
    kept_keys = mandatory_keys + added_keys
    kept_indices = np.asarray([selected[key] for key in kept_keys], dtype=np.int64)
    replaced_weak = sum(selected[key] >= original_count for key in mandatory_keys)
    added_count = len(added_keys)
    growth_ratio = added_count / max(len(original_voxels), 1)
    accepted_errors = np.asarray([point.error for point in accepted_new], dtype=np.float64)
    stats: dict[str, Any] = {
        "original_points": original_count,
        "original_voxel_points": len(original_voxels),
        "original_support_0": int(np.count_nonzero(original_supports == 0)),
        "original_support_1": int(np.count_nonzero(original_supports == 1)),
        "original_support_2_plus": int(np.count_nonzero(original_supports >= 2)),
        "triangulated_points": len(triangulated_points),
        "accepted_triangulated_points": len(accepted_new),
        "replaced_weak_original_voxels": int(replaced_weak),
        "added_voxel_points": int(added_count),
        "merged_points": int(len(kept_indices)),
        "growth_ratio": float(growth_ratio),
        "robust_scene_diagonal": robust_diagonal,
        "voxel_size": float(voxel_size),
        "max_reprojection_error": float(max_reprojection_error),
        "min_track_length": int(min_track_length),
        "voxel_divisor": float(voxel_divisor),
        "max_points": int(max_points),
        "accepted_error_median": (
            float(np.median(accepted_errors)) if len(accepted_errors) else None
        ),
        "accepted_error_p95": (
            float(np.quantile(accepted_errors, 0.95)) if len(accepted_errors) else None
        ),
    }
    return xyz[kept_indices], rgb[kept_indices], stats


def build_fixed_pose_point_cloud(
    image_dir: str | Path,
    sparse_dir: str | Path,
    output_ply: str | Path,
    colmap_executable: str = "colmap",
    max_reprojection_error: float = 2.5,
    min_track_length: int = 2,
    voxel_divisor: float = 6_000.0,
    max_points: int = 600_000,
    min_growth_ratio: float = 0.0,
) -> dict[str, Any]:
    """Triangulate lai tu train view va ghi PLY hop nhat cho khoi tao Gaussian."""
    image_dir = Path(image_dir)
    sparse_dir = Path(sparse_dir)
    output_ply = Path(output_ply)
    camera_values = read_intrinsics_binary(sparse_dir / "cameras.bin")
    if len(camera_values) != 1:
        raise ValueError(f"P1 yeu cau mot camera, nhan duoc {len(camera_values)}")
    camera = next(iter(camera_values.values()))
    if camera.model != "SIMPLE_RADIAL":
        raise ValueError(f"P1 phai chay truoc undistort, nhan duoc {camera.model}")
    if camera.model not in CAMERA_MODEL_NAMES:
        raise ValueError(f"Camera model khong duoc ho tro: {camera.model}")
    source_images = read_extrinsics_binary(sparse_dir / "images.bin")
    original_points = read_points3d_binary(sparse_dir / "points3D.bin")

    temp_root = Path(tempfile.mkdtemp(prefix="vai-retriangulate-"))
    try:
        database_path = temp_root / "database.db"
        camera_params = ",".join(f"{float(value):.17g}" for value in camera.params)
        _run_colmap(
            [
                colmap_executable,
                "feature_extractor",
                "--database_path",
                str(database_path),
                "--image_path",
                str(image_dir),
                "--ImageReader.single_camera",
                "1",
                "--ImageReader.camera_model",
                camera.model,
                "--ImageReader.camera_params",
                camera_params,
            ],
            "feature_extractor",
        )
        database_images = _database_image_rows(database_path)
        fixed_model = temp_root / "fixed_model"
        matched_images, train_source_ids = _write_fixed_pose_text_model(
            fixed_model,
            camera,
            source_images,
            database_images,
        )
        _run_colmap(
            [
                colmap_executable,
                "exhaustive_matcher",
                "--database_path",
                str(database_path),
            ],
            "exhaustive_matcher",
        )
        triangulated_model = temp_root / "triangulated_model"
        triangulated_model.mkdir()
        point_command = [
            colmap_executable,
            "point_triangulator",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--input_path",
            str(fixed_model),
            "--output_path",
            str(triangulated_model),
        ]
        help_result = subprocess.run(
            [colmap_executable, "point_triangulator", "-h"],
            check=False,
            capture_output=True,
            text=True,
            env=colmap_environment(),
        )
        help_text = (help_result.stdout or "") + (help_result.stderr or "")
        if "--Mapper.tri_ignore_two_view_tracks" in help_text:
            point_command.extend(["--Mapper.tri_ignore_two_view_tracks", "0"])
        _run_colmap(point_command, "point_triangulator")

        output_images = read_extrinsics_binary(triangulated_model / "images.bin")
        max_pose_delta = _pose_delta(output_images, matched_images)
        if max_pose_delta > 1e-8:
            raise ValueError(
                "COLMAP da thay doi pose trong P1: max delta={:.3e}".format(max_pose_delta)
            )
        output_cameras = read_intrinsics_binary(triangulated_model / "cameras.bin")
        if len(output_cameras) != 1:
            raise ValueError("Model triangulate khong con dung mot camera")
        output_camera = next(iter(output_cameras.values()))
        intrinsic_delta = float(
            np.max(np.abs(np.asarray(output_camera.params) - np.asarray(camera.params)))
        )
        if output_camera.model != camera.model or intrinsic_delta > 1e-10:
            raise ValueError(
                "COLMAP da thay doi intrinsic trong P1: model={} delta={:.3e}".format(
                    output_camera.model,
                    intrinsic_delta,
                )
            )

        triangulated_points = read_points3d_binary(
            triangulated_model / "points3D.bin"
        )
        xyz, rgb, stats = merge_sparse_points(
            original_points,
            triangulated_points,
            train_source_ids,
            max_reprojection_error=max_reprojection_error,
            min_track_length=min_track_length,
            voxel_divisor=voxel_divisor,
            max_points=max_points,
        )
        stats.update(
            {
                "enabled": True,
                "poses_fixed": True,
                "matching": "exhaustive",
                "database_images": len(database_images),
                "max_pose_delta": float(max_pose_delta),
                "max_intrinsic_delta": float(intrinsic_delta),
                "min_growth_ratio": float(min_growth_ratio),
            }
        )
        if stats["growth_ratio"] < float(min_growth_ratio):
            raise RuntimeError(
                "P1 dung som: point moi chi tang {:.1%}, thap hon nguong {:.1%}".format(
                    stats["growth_ratio"],
                    float(min_growth_ratio),
                )
            )
        output_ply.parent.mkdir(parents=True, exist_ok=True)
        write_point_cloud_ply(output_ply, xyz, rgb)
        return stats
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
