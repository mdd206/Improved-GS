"""Kiem thu cac phan VAI khong can COLMAP CLI hoac CUDA rasterizer."""
from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image as PilImage
from PIL import JpegImagePlugin

_GAUSSIAN_IO_SPEC = importlib.util.spec_from_file_location(
    "vai_gaussian_model_io",
    Path(__file__).resolve().parents[1] / "scene" / "gaussian_model_io.py",
)
assert _GAUSSIAN_IO_SPEC is not None and _GAUSSIAN_IO_SPEC.loader is not None
_GAUSSIAN_IO_MODULE = importlib.util.module_from_spec(_GAUSSIAN_IO_SPEC)
sys.modules[_GAUSSIAN_IO_SPEC.name] = _GAUSSIAN_IO_MODULE
_GAUSSIAN_IO_SPEC.loader.exec_module(_GAUSSIAN_IO_MODULE)
GaussianModelIOMixin = _GAUSSIAN_IO_MODULE.GaussianModelIOMixin

_CAMERA_UTILS_SPEC = importlib.util.spec_from_file_location(
    "vai_test_camera_utils",
    Path(__file__).resolve().parents[1] / "utils" / "camera_utils.py",
)
assert _CAMERA_UTILS_SPEC is not None and _CAMERA_UTILS_SPEC.loader is not None
_CAMERA_UTILS_MODULE = importlib.util.module_from_spec(_CAMERA_UTILS_SPEC)
sys.modules[_CAMERA_UTILS_SPEC.name] = _CAMERA_UTILS_MODULE
_CAMERA_CLASS_STUB = SimpleNamespace(Camera=object)
with patch.dict(
    sys.modules,
    {
        "scene": SimpleNamespace(cameras=_CAMERA_CLASS_STUB),
        "scene.cameras": _CAMERA_CLASS_STUB,
    },
):
    _CAMERA_UTILS_SPEC.loader.exec_module(_CAMERA_UTILS_MODULE)
camera_to_JSON = _CAMERA_UTILS_MODULE.camera_to_JSON

from vai.colmap_io import (
    Camera,
    Image,
    Point3D,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3d_binary,
    write_extrinsics_binary,
    write_intrinsics_binary,
    write_points3d_binary,
)
from vai.common import output_name_for_pose, read_pose_rows
from vai.distortion import redistort_and_crop, redistort_image
from vai.evaluation import compute_weighted_score
from vai.image_processing import save_render_image, sharpen_image
from vai.packaging import package_submission
from vai.preprocessing import _synchronize_and_filter_images, preprocess_scene
from vai.retriangulation import (
    _resolve_colmap_option,
    _run_colmap,
    merge_sparse_points,
)
from utils.coarse_to_fine import (
    build_training_resolution_scales,
    resolve_training_resolution_scale,
    validate_coarse_to_fine_schedule,
)
from utils.pose_aware_sampling import (
    CameraPose,
    build_pose_sampling_plan,
    build_pose_sampling_plan_v1,
    build_repeated_camera_pool,
    pose_from_csv_row,
)
from utils.simple_radial import (
    project_simple_radial,
    simple_radial_projection_hessians,
    simple_radial_projection_jacobian,
)


_EDGE_MODULE_SPEC = importlib.util.spec_from_file_location(
    "vai_test_densification_methods",
    Path(__file__).resolve().parents[1] / "scene" / "methods" / "densification_methods.py",
)
if _EDGE_MODULE_SPEC is None or _EDGE_MODULE_SPEC.loader is None:
    raise RuntimeError("Khong nap duoc densification_methods.py")
_EDGE_MODULE = importlib.util.module_from_spec(_EDGE_MODULE_SPEC)
_EDGE_MODULE_SPEC.loader.exec_module(_EDGE_MODULE)
prepare_edge_maps = _EDGE_MODULE.prepare_edge_maps


POSE_COLUMNS = [
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
]


def make_colmap_image(image_id: int, name: str, point_count: int = 0) -> Image:
    """Tao ban ghi COLMAP nho cho round-trip test."""
    xys = np.arange(point_count * 2, dtype=np.float64).reshape(point_count, 2)
    point_ids = np.arange(point_count, dtype=np.int64) + 10
    return Image(
        id=image_id,
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.array([1.0, 2.0, 3.0]),
        camera_id=1,
        name=name,
        xys=xys,
        point3D_ids=point_ids,
    )


class VaiCommonTests(unittest.TestCase):
    def test_output_name_supports_png_and_csv_extension(self) -> None:
        self.assertEqual(output_name_for_pose("folder/IMAGE.JPG", "png"), "IMAGE.png")
        self.assertEqual(output_name_for_pose("folder/IMAGE.JPG", "csv"), "IMAGE.JPG")

    def test_weighted_score_matches_competition_formula(self) -> None:
        score, normalized = compute_weighted_score(0.8, 30.0, 0.2, 40.0)
        self.assertAlmostEqual(normalized, 0.75)
        self.assertAlmostEqual(score, 0.785)

    def test_camera_json_accepts_preload_camera_info_dimensions(self) -> None:
        camera_info = SimpleNamespace(
            R=np.eye(3, dtype=np.float64),
            T=np.zeros(3, dtype=np.float64),
            image_name="train.JPG",
            width=640,
            height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            camera_model="SIMPLE_RADIAL",
            radial_k=-0.01,
        )
        entry = camera_to_JSON(7, camera_info)
        self.assertEqual(entry["width"], 640)
        self.assertEqual(entry["height"], 480)
        self.assertEqual(entry["camera_model"], "SIMPLE_RADIAL")


class CoarseToFineScheduleTests(unittest.TestCase):
    def test_disabled_schedule_always_uses_full_resolution(self) -> None:
        opt = SimpleNamespace(coarse_to_fine=False)
        self.assertEqual(build_training_resolution_scales(opt), [1.0])
        self.assertEqual(resolve_training_resolution_scale(1, opt), 1.0)

    def test_enabled_schedule_changes_at_configured_iterations(self) -> None:
        opt = SimpleNamespace(
            coarse_to_fine=True,
            coarse_to_fine_middle_iter=2_000,
            coarse_to_fine_full_iter=5_000,
        )
        self.assertEqual(build_training_resolution_scales(opt), [4.0, 2.0, 1.0])
        self.assertEqual(resolve_training_resolution_scale(1, opt), 4.0)
        self.assertEqual(resolve_training_resolution_scale(1_999, opt), 4.0)
        self.assertEqual(resolve_training_resolution_scale(2_000, opt), 2.0)
        self.assertEqual(resolve_training_resolution_scale(4_999, opt), 2.0)
        self.assertEqual(resolve_training_resolution_scale(5_000, opt), 1.0)

    def test_enabled_schedule_rejects_invalid_iteration_order(self) -> None:
        opt = SimpleNamespace(
            coarse_to_fine=True,
            coarse_to_fine_middle_iter=5_000,
            coarse_to_fine_full_iter=2_000,
        )
        with self.assertRaises(ValueError):
            validate_coarse_to_fine_schedule(opt)


class PoseAwareSamplingTests(unittest.TestCase):
    def test_identity_csv_pose_converts_translation_to_camera_center(self) -> None:
        pose = pose_from_csv_row(
            {
                "qw": "1",
                "qx": "0",
                "qy": "0",
                "qz": "0",
                "tx": "1",
                "ty": "2",
                "tz": "3",
            }
        )
        np.testing.assert_allclose(pose.center, [-1.0, -2.0, -3.0])
        np.testing.assert_allclose(pose.forward, [0.0, 0.0, 1.0])

    def test_sparse_test_neighbor_gets_one_extra_slot_without_losing_coverage(self) -> None:
        cameras = [
            SimpleNamespace(uid=index, camera_center=np.array([x, 0.0, 0.0]), R=np.eye(3))
            for index, x in enumerate([0.0, 1.0, 2.0, 10.0])
        ]
        test_poses = [CameraPose(center=np.array([12.0, 0.0, 0.0]), forward=np.array([0.0, 0.0, 1.0]))]
        plan = build_pose_sampling_plan(
            cameras,
            test_poses,
            position_neighbor_count=1,
            direction_neighbor_count=1,
            direction_radius=3.0,
            extra_fraction=0.25,
            max_repeat=2,
        )

        self.assertEqual(plan.repeat_counts, {0: 1, 1: 1, 2: 1, 3: 2})
        self.assertEqual(plan.extra_count, 1)
        self.assertEqual(plan.pool_size, 5)
        self.assertAlmostEqual(plan.median_train_spacing, 1.0)
        self.assertAlmostEqual(plan.max_test_gap, 2.0)
        self.assertAlmostEqual(plan.max_test_angle_gap_degrees, 0.0)
        pool = build_repeated_camera_pool(cameras, plan.repeat_counts)
        self.assertEqual(len(pool), 5)
        self.assertTrue(all(any(item is camera for item in pool) for camera in cameras))
        self.assertEqual(sum(item is cameras[3] for item in pool), 2)

    def test_v1_uses_old_combined_position_and_angle_cost(self) -> None:
        cameras = [
            SimpleNamespace(uid=0, camera_center=np.array([0.0, 0.0, 0.0]), R=np.eye(3)),
            SimpleNamespace(
                uid=1,
                camera_center=np.array([0.1, 0.0, 0.0]),
                R=np.diag([-1.0, 1.0, -1.0]),
            ),
        ]
        test_poses = [
            CameraPose(
                center=np.array([0.08, 0.0, 0.0]),
                forward=np.array([0.0, 0.0, 1.0]),
            )
        ]
        plan = build_pose_sampling_plan_v1(
            cameras,
            test_poses,
            neighbor_count=1,
            extra_fraction=0.5,
            max_repeat=2,
            angle_weight=0.25,
        )

        self.assertEqual(plan.repeat_counts, {0: 2, 1: 1})

    def test_view_direction_can_override_a_small_position_advantage(self) -> None:
        cameras = [
            SimpleNamespace(uid=0, camera_center=np.array([0.0, 0.0, 0.0]), R=np.eye(3)),
            SimpleNamespace(
                uid=1,
                camera_center=np.array([0.1, 0.0, 0.0]),
                R=np.diag([-1.0, 1.0, -1.0]),
            ),
        ]
        test_poses = [CameraPose(center=np.array([0.08, 0.0, 0.0]), forward=np.array([0.0, 0.0, 1.0]))]
        plan = build_pose_sampling_plan(
            cameras,
            test_poses,
            position_neighbor_count=1,
            direction_neighbor_count=1,
            direction_radius=3.0,
            extra_fraction=0.5,
            max_repeat=2,
        )

        self.assertEqual(plan.repeat_counts, {0: 2, 1: 1})

    def test_direction_neighbor_stays_inside_position_radius(self) -> None:
        cameras = [
            SimpleNamespace(uid=0, camera_center=np.array([0.0, 0.0, 0.0]), R=np.eye(3)),
            SimpleNamespace(
                uid=1,
                camera_center=np.array([1.0, 0.0, 0.0]),
                R=np.diag([-1.0, 1.0, -1.0]),
            ),
            SimpleNamespace(
                uid=2,
                camera_center=np.array([2.0, 0.0, 0.0]),
                R=np.diag([-1.0, 1.0, -1.0]),
            ),
            SimpleNamespace(uid=3, camera_center=np.array([10.0, 0.0, 0.0]), R=np.eye(3)),
        ]
        test_poses = [CameraPose(center=np.array([1.1, 0.0, 0.0]), forward=np.array([0.0, 0.0, 1.0]))]
        plan = build_pose_sampling_plan(
            cameras,
            test_poses,
            position_neighbor_count=1,
            direction_neighbor_count=1,
            direction_radius=3.0,
            extra_fraction=0.5,
            max_repeat=2,
        )

        self.assertEqual(plan.repeat_counts, {0: 2, 1: 2, 2: 1, 3: 1})


class ColmapIoTests(unittest.TestCase):
    def test_images_binary_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "images.bin"
            source = {
                4: make_colmap_image(4, "a.JPG", 0),
                9: make_colmap_image(9, "b.JPG", 2),
            }
            write_extrinsics_binary(source, path)
            loaded = read_extrinsics_binary(path)
            self.assertEqual(set(loaded), {4, 9})
            self.assertEqual(loaded[4].name, "a.JPG")
            self.assertEqual(loaded[9].name, "b.JPG")
            np.testing.assert_allclose(loaded[9].xys, source[9].xys)
            np.testing.assert_array_equal(loaded[9].point3D_ids, source[9].point3D_ids)

    def test_filter_removes_missing_and_syncs_png_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sparse_dir = root / "sparse"
            image_dir = root / "images"
            sparse_dir.mkdir()
            image_dir.mkdir()
            PilImage.new("RGBA", (3, 2), (1, 2, 3, 255)).save(image_dir / "a.png")
            write_extrinsics_binary(
                {
                    1: make_colmap_image(1, "a.JPG"),
                    2: make_colmap_image(2, "missing.JPG"),
                },
                sparse_dir / "images.bin",
            )
            count = _synchronize_and_filter_images(sparse_dir, image_dir)
            loaded = read_extrinsics_binary(sparse_dir / "images.bin")
            self.assertEqual(count, 1)
            self.assertEqual(list(loaded.values())[0].name, "a.png")

    def test_points3d_binary_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "points3D.bin"
            source = {
                7: Point3D(
                    id=7,
                    xyz=np.array([1.0, 2.0, 3.0]),
                    rgb=np.array([10, 20, 30], dtype=np.uint8),
                    error=0.75,
                    image_ids=np.array([2, 9], dtype=np.int32),
                    point2D_idxs=np.array([3, 4], dtype=np.int32),
                )
            }
            write_points3d_binary(source, path)
            loaded = read_points3d_binary(path)
            self.assertEqual(set(loaded), {7})
            np.testing.assert_allclose(loaded[7].xyz, source[7].xyz)
            np.testing.assert_array_equal(loaded[7].rgb, source[7].rgb)
            np.testing.assert_array_equal(loaded[7].image_ids, source[7].image_ids)
            self.assertAlmostEqual(loaded[7].error, 0.75)


class RetriangulationTests(unittest.TestCase):
    def test_colmap_uses_qt_offscreen_without_display(self) -> None:
        with patch.dict("vai.retriangulation.os.environ", {}, clear=True), patch(
            "vai.retriangulation.subprocess.run"
        ) as run_mock:
            _run_colmap(["colmap", "feature_extractor"], "feature_extractor")

        self.assertEqual(
            run_mock.call_args.kwargs["env"]["QT_QPA_PLATFORM"],
            "offscreen",
        )

    def test_colmap_device_option_supports_legacy_and_new_names(self) -> None:
        with patch(
            "vai.retriangulation.subprocess.run",
            return_value=SimpleNamespace(
                stdout="--SiftExtraction.use_gpu arg (=1)",
                stderr="",
            ),
        ):
            option = _resolve_colmap_option(
                "colmap",
                "feature_extractor",
                ("--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu"),
            )

        self.assertEqual(option, "--SiftExtraction.use_gpu")

    def test_colmap_gpu_uses_xvfb_without_display(self) -> None:
        with patch.dict("vai.retriangulation.os.environ", {}, clear=True), patch(
            "vai.retriangulation.shutil.which",
            return_value="/usr/bin/xvfb-run",
        ), patch("vai.retriangulation.subprocess.run") as run_mock:
            _run_colmap(
                ["colmap", "feature_extractor"],
                "feature_extractor",
                use_gpu=True,
            )

        self.assertEqual(
            run_mock.call_args.args[0],
            ["/usr/bin/xvfb-run", "-a", "colmap", "feature_extractor"],
        )
        self.assertNotIn(
            "QT_QPA_PLATFORM",
            run_mock.call_args.kwargs["env"],
        )

    def test_colmap_gpu_requires_xvfb_without_display(self) -> None:
        with patch.dict("vai.retriangulation.os.environ", {}, clear=True), patch(
            "vai.retriangulation.shutil.which",
            return_value=None,
        ):
            with self.assertRaises(FileNotFoundError):
                _run_colmap(
                    ["colmap", "feature_extractor"],
                    "feature_extractor",
                    use_gpu=True,
                )

    def test_merge_prefers_strong_original_then_new_then_weak_original(self) -> None:
        original = {
            1: Point3D(
                id=1,
                xyz=np.array([0.0, 0.0, 0.0]),
                rgb=np.array([255, 0, 0], dtype=np.uint8),
                error=0.5,
                image_ids=np.array([1, 2], dtype=np.int32),
                point2D_idxs=np.array([0, 0], dtype=np.int32),
            ),
            2: Point3D(
                id=2,
                xyz=np.array([10.0, 0.0, 0.0]),
                rgb=np.array([0, 255, 0], dtype=np.uint8),
                error=1.5,
                image_ids=np.array([99], dtype=np.int32),
                point2D_idxs=np.array([0], dtype=np.int32),
            ),
        }
        triangulated = {
            10: Point3D(
                id=10,
                xyz=np.array([10.001, 0.0, 0.0]),
                rgb=np.array([0, 0, 255], dtype=np.uint8),
                error=0.4,
                image_ids=np.array([1, 2], dtype=np.int32),
                point2D_idxs=np.array([0, 0], dtype=np.int32),
            ),
            11: Point3D(
                id=11,
                xyz=np.array([5.0, 0.0, 0.0]),
                rgb=np.array([255, 255, 0], dtype=np.uint8),
                error=0.6,
                image_ids=np.array([1, 2, 3], dtype=np.int32),
                point2D_idxs=np.array([0, 0, 0], dtype=np.int32),
            ),
            12: Point3D(
                id=12,
                xyz=np.array([6.0, 0.0, 0.0]),
                rgb=np.array([255, 0, 255], dtype=np.uint8),
                error=3.0,
                image_ids=np.array([1, 2], dtype=np.int32),
                point2D_idxs=np.array([0, 0], dtype=np.int32),
            ),
        }
        xyz, rgb, stats = merge_sparse_points(
            original,
            triangulated,
            {1, 2, 3},
            max_reprojection_error=2.5,
            min_track_length=2,
            voxel_divisor=100.0,
            max_points=10,
        )

        self.assertEqual(len(xyz), 3)
        self.assertEqual(stats["accepted_triangulated_points"], 2)
        self.assertEqual(stats["replaced_weak_original_voxels"], 1)
        self.assertEqual(stats["added_voxel_points"], 1)
        self.assertAlmostEqual(stats["growth_ratio"], 0.5)
        colors = {tuple(color.tolist()) for color in rgb}
        self.assertIn((255, 0, 0), colors)
        self.assertIn((0, 0, 255), colors)
        self.assertIn((255, 255, 0), colors)
        self.assertNotIn((0, 255, 0), colors)


class PreprocessingTests(unittest.TestCase):
    def test_native_simple_radial_keeps_raw_rgb_and_skips_colmap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_scene = root / "raw" / "HCM0204"
            source_images = source_scene / "train" / "images"
            source_sparse = source_scene / "train" / "sparse" / "0"
            test_dir = source_scene / "test"
            source_images.mkdir(parents=True)
            source_sparse.mkdir(parents=True)
            test_dir.mkdir(parents=True)
            PilImage.new("RGB", (4, 3), (10, 20, 30)).save(source_images / "train.JPG")
            write_intrinsics_binary(
                {
                    1: Camera(
                        id=1,
                        model="SIMPLE_RADIAL",
                        width=4,
                        height=3,
                        params=np.array([10.0, 2.0, 1.5, 0.01]),
                    )
                },
                source_sparse / "cameras.bin",
            )
            write_extrinsics_binary(
                {
                    1: make_colmap_image(1, "train.JPG"),
                    2: make_colmap_image(2, "missing.JPG"),
                },
                source_sparse / "images.bin",
            )
            (source_sparse / "points3D.bin").write_bytes(b"\x00" * 8)
            with open(test_dir / "test_poses.csv", "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=POSE_COLUMNS)
                writer.writeheader()
                writer.writerow(
                    {
                        "image_name": "test.JPG",
                        "qw": "1",
                        "qx": "0",
                        "qy": "0",
                        "qz": "0",
                        "tx": "0",
                        "ty": "0",
                        "tz": "0",
                        "fx": "10",
                        "fy": "10",
                        "cx": "2",
                        "cy": "1.5",
                        "width": "4",
                        "height": "3",
                    }
                )

            output_root = root / "native"
            with patch(
                "vai.preprocessing._check_colmap_executable",
                side_effect=AssertionError("native D3 must not check COLMAP"),
            ):
                result = preprocess_scene(
                    source_scene,
                    output_root,
                    native_simple_radial=True,
                )

            output_scene = output_root / "HCM0204"
            self.assertTrue(result["native_simple_radial"])
            self.assertEqual(result["train_images"], 1)
            self.assertTrue((output_scene / "images" / "train.JPG").is_file())
            self.assertFalse((output_scene / "images" / "train.png").exists())
            with PilImage.open(output_scene / "images" / "train.JPG") as image:
                self.assertEqual(image.mode, "RGB")
            camera = next(
                iter(read_intrinsics_binary(output_scene / "sparse" / "0" / "cameras.bin").values())
            )
            self.assertEqual(camera.model, "SIMPLE_RADIAL")
            self.assertEqual(
                [image.name for image in read_extrinsics_binary(output_scene / "sparse" / "0" / "images.bin").values()],
                ["train.JPG"],
            )
            with open(output_scene / "vai_metadata.json", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertTrue(metadata["native_simple_radial"])
            self.assertFalse(metadata["undistort"]["enabled"])
            self.assertEqual(metadata["training_camera"]["model"], "SIMPLE_RADIAL")
            self.assertNotIn("undistorted_camera", metadata)

    def test_scene_is_normalized_for_improvedgs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_scene = root / "raw" / "HCM0204"
            source_images = source_scene / "train" / "images"
            source_sparse = source_scene / "train" / "sparse" / "0"
            test_dir = source_scene / "test"
            source_images.mkdir(parents=True)
            source_sparse.mkdir(parents=True)
            test_dir.mkdir(parents=True)
            PilImage.new("RGB", (4, 3), (10, 20, 30)).save(source_images / "train.JPG")
            write_intrinsics_binary(
                {
                    1: Camera(
                        id=1,
                        model="SIMPLE_RADIAL",
                        width=4,
                        height=3,
                        params=np.array([10.0, 2.0, 1.5, 0.01]),
                    )
                },
                source_sparse / "cameras.bin",
            )
            write_extrinsics_binary(
                {
                    1: make_colmap_image(1, "train.JPG"),
                    2: make_colmap_image(2, "missing.JPG"),
                },
                source_sparse / "images.bin",
            )
            (source_sparse / "points3D.bin").write_bytes(b"\x00" * 8)
            with open(test_dir / "test_poses.csv", "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=POSE_COLUMNS)
                writer.writeheader()
                writer.writerow(
                    {
                        "image_name": "test.JPG",
                        "qw": "1",
                        "qx": "0",
                        "qy": "0",
                        "qz": "0",
                        "tx": "0",
                        "ty": "0",
                        "tz": "0",
                        "fx": "10",
                        "fy": "10",
                        "cx": "2",
                        "cy": "1.5",
                        "width": "4",
                        "height": "3",
                    }
                )

            def fake_undistorter(
                executable: str,
                image_path: Path,
                sparse_path: Path,
                output_path: Path,
                blank_pixels: float,
                min_scale: float,
                max_scale: float,
            ) -> None:
                del executable, blank_pixels, min_scale, max_scale
                shutil.copytree(image_path, output_path / "images")
                shutil.copytree(sparse_path, output_path / "sparse")
                write_intrinsics_binary(
                    {
                        1: Camera(
                            id=1,
                            model="PINHOLE",
                            width=4,
                            height=3,
                            params=np.array([10.0, 10.0, 2.0, 1.5]),
                        )
                    },
                    output_path / "sparse" / "cameras.bin",
                )

            fixed_pose_calls: list[dict[str, object]] = []

            def fake_fixed_pose_cloud(**kwargs: object) -> dict[str, object]:
                fixed_pose_calls.append(kwargs)
                output_ply = Path(kwargs["output_ply"])
                output_ply.write_bytes(b"ply\n")
                return {
                    "enabled": True,
                    "poses_fixed": True,
                    "original_points": 1,
                    "triangulated_points": 2,
                    "accepted_triangulated_points": 2,
                    "merged_points": 2,
                    "growth_ratio": 1.0,
                }

            output_root = root / "cleaned"
            with patch("vai.preprocessing._check_colmap_executable"), patch(
                "vai.preprocessing._run_colmap_undistorter",
                side_effect=fake_undistorter,
            ), patch(
                "vai.preprocessing.build_fixed_pose_point_cloud",
                side_effect=fake_fixed_pose_cloud,
            ):
                result = preprocess_scene(
                    source_scene,
                    output_root,
                    fixed_pose_retriangulation=True,
                )

            output_scene = output_root / "HCM0204"
            self.assertEqual(result["scene_name"], "HCM0204")
            self.assertEqual(result["train_images"], 1)
            self.assertEqual(result["initial_points"], 2)
            self.assertEqual(fixed_pose_calls[0]["sift_device"], "gpu")
            self.assertTrue((output_scene / "images" / "train.png").is_file())
            self.assertTrue((output_scene / "sparse" / "0" / "cameras.bin").is_file())
            self.assertTrue((output_scene / "sparse" / "0" / "points3D.ply").is_file())
            with PilImage.open(output_scene / "images" / "train.png") as image:
                self.assertEqual(image.mode, "RGBA")
            with open(output_scene / "vai_metadata.json", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["original_camera"]["model"], "SIMPLE_RADIAL")
            self.assertEqual(metadata["undistorted_camera"]["model"], "PINHOLE")
            self.assertTrue(metadata["fixed_pose_retriangulation"]["poses_fixed"])


class GaussianModelIOTests(unittest.TestCase):
    @staticmethod
    def _read_binary_ply(path: Path) -> tuple[list[str], np.ndarray]:
        property_names: list[str] = []
        point_count = None
        with path.open("rb") as source:
            while True:
                raw_line = source.readline()
                if not raw_line:
                    raise AssertionError("PLY header khong co end_header")
                line = raw_line.decode("ascii").strip()
                if line.startswith("element vertex "):
                    point_count = int(line.rsplit(" ", 1)[1])
                elif line.startswith("property float "):
                    property_names.append(line.rsplit(" ", 1)[1])
                elif line == "end_header":
                    break
            if point_count is None:
                raise AssertionError("PLY header khong co vertex count")
            dtype = np.dtype([(name, "<f4") for name in property_names])
            vertices = np.fromfile(source, dtype=dtype, count=point_count)
        return property_names, vertices

    @staticmethod
    def _make_small_gaussian_model(point_count: int = 5) -> GaussianModelIOMixin:
        model = GaussianModelIOMixin.__new__(GaussianModelIOMixin)
        model._xyz = torch.arange(point_count * 3, dtype=torch.float32).reshape(point_count, 3)
        model._features_dc = (
            torch.arange(point_count * 3, dtype=torch.float32).reshape(point_count, 1, 3)
            + 100.0
        )
        model._features_rest = (
            torch.arange(point_count * 6, dtype=torch.float32).reshape(point_count, 2, 3)
            + 200.0
        )
        model._opacity = torch.arange(point_count, dtype=torch.float32).reshape(point_count, 1)
        model._scaling = (
            torch.arange(point_count * 3, dtype=torch.float32).reshape(point_count, 3)
            + 300.0
        )
        model._rotation = (
            torch.arange(point_count * 4, dtype=torch.float32).reshape(point_count, 4)
            + 400.0
        )
        return model

    def test_save_ply_writes_binary_chunks_with_bounded_rows(self) -> None:
        model = self._make_small_gaussian_model()
        concatenate_rows: list[int] = []
        original_concatenate = np.concatenate

        def record_concatenate(arrays: tuple[np.ndarray, ...], axis: int) -> np.ndarray:
            result = original_concatenate(arrays, axis=axis)
            concatenate_rows.append(int(result.shape[0]))
            return result

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "point_cloud.ply"
            with patch("vai_gaussian_model_io.PLY_WRITE_CHUNK_SIZE", 2), patch(
                "vai_gaussian_model_io.np.concatenate",
                side_effect=record_concatenate,
            ), patch("builtins.print"):
                model.save_ply(str(output_path))

            self.assertEqual(concatenate_rows, [2, 2, 1])
            self.assertTrue(output_path.is_file())
            self.assertFalse(Path(f"{output_path}.tmp").exists())
            property_names, vertices = self._read_binary_ply(output_path)
            self.assertEqual(len(vertices), 5)
            self.assertIn("f_rest_5", property_names)
            xyz = np.column_stack(
                (
                    np.asarray(vertices["x"]),
                    np.asarray(vertices["y"]),
                    np.asarray(vertices["z"]),
                )
            )
            np.testing.assert_array_equal(xyz, model._xyz.numpy())
            np.testing.assert_array_equal(
                np.asarray(vertices["f_rest_5"]),
                model._features_rest[:, 1, 2].numpy(),
            )

    def test_save_ply_removes_partial_file_after_error(self) -> None:
        model = self._make_small_gaussian_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "point_cloud.ply"
            with patch(
                "vai_gaussian_model_io.np.concatenate",
                side_effect=MemoryError("test"),
            ), patch("builtins.print"):
                with self.assertRaises(MemoryError):
                    model.save_ply(str(output_path))

            self.assertFalse(output_path.exists())
            self.assertFalse(Path(f"{output_path}.tmp").exists())

    def test_load_ply_uses_float32_and_restores_feature_layout(self) -> None:
        point_count = 2
        coefficient_count = 3
        columns = {
            "x": np.array([1.0, 2.0], dtype=np.float32),
            "y": np.array([3.0, 4.0], dtype=np.float32),
            "z": np.array([5.0, 6.0], dtype=np.float32),
            "opacity": np.array([0.1, 0.2], dtype=np.float32),
            **{
                f"f_dc_{index}": np.array([index, index + 0.5], dtype=np.float32)
                for index in range(3)
            },
            **{
                f"f_rest_{index}": np.array([index, index + 0.5], dtype=np.float32)
                for index in range(3 * coefficient_count)
            },
            **{
                f"scale_{index}": np.array([index + 10.0, index + 10.5], dtype=np.float32)
                for index in range(3)
            },
            **{
                f"rot_{index}": np.array([index + 20.0, index + 20.5], dtype=np.float32)
                for index in range(4)
            },
        }

        class FakeVertices:
            data = np.empty(point_count, dtype=np.float32)
            properties = [SimpleNamespace(name=name) for name in columns]

            def __getitem__(self, name: str) -> np.ndarray:
                return columns[name]

        class FakePlyData:
            @staticmethod
            def read(_path: str) -> SimpleNamespace:
                return SimpleNamespace(elements=[FakeVertices()])

        model = GaussianModelIOMixin.__new__(GaussianModelIOMixin)
        model.max_sh_degree = 1
        empty_dtypes: list[object] = []
        original_empty = np.empty

        def record_empty(shape: object, dtype: object = float, *args: object, **kwargs: object) -> np.ndarray:
            empty_dtypes.append(dtype)
            return original_empty(shape, dtype=dtype, *args, **kwargs)

        with patch.dict(sys.modules, {"plyfile": SimpleNamespace(PlyData=FakePlyData)}), patch.object(
            torch.Tensor,
            "to",
            lambda tensor, *args, **kwargs: tensor,
        ), patch("vai_gaussian_model_io.np.empty", side_effect=record_empty):
            model.load_ply("fake.ply")

        self.assertEqual(empty_dtypes, [np.float32] * 6)
        self.assertEqual(tuple(model._features_dc.shape), (point_count, 1, 3))
        self.assertEqual(
            tuple(model._features_rest.shape),
            (point_count, coefficient_count, 3),
        )
        self.assertEqual(model._features_rest.dtype, torch.float32)
        self.assertEqual(float(model._features_rest[1, 2, 1]), 5.5)
        self.assertEqual(model.active_sh_degree, model.max_sh_degree)


class DistortionTests(unittest.TestCase):
    def test_zero_distortion_is_identity(self) -> None:
        image = torch.rand((3, 5, 7), dtype=torch.float32)
        output = redistort_image(image, focal=20.0, cx=3.0, cy=2.0, radial_k=0.0)
        torch.testing.assert_close(output, image, atol=1e-6, rtol=1e-6)

    def test_crop_uses_principal_point_offset(self) -> None:
        image = torch.arange(3 * 8 * 10, dtype=torch.float32).reshape(3, 8, 10)
        output = redistort_and_crop(
            image,
            focal=20.0,
            render_cx=5.0,
            render_cy=4.0,
            radial_k=0.0,
            target_cx=3.0,
            target_cy=2.0,
            target_width=6,
            target_height=4,
        )
        torch.testing.assert_close(output, image[:, 2:6, 2:8], atol=1e-5, rtol=1e-5)

    def test_bicubic_interpolation_differs_from_bilinear(self) -> None:
        image = torch.zeros((3, 9, 9), dtype=torch.float32)
        image[:, 4, 4] = 1.0
        bicubic = redistort_image(
            image,
            focal=8.0,
            cx=4.0,
            cy=4.0,
            radial_k=0.1,
            interpolation="bicubic",
        )
        bilinear = redistort_image(
            image,
            focal=8.0,
            cx=4.0,
            cy=4.0,
            radial_k=0.1,
            interpolation="bilinear",
        )
        self.assertGreater(float((bicubic - bilinear).abs().sum().item()), 0.0)


class SimpleRadialProjectionTests(unittest.TestCase):
    def test_projection_uses_colmap_pixel_centers(self) -> None:
        point = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64)
        projected = project_simple_radial(
            point,
            focal_x=10.0,
            focal_y=12.0,
            cx=2.0,
            cy=1.5,
            radial_k=0.1,
        )
        torch.testing.assert_close(
            projected,
            torch.tensor([1.5, 1.0], dtype=torch.float64),
        )

    def test_analytic_jacobian_matches_autograd(self) -> None:
        point = torch.tensor([0.3, -0.2, 2.0], dtype=torch.float64, requires_grad=True)
        expected = torch.autograd.functional.jacobian(
            lambda value: project_simple_radial(
                value,
                focal_x=850.0,
                focal_y=830.0,
                cx=512.0,
                cy=384.0,
                radial_k=-0.07,
            ),
            point,
        )
        actual = simple_radial_projection_jacobian(
            point,
            focal_x=850.0,
            focal_y=830.0,
            radial_k=-0.07,
        )
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)

    def test_analytic_hessians_match_autograd(self) -> None:
        point = torch.tensor([0.3, -0.2, 2.0], dtype=torch.float64, requires_grad=True)
        expected = torch.stack(
            [
                torch.autograd.functional.hessian(
                    lambda value, axis=axis: project_simple_radial(
                        value,
                        focal_x=850.0,
                        focal_y=830.0,
                        cx=512.0,
                        cy=384.0,
                        radial_k=-0.07,
                    )[axis],
                    point,
                )
                for axis in range(2)
            ]
        )
        actual = simple_radial_projection_hessians(
            point,
            focal_x=850.0,
            focal_y=830.0,
            radial_k=-0.07,
        )
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


class ImageProcessingTests(unittest.TestCase):
    def test_sharpen_uses_requested_amount_and_sigma(self) -> None:
        image = torch.zeros((3, 9, 9), dtype=torch.float32)
        image[:, 4, 4] = 0.6
        sharpened = sharpen_image(image, amount=1.0, sigma=0.60)
        self.assertGreater(float((sharpened - image).abs().sum().item()), 0.0)
        self.assertGreaterEqual(float(sharpened.min().item()), 0.0)
        self.assertLessEqual(float(sharpened.max().item()), 1.0)

    def test_jpeg_is_saved_with_quality_and_subsampling(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "sample.JPG"
            image = torch.rand((3, 8, 8), dtype=torch.float32)
            with patch("vai.image_processing.Image.Image.save", autospec=True) as save_mock:
                save_render_image(
                    image,
                    output_path,
                    jpeg_quality=95,
                    jpeg_subsampling=2,
                )
                save_kwargs = save_mock.call_args.kwargs
                self.assertEqual(save_kwargs["format"], "JPEG")
                self.assertEqual(save_kwargs["quality"], 95)
                self.assertEqual(save_kwargs["subsampling"], 2)

            save_render_image(image, output_path, jpeg_quality=95, jpeg_subsampling=2)
            with PilImage.open(output_path) as saved_image:
                self.assertEqual(saved_image.format, "JPEG")
                self.assertEqual(JpegImagePlugin.get_sampling(saved_image), 2)

    def test_png_is_saved_losslessly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "sample.png"
            pixels = torch.tensor(
                [
                    [[0, 64], [128, 255]],
                    [[255, 128], [64, 0]],
                    [[32, 96], [160, 224]],
                ],
                dtype=torch.float32,
            )
            image = pixels / 255.0
            save_render_image(image, output_path)
            with PilImage.open(output_path) as saved_image:
                self.assertEqual(saved_image.format, "PNG")
                saved = np.asarray(saved_image)
            np.testing.assert_array_equal(saved, pixels.permute(1, 2, 0).numpy().astype(np.uint8))

    def test_hcm0204_notebook_owns_runtime_config(self) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "vai_hcm0204.ipynb"
        with open(notebook_path, encoding="utf-8") as handle:
            notebook = json.load(handle)
        code_cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        for index, source in enumerate(code_cells):
            compile(source, f"vai_hcm0204.ipynb:cell_{index}", "exec")
        config_cells = [source for source in code_cells if "VAI_CONFIG =" in source]
        self.assertEqual(len(config_cells), 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            work_root = Path(temp_dir)
            phase_dir = work_root / "phase1"
            public_pose = phase_dir / "public_set" / "HCM0204" / "test" / "test_poses.csv"
            public_pose.parent.mkdir(parents=True)
            public_pose.touch()
            namespace = {
                "PHASE_DIR": phase_dir,
                "WORK_ROOT": work_root,
                "json": json,
                "sys": sys,
            }
            with patch("builtins.print"):
                exec(config_cells[0], namespace)
            config = namespace["VAI_CONFIG"]
            runtime_path = namespace["RUNTIME_CONFIG_PATH"]
            with open(runtime_path, encoding="utf-8") as handle:
                saved_config = json.load(handle)
            self.assertEqual(saved_config, config)

            private_root = phase_dir / "private_set1"
            for scene_name in ("PRIVATE_A", "PRIVATE_B"):
                pose_path = private_root / scene_name / "test" / "test_poses.csv"
                pose_path.parent.mkdir(parents=True)
                pose_path.touch()
            private_source = config_cells[0].replace(
                "SET_NAME = 'public_set'",
                "SET_NAME = 'private_set1'",
            ).replace(
                "SCENE_NAMES = ['HCM0204']",
                "SCENE_NAMES = ['PRIVATE_A', 'PRIVATE_B']",
            )
            private_namespace = {
                "PHASE_DIR": phase_dir,
                "WORK_ROOT": work_root,
                "json": json,
                "sys": sys,
            }
            with patch("builtins.print"):
                exec(private_source, private_namespace)
            private_config = private_namespace["VAI_CONFIG"]
            self.assertEqual(
                [scene["name"] for scene in private_config["scenes"]],
                ["PRIVATE_A", "PRIVATE_B"],
            )
            self.assertFalse(private_config["postprocess_args"]["evaluate"])
            self.assertFalse(private_config["postprocess_args"]["require_gt"])
            self.assertIn("private_set1", private_config["data_root"])

        render_config = config["postprocess_args"]
        train_config = config["train_args"]
        self.assertEqual(train_config["iterations"], 30000)
        self.assertEqual(train_config["save_iterations"], [30000])
        self.assertEqual(train_config["position_lr_max_steps"], 30000)
        self.assertFalse(train_config["coarse_to_fine"])
        self.assertFalse(train_config["pose_aware_sampling"])
        self.assertEqual(train_config["densify_grad_threshold"], 0.00025)
        self.assertEqual(train_config["budget"], 5_500_000)
        self.assertIn("vai_mvsplat_init_native_simple_radial", config["data_root"])
        self.assertIn(
            "mvsplat_init_native_simple_radial_improvedgs_30k_5m5_dense00025",
            config["output_root"],
        )
        self.assertEqual(render_config["redistort_interpolation"], "bicubic")
        self.assertEqual(render_config["sharpen_amount"], 1.0)
        self.assertEqual(render_config["sharpen_sigma"], 0.60)
        self.assertEqual(render_config["jpeg_quality"], 95)
        self.assertEqual(render_config["jpeg_subsampling"], 2)
        self.assertEqual(render_config["output_extension"], "csv")
        self.assertTrue(render_config["save_png"])
        self.assertIn("public_set", render_config["png_root"])
        self.assertIn(
            "mvsplat_init_native_simple_radial_improvedgs_30k_5m5_dense00025",
            render_config["png_root"],
        )
        notebook_source = "\n".join(code_cells)
        self.assertIn(
            "REPO_BRANCH = 'agent/mvsplat-init-improvedgs'",
            notebook_source,
        )
        self.assertIn(
            "'checkout', REPO_BRANCH",
            notebook_source,
        )
        self.assertIn(
            "'pull', '--ff-only', 'origin', REPO_BRANCH",
            notebook_source,
        )
        all_notebook_source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )
        self.assertIn(
            "MVSplat-init + native SIMPLE_RADIAL + ImprovedGS thuan, "
            "dense 0.00025, 30k, budget 5.5M",
            all_notebook_source,
        )
        self.assertIn("SCENE_NAMES = ['HCM0204']", notebook_source)
        self.assertIn("'--subset', *SELECTED_SCENES", notebook_source)
        self.assertIn("'--overwrite'", notebook_source)
        self.assertIn("'--native_simple_radial'", notebook_source)
        self.assertNotIn("'--fixed_pose_retriangulation'", notebook_source)
        self.assertNotIn("'--retriangulation_min_growth_ratio'", notebook_source)
        self.assertNotIn("'--retriangulation_sift_device'", notebook_source)
        self.assertIn("sys.executable, '-u', 'vai_preprocess.py'", notebook_source)
        self.assertIn("sys.executable, '-u', 'vai_mvsplat_init.py'", notebook_source)
        self.assertIn("'--mvsplat_repo', str(MVSPLAT_DIR)", notebook_source)
        self.assertIn("'--checkpoint_sha256', MVSPLAT_CHECKPOINT_SHA256", notebook_source)
        self.assertIn("'--max_pairs', str(MVSPLAT_MAX_PAIRS)", notebook_source)
        self.assertIn("'--mixed_precision', MVSPLAT_MIXED_PRECISION", notebook_source)
        self.assertIn(
            "MVSPLAT_COMMIT = '01f9a28edb5eb68416e7e63b01f8d90c3bdfbf01'",
            notebook_source,
        )
        self.assertNotIn("hf_edge_weighted_loss", notebook_source)
        self.assertNotIn("hf_scale_aware_refinement", notebook_source)
        self.assertIn("f'{SET_NAME}_{EXPERIMENT_NAME}_jpeg.zip'", notebook_source)
        self.assertIn("f'{SET_NAME}_{EXPERIMENT_NAME}_png.zip'", notebook_source)
        self.assertGreaterEqual(notebook_source.count("'vai_package.py'"), 2)
        self.assertNotIn("'--no-install-recommends', 'colmap'", notebook_source)
        self.assertNotIn("apt-get", notebook_source)
        self.assertNotIn("'xvfb'", notebook_source)
        self.assertNotIn("'xauth'", notebook_source)
        self.assertIn("'MAX_JOBS'] = '2'", notebook_source)
        self.assertNotIn("'numpy==1.26.1'", notebook_source)
        self.assertNotIn("'opencv-python==4.10.0.82'", notebook_source)
        self.assertNotIn("install_colmap_with_conda", notebook_source)
        self.assertNotIn("'install', '-y', '-qq', 'colmap'", notebook_source)
        self.assertNotIn("configs/vai_hcm0204.json", notebook_source)
        self.assertGreaterEqual(notebook_source.count("str(RUNTIME_CONFIG_PATH)"), 2)

    def test_hcm0204_template_matches_mvsplat_init_experiment(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "vai_hcm0204.json"
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)

        train_config = config["train_args"]
        self.assertEqual(train_config["iterations"], 30000)
        self.assertEqual(train_config["save_iterations"], [30000])
        self.assertEqual(train_config["position_lr_max_steps"], 30000)
        self.assertFalse(train_config["coarse_to_fine"])
        self.assertFalse(train_config["pose_aware_sampling"])
        self.assertEqual(train_config["densify_grad_threshold"], 0.00025)
        self.assertEqual(train_config["budget"], 5_500_000)
        self.assertIn("vai_mvsplat_init_native_simple_radial", config["data_root"])
        self.assertIn(
            "mvsplat_init_native_simple_radial_improvedgs_30k_5m5_dense00025",
            config["output_root"],
        )


class EdgeMaskTests(unittest.TestCase):
    def test_eas_ignores_invalid_alpha_region(self) -> None:
        camera = SimpleNamespace(
            original_image=torch.rand((3, 5, 7), dtype=torch.float32),
            alpha_mask=torch.zeros((1, 5, 7), dtype=torch.float32),
        )
        edge_map = prepare_edge_maps([camera], opt=None)[0]
        self.assertEqual(float(edge_map.abs().sum().item()), 0.0)


class PackagingTests(unittest.TestCase):
    def test_jpeg_and_png_are_packaged_into_separate_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            phase_dir = root / "phase1"
            pose_dir = phase_dir / "public_set" / "HCM0204" / "test"
            pose_dir.mkdir(parents=True)
            pose_path = pose_dir / "test_poses.csv"
            row = {
                "image_name": "sample.JPG",
                "qw": "1",
                "qx": "0",
                "qy": "0",
                "qz": "0",
                "tx": "0",
                "ty": "0",
                "tz": "0",
                "fx": "10",
                "fy": "10",
                "cx": "2",
                "cy": "1.5",
                "width": "4",
                "height": "3",
            }
            with open(pose_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=POSE_COLUMNS)
                writer.writeheader()
                writer.writerow(row)
            self.assertEqual(len(read_pose_rows(pose_path)), 1)

            jpeg_dir = root / "jpeg" / "HCM0204"
            png_dir = root / "png" / "HCM0204"
            jpeg_dir.mkdir(parents=True)
            png_dir.mkdir(parents=True)
            image = PilImage.new("RGB", (4, 3), (10, 20, 30))
            image.save(jpeg_dir / "sample.JPG")
            image.save(png_dir / "sample.png")

            jpeg_zip = root / "jpeg.zip"
            jpeg_counts = package_submission(
                phase_dir=phase_dir,
                set_name="public_set",
                submission_root=root / "jpeg",
                zip_path=jpeg_zip,
                subset=["HCM0204"],
                output_extension="csv",
            )
            png_zip = root / "png.zip"
            png_counts = package_submission(
                phase_dir=phase_dir,
                set_name="public_set",
                submission_root=root / "png",
                zip_path=png_zip,
                subset=["HCM0204"],
                output_extension="png",
            )
            self.assertEqual(jpeg_counts, {"HCM0204": 1})
            self.assertEqual(png_counts, {"HCM0204": 1})
            with zipfile.ZipFile(jpeg_zip) as archive:
                self.assertEqual(archive.namelist(), ["HCM0204/sample.JPG"])
            with zipfile.ZipFile(png_zip) as archive:
                self.assertEqual(archive.namelist(), ["HCM0204/sample.png"])


if __name__ == "__main__":
    unittest.main()
