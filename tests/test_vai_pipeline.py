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

from vai.colmap_io import (
    Camera,
    Image,
    read_extrinsics_binary,
    write_extrinsics_binary,
    write_intrinsics_binary,
)
from vai.common import output_name_for_pose, read_pose_rows
from vai.distortion import redistort_and_crop, redistort_image
from vai.evaluation import compute_weighted_score
from vai.image_processing import save_render_image, sharpen_image
from vai.packaging import package_submission
from vai.preprocessing import _synchronize_and_filter_images, preprocess_scene
from utils.coarse_to_fine import (
    build_training_resolution_scales,
    resolve_training_resolution_scale,
    validate_coarse_to_fine_schedule,
)
from utils.pose_aware_sampling import (
    CameraPose,
    build_pose_sampling_plan,
    build_repeated_camera_pool,
    pose_from_csv_row,
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
            neighbor_count=1,
            extra_fraction=0.25,
            max_repeat=2,
            angle_weight=0.25,
        )

        self.assertEqual(plan.repeat_counts, {0: 1, 1: 1, 2: 1, 3: 2})
        self.assertEqual(plan.extra_count, 1)
        self.assertEqual(plan.pool_size, 5)
        self.assertAlmostEqual(plan.median_train_spacing, 1.0)
        self.assertAlmostEqual(plan.max_test_gap, 2.0)
        pool = build_repeated_camera_pool(cameras, plan.repeat_counts)
        self.assertEqual(len(pool), 5)
        self.assertTrue(all(any(item is camera for item in pool) for camera in cameras))
        self.assertEqual(sum(item is cameras[3] for item in pool), 2)

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
            neighbor_count=1,
            extra_fraction=0.5,
            max_repeat=2,
            angle_weight=0.25,
        )

        self.assertEqual(plan.repeat_counts, {0: 2, 1: 1})


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


class PreprocessingTests(unittest.TestCase):
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

            output_root = root / "cleaned"
            with patch("vai.preprocessing._check_colmap_executable"), patch(
                "vai.preprocessing._run_colmap_undistorter",
                side_effect=fake_undistorter,
            ):
                result = preprocess_scene(source_scene, output_root)

            output_scene = output_root / "HCM0204"
            self.assertEqual(result["scene_name"], "HCM0204")
            self.assertEqual(result["train_images"], 1)
            self.assertTrue((output_scene / "images" / "train.png").is_file())
            self.assertTrue((output_scene / "sparse" / "0" / "cameras.bin").is_file())
            with PilImage.open(output_scene / "images" / "train.png") as image:
                self.assertEqual(image.mode, "RGBA")
            with open(output_scene / "vai_metadata.json", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["original_camera"]["model"], "SIMPLE_RADIAL")
            self.assertEqual(metadata["undistorted_camera"]["model"], "PINHOLE")


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
        self.assertTrue(train_config["coarse_to_fine"])
        self.assertEqual(train_config["coarse_to_fine_middle_iter"], 2000)
        self.assertEqual(train_config["coarse_to_fine_full_iter"], 5000)
        self.assertTrue(train_config["pose_aware_sampling"])
        self.assertEqual(train_config["pose_aware_k"], 3)
        self.assertEqual(train_config["pose_aware_extra_fraction"], 0.25)
        self.assertEqual(train_config["pose_aware_max_repeat"], 2)
        self.assertEqual(train_config["pose_aware_angle_weight"], 0.25)
        self.assertEqual(render_config["redistort_interpolation"], "bicubic")
        self.assertEqual(render_config["sharpen_amount"], 1.0)
        self.assertEqual(render_config["sharpen_sigma"], 0.60)
        self.assertEqual(render_config["jpeg_quality"], 95)
        self.assertEqual(render_config["jpeg_subsampling"], 2)
        self.assertEqual(render_config["output_extension"], "csv")
        self.assertTrue(render_config["save_png"])
        self.assertIn("public_set", render_config["png_root"])
        notebook_source = "\n".join(code_cells)
        self.assertIn("REPO_BRANCH = 'main'", notebook_source)
        all_notebook_source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )
        self.assertIn("ImprovedGS + C2F + pose-aware cho VAI public/private", all_notebook_source)
        self.assertIn("SCENE_NAMES = ['HCM0204']", notebook_source)
        self.assertIn("'--subset', *SELECTED_SCENES", notebook_source)
        self.assertIn("f'{SET_NAME}_jpeg.zip'", notebook_source)
        self.assertIn("f'{SET_NAME}_png.zip'", notebook_source)
        self.assertGreaterEqual(notebook_source.count("'vai_package.py'"), 2)
        self.assertIn("'--no-install-recommends', 'colmap'", notebook_source)
        self.assertIn("install_colmap_with_conda", notebook_source)
        self.assertNotIn("'install', '-y', '-qq', 'colmap'", notebook_source)
        self.assertNotIn("configs/vai_hcm0204.json", notebook_source)
        self.assertGreaterEqual(notebook_source.count("str(RUNTIME_CONFIG_PATH)"), 2)


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
