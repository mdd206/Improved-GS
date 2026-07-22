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
from utils.frequency_regularization import (
    compute_fregs_lite_loss,
    resolve_fregs_max_radius,
    resolve_fregs_window,
    should_run_fregs_lite,
    validate_fregs_lite_options,
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


class FreGSLiteTests(unittest.TestCase):
    def make_options(self, **overrides: object) -> SimpleNamespace:
        values = {
            "fregs_lite": True,
            "fregs_weight": 0.01,
            "fregs_phase_weight": 0.1,
            "fregs_detail_weight": 1.0,
            "fregs_start_iter": -1,
            "fregs_until_iter": -1,
            "fregs_interval": 1,
            "fregs_low_radius": 0.15,
            "fregs_middle_radius": 0.5,
            "fregs_hann_window": False,
            "fregs_epsilon": 1e-6,
            "densify_from_iter": 500,
            "densify_until_iter": 15_000,
            "coarse_to_fine": True,
            "coarse_to_fine_middle_iter": 2_000,
            "coarse_to_fine_full_iter": 5_000,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_window_and_frequency_radius_follow_c2f_stages(self) -> None:
        opt = self.make_options()
        self.assertEqual(resolve_fregs_window(opt), (500, 15_000))
        self.assertFalse(should_run_fregs_lite(opt, 499))
        self.assertTrue(should_run_fregs_lite(opt, 500))
        self.assertFalse(should_run_fregs_lite(opt, 15_001))
        self.assertAlmostEqual(resolve_fregs_max_radius(opt, 500), 0.15)
        self.assertAlmostEqual(resolve_fregs_max_radius(opt, 1_999), 0.15)
        self.assertAlmostEqual(resolve_fregs_max_radius(opt, 2_000), 0.15)
        self.assertAlmostEqual(resolve_fregs_max_radius(opt, 5_000), 0.5)
        self.assertAlmostEqual(resolve_fregs_max_radius(opt, 15_000), 1.0)

    def test_validation_rejects_invalid_band_or_weight(self) -> None:
        with self.assertRaises(ValueError):
            validate_fregs_lite_options(self.make_options(fregs_weight=0.0))
        with self.assertRaises(ValueError):
            validate_fregs_lite_options(self.make_options(fregs_low_radius=0.6))

    def test_identical_images_have_zero_frequency_loss(self) -> None:
        opt = self.make_options()
        image = torch.rand((3, 8, 10), dtype=torch.float32)
        outputs = compute_fregs_lite_loss(image, image.clone(), None, opt, 5_000)
        self.assertAlmostEqual(float(outputs["amplitude_loss"]), 0.0, places=7)
        self.assertAlmostEqual(float(outputs["phase_loss"]), 0.0, places=6)
        self.assertAlmostEqual(float(outputs["loss"]), 0.0, places=7)

    def test_alpha_mask_removes_invalid_border_from_frequency_loss(self) -> None:
        opt = self.make_options()
        rendered = torch.zeros((3, 8, 8), dtype=torch.float32)
        target = torch.ones_like(rendered)
        valid_mask = torch.zeros((1, 8, 8), dtype=torch.float32)
        valid_mask[:, 2:6, 2:6] = 1.0
        target[:, 2:6, 2:6] = 0.0
        outputs = compute_fregs_lite_loss(rendered, target, valid_mask, opt, 5_000)
        self.assertAlmostEqual(float(outputs["loss"]), 0.0, places=7)

    def test_progressive_band_reveals_checkerboard_detail(self) -> None:
        opt = self.make_options()
        rendered = torch.zeros((3, 16, 16), dtype=torch.float32)
        row = torch.arange(16, dtype=torch.float32)[:, None]
        column = torch.arange(16, dtype=torch.float32)[None, :]
        checkerboard = ((row + column) % 2.0) * 2.0 - 1.0
        target = checkerboard.unsqueeze(0).repeat(3, 1, 1)
        low_outputs = compute_fregs_lite_loss(rendered, target, None, opt, 500)
        full_outputs = compute_fregs_lite_loss(rendered, target, None, opt, 15_000)
        self.assertAlmostEqual(float(low_outputs["loss"]), 0.0, places=7)
        self.assertGreater(float(full_outputs["loss"]), 0.0)

    def test_frequency_loss_backpropagates_finite_gradients(self) -> None:
        opt = self.make_options(fregs_hann_window=True)
        rendered = torch.rand((3, 8, 10), dtype=torch.float32, requires_grad=True)
        target = torch.rand_like(rendered)
        outputs = compute_fregs_lite_loss(rendered, target, None, opt, 5_000)
        outputs["loss"].backward()
        self.assertIsNotNone(rendered.grad)
        self.assertTrue(bool(torch.isfinite(rendered.grad).all()))
        self.assertGreater(float(rendered.grad.abs().sum()), 0.0)


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
            (phase_dir / "public_set" / "HCM0204").mkdir(parents=True)
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

        render_config = config["postprocess_args"]
        train_config = config["train_args"]
        self.assertTrue(train_config["coarse_to_fine"])
        self.assertEqual(train_config["coarse_to_fine_middle_iter"], 2000)
        self.assertEqual(train_config["coarse_to_fine_full_iter"], 5000)
        self.assertTrue(train_config["fregs_lite"])
        self.assertEqual(train_config["fregs_weight"], 0.01)
        self.assertEqual(train_config["fregs_phase_weight"], 0.1)
        self.assertEqual(train_config["fregs_interval"], 1)
        self.assertEqual(train_config["fregs_low_radius"], 0.15)
        self.assertEqual(train_config["fregs_middle_radius"], 0.5)
        self.assertEqual(render_config["redistort_interpolation"], "bicubic")
        self.assertEqual(render_config["sharpen_amount"], 1.0)
        self.assertEqual(render_config["sharpen_sigma"], 0.60)
        self.assertEqual(render_config["jpeg_quality"], 95)
        self.assertEqual(render_config["jpeg_subsampling"], 2)
        self.assertEqual(render_config["output_extension"], "csv")
        notebook_source = "\n".join(code_cells)
        self.assertIn("REPO_BRANCH = 'agent/fregs-lite'", notebook_source)
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
    def test_package_contains_only_validated_scene_files(self) -> None:
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

            render_dir = root / "renders" / "HCM0204"
            render_dir.mkdir(parents=True)
            PilImage.new("RGB", (4, 3), (10, 20, 30)).save(render_dir / "sample.png")
            zip_path = root / "submission.zip"
            counts = package_submission(
                phase_dir=phase_dir,
                set_name="public_set",
                submission_root=root / "renders",
                zip_path=zip_path,
                subset=["HCM0204"],
                output_extension="png",
            )
            self.assertEqual(counts, {"HCM0204": 1})
            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(archive.namelist(), ["HCM0204/sample.png"])


if __name__ == "__main__":
    unittest.main()
