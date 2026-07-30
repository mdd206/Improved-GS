"""CPU-only tests for per-test-pose view selection and CLI defaults."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image as PilImage

from utils.pose_aware_sampling import CameraPose
from utils.test_pose_finetune import (
    configure_local_improvedgs_options,
    median_nearby_camera_spacing,
    pose_output_directory_name,
    select_top_training_views,
    validate_finetune_schedule,
)
from vai.finetune_progress import (
    build_progress_payload,
    save_progress,
    valid_completed_pose_result,
    validate_resume_manifest,
)


def make_camera(
    uid: int,
    center: tuple[float, float, float],
    forward: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> SimpleNamespace:
    rotation = np.eye(3, dtype=np.float64)
    rotation[:, 2] = np.asarray(forward, dtype=np.float64)
    return SimpleNamespace(
        uid=uid,
        image_name="train_{:03d}.JPG".format(uid),
        camera_center=np.asarray(center, dtype=np.float64),
        R=rotation,
    )


class TestPoseViewSelectionTests(unittest.TestCase):
    def test_median_spacing_uses_each_camera_nearest_neighbor(self) -> None:
        centers = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
            ]
        )
        # Nearest distances are [1, 1, 2, 5], whose median is 1.5.
        self.assertAlmostEqual(median_nearby_camera_spacing(centers), 1.5)

    def test_similarity_matches_requested_formula_exactly(self) -> None:
        cameras = [
            make_camera(0, (0.0, 0.0, 0.0)),
            make_camera(1, (2.0, 0.0, 0.0)),
            make_camera(2, (4.0, 0.0, 0.0)),
        ]
        test_pose = CameraPose(
            center=np.array([1.0, 0.0, 0.0]),
            forward=np.array([0.0, 0.0, 1.0]),
        )
        selection = select_top_training_views(
            cameras,
            test_pose,
            top_k=3,
            sigma_multiplier=3.0,
        )
        self.assertAlmostEqual(selection.sigma, 2.0)
        expected_distance_one = np.exp(-(1.0**2) / (2.0 * (3.0 * 2.0) ** 2))
        expected_distance_three = np.exp(-(3.0**2) / (2.0 * (3.0 * 2.0) ** 2))
        self.assertEqual(selection.selected_indices, (0, 1, 2))
        self.assertAlmostEqual(selection.views[0].score, expected_distance_one)
        self.assertAlmostEqual(selection.views[1].score, expected_distance_one)
        self.assertAlmostEqual(selection.views[2].score, expected_distance_three)

    def test_opposite_forward_direction_has_zero_score(self) -> None:
        cameras = [
            make_camera(0, (0.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
            make_camera(1, (10.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ]
        test_pose = CameraPose(
            center=np.array([0.0, 0.0, 0.0]),
            forward=np.array([0.0, 0.0, 1.0]),
        )
        selection = select_top_training_views(cameras, test_pose, top_k=2)
        self.assertEqual(selection.selected_indices, (1, 0))
        self.assertGreater(selection.views[0].score, 0.0)
        self.assertEqual(selection.views[1].score, 0.0)

    def test_direction_similarity_is_squared_cosine(self) -> None:
        cameras = [
            make_camera(0, (0.0, 0.0, 0.0), (np.sqrt(3.0) / 2.0, 0.0, 0.5)),
            make_camera(1, (2.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ]
        test_pose = CameraPose(
            center=np.array([0.0, 0.0, 0.0]),
            forward=np.array([0.0, 0.0, 1.0]),
        )
        selection = select_top_training_views(cameras, test_pose, top_k=2)
        first_view = next(view for view in selection.views if view.train_index == 0)
        self.assertAlmostEqual(first_view.cosine, 0.5)
        self.assertAlmostEqual(first_view.score, 0.25)

    def test_top_25_is_stable_and_contains_no_unselected_camera(self) -> None:
        cameras = [
            make_camera(index, (float(index), 0.0, 0.0))
            for index in range(30)
        ]
        test_pose = CameraPose(
            center=np.array([0.0, 0.0, 0.0]),
            forward=np.array([0.0, 0.0, 1.0]),
        )
        first = select_top_training_views(cameras, test_pose, top_k=25)
        second = select_top_training_views(cameras, test_pose, top_k=25)
        self.assertEqual(len(first.views), 25)
        self.assertEqual(first.selected_indices, tuple(range(25)))
        self.assertEqual(first.selected_indices, second.selected_indices)

    def test_schedule_and_pose_folder_validation(self) -> None:
        validate_finetune_schedule(3000, 0, 1500, 25, 3.0)
        with self.assertRaises(ValueError):
            validate_finetune_schedule(3000, 0, 3001, 25, 3.0)
        self.assertEqual(
            pose_output_directory_name(7, "nested/a strange image.JPG"),
            "0007_a_strange_image",
        )

    def test_local_schedule_forces_only_top_k_improvedgs_training_mode(self) -> None:
        original = SimpleNamespace(
            training_method="3dgs",
            coarse_to_fine=True,
            pose_aware_sampling=True,
            iterations=30_000,
            position_lr_max_steps=30_000,
            densify_from_iter=500,
            densify_until_iter=15_000,
            budget_warmup_until_offset=500,
        )
        local = configure_local_improvedgs_options(original, 3000, 0, 1500)
        self.assertEqual(local.training_method, "improvedgs")
        self.assertFalse(local.coarse_to_fine)
        self.assertFalse(local.pose_aware_sampling)
        self.assertEqual(local.iterations, 3001)
        self.assertEqual(local.position_lr_max_steps, 3000)
        self.assertEqual(local.mu_start_iter, 3001)
        self.assertEqual(local.mu_second_start_iter, 3002)
        self.assertEqual(local.densify_from_iter, 0)
        self.assertEqual(local.densify_until_iter, 1501)
        self.assertEqual(local.budget_warmup_until_offset, 1501)
        self.assertEqual(original.training_method, "3dgs")


class TestPoseFineTuneCliTests(unittest.TestCase):
    def test_help_exposes_exact_experiment_controls_without_cuda_imports(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, "vai_test_pose_finetune.py", "--help"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--base_model_path", completed.stdout)
        self.assertIn("--pose_start_index", completed.stdout)
        self.assertIn("--pose_count", completed.stdout)
        self.assertIn("--fine_tune_steps", completed.stdout)
        self.assertIn("--split_until_step", completed.stdout)
        self.assertIn("--top_k", completed.stdout)
        self.assertIn("--sigma_multiplier", completed.stdout)
        self.assertIn("--save_pose_models", completed.stdout)
        self.assertIn("--resume", completed.stdout)

    def test_kaggle_notebook_pins_the_requested_experiment(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        notebook_path = (
            repository_root / "notebooks" / "vai_test_pose_finetune.ipynb"
        )
        with open(notebook_path, encoding="utf-8") as handle:
            notebook = json.load(handle)
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )
        self.assertIn("'--fine_tune_steps', str(FINE_TUNE_STEPS)", source)
        self.assertIn("FINE_TUNE_STEPS = 3_000", source)
        self.assertIn("SPLIT_UNTIL_STEP = 1_500", source)
        self.assertIn("TOP_K = 25", source)
        self.assertIn("SIGMA_MULTIPLIER = 3.0", source)
        self.assertIn("SAVE_POSE_MODELS = False", source)
        self.assertIn("POSE_BATCH_INDEX = 0", source)
        self.assertIn("POSE_BATCH_SIZE = 15", source)
        self.assertIn("POSE_START_INDEX = POSE_BATCH_INDEX * POSE_BATCH_SIZE", source)
        self.assertIn("'--pose_start_index', str(POSE_START_INDEX)", source)
        self.assertIn("'--pose_count', str(POSE_COUNT)", source)
        self.assertIn("'--coarse_to_fine', 'false'", source)
        self.assertIn("'--pose_aware_sampling', 'false'", source)
        self.assertIn(
            "'co che nay se bi tat trong 3k step fine-tune.'",
            source,
        )
        self.assertNotIn(
            "raise ValueError(f'Base model dang bat {disabled_flag}",
            source,
        )
        self.assertIn("'--training_method', 'improvedgs'", source)
        self.assertIn("REPO_BRANCH = 'agent/test-pose-finetune'", source)
        self.assertIn("'git', 'clone', '--recursive', '--branch', REPO_BRANCH", source)
        self.assertIn("WORK_ROOT / 'vai_cleaned' / SET_NAME", source)
        self.assertIn("'--overwrite'", source)
        self.assertIn("'--no-install-recommends', 'colmap'", source)
        self.assertGreaterEqual(source.count("'vai_package.py'"), 2)
        self.assertNotIn("'--native_simple_radial'", source)
        self.assertNotIn("vai_native_simple_radial", source)
        self.assertNotIn("hfgs", source.lower())
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), str(notebook_path), "exec")

    def test_private_phase2_notebook_runs_all_poses_and_packages_png(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        notebook_path = (
            repository_root
            / "notebooks"
            / "vai_private_phase2_pose_finetune.ipynb"
        )
        with open(notebook_path, encoding="utf-8") as handle:
            notebook = json.load(handle)
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )
        self.assertIn("SCENE_NAMES = ['HCM0421']", source)
        for scene_name in (
            "bonsai",
            "chair",
            "HCM0421",
            "HCM0539",
            "HCM0540",
            "HCM0644",
            "HCM0674",
        ):
            self.assertIn(repr(scene_name), source)
        self.assertIn("BASE_MODEL_OVERRIDES = {", source)
        self.assertIn("BUDGET_OVERRIDES = {", source)
        self.assertIn("for scene_name in SELECTED_SCENES", source)
        self.assertGreaterEqual(source.count("'--pose_count', '-1'"), 2)
        self.assertIn("'--output_extension', 'png'", source)
        self.assertIn("'--save_png', 'false'", source)
        self.assertIn("RESUME_COMPLETED_POSES = True", source)
        self.assertIn(
            "'--resume', str(RESUME_COMPLETED_POSES).lower()",
            source,
        )
        self.assertIn(
            "PNG_ROOT / f'{scene_name}_finetune_progress.json'",
            source,
        )
        self.assertIn("progress['completed_pose_count']", source)
        self.assertIn("'--evaluate', 'false'", source)
        self.assertIn("'--coarse_to_fine', 'false'", source)
        self.assertIn("'--pose_aware_sampling', 'false'", source)
        self.assertIn("'--subset', *SELECTED_SCENES", source)
        self.assertIn("PHASE2_DIR.parent", source)
        self.assertIn("assert all(name.endswith('.png')", source)
        self.assertIn("SIMPLE_RADIAL", source)
        self.assertIn("SIMPLE_PINHOLE", source)
        self.assertNotIn("POSE_BATCH_INDEX", source)
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), str(notebook_path), "exec")

    def test_private_phase2_parameter_cell_discovers_data_and_scene_selection(
        self,
    ) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        notebook_path = (
            repository_root
            / "notebooks"
            / "vai_private_phase2_pose_finetune.ipynb"
        )
        with open(notebook_path, encoding="utf-8") as handle:
            notebook = json.load(handle)
        parameter_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if "SCENE_NAMES = ['HCM0421']" in "".join(cell.get("source", []))
        ]
        self.assertEqual(len(parameter_cells), 1)

        scene_names = (
            "bonsai",
            "chair",
            "HCM0421",
            "HCM0539",
            "HCM0540",
            "HCM0644",
            "HCM0674",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            kaggle_input = root / "input"
            kaggle_working = root / "working"
            phase2_root = kaggle_input / "dataset" / "phase2"
            for scene_name in scene_names:
                (phase2_root / scene_name / "train" / "images").mkdir(
                    parents=True
                )
                camera_path = (
                    phase2_root
                    / scene_name
                    / "train"
                    / "sparse"
                    / "0"
                    / "cameras.bin"
                )
                camera_path.parent.mkdir(parents=True)
                camera_path.touch()
                pose_path = (
                    phase2_root / scene_name / "test" / "test_poses.csv"
                )
                pose_path.parent.mkdir(parents=True)
                pose_path.touch()

            parameter_source = parameter_cells[0].replace(
                "WORK_ROOT = Path('/kaggle/working')",
                "WORK_ROOT = Path({})".format(repr(str(kaggle_working))),
            ).replace(
                "Path('/kaggle/input')",
                "Path({})".format(repr(str(kaggle_input))),
            )
            namespace: dict[str, object] = {}
            with patch("builtins.print"):
                exec(parameter_source, namespace)
            self.assertEqual(namespace["PHASE2_DIR"], phase2_root)
            self.assertEqual(namespace["SELECTED_SCENES"], ["HCM0421"])

            all_source = parameter_source.replace(
                "SCENE_NAMES = ['HCM0421']",
                "SCENE_NAMES = []",
            )
            all_namespace: dict[str, object] = {}
            with patch("builtins.print"):
                exec(all_source, all_namespace)
            self.assertEqual(
                all_namespace["SELECTED_SCENES"],
                sorted(scene_names),
            )


class FineTuneProgressTests(unittest.TestCase):
    @staticmethod
    def make_manifest() -> dict[str, object]:
        return {
            "scene_name": "HCM0421",
            "source_path": "/data/HCM0421",
            "base_model_path": "/models/HCM0421",
            "base_iteration": 30_000,
            "fine_tune_steps": 3_000,
            "split_from_step": 0,
            "split_until_step": 1_500,
            "top_k": 25,
            "sigma_multiplier": 3.0,
            "pose_start_index": 0,
            "pose_end_index_exclusive": 2,
            "output_extension": "png",
            "render_dir": "/renders/HCM0421",
            "test_pose_count": 2,
            "source_test_pose_count": 60,
            "poses": [
                {
                    "pose_index": 0,
                    "test_image_name": "pose_00.JPG",
                    "render_path": "/renders/HCM0421/pose_00.png",
                    "training_seconds": 12.5,
                }
            ],
        }

    def test_progress_payload_lists_completed_and_current_pose(self) -> None:
        manifest = self.make_manifest()
        current_pose = {
            "pose_index": 1,
            "test_image_name": "pose_01.JPG",
            "stage": "fine_tuning",
        }
        progress = build_progress_payload(manifest, "running", current_pose)
        self.assertEqual(progress["status"], "running")
        self.assertEqual(progress["completed_pose_count"], 1)
        self.assertEqual(progress["completed_pose_indices"], [0])
        self.assertEqual(progress["completed_images"], ["pose_00.JPG"])
        self.assertEqual(progress["current_pose"], current_pose)
        self.assertIn("updated_at_utc", progress)

    def test_progress_is_written_to_model_and_png_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [
                root / "model" / "finetune_progress.json",
                root / "png" / "HCM0421_finetune_progress.json",
            ]
            save_progress(paths, self.make_manifest(), "running")
            for path in paths:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["completed_pose_count"], 1)
                self.assertEqual(payload["scene_name"], "HCM0421")
            self.assertFalse(any(root.rglob("*.tmp")))

    def test_resume_rejects_changed_finetune_configuration(self) -> None:
        previous = self.make_manifest()
        current = dict(previous)
        current["fine_tune_steps"] = 2_000
        with self.assertRaisesRegex(ValueError, "fine_tune_steps"):
            validate_resume_manifest(previous, current)

    def test_resume_accepts_only_complete_manifest_and_valid_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            render_path = Path(temp_dir) / "pose_00.png"
            PilImage.new("RGB", (8, 6), (10, 20, 30)).save(render_path)
            row = {
                "image_name": "pose_00.JPG",
                "width": "8",
                "height": "6",
            }
            result = {
                "pose_index": 0,
                "test_image_name": "pose_00.JPG",
                "optimizer_updates": 3_000,
                "point_cloud_path": "",
            }
            self.assertTrue(
                valid_completed_pose_result(
                    result,
                    row,
                    pose_index=0,
                    render_path=render_path,
                    fine_tune_steps=3_000,
                    save_pose_models=False,
                )
            )
            result["optimizer_updates"] = 2_999
            self.assertFalse(
                valid_completed_pose_result(
                    result,
                    row,
                    pose_index=0,
                    render_path=render_path,
                    fine_tune_steps=3_000,
                    save_pose_models=False,
                )
            )
            result["optimizer_updates"] = 3_000
            render_path.write_bytes(b"truncated PNG")
            self.assertFalse(
                valid_completed_pose_result(
                    result,
                    row,
                    pose_index=0,
                    render_path=render_path,
                    fine_tune_steps=3_000,
                    save_pose_models=False,
                )
            )


if __name__ == "__main__":
    unittest.main()
