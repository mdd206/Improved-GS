"""CPU-only tests for per-test-pose view selection and CLI defaults."""
from __future__ import annotations

import subprocess
import sys
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from utils.pose_aware_sampling import CameraPose
from utils.test_pose_finetune import (
    configure_local_improvedgs_options,
    median_nearby_camera_spacing,
    pose_output_directory_name,
    select_top_training_views,
    validate_finetune_schedule,
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


if __name__ == "__main__":
    unittest.main()
