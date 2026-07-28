"""CPU tests for the HF-GS edge and scale components."""
from __future__ import annotations

import importlib.util
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from arguments import OptimizationParams


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HFGS = _load_module("test_hfgs_helpers", "scene/methods/hfgs.py")
TRAINING_CONFIG = _load_module(
    "test_hfgs_training_config",
    "scene/methods/training_config.py",
)


class HFGSEdgeTests(unittest.TestCase):
    def test_normalize_constant_map_is_zero(self) -> None:
        normalized = HFGS.normalize_prior_map(torch.full((5, 7), 3.0))
        torch.testing.assert_close(normalized, torch.zeros_like(normalized))

    def test_sobel_prior_responds_to_step_edge_without_constant_border(self) -> None:
        constant = torch.full((3, 9, 11), 0.5)
        torch.testing.assert_close(
            HFGS.compute_sobel_prior(constant),
            torch.zeros((9, 11)),
        )

        step = torch.zeros((3, 9, 11))
        step[:, :, 6:] = 1.0
        prior = HFGS.compute_sobel_prior(step)
        self.assertGreater(float(prior[:, 5:7].mean().item()), 0.9)
        self.assertEqual(tuple(prior.shape), (9, 11))
        self.assertGreaterEqual(float(prior.min().item()), 0.0)
        self.assertLessEqual(float(prior.max().item()), 1.0)

    def test_weighted_l1_preserves_standard_mean_convention(self) -> None:
        image = torch.zeros((3, 2, 2))
        target = torch.ones_like(image)
        ones = torch.ones((2, 2))
        emphasized = ones.clone()
        emphasized[0, 0] = 2.0

        self.assertEqual(float(HFGS.weighted_l1_loss(image, target, None)), 1.0)
        self.assertEqual(float(HFGS.weighted_l1_loss(image, target, ones)), 1.0)
        self.assertAlmostEqual(
            float(HFGS.weighted_l1_loss(image, target, emphasized)),
            1.25,
        )

    def test_official_pidinet_checkpoint_loads_and_infers(self) -> None:
        model = HFGS.load_pidinet(
            REPOSITORY_ROOT / "third_party/pidinet/table5_pidinet.pth",
            torch.device("cpu"),
        )
        prior = HFGS.compute_pidinet_prior(
            torch.rand((3, 32, 32)),
            model,
            torch.device("cpu"),
        )
        self.assertEqual(tuple(prior.shape), (32, 32))
        self.assertTrue(torch.isfinite(prior).all())
        self.assertGreaterEqual(float(prior.min().item()), 0.0)
        self.assertLessEqual(float(prior.max().item()), 1.0)


class HFGSScaleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scales = torch.tensor(
            [
                [1.0, 0.5, 0.25],
                [2.0, 1.0, 1.0],
                [3.0, 2.0, 1.0],
                [4.0, 1.0, 1.0],
                [8.0, 2.0, 1.0],
            ]
        )

    def test_reference_is_75th_percentile_of_max_axis(self) -> None:
        reference = HFGS.compute_scale_reference(self.scales, 0.75)
        self.assertEqual(float(reference.item()), 4.0)

    def test_large_gaussians_receive_lower_threshold(self) -> None:
        thresholds = HFGS.compute_scale_aware_thresholds(
            self.scales,
            base_threshold=0.00025,
            scale_reference=4.0,
            eta=0.2,
        )
        torch.testing.assert_close(
            thresholds[:4],
            torch.full((4,), 0.00025),
        )
        self.assertLess(float(thresholds[4]), 0.00025)
        self.assertAlmostEqual(float(thresholds[4]), 0.0002, places=8)

    def test_periodic_contraction_is_isotropic_and_bounded(self) -> None:
        ratios = HFGS.compute_scale_contraction_ratios(
            self.scales,
            scale_reference=4.0,
            gamma=1.0,
            minimum_ratio=0.70,
        )
        torch.testing.assert_close(ratios[:4], torch.ones(4))
        self.assertAlmostEqual(float(ratios[4]), 0.70, places=6)


class HFGSConfigurationTests(unittest.TestCase):
    @staticmethod
    def _options(**overrides):
        values = {
            "training_method": "improvedgs",
            "use_las": True,
            "use_eas": True,
            "use_mu": True,
            "use_rap": True,
            "hf_edge_weighted_loss": True,
            "hf_scale_aware_refinement": True,
            "hf_edge_alpha_p_ref": 0.12,
            "hf_edge_alpha_g_ref": 0.09,
            "hf_edge_epsilon": 1e-6,
            "hf_scale_quantile": 0.75,
            "hf_scale_eta": 0.2,
            "hf_scale_interval": 1000,
            "hf_scale_gamma": 0.005,
            "hf_scale_min_ratio": 0.70,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_components_are_independent_ablation_switches(self) -> None:
        config = TRAINING_CONFIG.build_training_method_config(
            self._options(hf_edge_weighted_loss=False)
        )
        self.assertFalse(config["hf_edge_weighted_loss"])
        self.assertTrue(config["hf_scale_aware_refinement"])

    def test_cli_exposes_paper_defaults_and_boolean_switches(self) -> None:
        parser = ArgumentParser()
        optimization_params = OptimizationParams(parser)
        namespace = parser.parse_args(
            [
                "--hf_edge_weighted_loss",
                "true",
                "--hf_scale_aware_refinement",
                "true",
            ]
        )
        options = optimization_params.extract(namespace)
        self.assertTrue(options.hf_edge_weighted_loss)
        self.assertTrue(options.hf_scale_aware_refinement)
        self.assertEqual(options.hf_edge_alpha_p_ref, 0.12)
        self.assertEqual(options.hf_edge_alpha_g_ref, 0.09)
        self.assertEqual(options.hf_scale_quantile, 0.75)
        self.assertEqual(options.hf_scale_interval, 1000)

    def test_hfgs_rejects_non_improvedgs_method(self) -> None:
        with self.assertRaisesRegex(ValueError, "training_method=improvedgs"):
            TRAINING_CONFIG.build_training_method_config(
                self._options(training_method="3dgs")
            )

    def test_scale_refinement_requires_las(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires ImprovedGS LAS"):
            TRAINING_CONFIG.build_training_method_config(
                self._options(use_las=False)
            )


if __name__ == "__main__":
    unittest.main()
