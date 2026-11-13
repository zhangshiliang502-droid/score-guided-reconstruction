from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = (
    ROOT / "03_evaluation" / "evaluate_highdim_reconstruction.py"
)


def load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "sgr_test_highdim_evaluator",
        EVALUATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HighDimensionalEvaluationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluator = load_evaluator()

    def test_spacing_uses_eighth_neighbor_excluding_self(self) -> None:
        reference = np.arange(10, dtype=np.float64)[:, None]
        _tree, count, spacing, threshold = self.evaluator.spacing_threshold(
            reference,
            np.random.default_rng(123),
            requested_count=10,
        )
        pairwise = np.abs(reference - reference.T)
        expected = float(np.median(np.sort(pairwise, axis=1)[:, 8]))
        self.assertEqual(count, 10)
        self.assertEqual(spacing, expected)
        self.assertEqual(threshold, 3.0 * expected)

    def test_spacing_does_not_silently_reduce_requested_sample(self) -> None:
        reference = np.arange(10, dtype=np.float64)[:, None]
        with self.assertRaisesRegex(ValueError, "only 10 are available"):
            self.evaluator.spacing_threshold(
                reference,
                np.random.default_rng(123),
                requested_count=11,
            )

    def test_input_projection_rejects_any_nonconverged_point(self) -> None:
        points = np.zeros((4, 2), dtype=np.float64)

        def fake_projection(_model, seeds, **_kwargs):
            fitted = seeds + 1.0
            accepted = np.ones(len(seeds), dtype=bool)
            accepted[-1] = False
            return (
                fitted,
                np.zeros(len(seeds), dtype=np.float32),
                np.ones(len(seeds), dtype=np.float32),
                accepted,
            )

        with mock.patch.object(
            self.evaluator,
            "project_learned_zero",
            side_effect=fake_projection,
        ):
            with self.assertRaisesRegex(RuntimeError, "only 3/4 converged"):
                self.evaluator.project_input_to_learned_surface(
                    None,
                    points,
                    mean=np.zeros(2),
                    scale=1.0,
                    low=np.full(2, -2.0, dtype=np.float32),
                    high=np.full(2, 2.0, dtype=np.float32),
                    device=torch.device("cpu"),
                    batch_size=4,
                    iterations=20,
                    max_step=0.1,
                    residual=1e-4,
                )

    def test_checkpoint_must_match_recorded_loss_weights(self) -> None:
        configuration = {
            "energy_training": {"sigma": 0.01},
            "sgr_training": {
                "loss": "score_pull_plus_global_cosine",
                "train_size": 220_000,
                "val_size": 30_000,
                "local_query_mode": "radial_shell",
                "local_shell_center_sigma": 2.5,
                "local_shell_std_sigma": 1.0,
                "lambda_score_pull": 1_000,
                "lambda_dir": 20,
                "local_residual_normalization": "none",
            },
        }
        meta = {
            "train_size": 220_000,
            "val_size": 30_000,
            "loss": {
                "name": "score_pull_plus_global_cosine",
                "sigma": 0.01,
                "local_query_mode": "radial_shell",
                "local_shell_center_sigma": 2.5,
                "local_shell_std_sigma": 1.0,
                "lambda_score_pull": 1_000,
                "lambda_dir": 60,
                "local_residual_normalization": "none",
            },
        }
        with self.assertRaisesRegex(ValueError, "lambda_dir=60.0"):
            self.evaluator.validate_checkpoint_configuration(
                meta,
                configuration,
            )


if __name__ == "__main__":
    unittest.main()
