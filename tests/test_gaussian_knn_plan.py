from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "01_lowdim_sgr" / "code" / "plan_gaussian_knn.py"
SPEC = importlib.util.spec_from_file_location("plan_gaussian_knn", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def random_sphere(count: int, seed: int = 20260801) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 3))
    return points / np.linalg.norm(points, axis=1, keepdims=True)


class GaussianKNNPlanTests(unittest.TestCase):
    def test_gaussian_mass_radius(self) -> None:
        for mass in (0.98, 0.99):
            radius = MODULE.gaussian_radius_for_mass(0.002, mass)
            recovered = 1.0 - math.exp(
                -(radius * radius) / (2.0 * 0.002**2)
            )
            self.assertAlmostEqual(recovered, mass, places=14)

    def test_knn_requirement_is_conservative_and_monotone(self) -> None:
        requirement_98 = MODULE.compute_knn_requirement(
            surface_area=6.0,
            num_samples=40_000_000,
            sigma=0.002,
            gaussian_mass=0.98,
            coverage_confidence=0.99,
            knn_round_up_multiple=256,
        )
        requirement_99 = MODULE.compute_knn_requirement(
            surface_area=6.0,
            num_samples=40_000_000,
            sigma=0.002,
            gaussian_mass=0.99,
            coverage_confidence=0.99,
            knn_round_up_multiple=256,
        )
        self.assertGreater(
            requirement_99.knn_k, requirement_98.knn_k
        )
        for requirement in (requirement_98, requirement_99):
            self.assertEqual(requirement.knn_k_rounded % 256, 0)
            self.assertGreaterEqual(
                requirement.achieved_coverage_confidence, 0.99
            )

    def test_multiscale_area_estimate_on_unit_sphere(self) -> None:
        points = random_sphere(30_000)
        estimate = MODULE.estimate_surface_area_knn(
            points,
            normalize=False,
            area_neighbors=(16, 32, 64),
            max_probes=12_000,
            seed=5,
            workers=1,
        )
        self.assertAlmostEqual(
            estimate.estimated_surface_area,
            4.0 * math.pi,
            delta=0.08 * 4.0 * math.pi,
        )
        self.assertLess(estimate.multiscale_relative_range, 0.12)

    def test_sgr_normalization_and_three_step_plan(self) -> None:
        points = 2.0 * random_sphere(20_000) + np.array([8.0, -3.0, 5.0])
        plan = MODULE.build_plan(
            points,
            num_samples=40_000_000,
            sigma=0.002,
            normalize=True,
            area_neighbors=(16, 32),
            area_probes=8_000,
            coverage_confidence=0.99,
            workers=1,
        )
        self.assertEqual(plan["schema"], "sgr-gaussian-knn-plan/v2")
        self.assertEqual(
            plan["step1_surface_area"]["coordinate_space"],
            "sgr_normalized",
        )
        self.assertAlmostEqual(
            plan["step1_surface_area"]["normalization_scale"],
            2.0,
            delta=0.04,
        )
        self.assertAlmostEqual(
            plan["step1_surface_area"][
                "estimated_surface_area_input_coordinates"
            ],
            plan["step1_surface_area"]["normalization_scale"] ** 2
            * plan["step1_surface_area"]["estimated_surface_area"],
            places=10,
        )
        self.assertEqual(plan["step2_sampling"]["num_samples"], 40_000_000)
        requirements = plan["step3_knn"]["requirements"]
        self.assertEqual(
            [value["gaussian_mass"] for value in requirements],
            [0.98, 0.99],
        )

    def test_npy_loader_uses_first_three_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points.npy"
            np.save(
                path,
                np.asarray(
                    [[1.0, 2.0, 3.0, 9.0], [4.0, 5.0, 6.0, 8.0]]
                ),
            )
            points, input_count = MODULE.load_point_cloud_xyz(path)
        self.assertEqual(input_count, 2)
        self.assertEqual(points.shape, (2, 3))
        self.assertTrue(np.allclose(points, [[1, 2, 3], [4, 5, 6]]))


if __name__ == "__main__":
    unittest.main()
