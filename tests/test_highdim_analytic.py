from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = (
    ROOT / "02_highdim_sgr" / "code" / "generate_paper_data.py"
)


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "sgr_test_paper_data_generator",
        GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HighDimensionalAnalyticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = load_generator()

    def test_parameterizations_match_implicit_zero_sets(self) -> None:
        for configuration in sorted(
            (ROOT / "example" / "highdim").glob(
                "example_*/config/training.json"
            )
        ):
            with self.subTest(example=configuration.parent.parent.name):
                config = self.generator.load_json(configuration)
                module, reference = self.generator.load_analytic_definition(
                    configuration,
                    config,
                )
                geometry = self.generator.GeometryAdapter(
                    config,
                    module,
                    reference,
                )
                rng = np.random.default_rng(12345)
                parameters = geometry.sample_parameters(rng, 32)
                points = geometry.parameterize(parameters)
                density = geometry.area_density(parameters)
                residual = np.abs(reference.implicit_value(points))
                self.assertEqual(points.shape, (32, int(config["ambient_dim"])))
                self.assertTrue(np.isfinite(points).all())
                self.assertTrue(np.isfinite(density).all())
                self.assertTrue((density > 0).all())
                self.assertLess(float(np.max(residual)), 1e-10)

    def test_newton_projection_reduces_analytic_residual(self) -> None:
        for configuration in sorted(
            (ROOT / "example" / "highdim").glob(
                "example_*/config/training.json"
            )
        ):
            with self.subTest(example=configuration.parent.parent.name):
                config = self.generator.load_json(configuration)
                module, reference = self.generator.load_analytic_definition(
                    configuration,
                    config,
                )
                geometry = self.generator.GeometryAdapter(
                    config,
                    module,
                    reference,
                )
                rng = np.random.default_rng(67890)
                surface = geometry.parameterize(
                    geometry.sample_parameters(rng, 4)
                )
                perturbed = surface + 0.01 * rng.standard_normal(surface.shape)
                before = np.abs(reference.implicit_value(perturbed))
                projected = reference.project_to_surface(perturbed, iterations=5)
                after = np.abs(reference.implicit_value(projected))
                self.assertLess(float(np.max(after)), float(np.max(before)))
                self.assertLess(float(np.max(after)), 1e-8)


if __name__ == "__main__":
    unittest.main()
