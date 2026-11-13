from __future__ import annotations

import argparse
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import torch
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path, *, models=None):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {name: module, **({"models": models} if models else {})}):
        spec.loader.exec_module(module)
    return module


class PaperFormulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.low_models = load_module("paper_low_models", "01_lowdim_sgr/code/models.py")
        cls.score = load_module("paper_low_score", "01_lowdim_sgr/code/construct_score.py")
        cls.driver = load_module("paper_low_driver", "01_lowdim_sgr/code/main.py")
        cls.low_train = load_module("paper_low_train", "01_lowdim_sgr/code/train_phi.py", models=cls.low_models)
        cls.high_models = load_module("paper_high_models", "02_highdim_sgr/code/models.py")
        cls.high_train = load_module("paper_high_train", "02_highdim_sgr/code/train_phi.py", models=cls.high_models)
        cls.projection = load_module("paper_projection", "02_highdim_sgr/code/projection.py", models=cls.high_models)

    def test_siren_analytic_gradient_and_parameter_derivatives(self):
        torch.manual_seed(42)
        model = self.low_models.SIRENImplicitSurface().double()
        self.assertEqual((model.hidden_dim, model.hidden_layers, model.omega0), (128, 4, 20))
        query = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        phi, analytic = model.phi_and_grad_analytic(query, create_graph=True)
        auto = torch.autograd.grad(model(query).sum(), query, create_graph=True)[0]
        torch.testing.assert_close(analytic, auto, rtol=1e-10, atol=1e-10)
        parameters = tuple(model.parameters())
        analytic_parameter_grad = torch.autograd.grad(phi.square().sum() + analytic.square().sum(), parameters)
        auto_parameter_grad = torch.autograd.grad(model(query).square().sum() + auto.square().sum(), parameters)
        for actual, expected in zip(analytic_parameter_grad, auto_parameter_grad):
            torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)

    def test_local_radius_is_clamped_and_not_folded_or_resampled(self):
        rng = mock.Mock()
        rng.standard_normal.return_value = np.array([[3, 4, 0], [0, 0, 2]], dtype=np.float32)
        rng.normal.return_value = np.array([[-0.1], [0.2]], dtype=np.float32)
        base = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        queries = self.score.sample_local_queries(rng, base, sigma=0.01)
        np.testing.assert_allclose(queries, [[1, 2, 3], [4, 5, 6.2]], atol=1e-6)
        self.assertEqual(rng.normal.call_args.kwargs["loc"], 0.025)
        self.assertEqual(rng.normal.call_args.kwargs["scale"], 0.01)
        self.assertEqual(rng.normal.call_count, 1)
        low, high = self.score.padded_bounds(base)
        np.testing.assert_allclose(low, [0.1, 1.1, 2.1], atol=1e-6)
        np.testing.assert_allclose(high, [4.9, 5.9, 6.9], atol=1e-6)

    def test_loss_sign_normalization_and_coefficients(self):
        class Plane:
            def phi_and_grad_analytic(self, points, **_kwargs):
                gradient = torch.zeros_like(points)
                gradient[:, 0] = 2
                return 2 * points[:, 0], gradient

        local = {
            "q_norm": torch.tensor([[2., 0, 0], [-1., 1, 0]]),
            "target_pull_norm": torch.tensor([[1., 1, 0], [-2., 0, 0]]),
        }
        global_batch = {
            "z_norm": torch.tensor([[1., 0, 0], [-1., 0, 0]]),
            "score_norm": torch.tensor([[-1., 0, 0], [-1., 0, 0]]),
        }
        loss = self.low_train.fixed_sgr_loss(Plane(), None, local, global_batch, lambda_global=7, create_graph=False)
        self.assertEqual(float(loss["score_pull"]), 5.0)
        self.assertEqual(float(loss["global_direction"]), 1.0)
        self.assertEqual(float(loss["total"]), 5007.0)

    def test_knn_score_equals_the_gaussian_log_density_gradient(self):
        samples = np.array([[0., 0, 0], [1., 0, 0], [0., 2, 0]], dtype=np.float32)
        queries = np.array([[0.2, 0.3, 0], [0.7, 1., 0.1]], dtype=np.float32)
        query_tensor = torch.tensor(queries, dtype=torch.float64, requires_grad=True)
        exponent = -(query_tensor[:, None, :] - torch.tensor(samples, dtype=torch.float64)).square().sum(dim=2) / (2 * 0.4**2)
        expected = torch.autograd.grad(torch.logsumexp(exponent, dim=1).sum(), query_tensor)[0].detach().numpy()
        with redirect_stdout(io.StringIO()):
            actual = self.score.empirical_score_knn(cKDTree(samples), samples, queries, sigma=0.4, k=3, chunk_size=1, workers=1, device=torch.device("cpu"))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_scaled_energy_has_the_paper_score_convention(self):
        class QuadraticEnergy(torch.nn.Module):
            def forward(self, points, sigma):
                return sigma * points.square().sum(dim=1) / 2

        points = torch.tensor([[1., 2, 3, 4], [-2., 0, 1, 3]])
        actual = self.high_train.energy_score(QuadraticEnergy(), points)
        torch.testing.assert_close(actual, -points)

    def test_newton_projection_bounds_step_and_rejects_flat_zeros(self):
        class Plane(torch.nn.Module):
            def forward(self, points):
                return points[:, 0]

        seeds = np.array([[0.25, 0, 0, 0]], dtype=np.float32)
        kwargs = dict(low=np.full(4, -1), high=np.full(4, 1), device=torch.device("cpu"), max_step=0.1, residual=1e-4)
        points, _, _, accepted = self.projection.project_learned_zero(Plane(), seeds, iterations=1, **kwargs)
        np.testing.assert_allclose(points[:, 0], [0.15], atol=1e-7)
        self.assertFalse(bool(accepted[0]))
        points, _, _, accepted = self.projection.project_learned_zero(Plane(), seeds, iterations=20, **kwargs)
        np.testing.assert_allclose(points[:, 0], [0], atol=1e-7)
        self.assertTrue(bool(accepted[0]))
        kwargs["low"] = np.array([0.2, -1, -1, -1])
        points, _, _, accepted = self.projection.project_learned_zero(Plane(), seeds, iterations=20, **kwargs)
        np.testing.assert_allclose(points[:, 0], [0.2], atol=1e-7)
        self.assertFalse(bool(accepted[0]))

        class Flat(torch.nn.Module):
            def forward(self, points):
                return 0 * points.sum(dim=1)

        _, _, _, accepted = self.projection.project_learned_zero(Flat(), seeds, iterations=20, **kwargs)
        self.assertFalse(bool(accepted[0]))

    def test_cache_sampling_alias_preserves_the_clamping_contract(self):
        args = argparse.Namespace(sigma=0.01, knn_k=2, local_cache_size=2, global_cache_size=2, seed=42)
        points = np.array([[0., 0, 0], [1., 1, 1]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            point_cloud = Path(directory) / "points.npy"
            np.save(point_cloud, points)
            cache_path = Path(directory) / "cache.pt"
            meta = self.score.cache_metadata(args, point_cloud=point_cloud, point_count=2, data_mean=np.zeros(3), data_scale=1, bg_low=points[0], bg_high=points[1], rng_seed=59)
            self.assertEqual(meta["local_sampling_mode"], "clamped_gaussian")
            for label in ("clamped_gaussian", "gaussian_abs"):
                meta["local_sampling_mode"] = label
                with redirect_stdout(io.StringIO()):
                    self.score.save_cache(cache_path, local_query=points, local_score=points, global_query=points, global_score=points, sigma=0.01, meta=meta)
                local, _, _ = self.low_train.load_score_cache(cache_path, torch.device("cpu"), expected_mean=np.zeros(3), expected_scale=1)
                np.testing.assert_allclose(local["target_pull_norm"].numpy(), -0.0001 * points)
                self.assertTrue(self.driver.score_cache_complete(cache_path, point_cloud=point_cloud, sigma=0.01, knn_k=2, local_cache_size=2, global_cache_size=2))
            meta["local_radius_negative_policy"] = "absolute_value"
            with redirect_stdout(io.StringIO()):
                self.score.save_cache(cache_path, local_query=points, local_score=points, global_query=points, global_score=points, sigma=0.01, meta=meta)
            with self.assertRaisesRegex(ValueError, "clamp negative local radii"):
                self.low_train.load_score_cache(cache_path, torch.device("cpu"), expected_mean=np.zeros(3), expected_scale=1)
            self.assertFalse(self.driver.score_cache_complete(cache_path, point_cloud=point_cloud, sigma=0.01, knn_k=2, local_cache_size=2, global_cache_size=2))


if __name__ == "__main__":
    unittest.main()
