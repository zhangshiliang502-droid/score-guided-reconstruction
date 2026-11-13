"""Analytic implicit expression for the 4D paper example."""

from __future__ import annotations

import numpy as np


def _plane_rotation(
    count: int,
    size: int,
    axis_a: int,
    axis_b: int,
    angle: np.ndarray,
) -> np.ndarray:
    result = np.broadcast_to(
        np.eye(size, dtype=np.float64), (count, size, size)
    ).copy()
    cosine = np.cos(angle)
    sine = np.sin(angle)
    result[:, axis_a, axis_a] = cosine
    result[:, axis_a, axis_b] = -sine
    result[:, axis_b, axis_a] = sine
    result[:, axis_b, axis_b] = cosine
    return result


class WavyTubeS1xS2_4D:
    """Wavy anisotropic tube with topology S^1 x S^2 in R^4."""

    name = "example_4d"
    family = "anisotropic multi-harmonic tube hypersurface"
    topology = "S^1 x S^2"
    ambient_dim = 4
    intrinsic_dim = 3

    @staticmethod
    def _components(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.asarray(theta, dtype=np.float64)
        center = np.column_stack(
            (
                1.48 + 0.17 * np.cos(3.0 * t) + 0.065 * np.sin(7.0 * t),
                0.085 * np.sin(2.0 * t + 0.35) + 0.030 * np.cos(5.0 * t),
                0.070 * np.cos(3.0 * t - 0.20) + 0.025 * np.sin(6.0 * t),
            )
        )
        radius = 0.31 * (
            1.0 + 0.17 * np.sin(2.0 * t) + 0.075 * np.cos(5.0 * t)
        )
        scales = np.column_stack(
            (
                1.00 + 0.18 * np.cos(t),
                0.87 + 0.14 * np.sin(3.0 * t),
                1.12 + 0.13 * np.cos(4.0 * t),
            )
        )
        first = _plane_rotation(
            len(t), 3, 0, 2, 0.52 * np.sin(2.0 * t + 0.25)
        )
        second = _plane_rotation(
            len(t), 3, 1, 2, 0.41 * np.cos(3.0 * t - 0.40)
        )
        rotation = np.einsum("nij,njk->nik", first, second)
        matrix = rotation * (radius[:, None] * scales)[:, None, :]
        return center, matrix

    def implicit_value(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        original_shape = values.shape[:-1]
        flat = values.reshape(-1, self.ambient_dim)
        theta = np.arctan2(flat[:, 1], flat[:, 0])
        center, matrix = self._components(theta)
        local = np.column_stack(
            (
                np.hypot(flat[:, 0], flat[:, 1]) - center[:, 0],
                flat[:, 2] - center[:, 1],
                flat[:, 3] - center[:, 2],
            )
        )
        sphere = np.linalg.solve(matrix, local[:, :, None])[:, :, 0]
        return (np.sum(sphere * sphere, axis=1) - 1.0).reshape(original_shape)

    def implicit_gradient(
        self, points: np.ndarray, epsilon: float = 2e-5
    ) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        gradient = np.empty_like(values)
        for axis in range(self.ambient_dim):
            plus = values.copy()
            minus = values.copy()
            plus[..., axis] += epsilon
            minus[..., axis] -= epsilon
            gradient[..., axis] = (
                self.implicit_value(plus) - self.implicit_value(minus)
            ) / (2.0 * epsilon)
        return gradient

    def project_to_surface(
        self, points: np.ndarray, iterations: int = 15
    ) -> np.ndarray:
        projected = np.asarray(points, dtype=np.float64).copy()
        for _ in range(iterations):
            value = self.implicit_value(projected)
            gradient = self.implicit_gradient(projected)
            denominator = np.maximum(
                np.sum(gradient * gradient, axis=-1), 1e-20
            )
            projected -= (value / denominator)[..., None] * gradient
        return projected

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "topology": self.topology,
            "ambient_dim": self.ambient_dim,
            "intrinsic_dim": self.intrinsic_dim,
            "codimension": 1,
        }
