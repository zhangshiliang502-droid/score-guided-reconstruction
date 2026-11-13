"""Analytic implicit expression for the 12D paper example."""

from __future__ import annotations

import numpy as np


class MobiusCliffordS5xS6_12D:
    """Conformally warped S^5 x S^6 hypersurface in R^12."""

    name = "example_12d"
    family = "conformally warped high-dimensional Clifford hypersurface"
    topology = "S^5 x S^6"
    ambient_dim = 12
    intrinsic_dim = 11
    major_radius = 1.86
    tube_radius = 1.10
    mobius_b = np.asarray(
        [
            0.024,
            -0.019,
            0.015,
            0.012,
            -0.009,
            0.006,
            0.004,
            -0.003,
            0.002,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )

    @classmethod
    def _mobius_inverse(cls, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        norm_squared = np.sum(values * values, axis=-1, keepdims=True)
        b_squared = float(np.dot(cls.mobius_b, cls.mobius_b))
        denominator = (
            1.0
            + 2.0 * np.sum(values * cls.mobius_b, axis=-1, keepdims=True)
            + b_squared * norm_squared
        )
        return (values + cls.mobius_b * norm_squared) / denominator

    def implicit_value(self, points: np.ndarray) -> np.ndarray:
        base = self._mobius_inverse(points)
        radial = np.linalg.norm(base[..., :6], axis=-1)
        return (
            (radial - self.major_radius) ** 2
            + np.sum(base[..., 6:] ** 2, axis=-1)
            - self.tube_radius**2
        )

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
            "geometry": {
                "major_radius": self.major_radius,
                "tube_radius": self.tube_radius,
                "mobius_b": self.mobius_b.tolist(),
            },
        }
