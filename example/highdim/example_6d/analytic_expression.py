"""Analytic implicit expression for the 6D paper example."""

from __future__ import annotations

import numpy as np


DEFAULT_PARAMS = {"R1": 1.45, "R2": 1.08, "r0": 0.24}


def _components(
    theta1: np.ndarray,
    theta2: np.ndarray,
    **params: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = {**DEFAULT_PARAMS, **params}
    t1 = np.asarray(theta1, dtype=np.float64)
    t2 = np.asarray(theta2, dtype=np.float64)
    center = np.column_stack(
        (
            p["R1"]
            + 0.20 * np.cos(2.0 * t2)
            + 0.080 * np.sin(3.0 * t1 + t2)
            + 0.045 * np.cos(5.0 * t1 - 2.0 * t2),
            p["R2"]
            + 0.16 * np.sin(3.0 * t1)
            + 0.070 * np.cos(2.0 * t1 - 3.0 * t2)
            + 0.040 * np.sin(t1 + 4.0 * t2),
            0.060 * np.sin(2.0 * t1 + 3.0 * t2),
            0.050 * np.cos(3.0 * t1 - 2.0 * t2),
        )
    )
    radius = p["r0"] * (
        1.0
        + 0.14 * np.cos(2.0 * t1 - t2)
        + 0.08 * np.sin(t1 + 3.0 * t2)
    )
    scales = np.column_stack(
        (
            1.00 + 0.22 * np.sin(t1 + 2.0 * t2),
            0.92 + 0.16 * np.cos(3.0 * t1 - t2),
            1.08 + 0.18 * np.sin(2.0 * t1 - 3.0 * t2),
            0.86 + 0.13 * np.cos(t1 + 4.0 * t2),
        )
    )
    alpha = 0.48 * np.sin(2.0 * t1 + t2)
    beta = 0.38 * np.cos(t1 - 3.0 * t2)
    return center, radius, scales, alpha, beta


class CoupledTwistedMultiharmonicTorus6D:
    """Coupled multiharmonic T^2 x S^3 hypersurface in R^6."""

    name = "example_6d"
    family = "coupled multiharmonic torus with anisotropic twisted fiber"
    topology = "T^2 x S^3"
    ambient_dim = 6
    intrinsic_dim = 5

    def implicit_value(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        theta1 = np.arctan2(values[..., 1], values[..., 0])
        theta2 = np.arctan2(values[..., 3], values[..., 2])
        center, radius, scales, alpha, beta = _components(
            theta1.reshape(-1), theta2.reshape(-1)
        )
        rho1 = np.hypot(values[..., 0], values[..., 1]).reshape(-1)
        rho2 = np.hypot(values[..., 2], values[..., 3]).reshape(-1)
        local = np.column_stack(
            (
                rho1 - center[:, 0],
                rho2 - center[:, 1],
                values[..., 4].reshape(-1) - center[:, 2],
                values[..., 5].reshape(-1) - center[:, 3],
            )
        )
        cosine_a, sine_a = np.cos(alpha), np.sin(alpha)
        cosine_b, sine_b = np.cos(beta), np.sin(beta)
        untwisted = np.empty_like(local)
        untwisted[:, 0] = cosine_a * local[:, 0] + sine_a * local[:, 2]
        untwisted[:, 2] = -sine_a * local[:, 0] + cosine_a * local[:, 2]
        untwisted[:, 1] = cosine_b * local[:, 1] + sine_b * local[:, 3]
        untwisted[:, 3] = -sine_b * local[:, 1] + cosine_b * local[:, 3]
        result = np.sum(
            (untwisted / (radius[:, None] * scales)) ** 2, axis=1
        ) - 1.0
        return result.reshape(theta1.shape)

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
            "geometry": dict(DEFAULT_PARAMS),
        }
