"""Analytic implicit expression for the 9D paper example."""

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


class CoupledT4xS4_9D:
    """Coupled T^4 x S^4 hypersurface in R^9."""

    name = "example_9d"
    family = "four-cycle coupled torus with anisotropic S4 fiber"
    topology = "T^4 x S^4"
    ambient_dim = 9
    intrinsic_dim = 8

    @staticmethod
    def _components(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = np.asarray(angles, dtype=np.float64)
        a0, a1, a2, a3 = (a[:, index] for index in range(4))
        center = np.column_stack(
            (
                1.68
                + 0.15 * np.cos(2.0 * a1)
                + 0.055 * np.sin(3.0 * a0 + a2 - a3),
                1.46
                + 0.13 * np.sin(3.0 * a0)
                + 0.050 * np.cos(2.0 * a1 - 3.0 * a2 + a3),
                1.25
                + 0.12 * np.cos(2.0 * a2)
                + 0.046 * np.sin(a0 + 4.0 * a1 - a3),
                1.08
                + 0.10 * np.sin(2.0 * a3)
                + 0.042 * np.cos(2.0 * a0 - a1 + 3.0 * a2),
                0.048 * np.sin(2.0 * a0 + a1 - 2.0 * a2 + 3.0 * a3),
            )
        )
        radius = 0.205 * (
            1.0
            + 0.10 * np.cos(2.0 * a0 - a1 + a3)
            + 0.070 * np.sin(a1 + 3.0 * a2 - a3)
            + 0.035 * np.cos(3.0 * a0 - 2.0 * a2 + 2.0 * a3)
        )
        scales = np.column_stack(
            (
                1.00 + 0.15 * np.sin(a0 + 2.0 * a1 - a3),
                0.90 + 0.12 * np.cos(3.0 * a1 - a2 + a3),
                1.11 + 0.13 * np.sin(2.0 * a2 - a0 + a3),
                0.85 + 0.10 * np.cos(a0 + a1 + a2 - 2.0 * a3),
                1.16 + 0.09 * np.sin(2.0 * a0 - 3.0 * a2 + a3),
            )
        )
        rotations = (
            _plane_rotation(
                len(a), 5, 0, 4, 0.41 * np.sin(2.0 * a0 + a1 - a3)
            ),
            _plane_rotation(
                len(a), 5, 1, 3, 0.35 * np.cos(a1 - 3.0 * a2 + a3)
            ),
            _plane_rotation(
                len(a), 5, 2, 4, 0.30 * np.sin(a0 + a1 - 2.0 * a2 + a3)
            ),
            _plane_rotation(
                len(a), 5, 0, 2, 0.24 * np.cos(2.0 * a0 - a2 - 2.0 * a3)
            ),
        )
        rotation = rotations[0]
        for next_rotation in rotations[1:]:
            rotation = np.einsum("nij,njk->nik", rotation, next_rotation)
        matrix = rotation * (radius[:, None] * scales)[:, None, :]
        return center, matrix

    def implicit_value(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        original_shape = values.shape[:-1]
        flat = values.reshape(-1, self.ambient_dim)
        angles = np.column_stack(
            [
                np.arctan2(flat[:, 2 * axis + 1], flat[:, 2 * axis])
                for axis in range(4)
            ]
        )
        center, matrix = self._components(angles)
        local = np.column_stack(
            [
                np.hypot(flat[:, 2 * axis], flat[:, 2 * axis + 1])
                - center[:, axis]
                for axis in range(4)
            ]
            + [flat[:, 8] - center[:, 4]]
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
