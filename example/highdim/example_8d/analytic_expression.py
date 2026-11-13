"""Analytic implicit expression for the 8D paper example."""

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


class CoupledT3xS4_8D:
    """Coupled T^3 x S^4 hypersurface in R^8."""

    name = "example_8d"
    family = "coupled multi-torus ellipsoidal-fiber hypersurface"
    topology = "T^3 x S^4"
    ambient_dim = 8
    intrinsic_dim = 7

    @staticmethod
    def _components(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = np.asarray(angles, dtype=np.float64)
        a0, a1, a2 = a[:, 0], a[:, 1], a[:, 2]
        center = np.column_stack(
            (
                1.58
                + 0.16 * np.cos(2.0 * a1)
                + 0.060 * np.sin(3.0 * a0 + a2),
                1.31
                + 0.14 * np.sin(3.0 * a0)
                + 0.055 * np.cos(2.0 * a1 - 3.0 * a2),
                1.12
                + 0.12 * np.cos(2.0 * a2)
                + 0.050 * np.sin(a0 + 4.0 * a1),
                0.055 * np.sin(2.0 * a0 + 3.0 * a1 - a2),
                0.045 * np.cos(a0 - 2.0 * a1 + 3.0 * a2),
            )
        )
        radius = 0.235 * (
            1.0
            + 0.11 * np.cos(2.0 * a0 - a1)
            + 0.075 * np.sin(a1 + 3.0 * a2)
            + 0.040 * np.cos(3.0 * a0 - 2.0 * a2)
        )
        scales = np.column_stack(
            (
                1.00 + 0.16 * np.sin(a0 + 2.0 * a1),
                0.91 + 0.12 * np.cos(3.0 * a1 - a2),
                1.10 + 0.14 * np.sin(2.0 * a2 - a0),
                0.86 + 0.11 * np.cos(a0 + a1 + a2),
                1.16 + 0.10 * np.sin(2.0 * a0 - 3.0 * a2),
            )
        )
        first = _plane_rotation(
            len(a), 5, 0, 3, 0.43 * np.sin(2.0 * a0 + a1)
        )
        second = _plane_rotation(
            len(a), 5, 1, 4, 0.37 * np.cos(a1 - 3.0 * a2)
        )
        third = _plane_rotation(
            len(a), 5, 2, 3, 0.31 * np.sin(a0 + a1 - 2.0 * a2)
        )
        rotation = np.einsum("nij,njk,nkl->nil", first, second, third)
        matrix = rotation * (radius[:, None] * scales)[:, None, :]
        return center, matrix

    def implicit_value(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        original_shape = values.shape[:-1]
        flat = values.reshape(-1, self.ambient_dim)
        angles = np.column_stack(
            (
                np.arctan2(flat[:, 1], flat[:, 0]),
                np.arctan2(flat[:, 3], flat[:, 2]),
                np.arctan2(flat[:, 5], flat[:, 4]),
            )
        )
        center, matrix = self._components(angles)
        local = np.column_stack(
            (
                np.hypot(flat[:, 0], flat[:, 1]) - center[:, 0],
                np.hypot(flat[:, 2], flat[:, 3]) - center[:, 1],
                np.hypot(flat[:, 4], flat[:, 5]) - center[:, 2],
                flat[:, 6] - center[:, 3],
                flat[:, 7] - center[:, 4],
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
