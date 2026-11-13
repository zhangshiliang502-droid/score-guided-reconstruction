#!/usr/bin/env python3
"""Generate the six analytic high-dimensional paper training sets.

The geometry is loaded from the ``analytic_expression.py`` adjacent to a paper
configuration.  This keeps the exact reference equation in one place while
adding the surface-area rejection sampler needed to reproduce the training
arrays.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


GENERATOR_SCHEMA = "paper-highdim-dataset/v1"
SUPPORTED_SOURCES = {
    "wavy_tube_s1xs2_4d",
    "coupled_twisted_multiharmonic_torus6d_v2",
    "coupled_t3xs4_8d",
    "coupled_t4xs4_9d",
    "mobius_clifford_s4xs5_10d",
    "mobius_clifford_s5xs6_12d",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def resolve_experiment_path(configuration: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (configuration.parent.parent / path).resolve()


def load_analytic_definition(
    configuration: Path,
    config: dict[str, Any],
) -> tuple[Any, Any]:
    definition = config.get("analytic_definition", {})
    module_path = resolve_experiment_path(configuration, definition.get("module", ""))
    class_name = str(definition.get("class", ""))
    if not module_path.is_file() or not class_name:
        raise ValueError(f"Invalid analytic_definition in {configuration}")
    module_name = f"sgr_paper_generator_{configuration.parent.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load analytic module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    reference_class = getattr(module, class_name, None)
    if reference_class is None:
        raise AttributeError(f"{module_path} does not define {class_name}")
    return module, reference_class()


def unit_sphere(
    rng: np.random.Generator,
    count: int,
    coordinate_dim: int,
) -> np.ndarray:
    values = rng.standard_normal((int(count), int(coordinate_dim)))
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-15)
    return values


def sphere_area(dimension: int) -> float:
    value = int(dimension)
    return float(
        2.0
        * math.pi ** ((value + 1.0) / 2.0)
        / math.gamma((value + 1.0) / 2.0)
    )


def sphere_tangent_basis(points: np.ndarray) -> np.ndarray:
    sphere = np.asarray(points, dtype=np.float64)
    coordinate_dim = int(sphere.shape[1])
    pole = np.zeros_like(sphere)
    pole[:, -1] = 1.0
    vector = pole - sphere
    denominator = np.sum(vector * vector, axis=1)
    seed = np.eye(coordinate_dim, dtype=np.float64)[:, : coordinate_dim - 1]
    basis = np.broadcast_to(
        seed[None, :, :],
        (len(sphere), *seed.shape),
    ).copy()
    regular = denominator > 1e-14
    if np.any(regular):
        selected = vector[regular]
        projection = np.einsum("bi,ij->bj", selected, seed)
        basis[regular] -= (
            2.0
            * selected[:, :, None]
            * (projection / denominator[regular, None])[:, None, :]
        )
    return basis


def area_density_from_jacobian(jacobian: np.ndarray) -> np.ndarray:
    gram = np.einsum("nai,naj->nij", jacobian, jacobian)
    sign, logdet = np.linalg.slogdet(gram)
    return np.where(sign > 0, np.exp(0.5 * logdet), 0.0)


class GeometryAdapter:
    """Sampling parameterization paired with one analytic reference class."""

    def __init__(
        self,
        config: dict[str, Any],
        module: Any,
        reference: Any,
    ) -> None:
        self.config = config
        self.module = module
        self.reference = reference
        self.source_name = str(config["source_name"])
        if self.source_name not in SUPPORTED_SOURCES:
            raise ValueError(
                f"Unsupported source_name={self.source_name!r}; "
                f"choices={sorted(SUPPORTED_SOURCES)}"
            )
        self.ambient_dim = int(config["ambient_dim"])
        self.intrinsic_dim = int(config["intrinsic_dim"])
        if int(getattr(reference, "ambient_dim", -1)) != self.ambient_dim:
            raise ValueError("Analytic class and configuration dimensions differ")

    @property
    def is_mobius(self) -> bool:
        return self.source_name.startswith("mobius_clifford_")

    @property
    def is_special_6d(self) -> bool:
        return self.source_name == "coupled_twisted_multiharmonic_torus6d_v2"

    @property
    def torus_count(self) -> int:
        return {
            "wavy_tube_s1xs2_4d": 1,
            "coupled_twisted_multiharmonic_torus6d_v2": 2,
            "coupled_t3xs4_8d": 3,
            "coupled_t4xs4_9d": 4,
        }.get(self.source_name, 0)

    @property
    def fiber_dim(self) -> int:
        if not self.torus_count:
            raise ValueError("Mobius products do not use the torus-fiber adapter")
        return self.ambient_dim - self.torus_count

    @property
    def parameter_volume(self) -> float:
        if self.is_mobius:
            left_coordinate_dim = self.ambient_dim // 2
            right_coordinate_dim = self.ambient_dim - left_coordinate_dim + 1
            return sphere_area(left_coordinate_dim - 1) * sphere_area(
                right_coordinate_dim - 1
            )
        return (2.0 * math.pi) ** self.torus_count * sphere_area(
            self.fiber_dim - 1
        )

    def sample_parameters(
        self,
        rng: np.random.Generator,
        count: int,
    ) -> dict[str, np.ndarray]:
        size = int(count)
        if self.source_name == "wavy_tube_s1xs2_4d":
            return {
                "theta": rng.uniform(0.0, 2.0 * math.pi, size=size),
                "sphere": unit_sphere(rng, size, 3),
            }
        if self.is_special_6d:
            return {
                "theta1": rng.uniform(0.0, 2.0 * math.pi, size=size),
                "theta2": rng.uniform(0.0, 2.0 * math.pi, size=size),
                "sphere": unit_sphere(rng, size, 4),
            }
        if self.source_name in {"coupled_t3xs4_8d", "coupled_t4xs4_9d"}:
            return {
                "angles": rng.uniform(
                    0.0,
                    2.0 * math.pi,
                    size=(size, self.torus_count),
                ),
                "sphere": unit_sphere(rng, size, 5),
            }
        left_coordinate_dim = self.ambient_dim // 2
        right_coordinate_dim = self.ambient_dim - left_coordinate_dim + 1
        return {
            "left": unit_sphere(rng, size, left_coordinate_dim),
            "right": unit_sphere(rng, size, right_coordinate_dim),
        }

    def angles(self, parameters: dict[str, np.ndarray]) -> np.ndarray:
        if self.source_name == "wavy_tube_s1xs2_4d":
            return np.asarray(parameters["theta"], dtype=np.float64)[:, None]
        if self.is_special_6d:
            return np.column_stack(
                (
                    np.asarray(parameters["theta1"], dtype=np.float64),
                    np.asarray(parameters["theta2"], dtype=np.float64),
                )
            )
        return np.asarray(parameters["angles"], dtype=np.float64)

    def torus_components(
        self,
        parameters: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        angles = self.angles(parameters)
        if self.source_name == "wavy_tube_s1xs2_4d":
            return self.reference._components(angles[:, 0])
        if self.source_name in {"coupled_t3xs4_8d", "coupled_t4xs4_9d"}:
            return self.reference._components(angles)
        center, radius, scales, alpha, beta = self.module._components(
            angles[:, 0],
            angles[:, 1],
        )
        matrix = np.zeros((len(angles), 4, 4), dtype=np.float64)
        scaled = radius[:, None] * scales
        cosine_alpha, sine_alpha = np.cos(alpha), np.sin(alpha)
        cosine_beta, sine_beta = np.cos(beta), np.sin(beta)
        matrix[:, 0, 0] = cosine_alpha * scaled[:, 0]
        matrix[:, 0, 2] = -sine_alpha * scaled[:, 2]
        matrix[:, 2, 0] = sine_alpha * scaled[:, 0]
        matrix[:, 2, 2] = cosine_alpha * scaled[:, 2]
        matrix[:, 1, 1] = cosine_beta * scaled[:, 1]
        matrix[:, 1, 3] = -sine_beta * scaled[:, 3]
        matrix[:, 3, 1] = sine_beta * scaled[:, 1]
        matrix[:, 3, 3] = cosine_beta * scaled[:, 3]
        return center, matrix

    def with_angles(
        self,
        parameters: dict[str, np.ndarray],
        angles: np.ndarray,
    ) -> dict[str, np.ndarray]:
        updated = dict(parameters)
        if self.source_name == "wavy_tube_s1xs2_4d":
            updated["theta"] = angles[:, 0]
        elif self.is_special_6d:
            updated["theta1"] = angles[:, 0]
            updated["theta2"] = angles[:, 1]
        else:
            updated["angles"] = angles
        return updated

    def parameterize(self, parameters: dict[str, np.ndarray]) -> np.ndarray:
        if self.is_mobius:
            left = np.asarray(parameters["left"], dtype=np.float64)
            right = np.asarray(parameters["right"], dtype=np.float64)
            radial = float(self.reference.major_radius) + float(
                self.reference.tube_radius
            ) * right[:, 0]
            base = np.column_stack(
                (
                    radial[:, None] * left,
                    float(self.reference.tube_radius) * right[:, 1:],
                )
            )
            norm_squared = np.sum(base * base, axis=1, keepdims=True)
            vector = np.asarray(self.reference.mobius_b, dtype=np.float64)
            vector_squared = float(np.dot(vector, vector))
            denominator = (
                1.0
                - 2.0 * np.sum(base * vector[None, :], axis=1, keepdims=True)
                + vector_squared * norm_squared
            )
            return (base - vector[None, :] * norm_squared) / denominator

        angles = self.angles(parameters)
        sphere = np.asarray(parameters["sphere"], dtype=np.float64)
        center, matrix = self.torus_components(parameters)
        fiber = np.einsum("nij,nj->ni", matrix, sphere)
        columns: list[np.ndarray] = []
        for axis in range(self.torus_count):
            radial = center[:, axis] + fiber[:, axis]
            columns.extend(
                (
                    radial * np.cos(angles[:, axis]),
                    radial * np.sin(angles[:, axis]),
                )
            )
        for axis in range(self.torus_count, self.fiber_dim):
            columns.append(center[:, axis] + fiber[:, axis])
        return np.column_stack(columns)

    def area_density(self, parameters: dict[str, np.ndarray]) -> np.ndarray:
        if self.is_mobius:
            left = np.asarray(parameters["left"], dtype=np.float64)
            right = np.asarray(parameters["right"], dtype=np.float64)
            radius = float(self.reference.major_radius)
            tube = float(self.reference.tube_radius)
            radial = radius + tube * right[:, 0]
            base = np.column_stack((radial[:, None] * left, tube * right[:, 1:]))
            vector = np.asarray(self.reference.mobius_b, dtype=np.float64)
            denominator = (
                1.0
                - 2.0 * np.sum(base * vector[None, :], axis=1)
                + float(np.dot(vector, vector)) * np.sum(base * base, axis=1)
            )
            left_sphere_dim = left.shape[1] - 1
            right_sphere_dim = right.shape[1] - 1
            return (
                radial**left_sphere_dim
                * tube**right_sphere_dim
                / denominator**self.intrinsic_dim
            )

        angles = self.angles(parameters)
        sphere = np.asarray(parameters["sphere"], dtype=np.float64)
        epsilon = 2e-5
        angle_tangents: list[np.ndarray] = []
        for axis in range(self.torus_count):
            plus = angles.copy()
            minus = angles.copy()
            plus[:, axis] += epsilon
            minus[:, axis] -= epsilon
            angle_tangents.append(
                (
                    self.parameterize(self.with_angles(parameters, plus))
                    - self.parameterize(self.with_angles(parameters, minus))
                )
                / (2.0 * epsilon)
            )
        _, matrix = self.torus_components(parameters)
        fiber_tangent = np.einsum(
            "nij,njk->nik",
            matrix,
            sphere_tangent_basis(sphere),
        )
        tangent = np.zeros(
            (
                len(angles),
                self.ambient_dim,
                self.fiber_dim - 1,
            ),
            dtype=np.float64,
        )
        for axis in range(self.torus_count):
            tangent[:, 2 * axis, :] = (
                np.cos(angles[:, axis])[:, None] * fiber_tangent[:, axis, :]
            )
            tangent[:, 2 * axis + 1, :] = (
                np.sin(angles[:, axis])[:, None] * fiber_tangent[:, axis, :]
            )
        output_axis = 2 * self.torus_count
        for fiber_axis in range(self.torus_count, self.fiber_dim):
            tangent[:, output_axis, :] = fiber_tangent[:, fiber_axis, :]
            output_axis += 1
        jacobian = np.concatenate(
            tuple(value[:, :, None] for value in angle_tangents) + (tangent,),
            axis=2,
        )
        return area_density_from_jacobian(jacobian)

    def metadata(self) -> dict[str, Any]:
        metadata_method = getattr(self.reference, "metadata", None)
        metadata = (
            metadata_method()
            if callable(metadata_method)
            else {
                "family": str(getattr(self.reference, "family", "")),
                "topology": str(self.config["topology"]),
            }
        )
        return {
            **metadata,
            "source_name": self.source_name,
            "ambient_dim": self.ambient_dim,
            "intrinsic_dim": self.intrinsic_dim,
            "codimension": 1,
        }


def normalization_statistics(
    points: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    # Match the training pipeline's float32 mean and scale contract exactly.
    mean = np.mean(points, axis=0, dtype=np.float32).astype(np.float32)
    low = np.full(points.shape[1], np.inf, dtype=np.float64)
    high = np.full(points.shape[1], -np.inf, dtype=np.float64)
    scale = 0.0
    for start in range(0, len(points), int(batch_size)):
        batch = np.asarray(points[start : start + int(batch_size)], dtype=np.float32)
        low = np.minimum(low, batch.min(axis=0))
        high = np.maximum(high, batch.max(axis=0))
        scale = max(
            scale,
            float(np.max(np.linalg.norm(batch - mean[None, :], axis=1))),
        )
    return mean, scale, low, high


def validate_density(density: np.ndarray) -> None:
    if not np.all(np.isfinite(density)) or float(np.min(density)) <= 0.0:
        raise RuntimeError("Area density contains non-positive or non-finite values")


def allocate_output(out: Path, count: int, dimension: int) -> np.ndarray:
    out.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        out,
        mode="w+",
        dtype=np.float32,
        shape=(int(count), int(dimension)),
    )


def generate_standard(
    geometry: GeometryAdapter,
    out: Path,
    *,
    count: int,
    seed: int,
    batch_size: int,
    pilot_size: int,
    density_bound_factor: float,
) -> tuple[np.ndarray, dict[str, Any], np.random.Generator]:
    rng = np.random.default_rng(int(seed))
    pilot = geometry.sample_parameters(rng, int(pilot_size))
    pilot_density = geometry.area_density(pilot)
    validate_density(pilot_density)
    density_bound = float(np.max(pilot_density) * float(density_bound_factor))
    acceptance_estimate = float(np.mean(pilot_density) / density_bound)
    points = allocate_output(out, count, geometry.ambient_dim)
    accepted = 0
    proposed = 0
    observed_max = float(np.max(pilot_density))
    started = time.perf_counter()
    while accepted < int(count):
        remaining = int(count) - accepted
        proposal_count = max(
            int(batch_size),
            int(math.ceil(remaining / max(acceptance_estimate * 0.90, 0.01))),
        )
        proposal_count = min(
            proposal_count,
            max(int(batch_size), 262_144),
        )
        parameters = geometry.sample_parameters(rng, proposal_count)
        density = geometry.area_density(parameters)
        validate_density(density)
        observed_max = max(observed_max, float(np.max(density)))
        if np.any(density > density_bound * (1.0 + 1e-10)):
            raise RuntimeError(
                "Area-density bound violated: "
                f"observed={float(np.max(density))}, bound={density_bound}; "
                "increase --density-bound-factor"
            )
        keep = rng.random(proposal_count) < density / density_bound
        if np.any(keep):
            selected = {key: value[keep] for key, value in parameters.items()}
            batch_points = geometry.parameterize(selected)
            take = min(len(batch_points), remaining)
            points[accepted : accepted + take] = batch_points[:take].astype(
                np.float32
            )
            accepted += take
        proposed += proposal_count
        print(
            json.dumps(
                {
                    "event": "sampling",
                    "source_name": geometry.source_name,
                    "accepted": accepted,
                    "target": int(count),
                    "proposed": proposed,
                    "elapsed_sec": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    points.flush()
    statistics = {
        "pilot_size": int(pilot_size),
        "pilot_density_mean": float(np.mean(pilot_density)),
        "pilot_density_max": float(np.max(pilot_density)),
        "density_bound_factor": float(density_bound_factor),
        "density_bound": density_bound,
        "observed_density_max": observed_max,
        "proposed": int(proposed),
        "acceptance_rate": float(count / proposed),
        "surface_measure_estimate": float(
            geometry.parameter_volume * np.mean(pilot_density)
        ),
    }
    return points, statistics, rng


def generate_special_6d(
    geometry: GeometryAdapter,
    out: Path,
    *,
    count: int,
    seed: int,
    batch_size: int,
    density_bound: float,
) -> tuple[np.ndarray, dict[str, Any], np.random.Generator]:
    # Preserve the original 6D sampler's random-number order and proposal rule.
    rng = np.random.default_rng(int(seed))
    points = allocate_output(out, count, geometry.ambient_dim)
    accepted = 0
    proposed = 0
    observed_max = 0.0
    while accepted < int(count):
        proposal_count = min(
            int(batch_size),
            max(20_000, 2 * (int(count) - accepted)),
        )
        parameters = geometry.sample_parameters(rng, proposal_count)
        density = geometry.area_density(parameters)
        validate_density(density)
        observed_max = max(observed_max, float(np.max(density)))
        if observed_max > float(density_bound) * (1.0 + 1e-10):
            raise RuntimeError(
                "Area-density bound violated: "
                f"observed={observed_max}, bound={density_bound}"
            )
        keep = rng.random(proposal_count) < density / float(density_bound)
        if np.any(keep):
            selected = {key: value[keep] for key, value in parameters.items()}
            batch_points = geometry.parameterize(selected)
            take = min(len(batch_points), int(count) - accepted)
            points[accepted : accepted + take] = batch_points[:take].astype(
                np.float32
            )
            accepted += take
        proposed += proposal_count
        print(
            json.dumps(
                {
                    "event": "sampling",
                    "source_name": geometry.source_name,
                    "accepted": accepted,
                    "target": int(count),
                    "proposed": proposed,
                }
            ),
            flush=True,
        )
    points.flush()
    return (
        points,
        {
            "proposed": int(proposed),
            "acceptance_rate": float(count / proposed),
            "area_density_bound": float(density_bound),
            "observed_area_density_max": observed_max,
        },
        rng,
    )


def build_metadata(
    geometry: GeometryAdapter,
    points: np.ndarray,
    rng: np.random.Generator,
    *,
    configuration: Path,
    out: Path,
    count: int,
    seed: int,
    batch_size: int,
    sampling: str,
    sampling_stats: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    mean, scale, low, high = normalization_statistics(points, batch_size)
    diagnostic_count = min(20_000, int(count))
    choice = rng.choice(int(count), size=diagnostic_count, replace=False)
    diagnostic = np.asarray(points[choice], dtype=np.float64)
    residual = np.abs(geometry.reference.implicit_value(diagnostic))
    gradient = geometry.reference.implicit_gradient(diagnostic)
    return {
        "schema": GENERATOR_SCHEMA,
        **geometry.metadata(),
        "configuration": str(configuration),
        "point_cloud": str(out),
        "num_points": int(count),
        "dtype": "float32",
        "seed": int(seed),
        "sampling": sampling,
        "sampling_target": "surface_area_uniform",
        "sampling_stats": sampling_stats,
        "normalization": {
            "mean_accumulator": "float32",
            "data_mean": mean.astype(float).tolist(),
            "data_scale_max_center_radius": float(scale),
        },
        "bounding_box": {
            "min": low.astype(float).tolist(),
            "max": high.astype(float).tolist(),
        },
        "surface_verification": {
            "implicit_abs_max": float(np.max(residual)),
            "implicit_abs_mean": float(np.mean(residual)),
            "gradient_norm_min_sample20k": float(
                np.min(np.linalg.norm(gradient, axis=1))
            ),
        },
        "elapsed_sec": float(elapsed),
        "points_per_sec": float(count / max(elapsed, 1e-12)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True)
    parser.add_argument(
        "--out",
        default="",
        help="Override data.path from the configuration.",
    )
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--pilot-size", type=int, default=65_536)
    parser.add_argument("--density-bound-factor", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_output_path(
    args: argparse.Namespace,
    configuration: Path,
    data_config: dict[str, Any],
) -> Path:
    if args.out:
        return Path(args.out).expanduser().resolve()
    if not data_config.get("path"):
        raise ValueError(f"{configuration} does not define data.path; provide --out")
    return resolve_experiment_path(configuration, data_config["path"])


def validate_output_paths(
    out: Path,
    metadata_path: Path,
    *,
    overwrite: bool,
) -> None:
    if (out.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {out}; pass --overwrite explicitly"
        )


def generation_parameters(
    args: argparse.Namespace,
    geometry: GeometryAdapter,
    data_config: dict[str, Any],
) -> tuple[int, int, int]:
    count = int(args.n if args.n is not None else data_config["samples"])
    seed = int(args.seed if args.seed is not None else data_config["seed"])
    if count <= 0:
        raise ValueError("--n must be positive")
    default_batch_size = 100_000 if geometry.is_special_6d else 32_768
    batch_size = int(args.batch_size or default_batch_size)
    if batch_size <= 0 or int(args.pilot_size) <= 0:
        raise ValueError("batch-size and pilot-size must be positive")
    return count, seed, batch_size


def generate_points(
    geometry: GeometryAdapter,
    out: Path,
    args: argparse.Namespace,
    data_config: dict[str, Any],
    *,
    count: int,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any], np.random.Generator]:
    if not geometry.is_special_6d:
        return generate_standard(
            geometry,
            out,
            count=count,
            seed=seed,
            batch_size=batch_size,
            pilot_size=int(args.pilot_size),
            density_bound_factor=float(args.density_bound_factor),
        )

    configured_stats = data_config.get("sampling_stats", {})
    density_bound = float(
        configured_stats.get("area_density_bound", float("nan"))
    )
    if not np.isfinite(density_bound) or density_bound <= 0.0:
        raise ValueError(
            "The 6D configuration must record sampling_stats.area_density_bound"
        )
    return generate_special_6d(
        geometry,
        out,
        count=count,
        seed=seed,
        batch_size=batch_size,
        density_bound=density_bound,
    )


def write_completion_event(
    out: Path,
    metadata_path: Path,
    count: int,
    ambient_dim: int,
) -> None:
    print(
        json.dumps(
            {
                "event": "done",
                "point_cloud": str(out),
                "metadata": str(metadata_path),
                "num_points": count,
                "ambient_dim": ambient_dim,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> None:
    args = build_arg_parser().parse_args()

    configuration = Path(args.configuration).expanduser().resolve()
    config = load_json(configuration)
    data_config = config.get("data", {})
    out = resolve_output_path(args, configuration, data_config)
    metadata_path = out.with_suffix(".json")
    validate_output_paths(out, metadata_path, overwrite=bool(args.overwrite))

    module, reference = load_analytic_definition(configuration, config)
    geometry = GeometryAdapter(config, module, reference)
    count, seed, batch_size = generation_parameters(args, geometry, data_config)

    started = time.perf_counter()
    points, sampling_stats, rng = generate_points(
        geometry,
        out,
        args,
        data_config,
        count=count,
        seed=seed,
        batch_size=batch_size,
    )
    metadata = build_metadata(
        geometry,
        points,
        rng,
        configuration=configuration,
        out=out,
        count=count,
        seed=seed,
        batch_size=batch_size,
        sampling=str(data_config["sampling"]),
        sampling_stats=sampling_stats,
        elapsed=time.perf_counter() - started,
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_completion_event(
        out,
        metadata_path,
        count,
        geometry.ambient_dim,
    )


if __name__ == "__main__":
    main()
