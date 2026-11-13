#!/usr/bin/env python3
"""Estimate surface area from a point cloud and plan Gaussian-KNN truncation.

The workflow is deliberately limited to a two-dimensional manifold embedded
in three dimensions:

1. load a point cloud and estimate its surface area from multi-scale KNN
   radii;
2. accept the intended surface-support sample count and Gaussian sigma; and
3. report the KNN sizes needed to reach beyond the radii containing 98% and
   99% of the tangent-plane Gaussian kernel mass.

Unless ``--no-normalize`` is passed, points are normalized exactly like the
public low-dimensional SGR cache builder. Consequently, ``sigma`` is also
interpreted in normalized coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import poisson


DEFAULT_AREA_NEIGHBORS = (16, 32, 64)
DEFAULT_AREA_PROBES = 100_000
DEFAULT_COVERAGE_CONFIDENCE = 0.99
TARGET_GAUSSIAN_MASSES = (0.98, 0.99)
SCHEMA = "sgr-gaussian-knn-plan/v2"


@dataclass(frozen=True)
class AreaEstimate:
    method: str
    coordinate_space: str
    input_points: int
    finite_points: int
    unique_points: int
    duplicate_points_removed: int
    probe_points: int
    area_neighbors: tuple[int, ...]
    area_by_neighbor_count: dict[str, float]
    estimated_surface_area: float
    estimated_surface_area_input_coordinates: float
    multiscale_relative_range: float
    normalization_mean: tuple[float, float, float] | None
    normalization_scale: float


@dataclass(frozen=True)
class GaussianKNNRequirement:
    gaussian_mass: float
    gaussian_radius: float
    expected_points_inside_radius: float
    expected_count_ceiling: int
    coverage_confidence_target: float
    knn_k: int
    knn_k_rounded: int
    achieved_coverage_confidence: float


def load_point_cloud_xyz(path: str | Path) -> tuple[np.ndarray, int]:
    """Load finite XYZ rows from an NPY, CSV, XYZ, or whitespace text file."""
    point_path = Path(path).expanduser().resolve()
    if not point_path.is_file():
        raise FileNotFoundError(point_path)

    if point_path.suffix.lower() == ".npy":
        raw = np.load(point_path, mmap_mode="r")
        points = np.asarray(raw)
    else:
        delimiter = "," if point_path.suffix.lower() == ".csv" else None
        first_nonempty = ""
        with point_path.open(
            "r", encoding="utf-8-sig", errors="replace"
        ) as stream:
            for line in stream:
                if line.strip():
                    first_nonempty = line.strip()
                    break
        if not first_nonempty:
            raise ValueError(f"Point-cloud file is empty: {point_path}")
        tokens = (
            first_nonempty.split(",")
            if delimiter == ","
            else first_nonempty.split()
        )
        try:
            [float(value) for value in tokens[:3]]
            skip_rows = 0
        except (TypeError, ValueError):
            skip_rows = 1
        points = np.loadtxt(
            point_path,
            delimiter=delimiter,
            skiprows=skip_rows,
            usecols=(0, 1, 2),
            dtype=np.float64,
        )

    points = np.asarray(points)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            f"Point cloud must have shape [N,3+], got {points.shape}: "
            f"{point_path}"
        )
    input_count = int(points.shape[0])
    points = np.asarray(points[:, :3], dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError(f"Point cloud contains no finite XYZ rows: {point_path}")
    return points, input_count


def normalize_like_sgr(
    points: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float], float]:
    """Apply the float32 mean/max-radius normalization used by SGR."""
    values = np.asarray(points, dtype=np.float32)
    mean = values.mean(axis=0, dtype=np.float32)
    centered = values - mean[None, :]
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Point cloud has invalid normalization scale: {scale}")
    normalized = np.asarray(centered / scale, dtype=np.float64)
    return normalized, tuple(float(value) for value in mean), scale


def parse_area_neighbors(text: str) -> tuple[int, ...]:
    """Parse a comma-separated sequence such as ``16,32,64``."""
    try:
        values = tuple(
            sorted(
                {
                    int(value.strip())
                    for value in str(text).split(",")
                    if value.strip()
                }
            )
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--area-neighbors must be comma-separated positive integers"
        ) from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(
            "--area-neighbors must be comma-separated positive integers"
        )
    return values


def _query_neighbor_distances(
    tree: cKDTree,
    probes: np.ndarray,
    *,
    maximum_neighbor: int,
    workers: int,
    chunk_size: int,
) -> Iterable[np.ndarray]:
    for start in range(0, len(probes), int(chunk_size)):
        chunk = probes[start : start + int(chunk_size)]
        try:
            distances, _ = tree.query(
                chunk,
                k=int(maximum_neighbor) + 1,
                workers=int(workers),
            )
        except TypeError:
            distances, _ = tree.query(
                chunk,
                k=int(maximum_neighbor) + 1,
            )
        distances = np.asarray(distances, dtype=np.float64)
        if distances.ndim != 2:
            raise RuntimeError("KNN area query returned an unexpected shape")
        yield distances


def prepare_area_coordinates(
    points: np.ndarray,
    *,
    normalize: bool,
) -> tuple[
    np.ndarray,
    int,
    tuple[float, float, float] | None,
    float,
    str,
]:
    """Filter finite rows, normalize if requested, and remove duplicates."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"points must have shape [N,3], got {values.shape}")
    finite = values[np.isfinite(values).all(axis=1)]
    finite_count = int(len(finite))
    if not finite_count:
        raise ValueError("points contains no finite rows")

    if normalize:
        working, mean, scale = normalize_like_sgr(finite)
        coordinate_space = "sgr_normalized"
    else:
        working = finite
        mean = None
        scale = 1.0
        coordinate_space = "input"
    return np.unique(working, axis=0), finite_count, mean, scale, coordinate_space


def validate_area_inputs(
    unique_count: int,
    area_neighbors: tuple[int, ...],
    *,
    max_probes: int,
    query_chunk_size: int,
) -> tuple[int, ...]:
    neighbors = tuple(sorted({int(value) for value in area_neighbors}))
    if not neighbors or any(value <= 0 for value in neighbors):
        raise ValueError("area_neighbors must contain positive integers")
    maximum_neighbor = max(neighbors)
    if unique_count <= maximum_neighbor:
        raise ValueError(
            f"Need more than {maximum_neighbor} unique points for area "
            f"estimation, found {unique_count}"
        )
    if int(max_probes) <= 0:
        raise ValueError("max_probes must be positive")
    if int(query_chunk_size) <= 0:
        raise ValueError("query_chunk_size must be positive")
    return neighbors


def choose_area_probes(
    points: np.ndarray,
    *,
    max_probes: int,
    seed: int,
) -> np.ndarray:
    probe_count = min(len(points), int(max_probes))
    if probe_count == len(points):
        return points
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(points), size=probe_count, replace=False)
    return points[indices]


def squared_knn_radius_sums(
    points: np.ndarray,
    probes: np.ndarray,
    neighbors: tuple[int, ...],
    *,
    workers: int,
    query_chunk_size: int,
) -> dict[int, float]:
    tree = cKDTree(points)
    sums = {neighbor: 0.0 for neighbor in neighbors}
    measured = 0
    for distances in _query_neighbor_distances(
        tree,
        probes,
        maximum_neighbor=max(neighbors),
        workers=int(workers),
        chunk_size=int(query_chunk_size),
    ):
        measured += int(len(distances))
        for neighbor in neighbors:
            radius = distances[:, neighbor]
            if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
                raise ValueError(
                    "KNN area radii are non-finite or zero; check for "
                    "degenerate/duplicate point-cloud coordinates"
                )
            sums[neighbor] += float(np.sum(np.square(radius), dtype=np.float64))
    if measured != len(probes):
        raise RuntimeError("KNN area probe count mismatch")
    return sums


def summarize_area_estimates(
    unique_count: int,
    probe_count: int,
    neighbors: tuple[int, ...],
    squared_radius_sums: dict[int, float],
) -> tuple[dict[str, float], float, float]:
    area_by_neighbor = {
        str(neighbor): float(
            math.pi
            * (unique_count - 1)
            * (squared_radius_sums[neighbor] / probe_count)
            / neighbor
        )
        for neighbor in neighbors
    }
    estimates = np.asarray(list(area_by_neighbor.values()), dtype=np.float64)
    estimated_area = float(np.median(estimates))
    if not math.isfinite(estimated_area) or estimated_area <= 0.0:
        raise ValueError(f"Invalid estimated surface area: {estimated_area}")
    relative_range = float(
        (float(np.max(estimates)) - float(np.min(estimates))) / estimated_area
    )
    return area_by_neighbor, estimated_area, relative_range


def estimate_surface_area_knn(
    points: np.ndarray,
    *,
    input_points: int | None = None,
    normalize: bool = True,
    area_neighbors: tuple[int, ...] = DEFAULT_AREA_NEIGHBORS,
    max_probes: int = DEFAULT_AREA_PROBES,
    seed: int = 20260801,
    workers: int = -1,
    query_chunk_size: int = 8192,
) -> AreaEstimate:
    """Estimate two-dimensional surface area from KNN disk radii.

    For a locally planar Poisson sample, ``rho*pi*r_k^2`` has mean ``k``.
    Averaging ``pi*(n-1)*r_k^2/k`` over query points therefore estimates the
    total surface area. Estimates at several neighborhood sizes expose the
    sensitivity to noise, curvature, boundaries, and non-manifold structure.
    """
    values = np.asarray(points, dtype=np.float64)
    (
        working,
        finite_count,
        normalization_mean,
        normalization_scale,
        coordinate_space,
    ) = prepare_area_coordinates(values, normalize=normalize)
    unique_count = int(len(working))
    neighbors = validate_area_inputs(
        unique_count,
        area_neighbors,
        max_probes=max_probes,
        query_chunk_size=query_chunk_size,
    )
    probes = choose_area_probes(working, max_probes=max_probes, seed=seed)
    squared_radius_sums = squared_knn_radius_sums(
        working,
        probes,
        neighbors,
        workers=int(workers),
        query_chunk_size=int(query_chunk_size),
    )
    area_by_neighbor, estimated_area, relative_range = summarize_area_estimates(
        unique_count,
        len(probes),
        neighbors,
        squared_radius_sums,
    )

    original_count = int(input_points if input_points is not None else len(values))
    return AreaEstimate(
        method="multiscale_knn_poisson_disk_2d",
        coordinate_space=coordinate_space,
        input_points=original_count,
        finite_points=finite_count,
        unique_points=unique_count,
        duplicate_points_removed=finite_count - unique_count,
        probe_points=int(len(probes)),
        area_neighbors=neighbors,
        area_by_neighbor_count=area_by_neighbor,
        estimated_surface_area=estimated_area,
        estimated_surface_area_input_coordinates=float(
            estimated_area * normalization_scale * normalization_scale
        ),
        multiscale_relative_range=relative_range,
        normalization_mean=normalization_mean,
        normalization_scale=float(normalization_scale),
    )


def gaussian_radius_for_mass(sigma: float, gaussian_mass: float) -> float:
    """Radius containing the requested 2D isotropic Gaussian kernel mass."""
    sigma = float(sigma)
    mass = float(gaussian_mass)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not 0.0 < mass < 1.0:
        raise ValueError("gaussian_mass must be in (0,1)")
    return sigma * math.sqrt(-2.0 * math.log1p(-mass))


def _round_up(value: int, multiple: int) -> int:
    if int(multiple) <= 0:
        raise ValueError("knn_round_up_multiple must be positive")
    return int(math.ceil(int(value) / int(multiple)) * int(multiple))


def compute_knn_requirement(
    *,
    surface_area: float,
    num_samples: int,
    sigma: float,
    gaussian_mass: float,
    coverage_confidence: float = DEFAULT_COVERAGE_CONFIDENCE,
    knn_round_up_multiple: int = 1,
) -> GaussianKNNRequirement:
    """Compute a conservative K whose K-th neighbor reaches the mass radius.

    Under uniform surface sampling, the count inside the target disk is
    modeled as Poisson. We choose ``K`` so that
    ``P[count_inside_radius <= K-1] >= coverage_confidence``. Thus the K-th
    neighbor reaches at least the requested Gaussian-mass radius with the
    stated confidence.
    """
    area = float(surface_area)
    count = int(num_samples)
    confidence = float(coverage_confidence)
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("surface_area must be finite and positive")
    if count <= 0:
        raise ValueError("num_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("coverage_confidence must be in (0,1)")

    radius = gaussian_radius_for_mass(float(sigma), float(gaussian_mass))
    density = count / area
    expected = density * math.pi * radius * radius
    poisson_quantile = int(poisson.ppf(confidence, expected))
    knn_k = poisson_quantile + 1
    rounded = _round_up(knn_k, int(knn_round_up_multiple))
    if rounded > count:
        raise ValueError(
            f"Required K={rounded} exceeds num_samples={count}; decrease "
            "sigma/confidence or increase the support sample count"
        )
    achieved = float(poisson.cdf(rounded - 1, expected))
    return GaussianKNNRequirement(
        gaussian_mass=float(gaussian_mass),
        gaussian_radius=float(radius),
        expected_points_inside_radius=float(expected),
        expected_count_ceiling=int(math.ceil(expected)),
        coverage_confidence_target=confidence,
        knn_k=knn_k,
        knn_k_rounded=rounded,
        achieved_coverage_confidence=achieved,
    )


def build_plan(
    points: np.ndarray,
    *,
    input_points: int | None = None,
    num_samples: int,
    sigma: float,
    normalize: bool = True,
    area_neighbors: tuple[int, ...] = DEFAULT_AREA_NEIGHBORS,
    area_probes: int = DEFAULT_AREA_PROBES,
    coverage_confidence: float = DEFAULT_COVERAGE_CONFIDENCE,
    knn_round_up_multiple: int = 1,
    seed: int = 20260801,
    workers: int = -1,
) -> dict[str, Any]:
    area = estimate_surface_area_knn(
        points,
        input_points=input_points,
        normalize=normalize,
        area_neighbors=area_neighbors,
        max_probes=int(area_probes),
        seed=int(seed),
        workers=int(workers),
    )
    requirements = [
        compute_knn_requirement(
            surface_area=area.estimated_surface_area,
            num_samples=int(num_samples),
            sigma=float(sigma),
            gaussian_mass=mass,
            coverage_confidence=float(coverage_confidence),
            knn_round_up_multiple=int(knn_round_up_multiple),
        )
        for mass in TARGET_GAUSSIAN_MASSES
    ]
    density = int(num_samples) / area.estimated_surface_area
    return {
        "schema": SCHEMA,
        "assumptions": {
            "intrinsic_dimension": 2,
            "embedding_dimension": 3,
            "surface_support_sampling": "uniform_with_respect_to_surface_area",
            "local_geometry": "approximately_planar_at_the_gaussian_scale",
            "neighbor_count_model": "homogeneous_poisson_on_local_tangent_plane",
            "gaussian_kernel": "exp(-r^2/(2*sigma^2))",
        },
        "step1_surface_area": asdict(area),
        "step2_sampling": {
            "num_samples": int(num_samples),
            "sigma": float(sigma),
            "coordinate_space": area.coordinate_space,
            "surface_density": float(density),
            "nominal_spacing_sqrt_area_over_n": float(
                math.sqrt(area.estimated_surface_area / int(num_samples))
            ),
        },
        "step3_knn": {
            "coverage_confidence": float(coverage_confidence),
            "requirements": [asdict(requirement) for requirement in requirements],
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--point-cloud",
        required=True,
        help="Input NPY/CSV/XYZ/TXT point cloud with shape [N,3+].",
    )
    parser.add_argument(
        "--num-samples",
        required=True,
        type=int,
        help="Number of surface-support samples used by Gaussian-KNN.",
    )
    parser.add_argument(
        "--sigma",
        required=True,
        type=float,
        help="Gaussian sigma in the same coordinate space as the area estimate.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help=(
            "Use input coordinates directly. By default the point cloud is "
            "normalized with the SGR mean/max-radius convention."
        ),
    )
    parser.add_argument(
        "--area-neighbors",
        type=parse_area_neighbors,
        default=DEFAULT_AREA_NEIGHBORS,
        help="K values for multi-scale area estimation (default: 16,32,64).",
    )
    parser.add_argument(
        "--area-probes",
        type=int,
        default=DEFAULT_AREA_PROBES,
        help="Maximum point-cloud rows used as area probes.",
    )
    parser.add_argument(
        "--coverage-confidence",
        type=float,
        default=DEFAULT_COVERAGE_CONFIDENCE,
        help="Probability that the K-th neighbor reaches the target radius.",
    )
    parser.add_argument(
        "--knn-round-up-multiple",
        type=int,
        default=1,
        help="Optionally round each recommended K up to this multiple.",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--json-out", type=Path)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    points, input_count = load_point_cloud_xyz(args.point_cloud)
    plan = build_plan(
        points,
        input_points=input_count,
        num_samples=int(args.num_samples),
        sigma=float(args.sigma),
        normalize=not bool(args.no_normalize),
        area_neighbors=tuple(args.area_neighbors),
        area_probes=int(args.area_probes),
        coverage_confidence=float(args.coverage_confidence),
        knn_round_up_multiple=int(args.knn_round_up_multiple),
        seed=int(args.seed),
        workers=int(args.workers),
    )
    output = json.dumps(plan, ensure_ascii=False, indent=2)
    print(output)
    if args.json_out:
        output_path = args.json_out.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
