#!/usr/bin/env python3
"""Evaluate one low-dimensional reconstruction with the paper protocol.

The script reports three quantities used in Tables 1--3:

* point-to-surface fitting RMSE from one million independent reference-mesh
  samples;
* squared bidirectional Chamfer L2 against deterministic held-out surface samples;
* connected-component ghost area ratio using the 3 x median 8-NN threshold.

Configuration paths are resolved through ``--asset-root``. Reconstructed
``.npz`` meshes and all reference data are mapped through the training
``data_mean``/``data_scale`` normalization before any distance metric is
computed. The finite training-cloud RMSE is retained as a diagnostic, but it
is not used as the paper Table 1 metric.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

if sys.version_info < (3, 10):
    raise SystemExit(
        "Python 3.10 or newer is required. Install requirements.txt before running this command."
    )

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


TEST_SEED_BASE = 20260717
RECONSTRUCTION_SEED_BASE = 20270717


class EvaluationSetup(NamedTuple):
    configuration_path: Path
    method: str
    dataset: str
    reference_config: dict[str, Any]
    seed_offset: int
    asset_root: Path
    mesh_path: Path
    reference_path: Path
    training: dict[str, Any]
    raw_reference: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    component_labels: np.ndarray
    mesh_scale: float
    mesh_translation: np.ndarray


class NormalizedSamples(NamedTuple):
    mean: np.ndarray
    scale: float
    reference: np.ndarray
    reconstruction: np.ndarray
    test: np.ndarray
    ground_truth_kind: str
    ground_truth_source: Path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def rebase_path(value: str | Path, asset_root: Path) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError("cannot resolve an empty asset path")
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (asset_root / path).resolve()


def load_csv_points(path: Path) -> np.ndarray:
    first = path.open(encoding="utf-8-sig", errors="replace").readline().split(",")
    try:
        [float(value) for value in first[:3]]
        skip_rows = 0
    except ValueError:
        skip_rows = 1
    points = np.loadtxt(
        path,
        delimiter=",",
        skiprows=skip_rows,
        usecols=(0, 1, 2),
        dtype=np.float64,
    )
    points = np.atleast_2d(points)
    return points[np.isfinite(points).all(axis=1)]


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path)
        vertex_key = "vertices" if "vertices" in data else "verts"
        face_key = "faces" if "faces" in data else "triangles"
        return (
            np.asarray(data[vertex_key], dtype=np.float64),
            np.asarray(data[face_key], dtype=np.int64),
        )
    loaded = trimesh.load(path, force="mesh", process=(suffix == ".stl"))
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
        raise ValueError(f"Cannot load a non-empty triangular mesh: {path}")
    if suffix == ".stl":
        loaded.merge_vertices()
        loaded.remove_unreferenced_vertices()
    return (
        np.asarray(loaded.vertices, dtype=np.float64),
        np.asarray(loaded.faces, dtype=np.int64),
    )


def compact_components(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
    faces = faces[valid]
    if not len(faces):
        raise ValueError("Mesh has no valid faces")
    used = np.unique(faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    vertices = vertices[used]
    faces = remap[faces]
    i, j, k = faces.T
    rows = np.concatenate((i, j, j, k, k, i))
    columns = np.concatenate((j, i, k, j, i, k))
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    _, labels = connected_components(graph, directed=False, return_labels=True)
    return vertices, faces, labels


def sample_surface(
    vertices: np.ndarray, faces: np.ndarray, count: int, seed: int
) -> np.ndarray:
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = np.isfinite(areas) & (areas > 0)
    triangles = triangles[valid]
    areas = areas[valid]
    if not len(triangles):
        raise ValueError("Mesh has no positive-area triangles")
    rng = np.random.default_rng(seed)
    chosen = triangles[
        rng.choice(len(triangles), size=int(count), replace=True, p=areas / areas.sum())
    ]
    u = rng.random(int(count))
    v = rng.random(int(count))
    reflected = u + v > 1.0
    u[reflected] = 1.0 - u[reflected]
    v[reflected] = 1.0 - v[reflected]
    return (
        chosen[:, 0]
        + u[:, None] * (chosen[:, 1] - chosen[:, 0])
        + v[:, None] * (chosen[:, 2] - chosen[:, 0])
    )


def nearest(tree: cKDTree, query: np.ndarray) -> np.ndarray:
    try:
        distances, _ = tree.query(query, k=1, workers=-1)
    except TypeError:  # SciPy < 1.6
        distances, _ = tree.query(query, k=1)
    return np.asarray(distances, dtype=np.float64)


def load_reference(path: Path, dataset: str) -> dict[str, Any]:
    value = load_json(path)
    if str(value.get("dataset")) != dataset:
        raise ValueError(f"Reference JSON does not match {dataset}: {path}")
    return value


def heldout_points(
    reference_config: dict[str, Any],
    asset_root: Path,
    count: int,
    seed: int,
) -> tuple[np.ndarray, str, Path]:
    reference_kind = str(reference_config.get("reference_kind", ""))
    if reference_kind != "independent_source_mesh":
        raise ValueError(
            "The paper evaluation protocol requires reference_kind="
            f"'independent_source_mesh', got {reference_kind!r}"
        )
    source = rebase_path(reference_config["asset_root_path"], asset_root)
    if not source.is_file():
        raise FileNotFoundError(source)
    vertices, faces = load_mesh(source)
    points = sample_surface(vertices, faces, count, seed)
    kind = "independent_mesh_surface_sample"
    transform = reference_config.get("coordinate_transform", {"mode": "identity"})
    if transform.get("mode") == "similarity_raw_to_reference":
        points = (
            points * float(transform["scale"])
            + np.asarray(transform["translation"], dtype=np.float64)[None, :]
        )
        kind = "independent_mesh_surface_sample_exact_similarity_aligned"
    elif transform.get("mode") != "identity":
        raise ValueError(f"Unsupported reference transform: {transform.get('mode')!r}")
    return points, kind, source


def spacing_threshold(reference: np.ndarray) -> tuple[float, float, int]:
    tree = cKDTree(reference)
    k = min(9, len(reference))
    distances = nearest_k(tree, reference, k)
    kth = distances[:, -1] if distances.ndim == 2 else distances
    median = float(np.median(kth))
    return median, 3.0 * median, len(reference)


def nearest_k(tree: cKDTree, query: np.ndarray, k: int) -> np.ndarray:
    try:
        distances, _ = tree.query(query, k=k, workers=-1)
    except TypeError:
        distances, _ = tree.query(query, k=k)
    return np.asarray(distances, dtype=np.float64)


def ghost_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    spacing, threshold, spacing_count = spacing_threshold(reference)
    distances = nearest(cKDTree(reference), vertices)
    triangles = vertices[faces]
    face_areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    face_labels = labels[faces[:, 0]]
    component_count = int(labels.max()) + 1 if len(labels) else 0
    face_counts = np.bincount(face_labels, minlength=component_count)
    areas = np.bincount(face_labels, weights=face_areas, minlength=component_count)
    components: list[dict[str, Any]] = []
    ghost_ids: list[int] = []
    for component_id in range(component_count):
        vertex_ids = np.flatnonzero(labels == component_id)
        component_distances = distances[vertex_ids]
        median_distance = float(np.median(component_distances))
        is_ghost = median_distance > threshold
        if is_ghost:
            ghost_ids.append(component_id)
        components.append(
            {
                "component_id": component_id,
                "vertices": int(len(vertex_ids)),
                "faces": int(face_counts[component_id]),
                "area": float(areas[component_id]),
                "area_ratio": float(areas[component_id] / max(float(areas.sum()), 1e-30)),
                "distance_to_reference_p50": median_distance,
                "distance_to_reference_p90": float(
                    np.quantile(component_distances, 0.9)
                ),
                "ghost_component": bool(is_ghost),
            }
        )
    ghost_area = float(sum(areas[index] for index in ghost_ids))
    return {
        "detector": "extracted_mesh_connected_component_median_distance",
        "spacing_sample_size": spacing_count,
        "spacing_k": 8,
        "spacing_median": spacing,
        "r_out_factor": 3.0,
        "r_out": threshold,
        "component_count": component_count,
        "ghost_component_count": len(ghost_ids),
        "ghost_area_ratio": ghost_area / max(float(areas.sum()), 1e-30),
        "components": sorted(components, key=lambda row: row["faces"], reverse=True),
    }


def prepare_evaluation(args: argparse.Namespace) -> EvaluationSetup:
    configuration_path = Path(args.configuration).expanduser().resolve()
    config = load_json(configuration_path)
    method_override = str(getattr(args, "method", "") or "").strip()
    method = method_override or str(config["method"])
    if method not in {"SGR", "DWG"}:
        raise ValueError(f"Unsupported low-dimensional method: {method!r}")
    dataset = str(config["dataset"])
    if args.reference_json:
        reference_config_path = Path(args.reference_json).expanduser().resolve()
    else:
        reference_config_path = (
            configuration_path.parent.parent / "dataset" / "reference.json"
        )
    reference_config = load_reference(reference_config_path, dataset)
    seed_offset = int(
        args.seed_offset
        if args.seed_offset is not None
        else reference_config.get("evaluation_seed_offset", 0)
    )
    asset_root = Path(args.asset_root).expanduser().resolve()
    mesh_path = Path(args.mesh).expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    training = config["training_parameters"]
    point_cloud_path = rebase_path(training["point_cloud"], asset_root)
    if not point_cloud_path.is_file():
        raise FileNotFoundError(point_cloud_path)
    raw_reference = load_csv_points(point_cloud_path)
    vertices, faces = load_mesh(mesh_path)
    mesh_scale = float(getattr(args, "mesh_scale", 1.0))
    mesh_translation = np.asarray(
        getattr(args, "mesh_translation", (0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    if not np.isfinite(mesh_scale) or mesh_scale <= 0:
        raise ValueError("--mesh-scale must be finite and positive")
    if mesh_translation.shape != (3,) or not np.isfinite(mesh_translation).all():
        raise ValueError("--mesh-translation must contain three finite values")
    vertices = vertices * mesh_scale + mesh_translation[None, :]
    vertices, faces, labels = compact_components(vertices, faces)
    return EvaluationSetup(
        configuration_path=configuration_path,
        method=method,
        dataset=dataset,
        reference_config=reference_config,
        seed_offset=seed_offset,
        asset_root=asset_root,
        mesh_path=mesh_path,
        reference_path=point_cloud_path,
        training=training,
        raw_reference=raw_reference,
        vertices=vertices,
        faces=faces,
        component_labels=labels,
        mesh_scale=mesh_scale,
        mesh_translation=mesh_translation,
    )


def normalization_contract(training: dict[str, Any]) -> tuple[np.ndarray, float]:
    mean = np.asarray(training["data_mean"], dtype=np.float64)
    scale = float(training["data_scale"])
    if mean.shape != (3,) or not np.isfinite(mean).all():
        raise ValueError("training data_mean must contain three finite values")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("training data_scale must be finite and positive")
    return mean, scale


def normalized_samples(
    args: argparse.Namespace,
    setup: EvaluationSetup,
) -> NormalizedSamples:
    mean, scale = normalization_contract(setup.training)

    reconstruction_points = sample_surface(
        setup.vertices,
        setup.faces,
        int(args.reconstruction_samples),
        RECONSTRUCTION_SEED_BASE + setup.seed_offset,
    )
    reference_normalized = (
        setup.raw_reference - mean[None, :]
    ) / scale
    reconstruction_normalized = (
        reconstruction_points - mean[None, :]
    ) / scale

    # Tables 1 and 2: one million independent reference-mesh points and one
    # million independently sampled reconstruction points.
    raw_test, ground_truth_kind, ground_truth_source = heldout_points(
        setup.reference_config,
        setup.asset_root,
        int(args.test_samples),
        TEST_SEED_BASE + setup.seed_offset,
    )
    return NormalizedSamples(
        mean=mean,
        scale=scale,
        reference=reference_normalized,
        reconstruction=reconstruction_normalized,
        test=(raw_test - mean[None, :]) / scale,
        ground_truth_kind=ground_truth_kind,
        ground_truth_source=ground_truth_source,
    )


def distance_metrics(samples: NormalizedSamples) -> dict[str, Any]:
    reconstruction_to_test = nearest(
        cKDTree(samples.test),
        samples.reconstruction,
    )
    test_to_reconstruction = nearest(
        cKDTree(samples.reconstruction),
        samples.test,
    )
    fit_error = float(np.mean(test_to_reconstruction**2))
    chamfer_l2 = float(
        np.mean(reconstruction_to_test**2) + np.mean(test_to_reconstruction**2)
    )

    # Diagnostic only: this uses every released training point, whose count is
    # generally below the one-million-point Table 1 contract.
    training_fit_distances = nearest(
        cKDTree(samples.reconstruction),
        samples.reference,
    )
    return {
        "reconstruction_to_test_mse": float(np.mean(reconstruction_to_test**2)),
        "test_to_reconstruction_mse": float(np.mean(test_to_reconstruction**2)),
        "fit_error": fit_error,
        "fit_rmse": float(np.sqrt(fit_error)),
        "chamfer_l2": chamfer_l2,
        "training_fit_error": float(np.mean(training_fit_distances**2)),
    }


def evaluation_result(
    args: argparse.Namespace,
    setup: EvaluationSetup,
    samples: NormalizedSamples,
    metrics: dict[str, Any],
    ghost: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    training_fit_error = float(metrics["training_fit_error"])
    test_seed = TEST_SEED_BASE + setup.seed_offset
    reconstruction_seed = RECONSTRUCTION_SEED_BASE + setup.seed_offset
    return {
        "schema": "lowdim-reconstruction-evaluation/v3",
        "dataset": setup.dataset,
        "method": setup.method,
        "configuration": str(setup.configuration_path),
        "mesh": str(setup.mesh_path),
        "mesh_to_raw_transform": {
            "definition": "raw_vertex = mesh_vertex * scale + translation",
            "scale": setup.mesh_scale,
            "translation": setup.mesh_translation.tolist(),
        },
        "reference_pointcloud": str(setup.reference_path),
        "reference_frame": "training_mean_maxradius_normalized",
        "vertices": int(len(setup.vertices)),
        "faces": int(len(setup.faces)),
        "normalization": {
            "contract": "training_mean_and_max_center_radius",
            "data_mean_raw": samples.mean.tolist(),
            "data_scale_raw": samples.scale,
            "applies_to": [
                "point_to_surface",
                "chamfer_l2",
                "ghost_zero",
                "training_input_diagnostic",
            ],
        },
        "point_to_surface": {
            "definition": "sqrt(mean_i dist(x_i, M)^2)",
            "canonical_point_kind": samples.ground_truth_kind,
            "canonical_point_source": str(samples.ground_truth_source),
            "canonical_points": int(len(samples.test)),
            "surface_samples": int(len(samples.reconstruction)),
            "canonical_seed": test_seed,
            "surface_seed": reconstruction_seed,
            "fitting_error": metrics["fit_error"],
            "fit_rmse": metrics["fit_rmse"],
        },
        "training_input_diagnostic": {
            "definition": "sqrt(mean_i dist(x_i, M)^2)",
            "role": "diagnostic_only_not_table_1",
            "input_points": int(len(samples.reference)),
            "surface_samples": int(len(samples.reconstruction)),
            "fitting_error": training_fit_error,
            "fit_rmse": float(np.sqrt(training_fit_error)),
        },
        "chamfer_l2": {
            "definition": "mean d(reconstruction,GT)^2 + mean d(GT,reconstruction)^2",
            "ground_truth_kind": samples.ground_truth_kind,
            "ground_truth_source": str(samples.ground_truth_source),
            "test_samples": int(len(samples.test)),
            "reconstruction_samples": int(len(samples.reconstruction)),
            "test_seed": test_seed,
            "reconstruction_seed": reconstruction_seed,
            "normalization_contract": "training_mean_and_max_center_radius",
            "normalization_scale_raw": samples.scale,
            "reconstruction_to_test_mse": metrics["reconstruction_to_test_mse"],
            "test_to_reconstruction_mse": metrics["test_to_reconstruction_mse"],
            "value": metrics["chamfer_l2"],
        },
        "ghost_zero": ghost,
        "elapsed_seconds": elapsed_seconds,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    setup = prepare_evaluation(args)
    samples = normalized_samples(args, setup)
    metrics = distance_metrics(samples)

    vertices_normalized = (
        setup.vertices - samples.mean[None, :]
    ) / samples.scale
    ghost = ghost_metrics(
        vertices_normalized,
        setup.faces,
        setup.component_labels,
        samples.reference,
    )
    return evaluation_result(
        args,
        setup,
        samples,
        metrics,
        ghost,
        time.time() - started,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configuration",
        required=True,
        help="Released training JSON defining input data and normalization.",
    )
    parser.add_argument(
        "--method",
        choices=("SGR", "DWG"),
        default="",
        help="Optional method override; defaults to configuration.method.",
    )
    parser.add_argument(
        "--reference-json",
        default="",
        help=(
            "Reference JSON. Defaults to dataset/reference.json next to the "
            "experiment configuration."
        ),
    )
    parser.add_argument("--mesh", required=True)
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=1.0,
        help="Scale mapping mesh coordinates into the raw released-data frame.",
    )
    parser.add_argument(
        "--mesh-translation",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("TX", "TY", "TZ"),
        help="Translation applied after --mesh-scale.",
    )
    parser.add_argument("--asset-root", required=True, help="Root containing datasets.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=None,
        help="Override the evaluation_seed_offset stored in the reference JSON.",
    )
    parser.add_argument("--test-samples", type=int, default=1_000_000)
    parser.add_argument("--reconstruction-samples", type=int, default=1_000_000)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = evaluate(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "dataset": result["dataset"],
                "method": result["method"],
                "fit_rmse": result["point_to_surface"]["fit_rmse"],
                "chamfer_l2": result["chamfer_l2"]["value"],
                "ghost_area_ratio": result["ghost_zero"]["ghost_area_ratio"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
