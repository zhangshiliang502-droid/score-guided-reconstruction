#!/usr/bin/env python3
"""Evaluate an extracted high-dimensional SGR zero-set point cloud.

The evaluator is independent from reconstruction. It supports two reference
modes:

* ``input`` compares against the observed N-by-D point cloud and works for any
  codimension-one dataset;
* ``analytic`` additionally loads an optional experiment definition and uses
  exact Newton projection to the known surface, matching the paper examples.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, NamedTuple

if sys.version_info < (3, 10):
    raise SystemExit(
        "Python 3.10 or newer is required. Install requirements.txt before running this command."
    )

import numpy as np
import torch
from scipy.spatial import cKDTree


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HIGHDIM_CODE = REPOSITORY_ROOT / "02_highdim_sgr" / "code"
sys.path.insert(0, str(HIGHDIM_CODE))

from projection import (  # noqa: E402
    load_model,
    nearest_distances,
    project_learned_zero,
    quantiles,
)


DEFAULT_SEED = 20260718
EVALUATION_POINTS = 100_000
SPACING_REFERENCE_POINTS = 250_000
PROJECTION_BATCH_SIZE = 65_536
ANALYTIC_BATCH_SIZE = 16_384
PROJECTION_ITERS = 20
PROJECTION_MAX_STEP = 0.10
PROJECTION_RESIDUAL = 1e-4
ANALYTIC_PROJECTION_ITERS = 15


class CheckpointContext(NamedTuple):
    path: Path
    model: torch.nn.Module
    payload: dict[str, Any]
    dimension: int
    mean: np.ndarray
    scale: float
    low: np.ndarray
    high: np.ndarray
    device: torch.device


class ReferenceContext(NamedTuple):
    mode: str
    analytic_reference: Any | None
    analytic_module_path: Path | None
    analytic_class: str


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_checkpoint_configuration(
    meta: dict[str, Any],
    configuration: dict[str, Any],
) -> None:
    """Require the evaluated checkpoint to match the recorded paper run."""
    sgr = configuration.get("sgr_training")
    if not isinstance(sgr, dict):
        return
    loss = meta.get("loss")
    if not isinstance(loss, dict):
        raise ValueError("Checkpoint metadata does not contain the configured SGR loss")

    string_fields = {
        "loss": "name",
        "local_query_mode": "local_query_mode",
        "local_residual_normalization": "local_residual_normalization",
    }
    for configuration_key, checkpoint_key in string_fields.items():
        if configuration_key in sgr and str(loss.get(checkpoint_key)) != str(
            sgr[configuration_key]
        ):
            raise ValueError(
                f"Checkpoint {checkpoint_key}={loss.get(checkpoint_key)!r} does not "
                f"match configuration {configuration_key}={sgr[configuration_key]!r}"
            )

    numeric_fields = {
        "local_shell_center_sigma": "local_shell_center_sigma",
        "local_shell_std_sigma": "local_shell_std_sigma",
        "lambda_score_pull": "lambda_score_pull",
        "lambda_dir": "lambda_dir",
    }
    for configuration_key, checkpoint_key in numeric_fields.items():
        if configuration_key not in sgr:
            continue
        actual = float(loss.get(checkpoint_key, float("nan")))
        expected = float(sgr[configuration_key])
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"Checkpoint {checkpoint_key}={actual} does not match configuration "
                f"{configuration_key}={expected}"
            )

    for key in ("train_size", "val_size"):
        if key in sgr and int(meta.get(key, -1)) != int(sgr[key]):
            raise ValueError(
                f"Checkpoint {key}={meta.get(key)!r} does not match "
                f"configuration {key}={sgr[key]!r}"
            )

    energy = configuration.get("energy_training")
    if isinstance(energy, dict) and "sigma" in energy:
        actual_sigma = float(loss.get("sigma", float("nan")))
        expected_sigma = float(energy["sigma"])
        if not math.isclose(
            actual_sigma, expected_sigma, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(
                f"Checkpoint sigma={actual_sigma} does not match "
                f"configuration sigma={expected_sigma}"
            )


def validate_evaluation_configuration(
    args: argparse.Namespace,
    configuration: dict[str, Any],
    *,
    input_count: int,
    reconstruction_count: int,
) -> None:
    """Reject command-line settings that change a recorded paper protocol."""
    if not configuration:
        return

    evaluation = configuration.get("evaluation")
    if isinstance(evaluation, dict):
        integer_fields = {
            "evaluation_points": "evaluation_points",
            "spacing_reference_points": "spacing_reference_points",
            "analytic_projection_iterations": "analytic_projection_iters",
        }
        for configuration_key, argument_key in integer_fields.items():
            if configuration_key in evaluation and int(
                getattr(args, argument_key)
            ) != int(evaluation[configuration_key]):
                raise ValueError(
                    f"--{argument_key.replace('_', '-')}={getattr(args, argument_key)} "
                    f"does not match configuration {configuration_key}="
                    f"{evaluation[configuration_key]}"
                )
        if str(evaluation.get("metric_space", "normalized")) != "normalized":
            raise ValueError("High-dimensional paper evaluation must be normalized")

    reconstruction = configuration.get("reconstruction")
    if isinstance(reconstruction, dict):
        integer_fields = {
            "projection_batch_size": "projection_batch_size",
            "projection_iterations": "projection_iters",
        }
        float_fields = {
            "projection_max_step": "projection_max_step",
            "projection_residual": "projection_residual",
        }
        for configuration_key, argument_key in integer_fields.items():
            if configuration_key in reconstruction and int(
                getattr(args, argument_key)
            ) != int(reconstruction[configuration_key]):
                raise ValueError(
                    f"--{argument_key.replace('_', '-')}={getattr(args, argument_key)} "
                    f"does not match configuration {configuration_key}="
                    f"{reconstruction[configuration_key]}"
                )
        for configuration_key, argument_key in float_fields.items():
            if configuration_key not in reconstruction:
                continue
            actual = float(getattr(args, argument_key))
            expected = float(reconstruction[configuration_key])
            if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    f"--{argument_key.replace('_', '-')}={actual} does not match "
                    f"configuration {configuration_key}={expected}"
                )
        if "zero_points" in reconstruction and reconstruction_count != int(
            reconstruction["zero_points"]
        ):
            raise ValueError(
                f"Reconstruction contains {reconstruction_count} points, but "
                f"configuration requires {reconstruction['zero_points']}"
            )

    data = configuration.get("data")
    if isinstance(data, dict) and "samples" in data and input_count != int(
        data["samples"]
    ):
        raise ValueError(
            f"Input contains {input_count} points, but configuration requires "
            f"{data['samples']}"
        )


def load_points(path: Path, dimension: int, *, minimum: int = 9) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    points = np.load(path, mmap_mode="r")
    if points.ndim != 2 or points.shape[1] != dimension or len(points) < minimum:
        raise ValueError(
            f"Expected at least {minimum} points with shape [N,{dimension}], got {points.shape}"
        )
    for start in range(0, len(points), 65_536):
        if not np.all(np.isfinite(points[start : start + 65_536])):
            raise ValueError(f"{path} contains non-finite coordinates")
    return points


def resolve_experiment_path(configuration: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    experiment_root = configuration.parent.parent
    return (experiment_root / path).resolve()


def resolve_analytic_module(configuration: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (
        configuration.parent.parent.parent / path,
        configuration.parent.parent / path,
        configuration.parent / path,
        Path.cwd() / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def load_analytic_reference(module_path: Path, class_name: str) -> Any:
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    module_name = f"sgr_analytic_reference_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load analytic module: {module_path}")
    sys.path.insert(0, str(module_path.parent))
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    reference_class = getattr(module, class_name, None)
    if reference_class is None:
        raise AttributeError(f"{module_path} does not define {class_name}")
    reference = reference_class()
    for name in ("project_to_surface", "implicit_value"):
        if not callable(getattr(reference, name, None)):
            raise TypeError(f"Analytic reference {class_name} does not implement {name}()")
    return reference


def project_analytic_batched(
    reference: Any,
    points: np.ndarray,
    *,
    iterations: int,
    batch_size: int,
) -> np.ndarray:
    projected = np.empty_like(points, dtype=np.float64)
    for start in range(0, len(points), int(batch_size)):
        end = min(start + int(batch_size), len(points))
        projected[start:end] = reference.project_to_surface(
            points[start:end], iterations=int(iterations)
        )
    return projected


def spacing_threshold(
    reference: np.ndarray,
    rng: np.random.Generator,
    requested_count: int,
) -> tuple[cKDTree, int, float, float]:
    count = int(requested_count)
    if len(reference) < count:
        raise ValueError(
            f"Requested {count} spacing reference points, but only "
            f"{len(reference)} are available"
        )
    if count < 9:
        raise ValueError(
            "At least nine reference points are required to estimate 8-NN spacing"
        )
    indices = rng.choice(len(reference), size=count, replace=False)
    sampled = np.asarray(reference[indices], dtype=np.float64)
    tree = cKDTree(sampled)
    try:
        distances, _ = tree.query(sampled, k=9, workers=-1)
    except TypeError:
        distances, _ = tree.query(sampled, k=9)
    distances = np.asarray(distances, dtype=np.float64)
    eighth_neighbor_distances = distances[:, -1]
    spacing = float(np.median(eighth_neighbor_distances))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Reference point cloud has invalid neighbor spacing")
    return tree, count, spacing, 3.0 * spacing


def project_input_to_learned_surface(
    model: torch.nn.Module,
    points_physical: np.ndarray,
    *,
    mean: np.ndarray,
    scale: float,
    low: np.ndarray,
    high: np.ndarray,
    device: torch.device,
    batch_size: int,
    iterations: int,
    max_step: float,
    residual: float,
) -> tuple[np.ndarray, float]:
    normalized = ((points_physical - mean[None, :]) / scale).astype(np.float32)
    distances: list[np.ndarray] = []
    accepted_count = 0
    for start in range(0, len(normalized), int(batch_size)):
        source = normalized[start : start + int(batch_size)]
        fitted, _, _, accepted = project_learned_zero(
            model,
            source,
            low=low,
            high=high,
            device=device,
            iterations=int(iterations),
            max_step=float(max_step),
            residual=float(residual),
        )
        distances.append(
            np.linalg.norm((fitted - source).astype(np.float64), axis=1)
        )
        accepted_count += int(np.sum(accepted))
    convergence_ratio = accepted_count / len(normalized)
    if accepted_count != len(normalized):
        raise RuntimeError(
            "Formal evaluation requires all sampled input points to converge to the "
            f"learned zero set, but only {accepted_count}/{len(normalized)} converged"
        )
    return np.concatenate(distances), convergence_ratio


def validate_positive_arguments(args: argparse.Namespace) -> None:
    for name in (
        "evaluation_points",
        "spacing_reference_points",
        "projection_batch_size",
        "analytic_batch_size",
        "projection_iters",
        "projection_max_step",
        "projection_residual",
        "analytic_projection_iters",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def load_configuration(
    value: str,
) -> tuple[Path | None, dict[str, Any]]:
    configuration_path = (
        Path(value).expanduser().resolve() if value else None
    )
    configuration = load_json(configuration_path) if configuration_path else {}
    return configuration_path, configuration


def load_checkpoint_context(
    args: argparse.Namespace,
    configuration: dict[str, Any],
) -> CheckpointContext:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    model, payload = load_model(checkpoint_path, device)
    meta = payload["meta"]
    validate_checkpoint_configuration(meta, configuration)

    dimension = int(meta["data_dim"])
    mean = np.asarray(meta["data_mean"], dtype=np.float64)
    scale = float(meta["data_scale"])
    low = np.asarray(meta["bg_low"], dtype=np.float32)
    high = np.asarray(meta["bg_high"], dtype=np.float32)
    expected_shape = (dimension,)
    if mean.shape != expected_shape or low.shape != expected_shape or high.shape != expected_shape:
        raise ValueError("Checkpoint normalization dimension is inconsistent")
    return CheckpointContext(
        path=checkpoint_path,
        model=model,
        payload=payload,
        dimension=dimension,
        mean=mean,
        scale=scale,
        low=low,
        high=high,
        device=device,
    )


def load_input_points(
    args: argparse.Namespace,
    configuration_path: Path | None,
    configuration: dict[str, Any],
    dimension: int,
) -> tuple[Path, np.ndarray]:
    if args.point_cloud:
        input_path = Path(args.point_cloud).expanduser().resolve()
    elif configuration_path and configuration.get("data", {}).get("path"):
        input_path = resolve_experiment_path(
            configuration_path,
            configuration["data"]["path"],
        )
    else:
        raise ValueError("Provide --point-cloud or a configuration containing data.path")
    return input_path, load_points(input_path, dimension)


def load_reconstruction(
    args: argparse.Namespace,
    dimension: int,
) -> tuple[Path, np.ndarray, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    if args.reconstruction_metadata:
        metadata = load_json(
            Path(args.reconstruction_metadata).expanduser().resolve()
        )
    if args.reconstruction:
        reconstruction_path = Path(args.reconstruction).expanduser().resolve()
    elif metadata.get("outputs", {}).get("physical"):
        reconstruction_path = Path(
            metadata["outputs"]["physical"]
        ).expanduser().resolve()
    else:
        raise ValueError(
            "Provide --reconstruction or reconstruction metadata with outputs.physical"
        )
    reconstruction = np.asarray(
        load_points(reconstruction_path, dimension, minimum=1),
        dtype=np.float64,
    )
    return reconstruction_path, reconstruction, metadata


def resolve_reference_context(
    args: argparse.Namespace,
    configuration_path: Path | None,
    configuration: dict[str, Any],
    dimension: int,
) -> ReferenceContext:
    analytic_definition = configuration.get("analytic_definition", {})
    module_value = args.analytic_module or analytic_definition.get("module", "")
    class_name = args.analytic_class or analytic_definition.get("class", "")
    if bool(module_value) != bool(class_name):
        raise ValueError("Analytic evaluation requires both a module and class")

    reference_mode = args.reference_mode
    if reference_mode == "auto":
        reference_mode = "analytic" if module_value else "input"
    if reference_mode == "analytic" and not module_value:
        raise ValueError(
            "Analytic reference mode requires --configuration or analytic overrides"
        )
    if reference_mode != "analytic":
        return ReferenceContext(reference_mode, None, None, str(class_name))

    module_path = (
        resolve_analytic_module(configuration_path, module_value)
        if configuration_path
        else Path(module_value).expanduser().resolve()
    )
    analytic_reference = load_analytic_reference(module_path, class_name)
    if int(getattr(analytic_reference, "ambient_dim", -1)) != dimension:
        raise ValueError("Analytic reference dimension does not match the checkpoint")
    return ReferenceContext(
        reference_mode,
        analytic_reference,
        module_path,
        str(class_name),
    )


def evaluation_rng(
    args: argparse.Namespace,
    configuration: dict[str, Any],
    reconstruction_metadata: dict[str, Any],
) -> tuple[int, np.random.Generator, str]:
    seed = int(
        args.seed
        if args.seed is not None
        else configuration.get("reconstruction", {}).get("seed", DEFAULT_SEED)
    )
    rng = np.random.default_rng(seed)
    source = "evaluation_seed"
    if reconstruction_metadata.get("evaluation_rng_state"):
        rng.bit_generator.state = reconstruction_metadata["evaluation_rng_state"]
        source = "reconstruction_continuation_state"
    return seed, rng, source


def evaluate_input_projection(
    args: argparse.Namespace,
    checkpoint: CheckpointContext,
    input_points: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, int]:
    evaluation_count = int(args.evaluation_points)
    if len(input_points) < evaluation_count:
        raise ValueError(
            f"Requested {evaluation_count} evaluation points, but only "
            f"{len(input_points)} are available"
        )
    indices = rng.choice(len(input_points), size=evaluation_count, replace=False)
    evaluation_points = np.asarray(input_points[indices], dtype=np.float64)
    distances, convergence_ratio = project_input_to_learned_surface(
        checkpoint.model,
        evaluation_points,
        mean=checkpoint.mean,
        scale=checkpoint.scale,
        low=checkpoint.low,
        high=checkpoint.high,
        device=checkpoint.device,
        batch_size=int(args.projection_batch_size),
        iterations=int(args.projection_iters),
        max_step=float(args.projection_max_step),
        residual=float(args.projection_residual),
    )
    return distances, convergence_ratio, evaluation_count


def evaluate_reference_distances(
    args: argparse.Namespace,
    checkpoint: CheckpointContext,
    reference: ReferenceContext,
    input_points: np.ndarray,
    reconstruction: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    tree, spacing_count, spacing_physical, outside_radius_physical = spacing_threshold(
        input_points,
        rng,
        int(args.spacing_reference_points),
    )
    spacing = spacing_physical / checkpoint.scale
    outside_radius = outside_radius_physical / checkpoint.scale

    analytic_residual = None
    if reference.analytic_reference is not None:
        projected_reference = project_analytic_batched(
            reference.analytic_reference,
            reconstruction,
            iterations=int(args.analytic_projection_iters),
            batch_size=int(args.analytic_batch_size),
        )
        distances = np.linalg.norm(
            reconstruction - projected_reference,
            axis=1,
        ) / checkpoint.scale
        analytic_residual = np.abs(
            reference.analytic_reference.implicit_value(projected_reference)
        )
    else:
        distances = nearest_distances(tree, reconstruction) / checkpoint.scale
    ghost_surface = {
        "spacing_reference_size": spacing_count,
        "spacing_k": 8,
        "spacing_median": spacing,
        "r_out_3x_spacing": outside_radius,
        "ratio_r_out": float(np.mean(distances > outside_radius)),
    }
    return distances, analytic_residual, ghost_surface


def analytic_report(
    reference: ReferenceContext,
    analytic_residual: np.ndarray | None,
) -> dict[str, Any] | None:
    if reference.analytic_reference is None:
        return None
    metadata_method = getattr(reference.analytic_reference, "metadata", None)
    metadata = (
        metadata_method()
        if callable(metadata_method)
        else {"class": reference.analytic_class}
    )
    return {
        "module": str(reference.analytic_module_path),
        "class": reference.analytic_class,
        "metadata": metadata,
        "projection_residual": quantiles(analytic_residual),
    }


def build_report(
    *,
    checkpoint: CheckpointContext,
    reference: ReferenceContext,
    input_path: Path,
    input_points: np.ndarray,
    reconstruction_path: Path,
    reconstruction: np.ndarray,
    seed: int,
    rng_source: str,
    evaluation_count: int,
    input_to_learned: np.ndarray,
    convergence_ratio: float,
    reconstruction_to_reference: np.ndarray,
    ghost_surface: dict[str, Any],
    analytic_details: dict[str, Any] | None,
) -> dict[str, Any]:
    point_to_surface_rmse = float(np.sqrt(np.mean(input_to_learned**2)))
    chamfer_l2 = float(
        np.mean(input_to_learned**2) + np.mean(reconstruction_to_reference**2)
    )
    report: dict[str, Any] = {
        "schema": "sgr-highdim-reconstruction-evaluation/v1",
        "reference_mode": reference.mode,
        "point_cloud": str(input_path),
        "reconstruction": str(reconstruction_path),
        "checkpoint": str(checkpoint.path),
        "checkpoint_step": int(checkpoint.payload["step"]),
        "ambient_dim": checkpoint.dimension,
        "coordinate_metrics": "normalized",
        "seed": seed,
        "rng_source": rng_source,
        "input_points": int(len(input_points)),
        "evaluation_points": evaluation_count,
        "zero_points": int(len(reconstruction)),
        "input_to_reconstruction": quantiles(input_to_learned),
        "input_to_reconstruction_convergence_ratio": convergence_ratio,
        "reconstruction_to_reference": quantiles(reconstruction_to_reference),
        "point_to_surface_rmse": point_to_surface_rmse,
        "chamfer_l2": chamfer_l2,
        "ghost_surface": ghost_surface,
        "normalization": {
            "data_mean": checkpoint.mean.tolist(),
            "data_scale": checkpoint.scale,
        },
        "bbox_normalized": {
            "low": checkpoint.low.tolist(),
            "high": checkpoint.high.tolist(),
        },
    }
    if analytic_details is not None:
        report["analytic_reference"] = analytic_details
    return report


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    validate_positive_arguments(args)
    configuration_path, configuration = load_configuration(args.configuration)
    checkpoint = load_checkpoint_context(args, configuration)
    input_path, input_points = load_input_points(
        args,
        configuration_path,
        configuration,
        checkpoint.dimension,
    )
    reconstruction_path, reconstruction, reconstruction_metadata = load_reconstruction(
        args,
        checkpoint.dimension,
    )
    validate_evaluation_configuration(
        args,
        configuration,
        input_count=len(input_points),
        reconstruction_count=len(reconstruction),
    )
    reference = resolve_reference_context(
        args,
        configuration_path,
        configuration,
        checkpoint.dimension,
    )
    seed, rng, rng_source = evaluation_rng(
        args,
        configuration,
        reconstruction_metadata,
    )
    input_to_learned, convergence_ratio, evaluation_count = evaluate_input_projection(
        args,
        checkpoint,
        input_points,
        rng,
    )
    reconstruction_to_reference, analytic_residual, ghost_surface = (
        evaluate_reference_distances(
            args,
            checkpoint,
            reference,
            input_points,
            reconstruction,
            rng,
        )
    )
    analytic_details = analytic_report(reference, analytic_residual)
    return build_report(
        checkpoint=checkpoint,
        reference=reference,
        input_path=input_path,
        input_points=input_points,
        reconstruction_path=reconstruction_path,
        reconstruction=reconstruction,
        seed=seed,
        rng_source=rng_source,
        evaluation_count=evaluation_count,
        input_to_learned=input_to_learned,
        convergence_ratio=convergence_ratio,
        reconstruction_to_reference=reconstruction_to_reference,
        ghost_surface=ghost_surface,
        analytic_details=analytic_details,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", default="")
    parser.add_argument("--point-cloud", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reconstruction", default="", help="Physical-coordinate NPY.")
    parser.add_argument("--reconstruction-metadata", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--reference-mode", choices=("auto", "input", "analytic"), default="auto"
    )
    parser.add_argument("--analytic-module", default="")
    parser.add_argument("--analytic-class", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--evaluation-points", type=int, default=EVALUATION_POINTS)
    parser.add_argument(
        "--spacing-reference-points", type=int, default=SPACING_REFERENCE_POINTS
    )
    parser.add_argument("--projection-batch-size", type=int, default=PROJECTION_BATCH_SIZE)
    parser.add_argument("--analytic-batch-size", type=int, default=ANALYTIC_BATCH_SIZE)
    parser.add_argument("--projection-iters", type=int, default=PROJECTION_ITERS)
    parser.add_argument("--projection-max-step", type=float, default=PROJECTION_MAX_STEP)
    parser.add_argument("--projection-residual", type=float, default=PROJECTION_RESIDUAL)
    parser.add_argument(
        "--analytic-projection-iters", type=int, default=ANALYTIC_PROJECTION_ITERS
    )
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = evaluate(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "done",
                "output": str(output),
                "reference_mode": report["reference_mode"],
                "point_to_surface_rmse": report["point_to_surface_rmse"],
                "chamfer_l2": report["chamfer_l2"],
                "ghost_surface_ratio": report["ghost_surface"]["ratio_r_out"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
