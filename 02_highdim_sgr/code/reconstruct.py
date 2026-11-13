#!/usr/bin/env python3
"""Reconstruct a zero-set point cloud from any observed N-by-D point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch

from projection import load_model, project_learned_zero, quantiles


SEED = 20260718
SEED_COUNT = 1_000_000
ZERO_POINTS = 250_000
SEED_INITIALIZATION = "uniform_bbox"
PROJECTION_BATCH_SIZE = 65_536
PROJECTION_ITERS = 20
PROJECTION_MAX_STEP = 0.10
PROJECTION_RESIDUAL = 1e-4


class ProjectionResult(NamedTuple):
    normalized_points: np.ndarray
    phi: np.ndarray
    gradient_norm: np.ndarray
    accepted_total: int
    processed: int


def load_points(path: Path, dimension: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    points = np.load(path, mmap_mode="r")
    if points.ndim != 2 or points.shape[1] != int(dimension) or len(points) < 9:
        raise ValueError(f"expected at least nine points with shape [N,{dimension}], got {points.shape}")
    for start in range(0, len(points), 65_536):
        if not np.all(np.isfinite(points[start : start + 65_536])):
            raise ValueError(f"{path} contains non-finite coordinates")
    return points


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct the formal learned zero set from an arbitrary N-by-D point cloud."
    )
    parser.add_argument("--point-cloud", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--seed-count", type=int, default=SEED_COUNT)
    parser.add_argument("--zero-points", type=int, default=ZERO_POINTS)
    parser.add_argument("--projection-batch-size", type=int, default=PROJECTION_BATCH_SIZE)
    parser.add_argument("--projection-iters", type=int, default=PROJECTION_ITERS)
    parser.add_argument("--projection-max-step", type=float, default=PROJECTION_MAX_STEP)
    parser.add_argument("--projection-residual", type=float, default=PROJECTION_RESIDUAL)
    parser.add_argument("--device", default="cuda")
    return parser


def validate_projection_args(args: argparse.Namespace) -> None:
    if int(args.seed_count) <= 0 or int(args.zero_points) <= 0:
        raise ValueError("seed-count and zero-points must be positive")
    if int(args.projection_batch_size) <= 0 or int(args.projection_iters) <= 0:
        raise ValueError("projection-batch-size and projection-iters must be positive")
    if float(args.projection_max_step) <= 0.0 or float(args.projection_residual) <= 0.0:
        raise ValueError("projection-max-step and projection-residual must be positive")


def checkpoint_geometry(
    meta: dict[str, Any],
    dimension: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean = np.asarray(meta["data_mean"], dtype=np.float64)
    scale = float(meta["data_scale"])
    low = np.asarray(meta["bg_low"], dtype=np.float32)
    high = np.asarray(meta["bg_high"], dtype=np.float32)
    expected_shape = (dimension,)
    if mean.shape != expected_shape or low.shape != expected_shape or high.shape != expected_shape:
        raise ValueError("checkpoint normalization dimension does not match the input data")
    if not np.isfinite(scale) or scale <= 0.0 or np.any(low >= high):
        raise ValueError("checkpoint contains invalid normalization or background bounds")
    return mean, scale, low, high


def project_seed_batches(
    model: torch.nn.Module,
    args: argparse.Namespace,
    *,
    dimension: int,
    low: np.ndarray,
    high: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
) -> ProjectionResult:
    kept_points: list[np.ndarray] = []
    kept_phi: list[np.ndarray] = []
    kept_gradient: list[np.ndarray] = []
    accepted_total = 0
    processed = 0
    for start in range(0, int(args.seed_count), int(args.projection_batch_size)):
        end = min(start + int(args.projection_batch_size), int(args.seed_count))
        seeds = rng.uniform(
            low[None, :],
            high[None, :],
            size=(end - start, dimension),
        ).astype(np.float32)
        projected, phi, gradient, accepted = project_learned_zero(
            model,
            seeds,
            low=low,
            high=high,
            device=device,
            iterations=int(args.projection_iters),
            max_step=float(args.projection_max_step),
            residual=float(args.projection_residual),
        )
        kept_points.append(projected[accepted])
        kept_phi.append(phi[accepted])
        kept_gradient.append(gradient[accepted])
        accepted_total += int(np.sum(accepted))
        processed = end
        print(
            json.dumps(
                {
                    "event": "projection",
                    "processed": processed,
                    "accepted": accepted_total,
                    "dimension": dimension,
                }
            ),
            flush=True,
        )
    return ProjectionResult(
        normalized_points=np.concatenate(kept_points, axis=0),
        phi=np.concatenate(kept_phi, axis=0),
        gradient_norm=np.concatenate(kept_gradient, axis=0),
        accepted_total=accepted_total,
        processed=processed,
    )


def select_zero_points(
    result: ProjectionResult,
    requested_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    available = len(result.normalized_points)
    if available < int(requested_count):
        raise RuntimeError(
            f"only {available} zero points passed after {result.processed} candidates; "
            f"required {requested_count}"
        )
    selected = rng.choice(available, size=int(requested_count), replace=False)
    return (
        result.normalized_points[selected],
        result.phi[selected],
        result.gradient_norm[selected],
    )


def save_reconstruction(
    args: argparse.Namespace,
    payload: dict[str, Any],
    *,
    input_point_count: int,
    dimension: int,
    mean: np.ndarray,
    scale: float,
    low: np.ndarray,
    high: np.ndarray,
    projected: ProjectionResult,
    zero_normalized: np.ndarray,
    zero_phi: np.ndarray,
    zero_gradient: np.ndarray,
    rng: np.random.Generator,
) -> None:
    zero_physical = zero_normalized.astype(np.float64) * scale + mean[None, :]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = output_dir / "zero_points_normalized.npy"
    physical_path = output_dir / "zero_points_physical.npy"
    np.save(normalized_path, zero_normalized)
    np.save(physical_path, zero_physical.astype(np.float32))

    report = {
        "schema": "generic-highdim-reconstruction/v2",
        "point_cloud": str(args.point_cloud),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload["step"]),
        "ambient_dim": dimension,
        "input_points": input_point_count,
        "zero_points": int(len(zero_normalized)),
        "seed": int(args.seed),
        "seed_count_requested": int(args.seed_count),
        "seed_count_processed": int(projected.processed),
        "projection_accepted_before_cap": int(projected.accepted_total),
        "projection_accept_ratio": float(
            projected.accepted_total / max(projected.processed, 1)
        ),
        "seed_initialization": SEED_INITIALIZATION,
        "projection": {
            "iterations": int(args.projection_iters),
            "max_step": float(args.projection_max_step),
            "residual": float(args.projection_residual),
        },
        "learned_zero_abs_phi": quantiles(np.abs(zero_phi)),
        "learned_zero_gradient_norm": quantiles(zero_gradient),
        "normalization": {"data_mean": mean.tolist(), "data_scale": scale},
        "bbox_normalized": {"low": low.tolist(), "high": high.tolist()},
        "evaluation_rng_state": rng.bit_generator.state,
        "outputs": {
            "normalized": str(normalized_path),
            "physical": str(physical_path),
        },
    }
    reconstruction_path = output_dir / "reconstruction.json"
    reconstruction_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "done",
                "reconstruction": str(reconstruction_path),
                **report["outputs"],
            }
        ),
        flush=True,
    )


def main() -> None:
    args = build_arg_parser().parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, payload = load_model(Path(args.checkpoint), device)
    meta = payload["meta"]
    dimension = int(meta["data_dim"])
    points = load_points(Path(args.point_cloud), dimension)
    mean, scale, low, high = checkpoint_geometry(meta, dimension)
    validate_projection_args(args)

    rng = np.random.default_rng(int(args.seed))
    projected = project_seed_batches(
        model,
        args,
        dimension=dimension,
        low=low,
        high=high,
        device=device,
        rng=rng,
    )
    zero_normalized, zero_phi, zero_gradient = select_zero_points(
        projected,
        int(args.zero_points),
        rng,
    )
    save_reconstruction(
        args,
        payload,
        input_point_count=int(len(points)),
        dimension=dimension,
        mean=mean,
        scale=scale,
        low=low,
        high=high,
        projected=projected,
        zero_normalized=zero_normalized,
        zero_phi=zero_phi,
        zero_gradient=zero_gradient,
        rng=rng,
    )


if __name__ == "__main__":
    main()
