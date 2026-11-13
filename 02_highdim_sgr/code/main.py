#!/usr/bin/env python3
"""Chain high-dimensional Energy score learning and Phi training.

Run conservative Energy-score learning followed by implicit-function
training, saving both checkpoints, stage logs and pipeline state.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


CODE_DIR = Path(__file__).resolve().parent
SIGMA = 0.01
FORMAL_LOSS = "score_pull_plus_global_cosine"
MINIMUM_INPUT_POINTS = 10


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare_metadata(data_path: Path, output_path: Path) -> dict[str, Any]:
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if data_path.suffix.lower() != ".npy":
        raise ValueError("high-dimensional --point-cloud must be an NPY array")
    points = np.load(data_path, mmap_mode="r")
    if points.ndim != 2 or points.shape[1] <= 3 or len(points) < MINIMUM_INPUT_POINTS:
        raise ValueError(
            f"input must have shape [N,D] with D>3 and N>={MINIMUM_INPUT_POINTS}, got {points.shape}"
        )
    dimension = int(points.shape[1])
    low = np.full(dimension, np.inf, dtype=np.float64)
    high = np.full(dimension, -np.inf, dtype=np.float64)
    for start in range(0, len(points), 65_536):
        batch = np.asarray(points[start : start + 65_536], dtype=np.float64)
        if not np.all(np.isfinite(batch)):
            raise ValueError(f"{data_path} contains non-finite coordinates")
        low = np.minimum(low, batch.min(axis=0))
        high = np.maximum(high, batch.max(axis=0))
    mean = np.mean(points, axis=0, dtype=np.float32).astype(np.float32)
    scale = 0.0
    for start in range(0, len(points), 65_536):
        batch = np.asarray(points[start : start + 65_536], dtype=np.float32)
        scale = max(scale, float(np.max(np.linalg.norm(batch - mean[None, :], axis=1))))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("input point cloud has zero or invalid scale")
    metadata = {
        "schema": "generic-highdim-input/v1",
        "point_cloud": str(data_path),
        "num_points": int(len(points)),
        "ambient_dim": dimension,
        "dtype": str(points.dtype),
        "normalization_mean_accumulator": "float32",
        "data_mean": mean.astype(float).tolist(),
        "data_scale_max_center_radius": scale,
        "bbox_min": low.tolist(),
        "bbox_max": high.tolist(),
    }
    atomic_json(output_path, metadata)
    return metadata


def split_sizes(point_count: int) -> dict[str, int]:
    energy_val = max(1, int(round(point_count * 0.10)))
    energy_train = point_count - energy_val
    phi_total = min(point_count, 250_000)
    phi_val = max(1, int(round(phi_total * 0.12)))
    return {
        "energy_train": energy_train,
        "energy_val": energy_val,
        "phi_train": phi_total - phi_val,
        "phi_val": phi_val,
    }


def energy_complete(path: Path, *, iterations: int, dimension: int) -> bool:
    latest_path = path.with_name("latest.pt")
    if not path.is_file() or not latest_path.is_file():
        return False
    try:
        best = torch.load(path, map_location="cpu", weights_only=True)
        latest = torch.load(latest_path, map_location="cpu", weights_only=True)
        kwargs = best.get("energy_model_kwargs", {})
        return (
            int(best.get("step", -1)) > 0
            and int(latest.get("step", -1)) >= int(iterations)
            and float(best.get("sigma", -1.0)) == SIGMA
            and float(latest.get("sigma", -1.0)) == SIGMA
            and str(best.get("model_type")) == "constrained_energy_score"
            and str(latest.get("model_type")) == "constrained_energy_score"
            and int(kwargs.get("data_dim", -1)) == int(dimension)
        )
    except Exception:
        return False


def implicit_checkpoint_complete(
    path: Path,
    *,
    iterations: int,
    dimension: int,
    point_cloud: Path,
    energy_checkpoint: Path,
    local_shell_center_sigma: float,
    lambda_dir: float,
) -> bool:
    latest_path = path.with_name("latest.pt")
    if not path.is_file() or not latest_path.is_file():
        return False
    try:
        best = torch.load(path, map_location="cpu", weights_only=True)
        latest = torch.load(latest_path, map_location="cpu", weights_only=True)
        meta = best.get("meta", {})
        loss = meta.get("loss", {})
        energy_meta = meta.get("learned_energy_score", {})
        return (
            int(best.get("step", -1)) > 0
            and int(latest.get("step", -1)) >= int(iterations)
            and str(meta.get("model_type")) == "siren"
            and int(meta.get("data_dim", -1)) == int(dimension)
            and str(loss.get("name")) == FORMAL_LOSS
            and Path(str(meta.get("point_cloud", ""))).resolve() == point_cloud.resolve()
            and Path(str(energy_meta.get("checkpoint", ""))).resolve()
            == energy_checkpoint.resolve()
            and float(loss.get("local_shell_center_sigma", -1.0))
            == float(local_shell_center_sigma)
            and str(loss.get("local_residual_normalization")) == "none"
            and float(loss.get("lambda_dir", -1.0)) == float(lambda_dir)
        )
    except Exception:
        return False


def run_stage(
    name: str,
    command: list[str],
    artifact: Path,
    validator: Callable[[Path], bool],
    *,
    output_dir: Path,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    if validator(artifact):
        state["stages"][name] = {"status": "skipped_complete", "artifact": str(artifact)}
        atomic_json(state_path, state)
        return
    log_path = output_dir / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state["stages"][name] = {"status": "running", "command": command}
    atomic_json(state_path, state)
    started = time.time()
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(json.dumps({"command": command}, ensure_ascii=False) + "\n")
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0 or not validator(artifact):
        state["stages"][name] = {
            "status": "failed",
            "return_code": int(completed.returncode),
            "log": str(log_path),
        }
        atomic_json(state_path, state)
        raise RuntimeError(f"stage {name} failed; see {log_path}")
    state["stages"][name] = {
        "status": "complete",
        "artifact": str(artifact),
        "elapsed_sec": time.time() - started,
    }
    atomic_json(state_path, state)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Input an N-by-D point cloud and output a learned implicit function checkpoint."
    )
    parser.add_argument("--point-cloud", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--energy-iterations", type=int, default=100_000)
    parser.add_argument("--phi-iterations", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--global-batch-size", type=int, default=4096)
    parser.add_argument("--eval-batches", type=int, default=12)
    parser.add_argument("--energy-learning-rate", type=float, default=2e-4)
    parser.add_argument("--phi-learning-rate", type=float, default=1e-4)
    parser.add_argument("--local-shell-center-sigma", type=float, default=2.5)
    parser.add_argument(
        "--lambda-dir",
        type=float,
        default=20.0,
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", default="cuda")
    return parser


def validate_positive_arguments(args: argparse.Namespace) -> None:
    for name in (
        "energy_iterations",
        "phi_iterations",
        "batch_size",
        "global_batch_size",
        "eval_batches",
        "energy_learning_rate",
        "phi_learning_rate",
        "local_shell_center_sigma",
        "lambda_dir",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def initial_state(point_cloud: Path, dimension: int) -> dict[str, Any]:
    return {
        "schema": "sgr-main-state/v1",
        "mode": "highdim",
        "point_cloud": str(point_cloud),
        "ambient_dim": dimension,
        "status": "running",
        "stages": {},
        "started_at_epoch": time.time(),
    }


def energy_training_command(
    args: argparse.Namespace,
    point_cloud: Path,
    metadata_path: Path,
    energy_dir: Path,
    splits: dict[str, int],
) -> list[str]:
    return [
        sys.executable,
        str(CODE_DIR / "construct_score.py"),
        "--point-cloud",
        str(point_cloud),
        "--normalization-meta",
        str(metadata_path),
        "--save",
        str(energy_dir / "latest.pt"),
        "--save-best",
        str(energy_dir / "best.pt"),
        "--train-size",
        str(splits["energy_train"]),
        "--val-size",
        str(splits["energy_val"]),
        "--batch-size",
        str(int(args.batch_size)),
        "--eval-batch-size",
        str(int(args.batch_size)),
        "--n-iters",
        str(int(args.energy_iterations)),
        "--lr",
        str(float(args.energy_learning_rate)),
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
    ]


def train_energy_score(
    args: argparse.Namespace,
    point_cloud: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    splits: dict[str, int],
    output_dir: Path,
    state: dict[str, Any],
    state_path: Path,
) -> Path:
    energy_dir = output_dir / "score"
    best_checkpoint = energy_dir / "best.pt"
    run_stage(
        "construct_score",
        energy_training_command(
            args,
            point_cloud,
            metadata_path,
            energy_dir,
            splits,
        ),
        best_checkpoint,
        lambda path: energy_complete(
            path,
            iterations=int(args.energy_iterations),
            dimension=int(metadata["ambient_dim"]),
        ),
        output_dir=output_dir,
        state=state,
        state_path=state_path,
    )
    return best_checkpoint


def phi_training_command(
    args: argparse.Namespace,
    point_cloud: Path,
    energy_checkpoint: Path,
    implicit_dir: Path,
    splits: dict[str, int],
) -> list[str]:
    return [
        sys.executable,
        str(CODE_DIR / "train_phi.py"),
        "--point-cloud",
        str(point_cloud),
        "--energy-checkpoint",
        str(energy_checkpoint),
        "--out",
        str(implicit_dir / "best.pt"),
        "--latest",
        str(implicit_dir / "latest.pt"),
        "--train-size",
        str(splits["phi_train"]),
        "--val-size",
        str(splits["phi_val"]),
        "--batch-size",
        str(int(args.batch_size)),
        "--global-batch-size",
        str(int(args.global_batch_size)),
        "--eval-batches",
        str(int(args.eval_batches)),
        "--n-iters",
        str(int(args.phi_iterations)),
        "--lr",
        str(float(args.phi_learning_rate)),
        "--local-shell-center-sigma",
        str(float(args.local_shell_center_sigma)),
        "--lambda-dir",
        str(float(args.lambda_dir)),
        "--seed",
        str(int(args.seed) + 1),
        "--device",
        str(args.device),
    ]


def train_implicit_function(
    args: argparse.Namespace,
    point_cloud: Path,
    energy_checkpoint: Path,
    metadata: dict[str, Any],
    splits: dict[str, int],
    output_dir: Path,
    state: dict[str, Any],
    state_path: Path,
) -> Path:
    implicit_dir = output_dir / "implicit_function"
    best_checkpoint = implicit_dir / "best.pt"
    run_stage(
        "train_phi",
        phi_training_command(
            args,
            point_cloud,
            energy_checkpoint,
            implicit_dir,
            splits,
        ),
        best_checkpoint,
        lambda path: implicit_checkpoint_complete(
            path,
            iterations=int(args.phi_iterations),
            dimension=int(metadata["ambient_dim"]),
            point_cloud=point_cloud,
            energy_checkpoint=energy_checkpoint,
            local_shell_center_sigma=float(args.local_shell_center_sigma),
            lambda_dir=float(args.lambda_dir),
        ),
        output_dir=output_dir,
        state=state,
        state_path=state_path,
    )
    return best_checkpoint


def write_pipeline_result(
    point_cloud: Path,
    metadata: dict[str, Any],
    energy_checkpoint: Path,
    implicit_checkpoint: Path,
    output_dir: Path,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    dimension = int(metadata["ambient_dim"])
    result = {
        "schema": "learned-implicit-function/v1",
        "mode": "highdim",
        "input_point_cloud": str(point_cloud),
        "input_points": int(metadata["num_points"]),
        "ambient_dim": dimension,
        "score_artifact": str(energy_checkpoint),
        "implicit_function": {
            "checkpoint": str(implicit_checkpoint),
            "model": "SIRENImplicitSurface",
            "field": f"Phi_theta: R^{dimension} -> R",
            "zero_set": "Phi_theta(x_normalized)=0",
        },
    }
    output_path = output_dir / "main_output.json"
    atomic_json(output_path, result)
    state.update(
        status="complete",
        output=str(output_path),
        finished_at_epoch=time.time(),
    )
    atomic_json(state_path, state)
    print(
        json.dumps(
            {"event": "done", "output": str(output_path), **result},
            ensure_ascii=False,
        )
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_positive_arguments(args)

    point_cloud = Path(args.point_cloud).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "input_metadata.json"
    metadata = prepare_metadata(point_cloud, metadata_path)
    splits = split_sizes(int(metadata["num_points"]))

    state_path = output_dir / "main_state.json"
    state = initial_state(point_cloud, int(metadata["ambient_dim"]))
    atomic_json(state_path, state)

    try:
        energy_checkpoint = train_energy_score(
            args,
            point_cloud,
            metadata_path,
            metadata,
            splits,
            output_dir,
            state,
            state_path,
        )
        implicit_checkpoint = train_implicit_function(
            args,
            point_cloud,
            energy_checkpoint,
            metadata,
            splits,
            output_dir,
            state,
            state_path,
        )
        write_pipeline_result(
            point_cloud,
            metadata,
            energy_checkpoint,
            implicit_checkpoint,
            output_dir,
            state,
            state_path,
        )
    except Exception as error:
        state.update(
            status="failed",
            error=str(error),
            finished_at_epoch=time.time(),
        )
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
