#!/usr/bin/env python3
"""Chain low-dimensional score construction and Phi training.

Run ``construct_score.py`` followed by ``train_phi.py`` and save the learned
implicit-function checkpoint, stage logs and pipeline state.
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
FORMAL_LOSS = "score_pull_plus_global_cosine"
CACHE_CONTRACT_VERSION = 4
LOCAL_RADIUS_CENTER_MULTIPLIER = 2.5
LOCAL_RADIUS_STD_MULTIPLIER = 1.0
LOCAL_RNG_SEED_OFFSET = 17


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_point_cloud(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        points = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    else:
        first = path.open(encoding="utf-8-sig", errors="replace").readline().split(",")
        try:
            [float(value) for value in first[:3]]
            skip_rows = 0
        except ValueError:
            skip_rows = 1
        points = np.loadtxt(path, delimiter=",", skiprows=skip_rows, dtype=np.float32)
    points = np.atleast_2d(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"{path} must contain an [N,3] point cloud, got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{path} contains non-finite coordinates")
    return points


def score_cache_complete(
    path: Path,
    *,
    point_cloud: Path,
    sigma: float,
    knn_k: int,
    local_cache_size: int,
    global_cache_size: int,
) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        meta = payload.get("meta", {})
        return (
            int(meta.get("cache_contract_version", -1)) == CACHE_CONTRACT_VERSION
            and Path(str(meta.get("point_cloud", ""))).resolve()
            == point_cloud.resolve()
            # The legacy label is valid only with the clamp_zero policy below.
            and str(meta.get("local_sampling_mode")) in {"clamped_gaussian", "gaussian_abs"}
            and str(meta.get("local_direction_mode")) == "normalized_isotropic_gaussian"
            and float(meta.get("local_radius_center_multiplier", -1.0))
            == LOCAL_RADIUS_CENTER_MULTIPLIER
            and float(meta.get("local_radius_std_multiplier", -1.0))
            == LOCAL_RADIUS_STD_MULTIPLIER
            and str(meta.get("local_radius_negative_policy")) == "clamp_zero"
            and int(meta.get("local_rng_seed_offset", -1)) == LOCAL_RNG_SEED_OFFSET
            and str(meta.get("normalization_mean_accumulator")) == "float32"
            and float(meta.get("sigma", -1.0)) == float(sigma)
            and int(meta.get("knn_k", -1)) == int(knn_k)
            and int(meta.get("local_cache_size", -1)) == int(local_cache_size)
            and int(meta.get("global_cache_size", -1)) == int(global_cache_size)
        )
    except Exception:
        return False


def implicit_checkpoint_complete(
    path: Path,
    *,
    point_cloud: Path,
    score_cache: Path,
    iterations: int,
    lambda_global: float,
    expected_cache_contract_version: int,
) -> bool:
    latest_path = path.with_name("latest.pt")
    if not path.is_file() or not latest_path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        best = torch.load(path, map_location="cpu", weights_only=True)
        latest = torch.load(latest_path, map_location="cpu", weights_only=True)
        meta = best.get("meta", {})
        loss = meta.get("loss", {})
        return (
            int(best.get("step", -1)) > 0
            and int(latest.get("step", -1)) >= int(iterations)
            and str(meta.get("model_type")) == "siren"
            and int(meta.get("data_dim", -1)) == 3
            and Path(str(meta.get("point_cloud", ""))).resolve() == point_cloud.resolve()
            and Path(str(meta.get("score_cache_pt", ""))).resolve() == score_cache.resolve()
            and str(loss.get("name")) == FORMAL_LOSS
            and float(loss.get("lambda_global", -1.0)) == float(lambda_global)
            and int(meta.get("score_cache_contract_version", -1))
            == int(expected_cache_contract_version)
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
        description="Input a 3D point cloud and output a learned implicit function checkpoint."
    )
    parser.add_argument("--point-cloud", required=True, help="Input [N,3] CSV or NPY point cloud.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--score-cache-pt",
        default="",
        help=(
            "Optional cache produced by this repository's construct_score.py. "
            "If omitted, construct one from the input point cloud."
        ),
    )
    parser.add_argument("--local-cache-size", type=int, default=250_000)
    parser.add_argument("--global-cache-size", type=int, default=250_000)
    parser.add_argument("--sigma", type=float, default=0.002)
    parser.add_argument("--knn-k", type=int, default=2048)
    parser.add_argument("--knn-workers", type=int, default=-1)
    parser.add_argument("--score-chunk-size", type=int, default=256)
    parser.add_argument("--lambda-global", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--global-batch-size", type=int, default=4096)
    parser.add_argument("--eval-batches", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", default="cuda")
    return parser


def initial_state(point_cloud: Path) -> dict[str, Any]:
    return {
        "schema": "sgr-main-state/v1",
        "mode": "lowdim",
        "point_cloud": str(point_cloud),
        "ambient_dim": 3,
        "status": "running",
        "stages": {},
        "started_at_epoch": time.time(),
    }


def score_construction_command(
    args: argparse.Namespace,
    point_cloud: Path,
    score_cache: Path,
) -> list[str]:
    return [
        sys.executable,
        str(CODE_DIR / "construct_score.py"),
        "--point-cloud",
        str(point_cloud),
        "--out-pt",
        str(score_cache),
        "--local-cache-size",
        str(int(args.local_cache_size)),
        "--global-cache-size",
        str(int(args.global_cache_size)),
        "--sigma",
        str(float(args.sigma)),
        "--knn-k",
        str(int(args.knn_k)),
        "--knn-workers",
        str(int(args.knn_workers)),
        "--chunk-size",
        str(int(args.score_chunk_size)),
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
    ]


def resolve_score_cache(
    args: argparse.Namespace,
    point_cloud: Path,
    output_dir: Path,
    state: dict[str, Any],
    state_path: Path,
) -> tuple[Path, int, int]:
    """Reuse a validated score cache or construct one from the input cloud."""
    if args.score_cache_pt:
        score_cache = Path(args.score_cache_pt).expanduser().resolve()
        if not score_cache.is_file():
            raise FileNotFoundError(score_cache)
        payload = torch.load(score_cache, map_location="cpu", weights_only=True)
        contract_version = int(
            payload.get("meta", {}).get("cache_contract_version", -1)
        )
        if contract_version != CACHE_CONTRACT_VERSION:
            raise ValueError(
                f"Score cache must use contract v{CACHE_CONTRACT_VERSION}; "
                f"got v{contract_version}"
            )
        state["stages"]["construct_score"] = {
            "status": "reused_external",
            "artifact": str(score_cache),
            "cache_contract_version": contract_version,
        }
        atomic_json(state_path, state)
        return score_cache, contract_version, int(args.seed)

    score_cache = output_dir / "score" / "score_cache.pt"
    run_stage(
        "construct_score",
        score_construction_command(args, point_cloud, score_cache),
        score_cache,
        lambda path: score_cache_complete(
            path,
            point_cloud=point_cloud,
            sigma=float(args.sigma),
            knn_k=int(args.knn_k),
            local_cache_size=int(args.local_cache_size),
            global_cache_size=int(args.global_cache_size),
        ),
        output_dir=output_dir,
        state=state,
        state_path=state_path,
    )
    return score_cache, CACHE_CONTRACT_VERSION, int(args.seed) + 1


def phi_training_command(
    args: argparse.Namespace,
    point_cloud: Path,
    score_cache: Path,
    implicit_dir: Path,
    training_seed: int,
) -> list[str]:
    return [
        sys.executable,
        str(CODE_DIR / "train_phi.py"),
        "--point-cloud",
        str(point_cloud),
        "--score-cache-pt",
        str(score_cache),
        "--out",
        str(implicit_dir / "best.pt"),
        "--latest",
        str(implicit_dir / "latest.pt"),
        "--batch-size",
        str(int(args.batch_size)),
        "--global-batch-size",
        str(int(args.global_batch_size)),
        "--eval-batches",
        str(int(args.eval_batches)),
        "--n-iters",
        str(int(args.iterations)),
        "--lr",
        str(float(args.learning_rate)),
        "--lambda-global",
        str(float(args.lambda_global)),
        "--seed",
        str(training_seed),
        "--device",
        str(args.device),
    ]


def train_implicit_function(
    args: argparse.Namespace,
    point_cloud: Path,
    score_cache: Path,
    cache_contract_version: int,
    training_seed: int,
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
            score_cache,
            implicit_dir,
            training_seed,
        ),
        best_checkpoint,
        lambda path: implicit_checkpoint_complete(
            path,
            point_cloud=point_cloud,
            score_cache=score_cache,
            iterations=int(args.iterations),
            lambda_global=float(args.lambda_global),
            expected_cache_contract_version=cache_contract_version,
        ),
        output_dir=output_dir,
        state=state,
        state_path=state_path,
    )
    return best_checkpoint


def write_pipeline_result(
    point_cloud: Path,
    point_count: int,
    score_cache: Path,
    best_checkpoint: Path,
    output_dir: Path,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    result = {
        "schema": "learned-implicit-function/v1",
        "mode": "lowdim",
        "input_point_cloud": str(point_cloud),
        "input_points": point_count,
        "ambient_dim": 3,
        "score_artifact": str(score_cache),
        "implicit_function": {
            "checkpoint": str(best_checkpoint),
            "model": "SIRENImplicitSurface",
            "field": "Phi_theta: R^3 -> R",
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
    point_cloud = Path(args.point_cloud).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    points = load_point_cloud(point_cloud)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = output_dir / "main_state.json"
    state = initial_state(point_cloud)
    atomic_json(state_path, state)

    try:
        score_cache, contract_version, training_seed = resolve_score_cache(
            args,
            point_cloud,
            output_dir,
            state,
            state_path,
        )
        best_checkpoint = train_implicit_function(
            args,
            point_cloud,
            score_cache,
            contract_version,
            training_seed,
            output_dir,
            state,
            state_path,
        )
        write_pipeline_result(
            point_cloud,
            int(len(points)),
            score_cache,
            best_checkpoint,
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
