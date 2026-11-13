#!/usr/bin/env python3
"""Construct an empirical Gaussian-KNN score cache from one 3D point cloud."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree


LOCAL_RADIUS_CENTER_MULTIPLIER = 2.5
LOCAL_RADIUS_STD_MULTIPLIER = 1.0
LOCAL_RNG_SEED_OFFSET = 17
BBOX_PAD_RATIO = 0.30
CACHE_CONTRACT_VERSION = 4


def load_points(path: Path) -> np.ndarray:
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


def normalize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    # Keep the input and NumPy mean accumulator in float32.
    mean = points.mean(axis=0).astype(np.float32)
    centered = points - mean[None, :]
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid normalization scale: {scale}")
    return (centered / scale).astype(np.float32), mean, scale


def empirical_score_knn(
    tree: cKDTree,
    samples: np.ndarray,
    query: np.ndarray,
    *,
    sigma: float,
    k: int,
    chunk_size: int,
    workers: int,
    device: torch.device,
) -> np.ndarray:
    """Evaluate the KNN-truncated empirical score from Section 6.3."""

    sigma_squared = np.float32(float(sigma) ** 2)
    output = np.empty_like(query, dtype=np.float32)
    k_effective = min(int(k), len(samples))
    samples_device = torch.from_numpy(samples).to(device) if device.type == "cuda" else None

    def query_chunk(start: int, end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = query[start:end].astype(np.float32)
        try:
            distance, index = tree.query(
                points.astype(np.float64), k=k_effective, workers=int(workers)
            )
        except TypeError:
            distance, index = tree.query(points.astype(np.float64), k=k_effective)
        distance = np.asarray(distance, dtype=np.float32)
        index = np.asarray(index)
        if index.ndim == 1:
            distance, index = distance[:, None], index[:, None]
        return points, distance, index

    def reduce_chunk(
        start: int,
        end: int,
        points: np.ndarray,
        distance: np.ndarray,
        index: np.ndarray,
    ) -> None:
        if samples_device is None:
            neighbors = samples[index]
            log_weight = -0.5 * np.square(distance) / sigma_squared
            log_weight -= log_weight.max(axis=1, keepdims=True)
            weight = np.exp(log_weight).astype(np.float32)
            weight /= np.maximum(weight.sum(axis=1, keepdims=True), np.float32(1e-30))
            weighted_mean = np.einsum("bk,bkd->bd", weight, neighbors)
            output[start:end] = (weighted_mean - points) / sigma_squared
            return
        with torch.inference_mode():
            distance_t = torch.from_numpy(distance).to(device)
            index_t = torch.from_numpy(index).to(device)
            points_t = torch.from_numpy(points).to(device)
            weight = torch.softmax(-0.5 * distance_t.square() / sigma_squared, dim=1)
            weighted_mean = torch.bmm(
                weight[:, None, :], samples_device[index_t]
            ).squeeze(1)
            output[start:end] = ((weighted_mean - points_t) / sigma_squared).cpu().numpy()

    ranges = [
        (start, min(start + int(chunk_size), len(query)))
        for start in range(0, len(query), int(chunk_size))
    ]
    if device.type == "cuda" and len(ranges) > 1:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(query_chunk, *ranges[0])
            for position, (start, end) in enumerate(ranges):
                points, distance, index = future.result()
                if position + 1 < len(ranges):
                    future = executor.submit(query_chunk, *ranges[position + 1])
                reduce_chunk(start, end, points, distance, index)
                print(json.dumps({"event": "score_chunk", "start": start, "end": end}), flush=True)
    else:
        for start, end in ranges:
            points, distance, index = query_chunk(start, end)
            reduce_chunk(start, end, points, distance, index)
            print(json.dumps({"event": "score_chunk", "start": start, "end": end}), flush=True)
    return output


def sample_local_queries(
    rng: np.random.Generator,
    base: np.ndarray,
    *,
    sigma: float,
) -> np.ndarray:
    direction = rng.standard_normal(base.shape, dtype=np.float32)
    direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = direction / np.maximum(direction_norm, np.float32(1e-12))
    radius = rng.normal(
        loc=LOCAL_RADIUS_CENTER_MULTIPLIER * float(sigma),
        scale=LOCAL_RADIUS_STD_MULTIPLIER * float(sigma),
        size=(base.shape[0], 1),
    ).astype(np.float32)
    radius = np.maximum(radius, np.float32(0.0))
    return (base + radius * direction).astype(np.float32)


def padded_bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the 30%-padded normalized box used for global score queries."""
    low = points.min(axis=0)
    high = points.max(axis=0)
    padding = BBOX_PAD_RATIO * (high - low)
    return (low - padding).astype(np.float32), (high + padding).astype(np.float32)


def cache_metadata(
    args: argparse.Namespace,
    *,
    point_cloud: Path,
    point_count: int,
    data_mean: np.ndarray,
    data_scale: float,
    bg_low: np.ndarray,
    bg_high: np.ndarray,
    rng_seed: int,
) -> dict[str, Any]:
    """Describe the normalization, sampling law, and KNN score contract."""
    return {
        "point_cloud": str(point_cloud),
        "num_points": point_count,
        "data_dim": 3,
        "data_mean": data_mean.astype(float).tolist(),
        "data_scale": float(data_scale),
        "normalization_mean_accumulator": "float32",
        "cache_contract_version": CACHE_CONTRACT_VERSION,
        "score_cache_type": "input_point_cloud_knn_empirical",
        "coordinate_space": "normalized",
        "score_space": "normalized",
        "sigma_space": "normalized",
        "local_base_source": "point_cloud",
        "global_bbox_source": "point_cloud",
        "local_sampling_mode": "clamped_gaussian",
        "local_direction_mode": "normalized_isotropic_gaussian",
        "local_radius_center_multiplier": LOCAL_RADIUS_CENTER_MULTIPLIER,
        "local_radius_std_multiplier": LOCAL_RADIUS_STD_MULTIPLIER,
        "local_radius_negative_policy": "clamp_zero",
        "local_rng_seed_offset": LOCAL_RNG_SEED_OFFSET,
        "local_radius_center": LOCAL_RADIUS_CENTER_MULTIPLIER * float(args.sigma),
        "local_radius_std": LOCAL_RADIUS_STD_MULTIPLIER * float(args.sigma),
        "sigma": float(args.sigma),
        "knn_k": int(args.knn_k),
        "local_cache_size": int(args.local_cache_size),
        "global_cache_size": int(args.global_cache_size),
        "bbox_pad_ratio": BBOX_PAD_RATIO,
        "bg_low": bg_low.astype(float).tolist(),
        "bg_high": bg_high.astype(float).tolist(),
        "seed": int(args.seed),
        "rng_seed": rng_seed,
    }


def save_cache(
    output_path: Path,
    *,
    local_query: np.ndarray,
    local_score: np.ndarray,
    global_query: np.ndarray,
    global_score: np.ndarray,
    sigma: float,
    meta: dict[str, Any],
) -> None:
    target_pull = (-(float(sigma) ** 2) * local_score).astype(np.float32)
    payload = {
        "local_cache": {
            "q_norm": torch.from_numpy(local_query),
            "target_pull_norm": torch.from_numpy(target_pull),
        },
        "global_cache": {
            "z_norm": torch.from_numpy(global_query),
            "score_norm": torch.from_numpy(global_score),
        },
        "meta": meta,
    }
    torch.save(payload, output_path)
    summary = {
        "meta": meta,
        "local_target_norm_mean": float(
            np.linalg.norm(target_pull, axis=1).mean()
        ),
        "global_score_norm_mean": float(np.linalg.norm(global_score, axis=1).mean()),
        "cache": str(output_path),
    }
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)


def build_cache(args: argparse.Namespace) -> None:
    """Sample local/global queries, evaluate scores, and save cache contract v4."""
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    point_cloud_path = Path(args.point_cloud).expanduser().resolve()
    output_path = Path(args.out_pt).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_points = load_points(point_cloud_path)
    points, data_mean, data_scale = normalize(raw_points)
    tree = cKDTree(points.astype(np.float64))
    # Keep cache sampling independent from the model-training random stream.
    rng_seed = int(args.seed) + LOCAL_RNG_SEED_OFFSET
    rng = np.random.default_rng(rng_seed)

    local_index = rng.integers(0, len(points), size=int(args.local_cache_size))
    local_query = sample_local_queries(
        rng,
        points[local_index],
        sigma=float(args.sigma),
    )
    local_score = empirical_score_knn(
        tree,
        points,
        local_query,
        sigma=float(args.sigma),
        k=int(args.knn_k),
        chunk_size=int(args.chunk_size),
        workers=int(args.knn_workers),
        device=device,
    )

    bg_low, bg_high = padded_bounds(points)
    global_query = bg_low[None, :] + (bg_high - bg_low)[None, :] * rng.random(
        (int(args.global_cache_size), 3), dtype=np.float32
    )
    global_score = empirical_score_knn(
        tree,
        points,
        global_query,
        sigma=float(args.sigma),
        k=int(args.knn_k),
        chunk_size=int(args.chunk_size),
        workers=int(args.knn_workers),
        device=device,
    )

    meta = cache_metadata(
        args,
        point_cloud=point_cloud_path,
        point_count=int(len(points)),
        data_mean=data_mean,
        data_scale=data_scale,
        bg_low=bg_low,
        bg_high=bg_high,
        rng_seed=rng_seed,
    )
    save_cache(
        output_path,
        local_query=local_query,
        local_score=local_score,
        global_query=global_query,
        global_score=global_score,
        sigma=float(args.sigma),
        meta=meta,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the paper's Gaussian-KNN score cache from an [N,3] point cloud."
    )
    parser.add_argument("--point-cloud", required=True)
    parser.add_argument("--out-pt", required=True)
    parser.add_argument("--local-cache-size", type=int, default=250_000)
    parser.add_argument("--global-cache-size", type=int, default=250_000)
    parser.add_argument("--sigma", type=float, default=0.002)
    parser.add_argument("--knn-k", type=int, default=2048)
    parser.add_argument("--knn-workers", type=int, default=-1)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "local_cache_size",
        "global_cache_size",
        "sigma",
        "knn_k",
        "chunk_size",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    build_cache(args)


if __name__ == "__main__":
    main()
