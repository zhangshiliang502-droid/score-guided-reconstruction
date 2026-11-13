"""Small-noise empirical-versus-trained score comparison on real 3D shapes.

Bunny and FlowerPetal are sampled from their reference meshes. Gargoyle is
sampled continuously from local tangent patches fitted to its oriented scan.
Every trained model and the fixed empirical KNN estimator therefore see the
same normalized surface distribution. Evaluation uses independent clean
surface samples and known normal offsets; no sphere-specific analytic score is
used in this experiment.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiment import (
    PROFILE_T,
    atomic_json,
    build_model,
    predict_epsilon,
    sigma_tag,
    update_ema,
)


DATASET_LABELS = {
    "bunny": "Bunny",
    "gargoyle": "Gargoyle",
    "flower": "Flower",
}
METHOD_LABELS = {
    "empirical_knn": "Empirical KNN",
    "trained_direct": "Trained DSM",
    "trained_energy": "Trained energy DSM",
}
METHOD_COLORS = {
    "empirical_knn": "#167A72",
    "trained_direct": "#D55E00",
    "trained_energy": "#7A5195",
}


def load_xyz_csv(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first = handle.readline().strip().lower()
    skiprows = 1 if any(character.isalpha() for character in first) else 0
    points = np.loadtxt(path, delimiter=",", skiprows=skiprows, dtype=np.float32)
    points = np.atleast_2d(points)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        raise ValueError(f"{path} must contain an [N,3] point cloud")
    points = np.asarray(points[:, :3], dtype=np.float32)
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{path} contains non-finite coordinates")
    return points


def sgr_normalization(points: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(points, dtype=np.float32)
    mean = values.mean(axis=0, dtype=np.float32)
    centered = values - mean[None, :]
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("invalid point-cloud normalization scale")
    return mean, scale


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class SurfaceSampler(Protocol):
    area: float

    def sample(
        self, count: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class MeshSampler:
    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        triangles = self.vertices[self.faces]
        crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        doubled_area = np.linalg.norm(crosses, axis=1)
        valid = doubled_area > 1e-15
        self.triangles = np.asarray(triangles[valid], dtype=np.float64)
        doubled_area = doubled_area[valid]
        self.normals = crosses[valid] / doubled_area[:, None]
        areas = 0.5 * doubled_area
        self.area = float(np.sum(areas))
        self.area_cdf = np.cumsum(areas)
        self.area_cdf /= self.area_cdf[-1]

    def sample(
        self, count: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        face_index = np.searchsorted(self.area_cdf, rng.random(count), side="right")
        triangle = self.triangles[face_index]
        root = np.sqrt(rng.random(count))
        second = rng.random(count)
        weights = np.column_stack(
            (1.0 - root, root * (1.0 - second), root * second)
        )
        points = np.sum(weights[:, :, None] * triangle, axis=1)
        return points.astype(np.float32), self.normals[face_index].astype(np.float32)


@dataclass
class OrientedPointSampler:
    points: np.ndarray
    normals: np.ndarray

    def __post_init__(self) -> None:
        from scipy.spatial import cKDTree

        self.points = np.asarray(self.points, dtype=np.float64)
        self.normals = np.asarray(self.normals, dtype=np.float64)
        self.normals /= np.linalg.norm(self.normals, axis=1, keepdims=True).clip(1e-12)
        tree = cKDTree(self.points)
        distances, _ = tree.query(self.points, k=9, workers=-1)
        eighth = np.asarray(distances[:, -1], dtype=np.float64)
        low, high = np.quantile(eighth, (0.02, 0.98))
        eighth = np.clip(eighth, low, high)
        self.patch_radius = 0.75 * eighth / math.sqrt(8.0)
        self.area = float(np.sum(math.pi * eighth * eighth / 8.0))

    def sample(
        self, count: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        index = rng.integers(0, len(self.points), size=count)
        normals = self.normals[index]
        axes = np.zeros_like(normals)
        use_z = np.abs(normals[:, 2]) < 0.9
        axes[use_z, 2] = 1.0
        axes[~use_z, 0] = 1.0
        tangent_u = np.cross(normals, axes)
        tangent_u /= np.linalg.norm(tangent_u, axis=1, keepdims=True).clip(1e-12)
        tangent_v = np.cross(normals, tangent_u)
        angle = 2.0 * math.pi * rng.random(count)
        radius = self.patch_radius[index] * np.sqrt(rng.random(count))
        offset = radius[:, None] * (
            np.cos(angle)[:, None] * tangent_u + np.sin(angle)[:, None] * tangent_v
        )
        return (self.points[index] + offset).astype(np.float32), normals.astype(np.float32)


def load_mesh_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".npz":
        payload = np.load(path)
        return np.asarray(payload["vertices"], dtype=np.float64), np.asarray(
            payload["faces"], dtype=np.int64
        )
    import trimesh

    loaded = trimesh.load(path, process=False, maintain_order=True)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"{path} is not a triangle mesh")
    return np.asarray(loaded.vertices, dtype=np.float64), np.asarray(
        loaded.faces, dtype=np.int64
    )


def load_oriented_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    loaded = trimesh.load(path, process=False, maintain_order=True)
    raw = loaded.metadata.get("_ply_raw", {}).get("vertex", {}).get("data")
    required = ("x", "y", "z", "nx", "ny", "nz")
    if raw is None or not all(name in raw.dtype.names for name in required):
        raise ValueError(f"{path} must contain x/y/z and nx/ny/nz vertex properties")
    points = np.column_stack([raw[name] for name in required[:3]])
    normals = np.column_stack([raw[name] for name in required[3:]])
    return np.asarray(points, dtype=np.float64), np.asarray(normals, dtype=np.float64)


def build_sampler(args: argparse.Namespace) -> tuple[SurfaceSampler, dict]:
    input_points = load_xyz_csv(Path(args.point_cloud))
    mean, scale = sgr_normalization(input_points)
    surface_path = Path(args.surface)
    if args.surface_kind == "mesh":
        vertices, faces = load_mesh_arrays(surface_path)
        vertices = (vertices - mean[None, :]) / scale
        sampler: SurfaceSampler = MeshSampler(vertices, faces)
        sampler_metadata = {
            "kind": "area_uniform_reference_mesh",
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
        }
    else:
        points, normals = load_oriented_ply(surface_path)
        points = (points - mean[None, :]) / scale
        sampler = OrientedPointSampler(points, normals)
        sampler_metadata = {
            "kind": "oriented_scan_local_tangent_patches",
            "oriented_points": int(len(points)),
            "patch_radius_rule": "0.75 * clipped_8NN_distance / sqrt(8)",
        }
    metadata = {
        "dataset": args.dataset,
        "dataset_label": DATASET_LABELS[args.dataset],
        "point_cloud": str(args.point_cloud),
        "point_cloud_sha256": sha256_file(Path(args.point_cloud)),
        "surface": str(surface_path),
        "surface_sha256": sha256_file(surface_path),
        "normalization": "float32_mean_and_max_center_radius",
        "normalization_mean": [float(value) for value in mean],
        "normalization_scale": float(scale),
        "normalized_surface_area": float(sampler.area),
        "sampler": sampler_metadata,
    }
    return sampler, metadata


def write_support(
    sampler: SurfaceSampler,
    output: Path,
    count: int,
    chunk_size: int,
    seed: int,
) -> None:
    if output.exists():
        existing = np.load(output, mmap_mode="r")
        if existing.shape == (count, 3) and existing.dtype == np.float32:
            print(json.dumps({"event": "reuse_support", "path": str(output)}), flush=True)
            return
        raise ValueError(f"existing support has unexpected shape/dtype: {existing.shape}/{existing.dtype}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    memory = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(count, 3)
    )
    rng = np.random.default_rng(seed)
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        points, _ = sampler.sample(end - start, rng)
        memory[start:end] = points
        if start == 0 or end == count or end % (10 * chunk_size) == 0:
            print(json.dumps({"event": "support", "written": end, "total": count}), flush=True)
    memory.flush()
    del memory
    temporary.replace(output)


def write_reference(
    sampler: SurfaceSampler, output: Path, count: int, seed: int
) -> None:
    if output.exists():
        payload = np.load(output)
        if payload["points"].shape == (count, 3) and payload["normals"].shape == (count, 3):
            print(json.dumps({"event": "reuse_reference", "path": str(output)}), flush=True)
            return
        raise ValueError(f"existing reference has unexpected shape: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    points, normals = sampler.sample(count, np.random.default_rng(seed))
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, points=points, normals=normals)
    temporary.replace(output)


def command_prepare(args: argparse.Namespace) -> None:
    sampler, metadata = build_sampler(args)
    output_dir = Path(args.output_dir)
    support_path = output_dir / "support_40m.npy"
    reference_path = output_dir / "reference_2m.npz"
    write_support(sampler, support_path, args.n_support, args.chunk_size, args.seed)
    write_reference(sampler, reference_path, args.n_reference, args.seed + 1)
    metadata.update(
        {
            "support_points": int(args.n_support),
            "support_seed": int(args.seed),
            "reference_points": int(args.n_reference),
            "reference_seed": int(args.seed + 1),
            "support_path": str(support_path),
            "reference_path": str(reference_path),
        }
    )
    atomic_json(output_dir / "metadata.json", metadata)


def load_support_to_ram(path: Path) -> np.ndarray:
    memory = np.load(path, mmap_mode="r")
    if memory.ndim != 2 or memory.shape[1] != 3 or memory.dtype != np.float32:
        raise ValueError(f"invalid support {path}: {memory.shape}/{memory.dtype}")
    return np.array(memory, dtype=np.float32, copy=True)


def validation_loss(
    model: nn.Module,
    model_type: str,
    clean: torch.Tensor,
    noise: torch.Tensor,
    sigma: float,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, len(clean), batch_size):
        end = min(start + batch_size, len(clean))
        noisy = clean[start:end] + float(sigma) * noise[start:end]
        with torch.enable_grad():
            prediction = predict_epsilon(
                model, noisy, sigma, model_type, create_graph=False
            )
        total += float(F.mse_loss(prediction, noise[start:end], reduction="sum").item())
        count += int(prediction.numel())
    return total / max(count, 1)


def command_train(args: argparse.Namespace) -> None:
    if args.sigma <= 0.0:
        raise ValueError("sigma must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    noise_generator = torch.Generator(device=device).manual_seed(args.seed)
    index_rng = np.random.default_rng(args.seed)
    support = load_support_to_ram(Path(args.support))
    model_kwargs = {
        "hidden_dim": args.hidden_dim,
        "time_dim": args.time_dim,
        "num_blocks": args.num_blocks,
    }
    model = build_model(args.model, model_kwargs).to(device)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    validation_index = index_rng.integers(0, len(support), size=args.val_size)
    validation_clean = torch.as_tensor(
        support[validation_index].copy(), dtype=torch.float32, device=device
    )
    validation_noise = torch.randn(
        validation_clean.shape, device=device, generator=noise_generator
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    last_loss = math.nan

    for step in range(1, args.steps + 1):
        model.train()
        index = index_rng.integers(0, len(support), size=args.batch_size)
        clean = torch.as_tensor(support[index].copy(), dtype=torch.float32, device=device)
        noise = torch.randn(clean.shape, device=device, generator=noise_generator)
        noisy = clean + float(args.sigma) * noise
        if args.model == "energy":
            noisy = noisy.detach().requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)
        prediction = predict_epsilon(
            model, noisy, args.sigma, args.model, create_graph=args.model == "energy"
        )
        loss = F.mse_loss(prediction, noise)
        learning_rate = args.lr * min(step / max(args.warmup, 1), 1.0)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        update_ema(ema, model, args.ema_decay)
        last_loss = float(loss.detach().cpu())
        if step == 1 or step % args.log_every == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "event": "train",
                        "dataset": args.dataset,
                        "model": args.model,
                        "sigma": args.sigma,
                        "seed": args.seed,
                        "step": step,
                        "steps": args.steps,
                        "eps_mse": last_loss,
                        "grad_norm": float(gradient_norm.detach().cpu()),
                        "steps_per_second": step / max(elapsed, 1e-12),
                    }
                ),
                flush=True,
            )

    final_validation = validation_loss(
        ema,
        args.model,
        validation_clean,
        validation_noise,
        args.sigma,
        args.eval_batch_size,
    )
    checkpoint = {
        "dataset": args.dataset,
        "model_type": args.model,
        "sigma": float(args.sigma),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "train_examples_seen": int(args.steps * args.batch_size),
        "support_points": int(len(support)),
        "model_kwargs": model_kwargs,
        "ema_state_dict": ema.state_dict(),
        "model_state_dict": model.state_dict(),
        "final_train_eps_mse": last_loss,
        "final_validation_eps_mse": final_validation,
        "score_convention": "network predicts epsilon; score=-epsilon/sigma",
    }
    torch.save(checkpoint, output)
    print(
        json.dumps(
            {"event": "done", "output": str(output), "validation_eps_mse": final_validation}
        ),
        flush=True,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["model_type"], checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["ema_state_dict"])
    model.eval().requires_grad_(False)
    return model, checkpoint


def trained_score_numpy(
    model: nn.Module,
    model_type: str,
    query: np.ndarray,
    sigma: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(query), batch_size):
        tensor = torch.as_tensor(
            query[start : start + batch_size], dtype=torch.float32, device=device
        )
        with torch.enable_grad():
            epsilon = predict_epsilon(
                model, tensor, sigma, model_type, create_graph=False
            )
        chunks.append((-epsilon / float(sigma)).detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def gaussian_mass_knn(
    n_points: int,
    surface_area: float,
    sigma: float,
    mass: float,
    safety: float,
) -> int:
    if not 0.0 < mass < 1.0:
        raise ValueError("mass must lie in (0,1)")
    radius = float(sigma) * math.sqrt(-2.0 * math.log(1.0 - mass))
    expected = float(n_points) * math.pi * radius * radius / float(surface_area)
    conservative = expected + 6.0 * math.sqrt(max(expected, 1.0)) + 32.0
    paper_density_floor = 2048.0 * (float(sigma) / 0.002) ** 2
    selected = safety * max(conservative, paper_density_floor)
    return min(int(n_points), max(64, int(math.ceil(selected))))


def empirical_score_numpy(
    tree: object,
    support: np.ndarray,
    query: np.ndarray,
    sigma: float,
    k: int,
    chunk_size: int,
    workers: int,
) -> np.ndarray:
    output = np.empty_like(query, dtype=np.float32)
    sigma_squared = float(sigma * sigma)
    for start in range(0, len(query), chunk_size):
        end = min(start + chunk_size, len(query))
        distances, indices = tree.query(query[start:end], k=k, workers=workers)
        distances = np.asarray(distances, dtype=np.float64)
        indices = np.asarray(indices)
        weights = np.exp(-0.5 * distances * distances / sigma_squared)
        neighbors = np.asarray(support[indices], dtype=np.float64)
        weighted_mean = np.sum(weights[:, :, None] * neighbors, axis=1)
        weighted_mean /= np.sum(weights, axis=1, keepdims=True).clip(1e-300)
        output[start:end] = ((weighted_mean - query[start:end]) / sigma_squared).astype(np.float32)
        if start == 0 or end == len(query) or end % (20 * chunk_size) == 0:
            print(
                json.dumps(
                    {"event": "empirical_query", "sigma": sigma, "done": end, "total": len(query), "k": k}
                ),
                flush=True,
            )
    return output


def metric_summary(
    dataset: str,
    method: str,
    sigma: float,
    estimated_score: np.ndarray,
    query: np.ndarray,
    clean: np.ndarray,
    surface_tree: object,
    seed: int | None,
) -> dict:
    pull = float(sigma * sigma) * estimated_score.astype(np.float64)
    denoised = query.astype(np.float64) + pull
    paired_error = denoised - clean.astype(np.float64)
    paired_norm = np.linalg.norm(paired_error, axis=1)
    target_pull = clean.astype(np.float64) - query.astype(np.float64)
    target_norm = np.linalg.norm(target_pull, axis=1)
    pull_error = pull - target_pull
    nearest, _ = surface_tree.query(denoised, k=1, workers=-1)
    valid = target_norm >= 0.25 * float(sigma)
    pull_norm = np.linalg.norm(pull[valid], axis=1)
    cosine = np.sum(pull[valid] * target_pull[valid], axis=1)
    cosine /= np.maximum(pull_norm * target_norm[valid], 1e-30)
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return {
        "dataset": dataset,
        "method": method,
        "sigma": float(sigma),
        "seed": seed,
        "n_queries": int(len(query)),
        "denoise_rmse_over_sigma": float(np.sqrt(np.mean(paired_norm**2)) / float(sigma)),
        "denoise_p90_over_sigma": float(np.quantile(paired_norm, 0.9) / float(sigma)),
        "pull_target_rmse_over_sigma": float(
            np.sqrt(np.mean(np.sum(pull_error * pull_error, axis=1))) / float(sigma)
        ),
        "surface_mae_over_sigma": float(np.mean(nearest) / float(sigma)),
        "surface_p90_over_sigma": float(np.quantile(nearest, 0.9) / float(sigma)),
        "angle_mean_deg": float(np.mean(angles)),
        "angle_p90_deg": float(np.quantile(angles, 0.9)),
    }


def profile_metrics(
    score: np.ndarray,
    normals: np.ndarray,
    sigma: float,
) -> tuple[dict[str, float], np.ndarray]:
    count = len(normals)
    tiled_normals = np.tile(normals.astype(np.float64), (len(PROFILE_T), 1))
    pull = float(sigma * sigma) * score.astype(np.float64)
    radial = np.sum(pull * tiled_normals, axis=1) / float(sigma)
    means = np.asarray(
        [np.mean(radial[index * count : (index + 1) * count]) for index in range(len(PROFILE_T))]
    )
    fit = np.abs(PROFILE_T) <= 1.0
    slope, intercept = np.polyfit(PROFILE_T[fit], means[fit], 1)
    offset = np.repeat(PROFILE_T, count)
    residual = offset + radial
    tangential = pull - np.sum(pull * tiled_normals, axis=1, keepdims=True) * tiled_normals
    return (
        {
            "normal_slope_ratio": float(-slope),
            "normal_intercept_over_sigma": float(intercept),
            "normal_residual_mae_over_sigma": float(np.mean(np.abs(residual[fit.repeat(count)]))),
            "profile_tangential_rms_over_sigma": float(
                np.sqrt(np.mean(np.sum(tangential * tangential, axis=1))) / float(sigma)
            ),
        },
        means,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_seed(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"checkpoint name lacks seed: {path}")
    return int(match.group(1))


def discover_checkpoints(root: Path, dataset: str, sigma: float) -> list[Path]:
    return sorted(root.glob(f"{dataset}/*/sigma_{sigma_tag(sigma)}/seed_*.pt"))


def append_profile_rows(
    rows: list[dict],
    dataset: str,
    method: str,
    sigma: float,
    seed: int | None,
    means: np.ndarray,
) -> None:
    for offset, value in zip(PROFILE_T, means):
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "sigma": float(sigma),
                "seed": seed,
                "offset_over_sigma": float(offset),
                "normal_pull_over_sigma": float(value),
            }
        )


def command_evaluate(args: argparse.Namespace) -> None:
    from scipy.spatial import cKDTree

    device = torch.device(args.device)
    all_rows: list[dict] = []
    profile_rows: list[dict] = []
    data_root = Path(args.data_root)
    checkpoint_root = Path(args.checkpoint_root)

    for dataset_index, dataset in enumerate(args.datasets):
        dataset_dir = data_root / dataset
        metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
        support = np.load(dataset_dir / "support_40m.npy", mmap_mode="r")
        reference_payload = np.load(dataset_dir / "reference_2m.npz")
        reference = np.asarray(reference_payload["points"], dtype=np.float32)
        normals = np.asarray(reference_payload["normals"], dtype=np.float32)
        print(json.dumps({"event": "build_reference_tree", "dataset": dataset}), flush=True)
        surface_tree = cKDTree(np.asarray(reference, dtype=np.float64))
        print(json.dumps({"event": "build_support_tree", "dataset": dataset, "n": len(support)}), flush=True)
        tree_started = time.perf_counter()
        support_tree = cKDTree(
            np.asarray(support, dtype=np.float64), compact_nodes=True, balanced_tree=True
        )
        print(
            json.dumps(
                {"event": "support_tree_ready", "dataset": dataset, "seconds": time.perf_counter() - tree_started}
            ),
            flush=True,
        )
        rng = np.random.default_rng(args.query_seed + dataset_index)
        selection = rng.choice(len(reference), size=args.n_queries + args.profile_directions, replace=False)
        clean = reference[selection[: args.n_queries]]
        noise = rng.normal(size=clean.shape).astype(np.float32)
        profile_clean = reference[selection[args.n_queries :]]
        profile_normals = normals[selection[args.n_queries :]]

        for sigma in args.sigmas:
            query = (clean + float(sigma) * noise).astype(np.float32)
            profile_query = np.concatenate(
                [profile_clean + float(sigma * offset) * profile_normals for offset in PROFILE_T]
            ).astype(np.float32)
            k = gaussian_mass_knn(
                len(support),
                float(metadata["normalized_surface_area"]),
                sigma,
                args.gaussian_mass,
                args.knn_safety,
            )
            empirical = empirical_score_numpy(
                support_tree,
                support,
                query,
                sigma,
                k,
                args.empirical_chunk,
                args.workers,
            )
            empirical_profile = empirical_score_numpy(
                support_tree,
                support,
                profile_query,
                sigma,
                k,
                args.empirical_chunk,
                args.workers,
            )
            row = metric_summary(
                dataset, "empirical_knn", sigma, empirical, query, clean, surface_tree, None
            )
            profile_result, means = profile_metrics(empirical_profile, profile_normals, sigma)
            row.update(profile_result)
            row.update({"knn_k": int(k), "support_n": int(len(support))})
            all_rows.append(row)
            append_profile_rows(profile_rows, dataset, "empirical_knn", sigma, None, means)

            for checkpoint_path in discover_checkpoints(checkpoint_root, dataset, sigma):
                model, checkpoint = load_checkpoint(checkpoint_path, device)
                model_type = str(checkpoint["model_type"])
                seed = checkpoint_seed(checkpoint_path)
                trained = trained_score_numpy(
                    model, model_type, query, sigma, device, args.eval_batch_size
                )
                trained_profile = trained_score_numpy(
                    model, model_type, profile_query, sigma, device, args.eval_batch_size
                )
                trained_row = metric_summary(
                    dataset,
                    f"trained_{model_type}",
                    sigma,
                    trained,
                    query,
                    clean,
                    surface_tree,
                    seed,
                )
                profile_result, means = profile_metrics(trained_profile, profile_normals, sigma)
                trained_row.update(profile_result)
                trained_row.update(
                    {
                        "checkpoint": str(checkpoint_path),
                        "steps": int(checkpoint["steps"]),
                        "train_examples_seen": int(checkpoint["train_examples_seen"]),
                        "validation_eps_mse": float(checkpoint["final_validation_eps_mse"]),
                    }
                )
                all_rows.append(trained_row)
                append_profile_rows(
                    profile_rows, dataset, f"trained_{model_type}", sigma, seed, means
                )
                print(
                    json.dumps(
                        {
                            "event": "evaluated",
                            "dataset": dataset,
                            "method": trained_row["method"],
                            "sigma": sigma,
                            "seed": seed,
                            "surface_mae_over_sigma": trained_row["surface_mae_over_sigma"],
                            "normal_slope_ratio": trained_row["normal_slope_ratio"],
                        }
                    ),
                    flush=True,
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        del support_tree, surface_tree, support, reference, normals, reference_payload
        gc.collect()

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "per_run.csv", all_rows)
    write_csv(output_dir / "normal_profiles.csv", profile_rows)
    atomic_json(output_dir / "per_run.json", all_rows)
    aggregate_outputs(all_rows, output_dir)


def aggregate_outputs(rows: list[dict], output_dir: Path) -> None:
    import pandas as pd

    frame = pd.DataFrame(rows)
    metrics = (
        "denoise_rmse_over_sigma",
        "denoise_p90_over_sigma",
        "pull_target_rmse_over_sigma",
        "surface_mae_over_sigma",
        "surface_p90_over_sigma",
        "angle_mean_deg",
        "angle_p90_deg",
        "normal_slope_ratio",
        "normal_intercept_over_sigma",
        "normal_residual_mae_over_sigma",
        "profile_tangential_rms_over_sigma",
        "validation_eps_mse",
    )
    summary: list[dict] = []
    for (dataset, method, sigma), group in frame.groupby(
        ["dataset", "method", "sigma"], sort=True
    ):
        result: dict[str, object] = {
            "dataset": dataset,
            "method": method,
            "sigma": float(sigma),
            "runs": int(len(group)),
        }
        for metric in metrics:
            values = pd.to_numeric(group.get(metric), errors="coerce").dropna()
            if len(values):
                result[f"{metric}_mean"] = float(values.mean())
                result[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(result)
    write_csv(output_dir / "summary.csv", summary)
    atomic_json(output_dir / "summary.json", summary)
    write_latex_table(summary, output_dir / "table_real_shapes.tex")
    write_figures(summary, output_dir)


def write_latex_table(rows: list[dict], path: Path) -> None:
    lines = [
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Shape & $\sigma$ & Score & Denoise RMSE$/\sigma$ $\downarrow$ & Surface residual$/\sigma$ $\downarrow$ & Slope ratio $\to 1$ \\",
        r"\midrule",
    ]
    present_datasets = [
        dataset for dataset in DATASET_LABELS if any(row["dataset"] == dataset for row in rows)
    ]
    for dataset_index, dataset in enumerate(present_datasets):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for sigma in sorted({float(row["sigma"]) for row in dataset_rows}, reverse=True):
            sigma_rows = [row for row in dataset_rows if float(row["sigma"]) == sigma]
            for row in sorted(sigma_rows, key=lambda item: str(item["method"])):
                lines.append(
                    f"{DATASET_LABELS[dataset]} & {sigma:g} & {METHOD_LABELS[str(row['method'])]} & "
                    f"{float(row['denoise_rmse_over_sigma_mean']):.4f} & "
                    f"{float(row['surface_mae_over_sigma_mean']):.4f} & "
                    f"{float(row['normal_slope_ratio_mean']):.3f} \\\\"
                )
        if dataset_index + 1 < len(present_datasets):
            lines.append(r"\addlinespace")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(rows: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, NullLocator

    def render(metric: str, ylabel: str, filename: str, reference_line: float | None = None) -> None:
        present_datasets = [
            dataset
            for dataset in DATASET_LABELS
            if any(row["dataset"] == dataset for row in rows)
        ]
        if not present_datasets:
            raise ValueError("cannot plot an empty result table")
        fig, axes = plt.subplots(
            1,
            len(present_datasets),
            figsize=(3.7 * len(present_datasets), 3.4),
            sharey=True,
        )
        axes = np.atleast_1d(axes)
        for axis, dataset in zip(axes, present_datasets):
            dataset_rows = [row for row in rows if row["dataset"] == dataset]
            for method in METHOD_LABELS:
                method_rows = sorted(
                    (row for row in dataset_rows if row["method"] == method),
                    key=lambda item: float(item["sigma"]),
                )
                if not method_rows:
                    continue
                x = np.asarray([float(row["sigma"]) for row in method_rows])
                y = np.asarray([float(row[f"{metric}_mean"]) for row in method_rows])
                std = np.asarray([float(row.get(f"{metric}_std", 0.0)) for row in method_rows])
                axis.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=1.8,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                )
                if np.any(std > 0):
                    axis.fill_between(x, y - std, y + std, color=METHOD_COLORS[method], alpha=0.14)
            ticks = sorted({float(row["sigma"]) for row in dataset_rows}, reverse=True)
            axis.set_xscale("log")
            axis.invert_xaxis()
            axis.set_xticks(ticks)
            axis.set_xticklabels([f"{value:g}" for value in ticks])
            axis.xaxis.set_minor_locator(NullLocator())
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_xlabel(r"Noise scale $\sigma$")
            axis.set_title(DATASET_LABELS[dataset])
            axis.grid(True, alpha=0.25)
            if reference_line is not None:
                axis.axhline(reference_line, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        axes[0].set_ylabel(ylabel)
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=240, bbox_inches="tight")
        plt.close(fig)

    render(
        "surface_mae_over_sigma",
        r"One-step surface residual $/\sigma$",
        "figure_real_shapes_surface.png",
    )
    render(
        "normal_slope_ratio",
        "Normal-slope ratio",
        "figure_real_shapes_slope.png",
        reference_line=1.0,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="create dense support and independent reference")
    prepare.add_argument("--dataset", choices=tuple(DATASET_LABELS), required=True)
    prepare.add_argument("--point-cloud", required=True)
    prepare.add_argument("--surface-kind", choices=("mesh", "oriented-points"), required=True)
    prepare.add_argument("--surface", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--n-support", type=int, default=40_000_000)
    prepare.add_argument("--n-reference", type=int, default=2_000_000)
    prepare.add_argument("--chunk-size", type=int, default=1_000_000)
    prepare.add_argument("--seed", type=int, default=20260828)
    prepare.set_defaults(function=command_prepare)

    train = subparsers.add_parser("train", help="train one fixed-sigma DSM model")
    train.add_argument("--dataset", choices=tuple(DATASET_LABELS), required=True)
    train.add_argument("--support", required=True)
    train.add_argument("--model", choices=("direct", "energy"), required=True)
    train.add_argument("--sigma", type=float, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--steps", type=int, default=30_000)
    train.add_argument("--batch-size", type=int, default=4096)
    train.add_argument("--val-size", type=int, default=50_000)
    train.add_argument("--eval-batch-size", type=int, default=4096)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--time-dim", type=int, default=64)
    train.add_argument("--num-blocks", type=int, default=4)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--warmup", type=int, default=2_000)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--ema-decay", type=float, default=0.999)
    train.add_argument("--log-every", type=int, default=1_000)
    train.add_argument("--device", default="cuda")
    train.add_argument("--output", required=True)
    train.set_defaults(function=command_train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate all real-shape checkpoints")
    evaluate.add_argument("--data-root", required=True)
    evaluate.add_argument("--checkpoint-root", required=True)
    evaluate.add_argument("--datasets", nargs="+", choices=tuple(DATASET_LABELS), default=list(DATASET_LABELS))
    evaluate.add_argument("--sigmas", type=float, nargs="+", default=(0.01, 0.005, 0.002))
    evaluate.add_argument("--n-queries", type=int, default=2_000)
    evaluate.add_argument("--profile-directions", type=int, default=256)
    evaluate.add_argument("--query-seed", type=int, default=20260829)
    evaluate.add_argument("--gaussian-mass", type=float, default=0.99)
    evaluate.add_argument("--knn-safety", type=float, default=1.10)
    evaluate.add_argument("--empirical-chunk", type=int, default=16)
    evaluate.add_argument("--workers", type=int, default=-1)
    evaluate.add_argument("--eval-batch-size", type=int, default=4096)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--output-dir", required=True)
    evaluate.set_defaults(function=command_evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
