#!/usr/bin/env python3
"""Train the paper's low-dimensional SGR model with its single fixed loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from models import SIRENImplicitSurface


MODEL_HIDDEN_DIM = 128
MODEL_HIDDEN_LAYERS = 4
MODEL_OMEGA0 = 20.0
LAMBDA_SCORE_PULL = 1000.0
GRAD_EPS = 1e-6
GRAD_CLIP = 10.0
LOSS_NAME = "score_pull_plus_global_cosine"
CACHE_CONTRACT_VERSION = 4
LOCAL_RADIUS_CENTER_MULTIPLIER = 2.5
LOCAL_RADIUS_STD_MULTIPLIER = 1.0
LOCAL_RNG_SEED_OFFSET = 17
INPUT_GRADIENT_BACKEND = "analytic_joint"
JIT_ANALYTIC_GRADIENT = True
FUSED_OPTIMIZER = True


class AnalyticGradientEvaluator(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.phi_and_grad_analytic(points, create_graph=True)


def load_training_points(path: Path) -> np.ndarray:
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
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{path} must contain an [N,3] point cloud, got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{path} contains non-finite coordinates")
    return points


def normalization(points: np.ndarray) -> tuple[np.ndarray, float]:
    # Match construct_score.py's float32 normalization exactly.
    mean = points.mean(axis=0).astype(np.float32)
    scale = float(np.max(np.linalg.norm(points - mean[None, :], axis=1)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid point-cloud normalization scale: {scale}")
    return mean, scale


def _cache_tensor(raw: dict[str, Any], key: str, device: torch.device) -> torch.Tensor:
    value = raw.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Score cache is missing tensor {key!r}")
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"Score-cache tensor {key!r} must have shape [N,3], got {tuple(value.shape)}")
    if value.shape[0] == 0:
        raise ValueError(f"Score-cache tensor {key!r} is empty")
    tensor = value.detach().to(device=device, dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"Score-cache tensor {key!r} contains non-finite values")
    return tensor


def load_score_cache(
    path: Path,
    device: torch.device,
    *,
    expected_mean: np.ndarray,
    expected_scale: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid score-cache payload: {path}")
    local_raw = payload.get("local_cache")
    global_raw = payload.get("global_cache")
    meta = payload.get("meta")
    if not isinstance(local_raw, dict) or not isinstance(global_raw, dict) or not isinstance(meta, dict):
        raise ValueError("Score cache must contain local_cache, global_cache, and meta dictionaries")

    local_cache = {
        "q_norm": _cache_tensor(local_raw, "q_norm", device),
        "target_pull_norm": _cache_tensor(local_raw, "target_pull_norm", device),
    }
    global_cache = {
        "z_norm": _cache_tensor(global_raw, "z_norm", device),
        "score_norm": _cache_tensor(global_raw, "score_norm", device),
    }
    if local_cache["q_norm"].shape != local_cache["target_pull_norm"].shape:
        raise ValueError("local q_norm and target_pull_norm shapes do not match")
    if global_cache["z_norm"].shape != global_cache["score_norm"].shape:
        raise ValueError("global z_norm and score_norm shapes do not match")

    for key in ("coordinate_space", "score_space", "sigma_space"):
        if str(meta.get(key)) != "normalized":
            raise ValueError(f"Score-cache {key} must be 'normalized', got {meta.get(key)!r}")
    sigma = float(meta.get("sigma", float("nan")))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Score cache has an invalid sigma")

    contract_version = int(meta.get("cache_contract_version", -1))
    if contract_version != CACHE_CONTRACT_VERSION:
        raise ValueError(
            f"Formal training requires score-cache contract v{CACHE_CONTRACT_VERSION}, "
            f"got v{contract_version}"
        )
    if str(meta.get("local_base_source")) != "point_cloud":
        raise ValueError("The score cache must anchor local queries on the input point cloud")
    if str(meta.get("global_bbox_source")) != "point_cloud":
        raise ValueError("The score cache must derive its global box from the input point cloud")
    # The legacy label still requires the explicit clamp_zero policy below.
    if str(meta.get("local_sampling_mode")) not in {"clamped_gaussian", "gaussian_abs"}:
        raise ValueError("The score cache must use a Gaussian radius clamped at zero")
    if str(meta.get("local_direction_mode")) != "normalized_isotropic_gaussian":
        raise ValueError("The score cache must use a normalized Gaussian direction")
    if float(meta.get("local_radius_center_multiplier", float("nan"))) != LOCAL_RADIUS_CENTER_MULTIPLIER:
        raise ValueError(
            f"The local radius must be centered at {LOCAL_RADIUS_CENTER_MULTIPLIER}*sigma"
        )
    if float(meta.get("local_radius_std_multiplier", float("nan"))) != LOCAL_RADIUS_STD_MULTIPLIER:
        raise ValueError(
            f"The local radius standard deviation must be {LOCAL_RADIUS_STD_MULTIPLIER}*sigma"
        )
    if str(meta.get("local_radius_negative_policy")) != "clamp_zero":
        raise ValueError("The score cache must clamp negative local radii to zero")
    if int(meta.get("local_rng_seed_offset", -1)) != LOCAL_RNG_SEED_OFFSET:
        raise ValueError(
            f"The score cache must use RNG seed offset {LOCAL_RNG_SEED_OFFSET}"
        )
    if str(meta.get("normalization_mean_accumulator")) != "float32":
        raise ValueError("The score cache must use a float32 normalization mean")
    meta["validated_cache_profile"] = "input_point_cloud_v4"

    cache_mean = np.asarray(meta.get("data_mean"), dtype=np.float64)
    cache_scale = float(meta.get("data_scale", float("nan")))
    if cache_mean.shape != expected_mean.shape or not np.all(np.isfinite(cache_mean)):
        raise ValueError("Score cache has invalid normalization mean")
    mean_error = float(np.linalg.norm(cache_mean - expected_mean.astype(np.float64)))
    scale_error = float(abs(cache_scale - expected_scale) / max(abs(expected_scale), 1e-12))
    if mean_error > 1e-5 * max(abs(expected_scale), 1.0) or scale_error > 1e-5:
        raise ValueError(
            "Score-cache normalization does not match the training cloud: "
            f"mean_error={mean_error:.6g}, scale_relative_error={scale_error:.6g}"
        )
    return local_cache, global_cache, meta


def sample_cache(cache: dict[str, torch.Tensor], size: int) -> dict[str, torch.Tensor]:
    count = next(iter(cache.values())).shape[0]
    indices = torch.randint(0, int(count), (int(size),), device=next(iter(cache.values())).device)
    return {key: value[indices] for key, value in cache.items()}


def fixed_sgr_loss(
    model: torch.nn.Module,
    gradient_evaluator: torch.nn.Module | None,
    local_batch: dict[str, torch.Tensor],
    global_batch: dict[str, torch.Tensor],
    *,
    lambda_global: float,
    create_graph: bool,
) -> dict[str, torch.Tensor]:
    """Compute the only formal low-dimensional loss.

    L = 1000 E||Phi grad(Phi)/||grad(Phi)|| + sigma^2 score||^2
        + lambda_global E[1-cos(-grad(Phi^2), score)].

    ``target_pull_norm`` stores ``-sigma^2 score``; therefore the local
    residual is ``predicted_pull - target_pull_norm``.
    """

    q = local_batch["q_norm"]
    z = global_batch["z_norm"]
    points = torch.cat((q, z), dim=0)
    context = torch.enable_grad() if create_graph else torch.no_grad()
    with context:
        if gradient_evaluator is None:
            phi, gradient = model.phi_and_grad_analytic(points, create_graph=create_graph)
        else:
            phi, gradient = gradient_evaluator(points)
    local_count = q.shape[0]
    phi_q, grad_q = phi[:local_count], gradient[:local_count]
    phi_z, grad_z = phi[local_count:], gradient[local_count:]

    grad_q_norm = grad_q.norm(dim=1, keepdim=True).clamp_min(GRAD_EPS)
    predicted_pull = phi_q[:, None] * grad_q / grad_q_norm
    score_pull = (predicted_pull - local_batch["target_pull_norm"]).square().sum(dim=1).mean()

    model_direction = -2.0 * phi_z[:, None] * grad_z
    cosine = F.cosine_similarity(
        model_direction,
        global_batch["score_norm"],
        dim=1,
        eps=GRAD_EPS,
    ).clamp(-1.0, 1.0)
    global_direction = (1.0 - cosine).mean()
    total = LAMBDA_SCORE_PULL * score_pull + float(lambda_global) * global_direction
    return {
        "total": total,
        "score_pull": score_pull,
        "global_direction": global_direction,
        "global_cosine": cosine.mean(),
    }


def evaluate(
    model: torch.nn.Module,
    gradient_evaluator: torch.nn.Module | None,
    local_cache: dict[str, torch.Tensor],
    global_cache: dict[str, torch.Tensor],
    *,
    batch_size: int,
    global_batch_size: int,
    eval_batches: int,
    lambda_global: float,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    for _ in range(int(eval_batches)):
        losses = fixed_sgr_loss(
            model,
            gradient_evaluator,
            sample_cache(local_cache, batch_size),
            sample_cache(global_cache, global_batch_size),
            lambda_global=lambda_global,
            create_graph=False,
        )
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
    model.train()
    return {key: value / int(eval_batches) for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    step: int,
    metrics: dict[str, float],
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "step": int(step),
            "eval": metrics,
            "best_metric": float(metrics["total"]),
            "meta": meta,
        },
        path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the fixed low-dimensional SGR loss.")
    parser.add_argument("--point-cloud", required=True)
    parser.add_argument("--score-cache-pt", required=True)
    parser.add_argument("--out", required=True, help="Best checkpoint path.")
    parser.add_argument("--latest", default="")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--global-batch-size", type=int, default=4096)
    parser.add_argument("--eval-batches", type=int, default=12)
    parser.add_argument("--n-iters", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-global", type=float, required=True)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "batch_size",
        "global_batch_size",
        "eval_batches",
        "n_iters",
        "lr",
        "lambda_global",
        "log_freq",
        "eval_freq",
    )
    for name in positive:
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def seed_everything(value: int) -> int:
    """Seed NumPy and PyTorch with the normalized 32-bit experiment seed."""
    seed = int(value) % (2**32 - 1) or 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def build_gradient_evaluator(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module | None:
    if not JIT_ANALYTIC_GRADIENT or device.type != "cuda":
        return None
    example_points = torch.empty(
        (int(args.batch_size) + int(args.global_batch_size), 3),
        dtype=torch.float32,
        device=device,
    )
    return torch.jit.trace(
        AnalyticGradientEvaluator(model),
        (example_points,),
        check_trace=False,
    )


def build_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.optim.Optimizer:
    """Use the recorded fused CUDA optimizer and a compatible CPU fallback."""
    use_fused = bool(FUSED_OPTIMIZER and device.type == "cuda")
    try:
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=0.0,
            fused=use_fused,
        )
    except (TypeError, RuntimeError):
        if use_fused:
            raise
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=0.0,
        )


def checkpoint_metadata(
    args: argparse.Namespace,
    *,
    points: np.ndarray,
    data_mean: np.ndarray,
    data_scale: float,
    cache_meta: dict[str, Any],
    seed: int,
    gradient_evaluator: torch.nn.Module | None,
    device: torch.device,
) -> dict[str, Any]:
    """Freeze every data, model, cache, and loss convention in the checkpoint."""
    score_type = str(
        cache_meta.get("score_cache_type", "input_point_cloud_knn_empirical")
    )
    return {
        "point_cloud": str(args.point_cloud),
        "num_points": int(points.shape[0]),
        "normalization": "mean_and_max_center_radius",
        "normalization_mean_accumulator": "float32",
        "data_mean": data_mean.astype(float).tolist(),
        "data_scale": float(data_scale),
        "model_type": "siren",
        "data_dim": 3,
        "hidden_dim": MODEL_HIDDEN_DIM,
        "hidden_layers": MODEL_HIDDEN_LAYERS,
        "omega0": MODEL_OMEGA0,
        "input_gradient_backend": INPUT_GRADIENT_BACKEND,
        "jit_analytic_gradient": bool(gradient_evaluator is not None),
        "fused_optimizer": bool(FUSED_OPTIMIZER and device.type == "cuda"),
        "score_source": score_type,
        "score_cache_pt": str(args.score_cache_pt),
        "score_cache_contract_version": int(cache_meta["cache_contract_version"]),
        "score_cache_profile": str(cache_meta["validated_cache_profile"]),
        "score_cache_type": score_type,
        "score_cache_knn_k": int(cache_meta["knn_k"]),
        "score_cache_local_sampling_mode": str(cache_meta["local_sampling_mode"]),
        "cache_sigma": float(cache_meta["sigma"]),
        "loss": {
            "name": LOSS_NAME,
            "lambda_score_pull": LAMBDA_SCORE_PULL,
            "lambda_global": float(args.lambda_global),
            "local_residual_normalization": "none",
            "global_direction": "cosine",
        },
        "seed": seed,
    }


def training_step(
    model: torch.nn.Module,
    gradient_evaluator: torch.nn.Module | None,
    local_cache: dict[str, torch.Tensor],
    global_cache: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = fixed_sgr_loss(
        model,
        gradient_evaluator,
        sample_cache(local_cache, int(args.batch_size)),
        sample_cache(global_cache, int(args.global_batch_size)),
        lambda_global=float(args.lambda_global),
        create_graph=True,
    )
    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    scheduler.step()
    return losses


def fit(
    model: torch.nn.Module,
    gradient_evaluator: torch.nn.Module | None,
    local_cache: dict[str, torch.Tensor],
    global_cache: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    meta: dict[str, Any],
    args: argparse.Namespace,
) -> float:
    best_total = float("inf")
    best_metrics: dict[str, float] | None = None
    for step in range(1, int(args.n_iters) + 1):
        losses = training_step(
            model,
            gradient_evaluator,
            local_cache,
            global_cache,
            optimizer,
            scheduler,
            args,
        )
        if step == 1 or step % int(args.log_freq) == 0:
            print(
                json.dumps(
                    {
                        "event": "train",
                        "step": step,
                        "lr": float(scheduler.get_last_lr()[0]),
                        "losses": {
                            key: float(value.detach().cpu())
                            for key, value in losses.items()
                        },
                    }
                ),
                flush=True,
            )

        should_evaluate = (
            step == 1
            or step % int(args.eval_freq) == 0
            or step == int(args.n_iters)
        )
        if not should_evaluate:
            continue
        metrics = evaluate(
            model,
            gradient_evaluator,
            local_cache,
            global_cache,
            batch_size=int(args.batch_size),
            global_batch_size=int(args.global_batch_size),
            eval_batches=int(args.eval_batches),
            lambda_global=float(args.lambda_global),
        )
        print(json.dumps({"event": "eval", "step": step, "metrics": metrics}), flush=True)
        if metrics["total"] < best_total:
            best_total = metrics["total"]
            best_metrics = metrics
            save_checkpoint(Path(args.out), model, step=step, metrics=metrics, meta=meta)
            print(json.dumps({"event": "best", "step": step, "total": best_total}), flush=True)
        if args.latest:
            save_checkpoint(Path(args.latest), model, step=step, metrics=metrics, meta=meta)

    if best_metrics is None:
        raise RuntimeError("Training finished without an evaluation checkpoint")
    return best_total


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    seed = seed_everything(int(args.seed))
    points = load_training_points(Path(args.point_cloud))
    data_mean, data_scale = normalization(points)
    local_cache, global_cache, cache_meta = load_score_cache(
        Path(args.score_cache_pt),
        device,
        expected_mean=data_mean,
        expected_scale=data_scale,
    )

    model = SIRENImplicitSurface(
        hidden_dim=MODEL_HIDDEN_DIM,
        hidden_layers=MODEL_HIDDEN_LAYERS,
        omega0=MODEL_OMEGA0,
        input_dim=3,
    ).to(device)
    gradient_evaluator = build_gradient_evaluator(model, args, device)
    optimizer = build_optimizer(model, args, device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(args.n_iters),
        eta_min=float(args.lr) * 0.03,
    )

    meta = checkpoint_metadata(
        args,
        points=points,
        data_mean=data_mean,
        data_scale=data_scale,
        cache_meta=cache_meta,
        seed=seed,
        gradient_evaluator=gradient_evaluator,
        device=device,
    )
    print(
        json.dumps(
            {
                "event": "start",
                "loss": meta["loss"],
                "local_cache_size": int(local_cache["q_norm"].shape[0]),
                "global_cache_size": int(global_cache["z_norm"].shape[0]),
                "device": str(device),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    best_total = fit(
        model,
        gradient_evaluator,
        local_cache,
        global_cache,
        optimizer,
        scheduler,
        meta,
        args,
    )
    print(json.dumps({"event": "done", "best_total": best_total, "best": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
