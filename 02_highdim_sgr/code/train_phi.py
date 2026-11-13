#!/usr/bin/env python3
"""Train the formal high-dimensional Energy-SGR model with one fixed loss."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from models import EnergyMLP, SIRENImplicitSurface


SIGMA = 0.01
DEFAULT_LOCAL_SHELL_CENTER_SIGMA = 2.5
LOCAL_SHELL_STD_SIGMA = 1.0
BBOX_PAD_RATIO = 0.30
MODEL_HIDDEN_DIM = 128
MODEL_HIDDEN_LAYERS = 4
MODEL_OMEGA0 = 20.0
LAMBDA_SCORE_PULL = 1000.0
GRAD_EPS = 1e-6
GRAD_CLIP = 10.0
LOSS_NAME = "score_pull_plus_global_cosine"


class TrainingData(NamedTuple):
    points: np.ndarray
    mean: np.ndarray
    scale: float
    train_points: torch.Tensor
    validation_points: torch.Tensor
    background_low: torch.Tensor
    background_high: torch.Tensor


def load_points(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    points = np.load(path, mmap_mode="r")
    if points.ndim != 2 or points.shape[1] <= 3:
        raise ValueError(f"Expected an [N,D] point array with D>3, got {points.shape}")
    data = np.asarray(points, dtype=np.float32)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path} contains non-finite coordinates")
    return data


def normalization(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    mean = points.mean(axis=0).astype(np.float32)
    normalized = points - mean[None, :]
    scale = float(np.max(np.linalg.norm(normalized, axis=1)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid normalization scale: {scale}")
    return mean, scale, (normalized / scale).astype(np.float32)


def load_energy_model(
    checkpoint_path: Path,
    device: torch.device,
    *,
    data_dim: int,
    data_mean: np.ndarray,
    data_scale: float,
) -> tuple[EnergyMLP, dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid Energy checkpoint: {checkpoint_path}")
    checkpoint_sigma = float(checkpoint.get("sigma", float("nan")))
    if not math.isclose(checkpoint_sigma, SIGMA, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError(f"Formal Energy checkpoint must use sigma={SIGMA}, got {checkpoint_sigma}")
    model_kwargs = checkpoint.get("energy_model_kwargs")
    if not isinstance(model_kwargs, dict) or int(model_kwargs.get("data_dim", -1)) != int(data_dim):
        raise ValueError("Energy checkpoint dimension does not match the hypersurface data")
    expected_model = {"time_dim": 64, "hidden_dim": 256, "num_blocks": 4}
    if any(int(model_kwargs.get(key, -1)) != value for key, value in expected_model.items()):
        raise ValueError(f"Energy checkpoint must use the formal architecture {expected_model}")
    if str(checkpoint.get("model_type")) != "constrained_energy_score":
        raise ValueError("Energy checkpoint has the wrong model type")
    if int(checkpoint.get("step", -1)) <= 0:
        raise ValueError("Energy checkpoint has no completed training step")
    normalization_meta = checkpoint.get("normalization")
    if not isinstance(normalization_meta, dict):
        raise ValueError("Energy checkpoint is missing normalization metadata")
    energy_mean = np.asarray(normalization_meta.get("data_mean"), dtype=np.float64)
    energy_scale = float(normalization_meta.get("data_scale_max_center_radius", float("nan")))
    if energy_mean.shape != data_mean.shape or not np.allclose(
        energy_mean,
        data_mean.astype(np.float64),
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError("Energy and SGR normalization means do not match")
    if not math.isclose(energy_scale, data_scale, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("Energy and SGR normalization scales do not match")
    state = checkpoint.get("model_ema")
    if not isinstance(state, dict):
        raise ValueError("Formal Energy checkpoint must contain the EMA model state")
    model = EnergyMLP(**model_kwargs).to(device)
    model.load_state_dict(state)
    model.eval()
    model.requires_grad_(False)
    return model, {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "sigma": SIGMA,
        "model_kwargs": model_kwargs,
        "normalization_contract": "validated",
        "score_convention": "score=-grad(U)/sigma",
    }


def energy_score(model: EnergyMLP, points: torch.Tensor) -> torch.Tensor:
    query = points.detach().requires_grad_(True)
    time = torch.full((query.shape[0],), SIGMA, dtype=query.dtype, device=query.device)
    energy = model(query, time)
    gradient = torch.autograd.grad(energy.sum(), query, create_graph=False)[0]
    return (-gradient / SIGMA).detach()


def sample_background(count: int, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return low[None, :] + (high - low)[None, :] * torch.rand(
        (int(count), low.numel()),
        dtype=low.dtype,
        device=low.device,
    )


def fixed_energy_sgr_loss(
    model: SIRENImplicitSurface,
    energy_model: EnergyMLP,
    surface_points: torch.Tensor,
    *,
    global_batch_size: int,
    bg_low: torch.Tensor,
    bg_high: torch.Tensor,
    local_shell_center_sigma: float,
    lambda_dir: float,
    create_graph: bool,
) -> dict[str, torch.Tensor]:
    """Compute the only formal Energy-SGR loss.

    Local samples use a random unit direction and a Gaussian radius
    clamped at zero with a configurable centre. The global term is
    mean(1-cos(-grad(Phi^2), score)).
    """

    direction = F.normalize(torch.randn_like(surface_points), dim=1, eps=1e-12)
    radius_sigma = (
        float(local_shell_center_sigma)
        + LOCAL_SHELL_STD_SIGMA
        * torch.randn(
            (surface_points.shape[0], 1),
            dtype=surface_points.dtype,
            device=surface_points.device,
        )
    ).clamp_min(0.0)
    q = surface_points + SIGMA * radius_sigma * direction
    z = sample_background(global_batch_size, bg_low, bg_high)
    with torch.enable_grad():
        local_score = energy_score(energy_model, q)
        global_score = energy_score(energy_model, z)
    target_pull = -(SIGMA**2) * local_score

    points = torch.cat((q, z), dim=0)
    phi, gradient = model.phi_and_grad(points, create_graph=create_graph)
    local_count = q.shape[0]
    phi_q, grad_q = phi[:local_count], gradient[:local_count]
    phi_z, grad_z = phi[local_count:], gradient[local_count:]

    predicted_pull = phi_q[:, None] * grad_q / grad_q.norm(dim=1, keepdim=True).clamp_min(GRAD_EPS)
    score_pull = (predicted_pull - target_pull).square().sum(dim=1).mean()

    model_direction = -2.0 * phi_z[:, None] * grad_z
    cosine = F.cosine_similarity(model_direction, global_score, dim=1, eps=GRAD_EPS).clamp(-1.0, 1.0)
    global_direction = (1.0 - cosine).mean()
    total = LAMBDA_SCORE_PULL * score_pull + float(lambda_dir) * global_direction
    return {
        "total": total,
        "score_pull": score_pull,
        "global_direction": global_direction,
        "global_cosine": cosine.mean(),
    }


def evaluate(
    model: SIRENImplicitSurface,
    energy_model: EnergyMLP,
    validation_points: torch.Tensor,
    *,
    batch_size: int,
    global_batch_size: int,
    eval_batches: int,
    bg_low: torch.Tensor,
    bg_high: torch.Tensor,
    local_shell_center_sigma: float,
    lambda_dir: float,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    for _ in range(int(eval_batches)):
        indices = torch.randint(
            0,
            validation_points.shape[0],
            (int(batch_size),),
            device=validation_points.device,
        )
        losses = fixed_energy_sgr_loss(
            model,
            energy_model,
            validation_points[indices],
            global_batch_size=global_batch_size,
            bg_low=bg_low,
            bg_high=bg_high,
            local_shell_center_sigma=local_shell_center_sigma,
            lambda_dir=lambda_dir,
            create_graph=False,
        )
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
    model.train()
    return {key: value / int(eval_batches) for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: SIRENImplicitSurface,
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
    parser = argparse.ArgumentParser(description="Train the single formal high-dimensional Energy-SGR loss.")
    parser.add_argument("--point-cloud", required=True)
    parser.add_argument("--energy-checkpoint", required=True)
    parser.add_argument("--out", required=True, help="Best checkpoint path.")
    parser.add_argument("--latest", default="")
    parser.add_argument("--train-size", type=int, default=220000)
    parser.add_argument("--val-size", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--global-batch-size", type=int, default=4096)
    parser.add_argument("--eval-batches", type=int, default=12)
    parser.add_argument("--n-iters", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--local-shell-center-sigma",
        type=float,
        default=DEFAULT_LOCAL_SHELL_CENTER_SIGMA,
    )
    parser.add_argument(
        "--lambda-dir",
        type=float,
        default=20.0,
    )
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def validate_args(args: argparse.Namespace, point_count: int) -> None:
    positive = (
        "train_size",
        "val_size",
        "batch_size",
        "global_batch_size",
        "eval_batches",
        "n_iters",
        "lr",
        "local_shell_center_sigma",
        "lambda_dir",
        "log_freq",
        "eval_freq",
    )
    for name in positive:
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.train_size) + int(args.val_size) > int(point_count):
        raise ValueError("train_size + val_size exceeds the input point count")


def seed_everything(value: int) -> int:
    """Seed NumPy and PyTorch with the normalized 32-bit experiment seed."""
    seed = int(value) % (2**32 - 1) or 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def prepare_training_data(
    points: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> TrainingData:
    mean, scale, normalized = normalization(points)
    permutation = np.random.default_rng(seed).permutation(len(normalized))
    train_end = int(args.train_size)
    validation_end = train_end + int(args.val_size)
    train_points = torch.from_numpy(normalized[permutation[:train_end]]).to(device)
    validation_points = torch.from_numpy(
        normalized[permutation[train_end:validation_end]]
    ).to(device)

    low = normalized.min(axis=0)
    high = normalized.max(axis=0)
    padding = BBOX_PAD_RATIO * (high - low)
    background_low = torch.as_tensor(
        low - padding,
        dtype=torch.float32,
        device=device,
    )
    background_high = torch.as_tensor(
        high + padding,
        dtype=torch.float32,
        device=device,
    )
    return TrainingData(
        points=points,
        mean=mean,
        scale=scale,
        train_points=train_points,
        validation_points=validation_points,
        background_low=background_low,
        background_high=background_high,
    )


def checkpoint_metadata(
    args: argparse.Namespace,
    data: TrainingData,
    energy_meta: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    dimension = int(data.train_points.shape[1])
    return {
        "point_cloud": str(args.point_cloud),
        "num_points": int(len(data.points)),
        "train_size": int(args.train_size),
        "val_size": int(args.val_size),
        "normalization": "mean_and_max_center_radius",
        "data_mean": data.mean.astype(float).tolist(),
        "data_scale": float(data.scale),
        "model_type": "siren",
        "data_dim": dimension,
        "hidden_dim": MODEL_HIDDEN_DIM,
        "hidden_layers": MODEL_HIDDEN_LAYERS,
        "omega0": MODEL_OMEGA0,
        "bg_low": data.background_low.detach().cpu().numpy().astype(float).tolist(),
        "bg_high": data.background_high.detach().cpu().numpy().astype(float).tolist(),
        "score_source": "fixed_energy_ema_gradient",
        "learned_energy_score": energy_meta,
        "loss": {
            "name": LOSS_NAME,
            "sigma": SIGMA,
            "local_query_mode": "radial_shell",
            "local_shell_center_sigma": float(args.local_shell_center_sigma),
            "local_shell_std_sigma": LOCAL_SHELL_STD_SIGMA,
            "lambda_score_pull": LAMBDA_SCORE_PULL,
            "lambda_dir": float(args.lambda_dir),
            "local_residual_normalization": "none",
            "global_direction": "cosine",
        },
        "seed": seed,
    }


def training_step(
    model: SIRENImplicitSurface,
    energy_model: EnergyMLP,
    data: TrainingData,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    indices = torch.randint(
        0,
        data.train_points.shape[0],
        (int(args.batch_size),),
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    losses = fixed_energy_sgr_loss(
        model,
        energy_model,
        data.train_points[indices],
        global_batch_size=int(args.global_batch_size),
        bg_low=data.background_low,
        bg_high=data.background_high,
        local_shell_center_sigma=float(args.local_shell_center_sigma),
        lambda_dir=float(args.lambda_dir),
        create_graph=True,
    )
    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    scheduler.step()
    return losses


def fit(
    model: SIRENImplicitSurface,
    energy_model: EnergyMLP,
    data: TrainingData,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    meta: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    best_total = float("inf")
    best_metrics: dict[str, float] | None = None
    for step in range(1, int(args.n_iters) + 1):
        losses = training_step(
            model,
            energy_model,
            data,
            optimizer,
            scheduler,
            args,
            device,
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
            energy_model,
            data.validation_points,
            batch_size=int(args.batch_size),
            global_batch_size=int(args.global_batch_size),
            eval_batches=int(args.eval_batches),
            bg_low=data.background_low,
            bg_high=data.background_high,
            local_shell_center_sigma=float(args.local_shell_center_sigma),
            lambda_dir=float(args.lambda_dir),
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
    points = load_points(Path(args.point_cloud))
    validate_args(args, len(points))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    seed = seed_everything(int(args.seed))
    data = prepare_training_data(points, args, device, seed)

    energy_model, energy_meta = load_energy_model(
        Path(args.energy_checkpoint),
        device,
        data_dim=data.train_points.shape[1],
        data_mean=data.mean,
        data_scale=data.scale,
    )
    model = SIRENImplicitSurface(
        hidden_dim=MODEL_HIDDEN_DIM,
        hidden_layers=MODEL_HIDDEN_LAYERS,
        omega0=MODEL_OMEGA0,
        input_dim=data.train_points.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(args.n_iters),
        eta_min=float(args.lr) * 0.03,
    )

    meta = checkpoint_metadata(args, data, energy_meta, seed)
    print(json.dumps({"event": "start", "meta": meta}, ensure_ascii=False), flush=True)

    best_total = fit(
        model,
        energy_model,
        data,
        optimizer,
        scheduler,
        meta,
        args,
        device,
    )
    print(json.dumps({"event": "done", "best_total": best_total, "best": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
