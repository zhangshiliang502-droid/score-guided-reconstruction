from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from models import EnergyMLP, energy_gradient


SIGMA = 0.01
WARMUP_STEPS = 2_000
GRAD_CLIP = 1.0
EMA_DECAY = 0.999
TIME_DIM = 64
HIDDEN_DIM = 256
NUM_BLOCKS = 4


class TrainingData(NamedTuple):
    points: np.ndarray
    data_dim: int
    mean: torch.Tensor
    scale: float
    validation_points: torch.Tensor
    validation_noise: torch.Tensor
    normalization: dict[str, object]


@torch.no_grad()
def ema_update(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.mul_(float(decay)).add_(param, alpha=1.0 - float(decay))


def load_normalization(path: Path, expected_dim: int) -> tuple[np.ndarray, float, dict[str, object]]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    mean = np.asarray(meta["data_mean"], dtype=np.float32)
    scale = float(meta["data_scale_max_center_radius"])
    if mean.shape != (int(expected_dim),) or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid normalization metadata: {path}")
    ambient_dim = int(meta.get("ambient_dim", expected_dim))
    if ambient_dim != int(expected_dim):
        raise ValueError(f"metadata ambient_dim={ambient_dim}, dataset dimension={expected_dim}")
    return mean, scale, meta


def validate_dataset_metadata(
    points: np.ndarray,
    mean: np.ndarray,
    scale: float,
    meta: dict[str, object],
    *,
    batch_size: int = 65_536,
) -> None:
    if int(meta.get("num_points", -1)) != len(points):
        raise ValueError("normalization metadata point count does not match the dataset")
    for start in range(0, len(points), int(batch_size)):
        batch = np.asarray(points[start : start + int(batch_size)], dtype=np.float32)
        if not np.all(np.isfinite(batch)):
            raise ValueError("dataset contains non-finite coordinates")
    measured_mean = np.mean(points, axis=0, dtype=np.float32).astype(np.float32)
    measured_scale = 0.0
    for start in range(0, len(points), int(batch_size)):
        batch = np.asarray(points[start : start + int(batch_size)], dtype=np.float32)
        measured_scale = max(
            measured_scale,
            float(np.max(np.linalg.norm(batch - measured_mean[None, :], axis=1))),
        )
    if not np.allclose(measured_mean, mean.astype(np.float64), rtol=1e-6, atol=1e-7):
        raise ValueError("normalization metadata mean does not match the dataset")
    if not math.isclose(measured_scale, float(scale), rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("normalization metadata scale does not match the dataset")


def predicted_epsilon(model: torch.nn.Module, noisy: torch.Tensor, sigma: float) -> torch.Tensor:
    t = torch.full((len(noisy),), float(sigma), dtype=noisy.dtype, device=noisy.device)
    _, gradient, _ = energy_gradient(model, noisy, t, create_graph=False)
    return gradient


def evaluate(
    model: torch.nn.Module,
    val_points: torch.Tensor,
    val_noise: torch.Tensor,
    sigma: float,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, len(val_points), int(batch_size)):
        end = min(start + int(batch_size), len(val_points))
        x0 = val_points[start:end]
        eps = val_noise[start:end]
        noisy = (x0 + float(sigma) * eps).detach().requires_grad_(True)
        with torch.enable_grad():
            pred = predicted_epsilon(model, noisy, float(sigma))
        total += float(F.mse_loss(pred, eps, reduction="sum").item())
        count += int(eps.numel())
    return total / max(count, 1)


def payload(
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    args: argparse.Namespace,
    model_kwargs: dict[str, int],
    step: int,
    train_loss: float,
    eval_loss: float | None,
    best_eval: float,
    normalization: dict[str, object],
) -> dict[str, object]:
    return {
        "model": model.state_dict(),
        "model_ema": ema_model.state_dict(),
        "args": vars(args).copy(),
        "sigma": SIGMA,
        "energy_model_kwargs": model_kwargs,
        "model_type": "constrained_energy_score",
        "step": int(step),
        "train_eps_loss": float(train_loss),
        "eval_eps_loss": None if eval_loss is None else float(eval_loss),
        "best_ema_eval_loss": float(best_eval),
        "normalization": normalization,
        "energy_convention": "model outputs U=sigma*E; predicted_epsilon=grad_x_U; score=-grad_x_U/sigma=-grad_x_E",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the single formal conservative Energy score at sigma=0.01."
    )
    parser.add_argument("--point-cloud", required=True)
    parser.add_argument("--normalization-meta", required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--save-best", required=True)
    parser.add_argument("--train-size", type=int, default=900_000)
    parser.add_argument("--val-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--n-iters", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--eval-freq", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", default="cuda")
    return parser


def validate_positive_arguments(args: argparse.Namespace) -> None:
    """Reject invalid schedules before loading the potentially large dataset."""

    for name in (
        "train_size",
        "val_size",
        "batch_size",
        "eval_batch_size",
        "n_iters",
        "lr",
        "log_freq",
        "eval_freq",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def prepare_training_data(
    args: argparse.Namespace,
    device: torch.device,
) -> TrainingData:
    points = np.load(args.point_cloud, mmap_mode="r")
    total_needed = int(args.train_size) + int(args.val_size)
    if points.ndim != 2 or points.shape[1] <= 3:
        raise ValueError(f"expected [N,D] with D>3, got {points.shape}")
    data_dim = int(points.shape[1])
    if total_needed > len(points):
        raise ValueError(
            f"invalid dataset shape/size: {points.shape}, requested={total_needed}"
        )

    mean_np, scale, normalization = load_normalization(
        Path(args.normalization_meta),
        data_dim,
    )
    validate_dataset_metadata(points, mean_np, scale, normalization)
    mean = torch.as_tensor(mean_np, dtype=torch.float32, device=device)

    validation_slice = slice(int(args.train_size), total_needed)
    validation_raw = np.asarray(points[validation_slice], dtype=np.float32).copy()
    validation_points = (
        torch.from_numpy(validation_raw).to(device) - mean[None, :]
    ) / float(scale)
    validation_generator = torch.Generator(device=device).manual_seed(
        int(args.seed) + 1
    )
    validation_noise = torch.randn(
        validation_points.shape,
        dtype=validation_points.dtype,
        device=device,
        generator=validation_generator,
    )
    return TrainingData(
        points=points,
        data_dim=data_dim,
        mean=mean,
        scale=scale,
        validation_points=validation_points,
        validation_noise=validation_noise,
        normalization=normalization,
    )


def build_model(
    data_dim: int,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, int]]:
    model_kwargs = {
        "data_dim": data_dim,
        "time_dim": TIME_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_blocks": NUM_BLOCKS,
    }
    model = EnergyMLP(**model_kwargs).to(device)
    ema_model = copy.deepcopy(model).eval()
    return model, ema_model, model_kwargs


def checkpoint_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    latest = Path(args.save)
    best = Path(args.save_best)
    latest.parent.mkdir(parents=True, exist_ok=True)
    best.parent.mkdir(parents=True, exist_ok=True)
    return latest, best


def print_start_event(
    args: argparse.Namespace,
    data: TrainingData,
    device: torch.device,
    model_kwargs: dict[str, int],
) -> None:
    print(
        json.dumps(
            {
                "event": "start",
                "model_type": "constrained_energy_score",
                "point_cloud": str(args.point_cloud),
                "sigma": SIGMA,
                "score_convention": "model U=sigma*E; score=-grad_U/sigma",
                "model_kwargs": model_kwargs,
                "dataset_points": int(len(data.points)),
                "train_size": int(args.train_size),
                "val_size": int(args.val_size),
                "n_iters": int(args.n_iters),
                "batch_size": int(args.batch_size),
                "seed": int(args.seed),
                "device": str(device),
            }
        ),
        flush=True,
    )


def training_step(
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: TrainingData,
    args: argparse.Namespace,
    rng: np.random.Generator,
    step: int,
    device: torch.device,
) -> tuple[float, float, float]:
    model.train()
    indices = rng.integers(
        0,
        int(args.train_size),
        size=int(args.batch_size),
        dtype=np.int64,
    )
    batch = np.asarray(data.points[indices], dtype=np.float32)
    clean = (
        torch.from_numpy(batch).to(device) - data.mean[None, :]
    ) / float(data.scale)
    epsilon = torch.randn_like(clean)
    noisy = (clean + SIGMA * epsilon).detach().requires_grad_(True)
    time_values = torch.full(
        (len(noisy),),
        SIGMA,
        dtype=noisy.dtype,
        device=device,
    )

    optimizer.zero_grad(set_to_none=True)
    _, gradient, _ = energy_gradient(
        model,
        noisy,
        time_values,
        create_graph=True,
    )
    loss = F.mse_loss(gradient, epsilon)
    learning_rate = float(args.lr) * min(float(step) / WARMUP_STEPS, 1.0)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    loss.backward()
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        .detach()
        .cpu()
    )
    optimizer.step()
    ema_update(ema_model, model, EMA_DECAY)
    return float(loss.detach().cpu()), gradient_norm, learning_rate


def maybe_log_training(
    args: argparse.Namespace,
    step: int,
    loss: float,
    gradient_norm: float,
    learning_rate: float,
    started: float,
) -> None:
    if step != 1 and step % int(args.log_freq) != 0:
        return
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "event": "train",
                "step": step,
                "total_steps": int(args.n_iters),
                "train_eps_loss": loss,
                "grad_norm": gradient_norm,
                "lr": learning_rate,
                "steps_per_sec": step / max(elapsed, 1e-12),
            }
        ),
        flush=True,
    )


def evaluate_checkpoint(
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    args: argparse.Namespace,
    data: TrainingData,
    model_kwargs: dict[str, int],
    best_path: Path,
    step: int,
    train_loss: float,
    best_eval: float,
) -> tuple[float | None, float]:
    should_evaluate = step % int(args.eval_freq) == 0 or step == int(args.n_iters)
    if not should_evaluate:
        return None, best_eval

    eval_loss = evaluate(
        ema_model,
        data.validation_points,
        data.validation_noise,
        SIGMA,
        int(args.eval_batch_size),
    )
    if not math.isfinite(eval_loss):
        raise RuntimeError(f"non-finite validation loss at step {step}: {eval_loss}")
    improved = eval_loss < best_eval
    best_eval = min(best_eval, eval_loss)
    if improved:
        torch.save(
            payload(
                model,
                ema_model,
                args,
                model_kwargs,
                step,
                train_loss,
                eval_loss,
                best_eval,
                data.normalization,
            ),
            best_path,
        )
    print(
        json.dumps(
            {
                "event": "eval",
                "step": step,
                "eval_eps_loss": eval_loss,
                "best_eval_eps_loss": best_eval,
                "best_saved": improved,
            }
        ),
        flush=True,
    )
    return eval_loss, best_eval


def train(
    args: argparse.Namespace,
    data: TrainingData,
    device: torch.device,
) -> None:
    model, ema_model, model_kwargs = build_model(data.data_dim, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    latest_path, best_path = checkpoint_paths(args)
    print_start_event(args, data, device, model_kwargs)

    rng = np.random.default_rng(int(args.seed))
    best_eval = math.inf
    last_loss = math.inf
    started = time.perf_counter()
    for step in range(1, int(args.n_iters) + 1):
        last_loss, gradient_norm, learning_rate = training_step(
            model,
            ema_model,
            optimizer,
            data,
            args,
            rng,
            step,
            device,
        )
        maybe_log_training(
            args,
            step,
            last_loss,
            gradient_norm,
            learning_rate,
            started,
        )
        eval_loss, best_eval = evaluate_checkpoint(
            model,
            ema_model,
            args,
            data,
            model_kwargs,
            best_path,
            step,
            last_loss,
            best_eval,
        )
        if step == int(args.n_iters):
            state = payload(
                model,
                ema_model,
                args,
                model_kwargs,
                step,
                last_loss,
                eval_loss,
                best_eval,
                data.normalization,
            )
            torch.save(state, latest_path)
            print(
                json.dumps(
                    {"event": "checkpoint", "step": step, "latest": str(latest_path)}
                ),
                flush=True,
            )

    print(
        json.dumps(
            {
                "event": "done",
                "step": int(args.n_iters),
                "best_eval_eps_loss": best_eval,
                "latest": str(latest_path),
                "best": str(best_path),
            }
        ),
        flush=True,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_positive_arguments(args)
    torch.manual_seed(int(args.seed))
    device = resolve_device(args.device)
    data = prepare_training_data(args, device)
    train(args, data, device)


if __name__ == "__main__":
    main()
