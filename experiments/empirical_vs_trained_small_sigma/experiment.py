"""Controlled small-sigma comparison of empirical and DSM-trained scores.

The data distribution is uniform surface measure on the unit sphere in R^3.
Its Gaussian-smoothed score is analytic, which gives an unambiguous population
target for both the trained and finite-point empirical estimators.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SIGMAS_DEFAULT = (0.02, 0.01, 0.005, 0.002)
PROFILE_T = np.asarray((-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0))


def sigma_tag(sigma: float) -> str:
    return f"{float(sigma):g}".replace(".", "p")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def sample_unit_sphere_torch(
    count: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    points = torch.randn((int(count), 3), device=device, generator=generator)
    return points / torch.linalg.norm(points, dim=1, keepdim=True).clamp_min(1e-12)


def exact_sphere_score_numpy(query: np.ndarray, sigma: float) -> np.ndarray:
    """Population score of uniform S^2 convolved with N(0, sigma^2 I)."""
    query64 = np.asarray(query, dtype=np.float64)
    radius = np.linalg.norm(query64, axis=1).clip(1e-12)
    argument = radius / float(sigma * sigma)
    coth = np.ones_like(argument)
    stable = argument < 20.0
    coth[stable] = 1.0 / np.tanh(argument[stable])
    radial = -radius / float(sigma * sigma) + coth / float(sigma * sigma) - 1.0 / radius
    return (radial[:, None] * query64 / radius[:, None]).astype(np.float32)


class SinusoidalSigmaEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = int(dimension)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                half,
                dtype=sigma.dtype,
                device=sigma.device,
            )
        )
        angles = 2.0 * math.pi * sigma.reshape(-1, 1) * frequencies.reshape(1, -1)
        embedded = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
        if self.dimension % 2:
            embedded = F.pad(embedded, (0, 1))
        return embedded


class ResidualBackbone(nn.Module):
    def __init__(self, hidden_dim: int, time_dim: int, num_blocks: int) -> None:
        super().__init__()
        self.time = nn.Sequential(
            SinusoidalSigmaEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.input = nn.Linear(3, hidden_dim)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_blocks)
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        time_features = self.time(sigma)
        hidden = F.silu(self.input(x))
        for block in self.blocks:
            hidden = F.silu(hidden + block(torch.cat((hidden, time_features), dim=1)))
        return hidden


class DirectDSM(nn.Module):
    def __init__(self, hidden_dim: int = 256, time_dim: int = 64, num_blocks: int = 4) -> None:
        super().__init__()
        self.backbone = ResidualBackbone(hidden_dim, time_dim, num_blocks)
        self.output = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return self.output(self.backbone(x, sigma))


class EnergyDSM(nn.Module):
    def __init__(self, hidden_dim: int = 256, time_dim: int = 64, num_blocks: int = 4) -> None:
        super().__init__()
        self.backbone = ResidualBackbone(hidden_dim, time_dim, num_blocks)
        self.output = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return self.output(self.backbone(x, sigma)).squeeze(1)


def build_model(model_type: str, model_kwargs: dict[str, int]) -> nn.Module:
    if model_type == "direct":
        return DirectDSM(**model_kwargs)
    if model_type == "energy":
        return EnergyDSM(**model_kwargs)
    raise ValueError(f"unknown model type: {model_type}")


def predict_epsilon(
    model: nn.Module,
    noisy: torch.Tensor,
    sigma: float,
    model_type: str,
    *,
    create_graph: bool,
) -> torch.Tensor:
    sigma_tensor = torch.full(
        (len(noisy),), float(sigma), dtype=noisy.dtype, device=noisy.device
    )
    if model_type == "direct":
        return model(noisy, sigma_tensor)
    query = noisy if noisy.requires_grad else noisy.detach().requires_grad_(True)
    potential = model(query, sigma_tensor)
    return torch.autograd.grad(potential.sum(), query, create_graph=create_graph)[0]


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
        ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)


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
    generator = torch.Generator(device=device).manual_seed(args.seed)
    validation_generator = torch.Generator(device=device).manual_seed(args.seed + 10_000)

    model_kwargs = {
        "hidden_dim": args.hidden_dim,
        "time_dim": args.time_dim,
        "num_blocks": args.num_blocks,
    }
    model = build_model(args.model, model_kwargs).to(device)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    validation_clean = sample_unit_sphere_torch(
        args.val_size, device=device, generator=validation_generator
    )
    validation_noise = torch.randn(
        validation_clean.shape, device=device, generator=validation_generator
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    last_loss = math.nan
    for step in range(1, args.steps + 1):
        model.train()
        clean = sample_unit_sphere_torch(
            args.batch_size, device=device, generator=generator
        )
        noise = torch.randn(clean.shape, device=device, generator=generator)
        noisy = clean + args.sigma * noise
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
        "model_type": args.model,
        "sigma": args.sigma,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "train_examples_seen": args.steps * args.batch_size,
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
            {
                "event": "done",
                "output": str(output),
                "validation_eps_mse": final_validation,
            }
        ),
        flush=True,
    )


def command_make_support(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = np.load(output, mmap_mode="r")
        if existing.shape == (args.n_points, 3) and existing.dtype == np.float32:
            print(json.dumps({"event": "reuse_support", "path": str(output)}))
            return
        raise ValueError(f"existing support has unexpected shape/dtype: {existing.shape}/{existing.dtype}")

    temporary = output.with_suffix(output.suffix + ".tmp")
    support = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(args.n_points, 3)
    )
    rng = np.random.default_rng(args.seed)
    for start in range(0, args.n_points, args.chunk_size):
        end = min(start + args.chunk_size, args.n_points)
        points = rng.normal(size=(end - start, 3)).astype(np.float32)
        points /= np.linalg.norm(points, axis=1, keepdims=True).clip(1e-12)
        support[start:end] = points
        if start == 0 or end == args.n_points or end % (10 * args.chunk_size) == 0:
            print(json.dumps({"event": "support", "written": end, "total": args.n_points}), flush=True)
    support.flush()
    del support
    temporary.replace(output)
    atomic_json(
        output.with_suffix(".json"),
        {
            "distribution": "uniform_unit_sphere_S2",
            "n_points": args.n_points,
            "seed": args.seed,
            "dtype": "float32",
        },
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
        noisy = torch.as_tensor(
            query[start : start + batch_size], dtype=torch.float32, device=device
        )
        with torch.enable_grad():
            epsilon = predict_epsilon(
                model, noisy, sigma, model_type, create_graph=False
            )
        chunks.append((-epsilon / float(sigma)).detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def gaussian_mass_knn(n_points: int, sigma: float, mass: float, safety: float) -> int:
    if not 0.0 < mass < 1.0:
        raise ValueError("mass must lie in (0,1)")
    tangent_radius = float(sigma) * math.sqrt(-2.0 * math.log(1.0 - mass))
    expected = float(n_points) * tangent_radius * tangent_radius / 4.0
    conservative = expected + 6.0 * math.sqrt(max(expected, 1.0)) + 32.0
    return min(int(n_points), max(64, int(math.ceil(safety * conservative))))


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
        print(
            json.dumps(
                {"event": "empirical_query", "sigma": sigma, "done": end, "total": len(query), "k": k}
            ),
            flush=True,
        )
    return output


def metric_summary(
    method: str,
    sigma: float,
    estimated_score: np.ndarray,
    exact_score: np.ndarray,
    query: np.ndarray,
    seed: int | None,
) -> dict[str, float | int | str | None]:
    estimated_pull = float(sigma * sigma) * estimated_score.astype(np.float64)
    exact_pull = float(sigma * sigma) * exact_score.astype(np.float64)
    error = estimated_pull - exact_pull
    error_norm = np.linalg.norm(error, axis=1)
    truth_norm = np.linalg.norm(exact_pull, axis=1)
    estimate_norm = np.linalg.norm(estimated_pull, axis=1)
    aggregate_relative = math.sqrt(
        float(np.sum(error * error)) / max(float(np.sum(exact_pull * exact_pull)), 1e-30)
    )
    valid = truth_norm >= 0.25 * float(sigma)
    cosine = np.sum(estimated_pull[valid] * exact_pull[valid], axis=1)
    cosine /= np.maximum(estimate_norm[valid] * truth_norm[valid], 1e-30)
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    denoised = query.astype(np.float64) + estimated_pull
    surface_residual = np.abs(np.linalg.norm(denoised, axis=1) - 1.0)
    return {
        "method": method,
        "sigma": float(sigma),
        "seed": seed,
        "n_queries": int(len(query)),
        "pull_nrmse": float(np.sqrt(np.mean(error_norm * error_norm)) / float(sigma)),
        "pull_relative_l2": aggregate_relative,
        "pull_error_p90_over_sigma": float(np.quantile(error_norm, 0.9) / float(sigma)),
        "angle_mean_deg": float(np.mean(angles)),
        "angle_p90_deg": float(np.quantile(angles, 0.9)),
        "angle_valid_fraction": float(np.mean(valid)),
        "surface_mae_over_sigma": float(np.mean(surface_residual) / float(sigma)),
        "surface_p90_over_sigma": float(np.quantile(surface_residual, 0.9) / float(sigma)),
    }


def profile_metrics(
    score: np.ndarray,
    exact_score: np.ndarray,
    directions: np.ndarray,
    sigma: float,
) -> dict[str, float]:
    count = len(directions)
    t_repeated = np.repeat(PROFILE_T, count)
    directions_tiled = np.tile(directions, (len(PROFILE_T), 1)).astype(np.float64)
    predicted_pull = float(sigma * sigma) * score.astype(np.float64)
    exact_pull = float(sigma * sigma) * exact_score.astype(np.float64)
    predicted_radial = np.sum(predicted_pull * directions_tiled, axis=1) / float(sigma)
    exact_radial = np.sum(exact_pull * directions_tiled, axis=1) / float(sigma)
    predicted_means = np.asarray(
        [np.mean(predicted_radial[t_repeated == value]) for value in PROFILE_T]
    )
    exact_means = np.asarray(
        [np.mean(exact_radial[t_repeated == value]) for value in PROFILE_T]
    )
    fit = np.abs(PROFILE_T) <= 1.0
    predicted_slope, predicted_intercept = np.polyfit(PROFILE_T[fit], predicted_means[fit], 1)
    exact_slope, exact_intercept = np.polyfit(PROFILE_T[fit], exact_means[fit], 1)
    tangential = predicted_pull - np.sum(
        predicted_pull * directions_tiled, axis=1, keepdims=True
    ) * directions_tiled
    return {
        "normal_slope_ratio": float(predicted_slope / exact_slope),
        "normal_slope_predicted": float(predicted_slope),
        "normal_slope_exact": float(exact_slope),
        "surface_bias_predicted_over_sigma": float(predicted_intercept),
        "surface_bias_exact_over_sigma": float(exact_intercept),
        "profile_tangential_rms_over_sigma": float(
            np.sqrt(np.mean(np.sum(tangential * tangential, axis=1))) / float(sigma)
        ),
    }


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


def discover_checkpoints(root: Path, sigma: float) -> list[Path]:
    tag = sigma_tag(sigma)
    return sorted(root.glob(f"*/sigma_{tag}/seed_*.pt"))


def checkpoint_seed(path: Path) -> int:
    match = re.search(r"seed_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"checkpoint name lacks seed: {path}")
    return int(match.group(1))


def command_evaluate(args: argparse.Namespace) -> None:
    from scipy.spatial import cKDTree

    device = torch.device(args.device)
    support = np.load(args.support, mmap_mode="r")
    if support.ndim != 2 or support.shape[1] != 3:
        raise ValueError(f"invalid support: {support.shape}")
    print(json.dumps({"event": "build_kdtree", "n_support": len(support)}), flush=True)
    tree_started = time.perf_counter()
    tree = cKDTree(np.asarray(support, dtype=np.float64), compact_nodes=True, balanced_tree=True)
    print(json.dumps({"event": "kdtree_ready", "seconds": time.perf_counter() - tree_started}), flush=True)

    rng = np.random.default_rng(args.query_seed)
    clean = rng.normal(size=(args.n_queries, 3))
    clean /= np.linalg.norm(clean, axis=1, keepdims=True).clip(1e-12)
    noise = rng.normal(size=(args.n_queries, 3))
    profile_directions = rng.normal(size=(args.profile_directions, 3))
    profile_directions /= np.linalg.norm(profile_directions, axis=1, keepdims=True).clip(1e-12)

    rows: list[dict] = []
    profile_rows: list[dict] = []
    for sigma in args.sigmas:
        query = (clean + float(sigma) * noise).astype(np.float32)
        exact = exact_sphere_score_numpy(query, sigma)
        profile_query = np.concatenate(
            [(1.0 + float(sigma) * value) * profile_directions for value in PROFILE_T]
        ).astype(np.float32)
        profile_exact = exact_sphere_score_numpy(profile_query, sigma)

        k = gaussian_mass_knn(len(support), sigma, args.gaussian_mass, args.knn_safety)
        empirical = empirical_score_numpy(
            tree,
            support,
            query,
            sigma,
            k,
            args.empirical_chunk,
            args.workers,
        )
        empirical_profile = empirical_score_numpy(
            tree,
            support,
            profile_query,
            sigma,
            k,
            args.empirical_chunk,
            args.workers,
        )
        empirical_row = metric_summary("empirical_knn", sigma, empirical, exact, query, None)
        empirical_row.update({"knn_k": k, "support_n": len(support)})
        empirical_row.update(
            profile_metrics(empirical_profile, profile_exact, profile_directions, sigma)
        )
        rows.append(empirical_row)

        for checkpoint_path in discover_checkpoints(Path(args.checkpoint_root), sigma):
            model, checkpoint = load_checkpoint(checkpoint_path, device)
            model_type = str(checkpoint["model_type"])
            seed = checkpoint_seed(checkpoint_path)
            trained = trained_score_numpy(
                model, model_type, query, sigma, device, args.eval_batch_size
            )
            trained_profile = trained_score_numpy(
                model, model_type, profile_query, sigma, device, args.eval_batch_size
            )
            row = metric_summary(
                f"trained_{model_type}", sigma, trained, exact, query, seed
            )
            row.update(
                {
                    "checkpoint": str(checkpoint_path),
                    "steps": int(checkpoint["steps"]),
                    "train_examples_seen": int(checkpoint["train_examples_seen"]),
                    "validation_eps_mse": float(checkpoint["final_validation_eps_mse"]),
                }
            )
            row.update(
                profile_metrics(trained_profile, profile_exact, profile_directions, sigma)
            )
            rows.append(row)
            print(
                json.dumps(
                    {
                        "event": "evaluated",
                        "method": row["method"],
                        "sigma": sigma,
                        "seed": seed,
                        "pull_nrmse": row["pull_nrmse"],
                        "normal_slope_ratio": row["normal_slope_ratio"],
                    }
                ),
                flush=True,
            )

        for method, score in (("exact", profile_exact), ("empirical_knn", empirical_profile)):
            radial = float(sigma * sigma) * np.sum(
                score.astype(np.float64)
                * np.tile(profile_directions, (len(PROFILE_T), 1)),
                axis=1,
            ) / float(sigma)
            for index, value in enumerate(PROFILE_T):
                selection = slice(index * len(profile_directions), (index + 1) * len(profile_directions))
                profile_rows.append(
                    {
                        "method": method,
                        "sigma": sigma,
                        "seed": None,
                        "offset_over_sigma": value,
                        "radial_pull_over_sigma": float(np.mean(radial[selection])),
                    }
                )

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "per_run.csv", rows)
    write_csv(output_dir / "normal_profiles.csv", profile_rows)
    atomic_json(output_dir / "per_run.json", rows)
    aggregate_outputs(rows, output_dir)


def aggregate_outputs(rows: list[dict], output_dir: Path) -> None:
    import pandas as pd

    frame = pd.DataFrame(rows)
    numeric_metrics = [
        "pull_nrmse",
        "pull_relative_l2",
        "pull_error_p90_over_sigma",
        "angle_mean_deg",
        "angle_p90_deg",
        "surface_mae_over_sigma",
        "surface_p90_over_sigma",
        "normal_slope_ratio",
        "profile_tangential_rms_over_sigma",
        "validation_eps_mse",
    ]
    summary_rows: list[dict] = []
    for (method, sigma), group in frame.groupby(["method", "sigma"], sort=True):
        result: dict[str, object] = {
            "method": method,
            "sigma": float(sigma),
            "runs": int(len(group)),
        }
        for metric in numeric_metrics:
            values = pd.to_numeric(group.get(metric), errors="coerce").dropna()
            if len(values):
                result[f"{metric}_mean"] = float(values.mean())
                result[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(result)

    write_csv(output_dir / "summary.csv", summary_rows)
    atomic_json(output_dir / "summary.json", summary_rows)
    write_latex_table(summary_rows, output_dir / "table_sphere_score_comparison.tex")
    write_figure(summary_rows, output_dir / "figure_sigma_sweep.png")


def write_latex_table(rows: list[dict], path: Path) -> None:
    labels = {
        "empirical_knn": "Empirical KNN",
        "trained_direct": "Trained DSM",
        "trained_energy": "Trained energy DSM",
    }
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"$\sigma$ & Score & Pull NRMSE $\downarrow$ & Rel. $\ell_2$ $\downarrow$ & Surface residual$/\sigma$ $\downarrow$ & Slope ratio $\to 1$ \\",
        r"\midrule",
    ]
    for sigma in sorted({float(row["sigma"]) for row in rows}, reverse=True):
        sigma_rows = [row for row in rows if float(row["sigma"]) == sigma]
        for row in sorted(sigma_rows, key=lambda item: str(item["method"])):
            method = labels.get(str(row["method"]), str(row["method"]))
            lines.append(
                f"{sigma:g} & {method} & "
                f"{float(row['pull_nrmse_mean']):.4f} & "
                f"{float(row['pull_relative_l2_mean']):.4f} & "
                f"{float(row['surface_mae_over_sigma_mean']):.4f} & "
                f"{float(row['normal_slope_ratio_mean']):.3f} \\\\"
            )
        lines.append(r"\addlinespace")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "empirical_knn": "Empirical KNN",
        "trained_direct": "Trained DSM",
        "trained_energy": "Trained energy DSM",
    }
    colors = {
        "empirical_knn": "#167A72",
        "trained_direct": "#D55E00",
        "trained_energy": "#7A5195",
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    for method in labels:
        method_rows = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda item: float(item["sigma"]),
        )
        if not method_rows:
            continue
        x = np.asarray([float(row["sigma"]) for row in method_rows])
        for axis, metric in zip(
            axes,
            ("pull_nrmse", "surface_mae_over_sigma", "normal_slope_ratio"),
        ):
            y = np.asarray([float(row[f"{metric}_mean"]) for row in method_rows])
            std = np.asarray([float(row.get(f"{metric}_std", 0.0)) for row in method_rows])
            axis.plot(x, y, marker="o", linewidth=1.8, color=colors[method], label=labels[method])
            if np.any(std > 0):
                axis.fill_between(x, y - std, y + std, color=colors[method], alpha=0.14)
    for axis in axes:
        axis.set_xscale("log")
        axis.invert_xaxis()
        tick_values = sorted({float(row["sigma"]) for row in rows}, reverse=True)
        axis.set_xticks(tick_values)
        axis.set_xticklabels([f"{value:g}" for value in tick_values])
        axis.set_xlabel(r"Noise scale $\sigma$")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Pull NRMSE")
    axes[1].set_ylabel(r"One-step surface residual $/\sigma$")
    axes[2].set_ylabel("Normal-slope ratio")
    axes[2].axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
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
    train.set_defaults(func=command_train)

    support = commands.add_parser("make-support")
    support.add_argument("--output", required=True)
    support.add_argument("--n-points", type=int, default=40_000_000)
    support.add_argument("--chunk-size", type=int, default=1_000_000)
    support.add_argument("--seed", type=int, default=20260826)
    support.set_defaults(func=command_make_support)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint-root", required=True)
    evaluate.add_argument("--support", required=True)
    evaluate.add_argument("--sigmas", type=float, nargs="+", default=SIGMAS_DEFAULT)
    evaluate.add_argument("--n-queries", type=int, default=5_000)
    evaluate.add_argument("--profile-directions", type=int, default=512)
    evaluate.add_argument("--query-seed", type=int, default=20260829)
    evaluate.add_argument("--gaussian-mass", type=float, default=0.99)
    evaluate.add_argument("--knn-safety", type=float, default=1.10)
    evaluate.add_argument("--empirical-chunk", type=int, default=128)
    evaluate.add_argument("--workers", type=int, default=-1)
    evaluate.add_argument("--eval-batch-size", type=int, default=4096)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--output-dir", required=True)
    evaluate.set_defaults(func=command_evaluate)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
