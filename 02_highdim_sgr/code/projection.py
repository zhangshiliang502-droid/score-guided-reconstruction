"""Shared zero-set projection and metric primitives for formal evaluation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from models import SIRENImplicitSurface


FORMAL_LOSS = "score_pull_plus_global_cosine"
FORMAL_ARCHITECTURE = {"hidden_dim": 128, "hidden_layers": 4, "omega0": 20.0}
FORMAL_SIGMA = 0.01
FORMAL_LOCAL_SHELL_STD_SIGMA = 1.0


def validate_formal_loss(meta: dict) -> None:
    """Validate checkpoint architecture and loss settings against the experiment."""
    loss = meta.get("loss")
    if not isinstance(loss, dict) or str(loss.get("name")) != FORMAL_LOSS:
        raise ValueError(f"checkpoint does not use the formal loss {FORMAL_LOSS}")
    if str(loss.get("local_query_mode")) != "radial_shell":
        raise ValueError("checkpoint does not use the table's radial-shell local queries")
    local_shell_center_sigma = float(
        loss.get("local_shell_center_sigma", float("nan"))
    )
    if not math.isfinite(local_shell_center_sigma) or local_shell_center_sigma <= 0.0:
        raise ValueError("checkpoint has an invalid local_shell_center_sigma")
    if not math.isclose(
        float(loss.get("local_shell_std_sigma", float("nan"))),
        FORMAL_LOCAL_SHELL_STD_SIGMA,
    ):
        raise ValueError("checkpoint does not use local_shell_std_sigma=1")
    if str(loss.get("local_residual_normalization")) != "none":
        raise ValueError("checkpoint must use the unscaled local residual")
    lambda_dir = float(loss.get("lambda_dir", float("nan")))
    if not math.isfinite(lambda_dir) or lambda_dir <= 0.0:
        raise ValueError("checkpoint has an invalid lambda_dir")
    if not math.isclose(float(loss.get("sigma", float("nan"))), FORMAL_SIGMA):
        raise ValueError("checkpoint does not use sigma=0.01")


def quantiles(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("cannot summarize an empty array")
    return {
        "mean": float(np.mean(data)),
        "rmse": float(np.sqrt(np.mean(data**2))),
        "p50": float(np.quantile(data, 0.50)),
        "p90": float(np.quantile(data, 0.90)),
        "p99": float(np.quantile(data, 0.99)),
        "max": float(np.max(data)),
    }


def load_model(path: Path, device: torch.device) -> tuple[SIRENImplicitSurface, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("meta"), dict):
        raise ValueError(f"invalid SGR checkpoint: {path}")
    meta = payload["meta"]
    required = ("hidden_dim", "hidden_layers", "omega0", "data_dim")
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(f"checkpoint metadata is missing {missing}")
    if str(meta.get("model_type")) != "siren":
        raise ValueError("checkpoint is not a SIREN implicit surface")
    if int(meta["data_dim"]) <= 3:
        raise ValueError("high-dimensional reconstruction requires data_dim > 3")
    validate_formal_loss(meta)
    for key, expected in FORMAL_ARCHITECTURE.items():
        if float(meta[key]) != float(expected):
            raise ValueError(f"checkpoint does not use formal {key}={expected}")
    if int(payload.get("step", -1)) <= 0:
        raise ValueError("checkpoint has no completed training step")
    model = SIRENImplicitSurface(
        hidden_dim=int(meta["hidden_dim"]),
        hidden_layers=int(meta["hidden_layers"]),
        omega0=float(meta["omega0"]),
        input_dim=int(meta["data_dim"]),
    ).to(device)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint is missing model_state_dict")
    model.load_state_dict(state)
    model.eval()
    return model, payload


def eval_phi_grad(
    model: SIRENImplicitSurface,
    points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = points.detach().requires_grad_(True)
    phi = model(query)
    gradient = torch.autograd.grad(phi.sum(), query, create_graph=False)[0]
    return phi.detach(), gradient.detach()


def project_learned_zero(
    model: SIRENImplicitSurface,
    seeds: np.ndarray,
    *,
    low: np.ndarray,
    high: np.ndarray,
    device: torch.device,
    iterations: int,
    max_step: float,
    residual: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = torch.as_tensor(seeds, dtype=torch.float32, device=device)
    low_tensor = torch.as_tensor(low, dtype=torch.float32, device=device)
    high_tensor = torch.as_tensor(high, dtype=torch.float32, device=device)
    for _ in range(int(iterations)):
        phi, gradient = eval_phi_grad(model, points)
        gradient_squared = gradient.square().sum(dim=1).clamp_min(1e-12)
        step = (phi / gradient_squared)[:, None] * gradient
        step_norm = step.norm(dim=1, keepdim=True).clamp_min(1e-12)
        step = step * torch.clamp(float(max_step) / step_norm, max=1.0)
        points = (points - step).clamp(low_tensor, high_tensor).detach()
    phi, gradient = eval_phi_grad(model, points)
    gradient_norm = gradient.norm(dim=1)
    accepted = (phi.abs() <= float(residual)) & (gradient_norm > 1e-6)
    return (
        points.cpu().numpy().astype(np.float32),
        phi.cpu().numpy().astype(np.float32),
        gradient_norm.cpu().numpy().astype(np.float32),
        accepted.cpu().numpy(),
    )


def nearest_distances(tree: cKDTree, points: np.ndarray) -> np.ndarray:
    try:
        distance, _ = tree.query(points.astype(np.float64), k=1, workers=-1)
    except TypeError:
        distance, _ = tree.query(points.astype(np.float64), k=1)
    return np.asarray(distance, dtype=np.float64)
