#!/usr/bin/env python3
"""Extract the complete 360^3 zero-level mesh used by the paper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from models import SIRENImplicitSurface


GRID_RESOLUTION = 360
LEVEL = 0.0
DEFAULT_BBOX_PAD_FRACTION_EACH_SIDE = 0.10
FORMAL_LOSS = "score_pull_plus_global_cosine"
FORMAL_ARCHITECTURE = {"hidden_dim": 128, "hidden_layers": 4, "omega0": 20.0}


def parse_bounds(text: str) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    # Accept both the public xmin,...,zmax form and the batch runner's
    # xlow,xhigh;ylow,yhigh;zlow,zhigh form.
    values = [float(value.strip()) for value in text.replace(";", ",").split(",") if value.strip()]
    if len(values) != 6:
        raise ValueError("--bounds requires xmin,xmax,ymin,ymax,zmin,zmax")
    bounds = ((values[0], values[1]), (values[2], values[3]), (values[4], values[5]))
    if any(not np.isfinite(low) or not np.isfinite(high) or low >= high for low, high in bounds):
        raise ValueError(f"Invalid extraction bounds: {bounds}")
    return bounds


def load_point_cloud_xyz(path: Path) -> np.ndarray:
    """Load XYZ from an NPY array or the first columns of a text point cloud."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        points = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64)
        points = np.atleast_2d(points)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(
                f"NPY point cloud must have shape [N,3+] for extraction: {path}"
            )
        points = points[:, :3]
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            raise ValueError(f"Point cloud contains no finite XYZ rows: {path}")
        return points
    delimiter = "," if path.suffix.lower() == ".csv" else None
    first = path.open(encoding="utf-8-sig", errors="replace").readline()
    tokens = first.split(",") if delimiter == "," else first.split()
    try:
        [float(value) for value in tokens[:3]]
        skip_rows = 0
    except ValueError:
        skip_rows = 1
    points = np.loadtxt(
        path,
        delimiter=delimiter,
        skiprows=skip_rows,
        usecols=(0, 1, 2),
        dtype=np.float64,
    )
    points = np.atleast_2d(points)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError(f"Point cloud contains no finite XYZ rows: {path}")
    return points


def padded_point_cloud_bounds(
    points: np.ndarray,
    pad_fraction_each_side: float = DEFAULT_BBOX_PAD_FRACTION_EACH_SIDE,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return per-axis input bounds padded by a fraction of that axis span."""
    fraction = float(pad_fraction_each_side)
    if not np.isfinite(fraction) or fraction < 0:
        raise ValueError("--bbox-pad-fraction-each-side must be finite and non-negative")
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    span = high - low
    if np.any(~np.isfinite(span)) or np.any(span <= 0):
        raise ValueError(f"Point cloud has a degenerate coordinate span: {span.tolist()}")
    padding = fraction * span
    padded_low = low - padding
    padded_high = high + padding
    return tuple(
        (float(padded_low[index]), float(padded_high[index]))
        for index in range(3)
    )


def load_model(path: Path, device: torch.device) -> tuple[SIRENImplicitSurface, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Formal extraction requires a checkpoint dictionary")
    meta = checkpoint.get("meta")
    if not isinstance(meta, dict) or str(meta.get("model_type")) != "siren":
        raise ValueError("Formal extraction requires an SGR SIREN checkpoint")
    if int(meta.get("data_dim", 3)) != 3:
        raise ValueError("Low-dimensional extraction requires a 3D checkpoint")
    if str(meta.get("loss", {}).get("name")) != FORMAL_LOSS:
        raise ValueError(f"Formal extraction requires loss={FORMAL_LOSS}")
    for key, expected in FORMAL_ARCHITECTURE.items():
        if float(meta.get(key, float("nan"))) != float(expected):
            raise ValueError(f"Formal extraction requires {key}={expected}")
    if int(checkpoint.get("step", -1)) <= 0:
        raise ValueError("Formal extraction requires a completed training checkpoint")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Formal extraction checkpoint is missing model_state_dict")
    model = SIRENImplicitSurface(
        hidden_dim=int(meta["hidden_dim"]),
        hidden_layers=int(meta["hidden_layers"]),
        omega0=float(meta["omega0"]),
        input_dim=3,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def evaluate_grid(
    model: SIRENImplicitSurface,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    data_mean: np.ndarray,
    data_scale: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    x_axis, y_axis, z_axis = axes
    grid = np.empty((GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION), dtype=np.float32)
    xy = np.stack(np.meshgrid(x_axis, y_axis, indexing="ij"), axis=-1).reshape(-1, 2)
    mean = torch.as_tensor(data_mean, dtype=torch.float32, device=device)
    for index, z_value in enumerate(z_axis):
        physical = np.column_stack(
            (xy, np.full((len(xy),), z_value, dtype=np.float32))
        ).astype(np.float32)
        values: list[np.ndarray] = []
        for start in range(0, len(physical), int(batch_size)):
            points = torch.from_numpy(physical[start : start + int(batch_size)]).to(device)
            normalized = (points - mean[None, :]) / float(data_scale)
            values.append(model(normalized).cpu().numpy().astype(np.float32))
        grid[:, :, index] = np.concatenate(values).reshape(GRID_RESOLUTION, GRID_RESOLUTION)
        if index == 0 or (index + 1) % 20 == 0 or index + 1 == GRID_RESOLUTION:
            print(json.dumps({"event": "grid_slice", "completed": index + 1, "total": GRID_RESOLUTION}), flush=True)
    return grid


def marching_cubes(
    field: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from skimage import measure
    except ImportError as exc:
        raise RuntimeError("scikit-image is required for 360^3 extraction") from exc
    if float(field.min()) > LEVEL or float(field.max()) < LEVEL:
        raise RuntimeError(
            f"The requested bounds do not contain Phi=0: field range [{field.min()}, {field.max()}]"
        )
    spacing = tuple(float(axis[1] - axis[0]) for axis in axes)
    vertices, faces, _, _ = measure.marching_cubes(field, level=LEVEL, spacing=spacing)
    origin = np.asarray([axis[0] for axis in axes], dtype=np.float32)
    return vertices.astype(np.float32) + origin[None, :], faces.astype(np.int64)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract the formal complete SGR Phi=0 mesh at 360^3.")
    parser.add_argument("--checkpoint", required=True)
    domain = parser.add_mutually_exclusive_group(required=True)
    domain.add_argument(
        "--point-cloud",
        help=(
            "Released input point cloud used to derive the extraction domain. "
            "Each coordinate interval is padded on both sides by 10%% by default."
        ),
    )
    domain.add_argument(
        "--bounds",
        help="Explicit xmin,xmax,ymin,ymax,zmin,zmax override.",
    )
    parser.add_argument(
        "--bbox-pad-fraction-each-side",
        type=float,
        default=DEFAULT_BBOX_PAD_FRACTION_EACH_SIDE,
        help=(
            "Per-axis padding fraction applied on each side of --point-cloud "
            f"(default: {DEFAULT_BBOX_PAD_FRACTION_EACH_SIDE:.2f})."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-mesh-npz", required=True)
    parser.add_argument("--out-stats", required=True)
    return parser


def normalize_bounds_argument(argv: list[str]) -> list[str]:
    """Protect a negative comma-delimited bounds value from argparse option parsing."""
    normalized = list(argv)
    if "--bounds" in normalized:
        index = normalized.index("--bounds")
        if index + 1 < len(normalized):
            normalized[index : index + 2] = [f"--bounds={normalized[index + 1]}"]
    return normalized


def main() -> None:
    args = build_arg_parser().parse_args(normalize_bounds_argument(sys.argv[1:]))
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.point_cloud:
        point_cloud_path = Path(args.point_cloud).expanduser().resolve()
        bounds = padded_point_cloud_bounds(
            load_point_cloud_xyz(point_cloud_path),
            float(args.bbox_pad_fraction_each_side),
        )
        bounds_source = "input_point_cloud_per_axis_span"
    else:
        point_cloud_path = None
        bounds = parse_bounds(args.bounds)
        bounds_source = "explicit_override"
    axes = tuple(
        np.linspace(low, high, GRID_RESOLUTION, dtype=np.float32)
        for low, high in bounds
    )
    model, checkpoint = load_model(Path(args.checkpoint), device)
    meta = checkpoint["meta"]
    field = evaluate_grid(
        model,
        axes,
        data_mean=np.asarray(meta["data_mean"], dtype=np.float32),
        data_scale=float(meta["data_scale"]),
        batch_size=int(args.batch_size),
        device=device,
    )
    vertices, faces = marching_cubes(field, axes)
    mesh_path = Path(args.out_mesh_npz)
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mesh_path, vertices=vertices, faces=faces, level=LEVEL)
    stats = {
        "extractor": "formal_sgr_complete_marching_cubes",
        "checkpoint": str(args.checkpoint),
        "step": int(checkpoint.get("step", -1)),
        "grid_n": GRID_RESOLUTION,
        "axis_ranges": [[float(axis[0]), float(axis[-1])] for axis in axes],
        "bounds_source": bounds_source,
        "point_cloud": str(point_cloud_path) if point_cloud_path is not None else None,
        "bbox_pad_fraction_each_side": (
            float(args.bbox_pad_fraction_each_side)
            if point_cloud_path is not None
            else None
        ),
        "level": LEVEL,
        "field_min": float(field.min()),
        "field_max": float(field.max()),
        "component_policy": "all_components",
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "out_mesh_npz": str(mesh_path),
    }
    stats_path = Path(args.out_stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", **stats}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
