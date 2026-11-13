#!/usr/bin/env python3
"""Generate independent analytic reference meshes for three synthetic datasets.

The meshes are evaluated independently of the released training point clouds:
their vertices are obtained directly from the analytic parameterizations used
to define Sphere, CV2Radial_a020_f16, and FlowerPetal.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np


RadiusFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


def sphere_radius(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    del phi
    return np.ones_like(theta, dtype=np.float64)


def cv2_radius(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    st = np.sin(theta)
    frequency = 16.0
    wave = (
        0.62 * np.sin(frequency * theta) * np.cos(frequency * phi)
        + 0.25
        * np.cos((frequency + 1.0) * theta + 0.35)
        * np.sin((frequency + 2.0) * phi)
        + 0.13 * np.sin((2.0 * frequency + 1.0) * phi + 0.2)
    )
    return np.maximum(1.0 + 0.2 * st**2 * wave, 0.72)


def flower_radius(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    st = np.sin(theta)
    a = 16.0 * phi + 0.65 * np.sin(3.0 * theta)
    b = 5.0 * theta + 1.1 * np.sin(8.0 * phi)
    main = st**1.8 * np.cos(8.0 * phi)
    secondary = st**2.5 * np.cos(a)
    vertical = (
        0.5 * np.cos(4.0 * theta)
        + 0.5 * np.sin(3.0 * theta + 0.4) * np.cos(4.0 * phi)
    )
    twist = st**2 * np.sin(b)
    raw = (
        1.0
        + 0.26 * main
        + 0.10 * secondary
        + 0.075 * vertical
        + 0.055 * twist
    )
    return np.maximum(raw, 0.62)


DATASETS: dict[str, tuple[str, RadiusFunction, str]] = {
    "sphere": (
        "Sphere",
        sphere_radius,
        "r(theta,phi)=1",
    ),
    "cv2_radial_a020_f16": (
        "CV2Radial_a020_f16",
        cv2_radius,
        "r=max(1+0.2 sin(theta)^2 wave_16(theta,phi),0.72)",
    ),
    "flower_petal": (
        "FlowerPetal",
        flower_radius,
        "the released analytic eight-lobe radial FlowerPetal formula",
    ),
}


def radial_vertices(
    radius_function: RadiusFunction,
    latitude_segments: int,
    longitude_segments: int,
) -> np.ndarray:
    theta = (
        np.arange(1, latitude_segments, dtype=np.float64)
        * (math.pi / latitude_segments)
    )[:, None]
    phi = (
        np.arange(longitude_segments, dtype=np.float64)
        * (2.0 * math.pi / longitude_segments)
    )[None, :]
    theta_grid = np.broadcast_to(theta, (latitude_segments - 1, longitude_segments))
    phi_grid = np.broadcast_to(phi, theta_grid.shape)
    radius = radius_function(theta_grid, phi_grid)
    sin_theta = np.sin(theta_grid)
    rings = np.stack(
        (
            radius * sin_theta * np.cos(phi_grid),
            radius * sin_theta * np.sin(phi_grid),
            radius * np.cos(theta_grid),
        ),
        axis=-1,
    )

    # The analytic parameterizations are defined almost everywhere on S^2.
    # A single cap vertex closes each measure-zero polar seam. Using the mean
    # of the adjacent ring avoids introducing a point sampled from a training
    # cloud and also handles FlowerPetal's phi-dependent limiting radius.
    north = rings[0].mean(axis=0, dtype=np.float64)
    south = rings[-1].mean(axis=0, dtype=np.float64)
    vertices = np.concatenate(
        (
            north[None, :],
            rings.reshape(-1, 3),
            south[None, :],
        ),
        axis=0,
    ).astype(np.float32)
    return vertices


def radial_faces(
    latitude_segments: int,
    longitude_segments: int,
    vertex_count: int,
) -> np.ndarray:
    """Triangulate polar caps and consecutive latitude rings."""

    rings_count = latitude_segments - 1
    first_ring = 1
    south_index = vertex_count - 1
    current = np.arange(longitude_segments, dtype=np.int64)
    following = np.roll(current, -1)

    north_faces = np.column_stack(
        (
            np.zeros(longitude_segments, dtype=np.int64),
            first_ring + following,
            first_ring + current,
        )
    )
    bands: list[np.ndarray] = []
    for ring_index in range(rings_count - 1):
        lower = first_ring + ring_index * longitude_segments
        upper = lower + longitude_segments
        bands.append(
            np.concatenate(
                (
                    np.column_stack(
                        (
                            lower + current,
                            lower + following,
                            upper + following,
                        )
                    ),
                    np.column_stack(
                        (
                            lower + current,
                            upper + following,
                            upper + current,
                        )
                    ),
                ),
                axis=0,
            )
        )
    last_ring = first_ring + (rings_count - 1) * longitude_segments
    south_faces = np.column_stack(
        (
            np.full(longitude_segments, south_index, dtype=np.int64),
            last_ring + current,
            last_ring + following,
        )
    )
    faces = np.concatenate((north_faces, *bands, south_faces), axis=0)
    return faces.astype(np.int32)


def radial_mesh(
    radius_function: RadiusFunction,
    latitude_segments: int,
    longitude_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    if latitude_segments < 4 or longitude_segments < 8:
        raise ValueError("mesh resolution is too small")
    vertices = radial_vertices(
        radius_function,
        latitude_segments,
        longitude_segments,
    )
    faces = radial_faces(
        latitude_segments,
        longitude_segments,
        len(vertices),
    )
    return vertices, faces


def write_mesh(
    output: Path,
    dataset: str,
    formula: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    latitude_segments: int,
    longitude_segments: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, vertices=vertices, faces=faces)
    metadata = {
        "schema": "paper-independent-analytic-reference-mesh/v1",
        "dataset": dataset,
        "source_kind": "analytic_parameterization_not_training_pointcloud",
        "formula": formula,
        "latitude_segments": latitude_segments,
        "longitude_segments": longitude_segments,
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "coordinate_space": "raw_dataset_coordinates",
        "mesh": str(output),
        "bounds_min": vertices.min(axis=0).astype(float).tolist(),
        "bounds_max": vertices.max(axis=0).astype(float).tolist(),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        required=True,
        help="Asset root containing datasets/source_meshes.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted((*DATASETS, "all")),
        default="all",
    )
    parser.add_argument("--latitude-segments", type=int, default=512)
    parser.add_argument("--longitude-segments", type=int, default=1024)
    args = parser.parse_args()

    asset_root = Path(args.asset_root).expanduser().resolve()
    selected = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    for key, (dataset, radius_function, formula) in selected.items():
        vertices, faces = radial_mesh(
            radius_function,
            int(args.latitude_segments),
            int(args.longitude_segments),
        )
        filename = (
            f"{dataset}_analytic_reference_"
            f"{args.latitude_segments}x{args.longitude_segments}.npz"
        )
        output = asset_root / "A" / "datasets" / "source_meshes" / dataset / filename
        write_mesh(
            output,
            dataset,
            formula,
            vertices,
            faces,
            int(args.latitude_segments),
            int(args.longitude_segments),
        )
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "mesh": str(output),
                    "vertices": int(len(vertices)),
                    "faces": int(len(faces)),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
