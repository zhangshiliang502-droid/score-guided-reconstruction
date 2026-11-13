#!/usr/bin/env python3
"""Run a small end-to-end low-dimensional SGR training smoke test on CPU."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def synthetic_sphere(count: int = 256) -> np.ndarray:
    rng = np.random.default_rng(20260810)
    points = rng.normal(size=(count, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points.astype(np.float32)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sgr-lowdim-smoke-") as directory:
        workspace = Path(directory)
        point_cloud = workspace / "sphere.npy"
        output_dir = workspace / "run"
        np.save(point_cloud, synthetic_sphere())
        command = [
            sys.executable,
            str(ROOT / "01_lowdim_sgr" / "code" / "main.py"),
            "--point-cloud",
            str(point_cloud),
            "--output-dir",
            str(output_dir),
            "--local-cache-size",
            "128",
            "--global-cache-size",
            "128",
            "--knn-k",
            "16",
            "--knn-workers",
            "1",
            "--score-chunk-size",
            "64",
            "--iterations",
            "2",
            "--batch-size",
            "32",
            "--global-batch-size",
            "32",
            "--eval-batches",
            "1",
            "--seed",
            "20260810",
            "--device",
            "cpu",
        ]
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "low-dimensional CPU smoke test failed:\n" + completed.stdout
            )

        state = json.loads((output_dir / "main_state.json").read_text())
        if state.get("status") != "complete":
            raise RuntimeError(f"unexpected main state: {state}")
        checkpoint_path = output_dir / "implicit_function" / "best.pt"
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if int(checkpoint.get("step", 0)) < 1:
            raise RuntimeError("smoke checkpoint has no completed training step")
        if not isinstance(checkpoint.get("model_state_dict"), dict):
            raise RuntimeError("smoke checkpoint has no model state")
        print(
            json.dumps(
                {
                    "status": "ok",
                    "device": "cpu",
                    "input_points": 256,
                    "training_steps": int(checkpoint["step"]),
                }
            )
        )


if __name__ == "__main__":
    main()
