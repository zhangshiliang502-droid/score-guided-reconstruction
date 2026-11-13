#!/usr/bin/env python3
"""Re-evaluate all 27 released low-dimensional SGR meshes.

The command writes one ``result/evaluation.json`` per example and refreshes
the evaluation block in each ``result/summary.json``. The retained aggregate
comparison is generated separately by ``evaluate_lowdim_comparison.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import evaluate_lowdim_reconstruction as single


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def discover_examples(root: Path) -> list[tuple[int, Path, dict[str, Any]]]:
    examples: list[tuple[int, Path, dict[str, Any]]] = []
    for configuration in sorted((root / "example" / "lowdim").glob(
        "example_*/config/training.json"
    )):
        example = configuration.parent.parent
        reference = load_json(example / "dataset" / "reference.json")
        examples.append(
            (
                int(reference.get("evaluation_seed_offset", 0)),
                example,
                reference,
            )
        )
    examples.sort(key=lambda row: (row[0], row[1].name))
    if len(examples) != 27:
        raise RuntimeError(f"Expected 27 low-dimensional examples, found {len(examples)}")
    datasets = [str(row[2].get("dataset")) for row in examples]
    if len(set(datasets)) != len(datasets):
        raise RuntimeError("Low-dimensional dataset names are not unique")
    return examples


def progress_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "fit_rmse": result["point_to_surface"]["fit_rmse"],
        "chamfer_l2": result["chamfer_l2"]["value"],
        "component_count": result["ghost_zero"]["component_count"],
        "ghost_component_count": result["ghost_zero"]["ghost_component_count"],
        "elapsed_seconds": result["elapsed_seconds"],
    }


def refresh_summary(path: Path, result: dict[str, Any]) -> None:
    summary = load_json(path)
    summary["schema"] = "paper-lowdim-best-result-summary/v4"
    summary.setdefault("result_files", {})["evaluation"] = "result/evaluation.json"
    summary["evaluation"] = {
        "protocol": "paper_bbox10_independent_mesh_1m_training_normalization_v3",
        "rmse": result["point_to_surface"]["fit_rmse"],
        "chamfer_l2": result["chamfer_l2"]["value"],
        "component_count": result["ghost_zero"]["component_count"],
        "ghost_component_count": result["ghost_zero"]["ghost_component_count"],
        "ghost_area_ratio": result["ghost_zero"]["ghost_area_ratio"],
        "training_input_rmse_diagnostic": result["training_input_diagnostic"][
            "fit_rmse"
        ],
    }
    summary.pop("paper_table_comparison", None)
    write_json(path, summary)


def make_example_paths_portable(
    result: dict[str, Any],
    configuration: dict[str, Any],
    reference: dict[str, Any],
) -> None:
    point_cloud = str(configuration["training_parameters"]["point_cloud"])
    source_mesh = str(reference["asset_root_path"])
    result["configuration"] = "config/training.json"
    result["mesh"] = "result/reconstruction/zero_mesh.npz"
    result["reference_pointcloud"] = point_cloud
    result["point_to_surface"]["canonical_point_source"] = source_mesh
    result["chamfer_l2"]["ground_truth_source"] = source_mesh
    result.pop("elapsed_seconds", None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--test-samples", type=int, default=1_000_000)
    parser.add_argument("--reconstruction-samples", type=int, default=1_000_000)
    args = parser.parse_args()

    repository_root = Path(args.repository_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    examples = discover_examples(repository_root)

    for index, (_, example, reference) in enumerate(examples, start=1):
        dataset = str(reference["dataset"])
        print(f"[{index:02d}/{len(examples)}] {dataset}", flush=True)
        configuration = example / "config" / "training.json"
        mesh = example / "result" / "reconstruction" / "zero_mesh.npz"
        output = example / "result" / "evaluation.json"
        result = single.evaluate(
            SimpleNamespace(
                configuration=str(configuration),
                reference_json=str(example / "dataset" / "reference.json"),
                mesh=str(mesh),
                asset_root=str(asset_root),
                output=str(output),
                seed_offset=None,
                test_samples=int(args.test_samples),
                reconstruction_samples=int(args.reconstruction_samples),
                method="SGR",
                mesh_scale=1.0,
                mesh_translation=(0.0, 0.0, 0.0),
            )
        )
        row = progress_row(result)
        make_example_paths_portable(result, load_json(configuration), reference)
        write_json(output, result)
        refresh_summary(example / "result" / "summary.json", result)
        print(
            "  RMSE={fit_rmse:.8e} Chamfer={chamfer_l2:.8e} "
            "components={component_count} ghosts={ghost_component_count} "
            "elapsed={elapsed_seconds:.1f}s".format(**row),
            flush=True,
        )

    print(f"complete datasets={len(examples)}", flush=True)


if __name__ == "__main__":
    main()
