#!/usr/bin/env python3
"""Measure SGR and DWG with one independent-mesh, normalized 10% protocol."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
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
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def discover_examples(repository_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for configuration in sorted(
        (repository_root / "example" / "lowdim").glob(
            "example_*/config/training.json"
        )
    ):
        example = configuration.parent.parent
        reference = load_json(example / "dataset" / "reference.json")
        rows.append((example, reference))
    rows.sort(
        key=lambda row: (
            int(row[1].get("evaluation_seed_offset", 0)),
            str(row[1]["dataset"]),
        )
    )
    if len(rows) != 27:
        raise RuntimeError(f"Expected 27 examples, found {len(rows)}")
    return rows


def load_done(root: Path, dataset: str, method: str) -> tuple[dict[str, Any], Path]:
    directory = root / dataset
    done_path = directory / "DONE.json"
    if not done_path.is_file():
        raise FileNotFoundError(done_path)
    done = load_json(done_path)
    if done.get("status") != "ok":
        raise RuntimeError(f"{method} extraction did not finish successfully: {done_path}")
    parameters = done.get("parameters", {})
    if method == "DWG":
        padding = float(parameters.get("bbox_pad_ratio", -1.0))
    else:
        stats_path = Path(str(done.get("stats", "")))
        if not stats_path.is_file():
            candidates = sorted(directory.rglob("stats.json"))
            if not candidates:
                raise FileNotFoundError(f"Cannot locate SGR stats under {directory}")
            stats_path = candidates[0]
        padding = float(load_json(stats_path).get("bbox_pad_fraction_each_side", -1.0))
    if padding != 0.10:
        raise ValueError(f"{method} {dataset} is not a 10% extraction: {padding}")

    declared = Path(str(done.get("mesh", "")))
    if declared.is_file():
        mesh = declared
    else:
        suffix = ".stl" if method == "DWG" else ".npz"
        candidates = sorted(directory.rglob(f"*{suffix}"))
        if not candidates:
            raise FileNotFoundError(f"Cannot locate {method} mesh under {directory}")
        mesh = candidates[0]
    return done, mesh.resolve()


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "rmse": float(result["point_to_surface"]["fit_rmse"]),
        "chamfer_l2": float(result["chamfer_l2"]["value"]),
        "component_count": int(result["ghost_zero"]["component_count"]),
        "ghost_component_count": int(result["ghost_zero"]["ghost_component_count"]),
        "ghost_area_ratio": float(result["ghost_zero"]["ghost_area_ratio"]),
        "training_input_rmse_diagnostic": float(
            result["training_input_diagnostic"]["fit_rmse"]
        ),
        "vertices": int(result["vertices"]),
        "faces": int(result["faces"]),
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Unified 10% low-dimensional comparison",
        "",
        (
            "Both methods use a 360³ grid over their input point-cloud bounding "
            "box padded by 10% of each coordinate span on each side. RMSE and "
            "Chamfer use independent reference-mesh samples and the released "
            "training normalization; ghost spacing uses the complete input cloud."
        ),
        "",
        (
            "| Dataset | SGR RMSE | DWG RMSE | SGR Chamfer | DWG Chamfer | "
            "SGR comp./ghost | DWG comp./ghost | SGR ghost area (%) | "
            "DWG ghost area (%) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {sgr_rmse:.8e} | {dwg_rmse:.8e} | "
            "{sgr_chamfer_l2:.8e} | {dwg_chamfer_l2:.8e} | "
            "{sgr_component_count}/{sgr_ghost_component_count} | "
            "{dwg_component_count}/{dwg_ghost_component_count} | "
            "{sgr_ghost_percent:.6f} | {dwg_ghost_percent:.6f} |".format(
                **row,
                sgr_ghost_percent=100.0 * row["sgr_ghost_area_ratio"],
                dwg_ghost_percent=100.0 * row["dwg_ghost_area_ratio"],
            )
        )
    return "\n".join(lines) + "\n"


def latex_table(
    rows: list[dict[str, Any]],
    *,
    sgr_key: str,
    dwg_key: str,
    caption: str,
    label: str,
) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Dataset & DWG & SGR & DWG/SGR & Lower \\",
        r"\midrule",
    ]
    for row in rows:
        sgr = float(row[sgr_key])
        dwg = float(row[dwg_key])
        sgr_text = rf"\textbf{{{sgr:.4e}}}" if sgr <= dwg else f"{sgr:.4e}"
        dwg_text = rf"\textbf{{{dwg:.4e}}}" if dwg < sgr else f"{dwg:.4e}"
        ratio = dwg / sgr
        lower = "DWG" if dwg < sgr else "SGR"
        dataset = str(row["dataset"]).replace("_", r"\_")
        lines.append(
            f"{dataset} & {dwg_text} & {sgr_text} & {ratio:.3f} & {lower} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def latex_ghost_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (
            r"\caption{Ghost-component detection on the complete meshes "
            r"extracted from the input bounding box with 10\% padding per side.}"
        ),
        r"\label{tab:dwg-ghost}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{DWG} & \multicolumn{3}{c}{SGR} \\",
        r"Dataset & Comp. & Ghost & $G_{\rm area}$ (\%) & Comp. & Ghost & $G_{\rm area}$ (\%) \\",
        r"\midrule",
    ]
    for row in rows:
        dataset = str(row["dataset"]).replace("_", r"\_")
        lines.append(
            f"{dataset} & {row['dwg_component_count']} & "
            f"{row['dwg_ghost_component_count']} & "
            f"{100.0 * row['dwg_ghost_area_ratio']:.3f} & "
            f"{row['sgr_component_count']} & "
            f"{row['sgr_ghost_component_count']} & "
            f"{100.0 * row['sgr_ghost_area_ratio']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--sgr-extraction-root", required=True)
    parser.add_argument("--dwg-extraction-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-samples", type=int, default=1_000_000)
    parser.add_argument("--reconstruction-samples", type=int, default=1_000_000)
    return parser


def evaluate_method(
    *,
    method: str,
    configuration: Path,
    reference_json: Path,
    mesh: Path,
    asset_root: Path,
    test_samples: int,
    reconstruction_samples: int,
    scale: float,
    translation: tuple[float, float, float],
) -> dict[str, Any]:
    result = single.evaluate(
        SimpleNamespace(
            configuration=str(configuration),
            reference_json=str(reference_json),
            mesh=str(mesh),
            asset_root=str(asset_root),
            output="",
            seed_offset=None,
            test_samples=test_samples,
            reconstruction_samples=reconstruction_samples,
            method=method,
            mesh_scale=scale,
            mesh_translation=translation,
        )
    )
    return compact_result(result)


def comparison_row(
    example: Path,
    reference: dict[str, Any],
    *,
    asset_root: Path,
    sgr_root: Path,
    dwg_root: Path,
    test_samples: int,
    reconstruction_samples: int,
) -> dict[str, Any]:
    dataset = str(reference["dataset"])
    configuration = example / "config" / "training.json"
    reference_json = example / "dataset" / "reference.json"
    _, sgr_mesh = load_done(sgr_root, dataset, "SGR")
    dwg_done, dwg_mesh = load_done(dwg_root, dataset, "DWG")
    transform = dwg_done.get("input_transform")
    if not isinstance(transform, dict):
        raise ValueError(f"DWG transform missing for {dataset}")
    dwg_scale = float(transform["dwg_scale_raw"])
    dwg_translation = tuple(float(value) for value in transform["dwg_center_raw"])

    sgr = evaluate_method(
        method="SGR",
        configuration=configuration,
        reference_json=reference_json,
        mesh=sgr_mesh,
        asset_root=asset_root,
        test_samples=test_samples,
        reconstruction_samples=reconstruction_samples,
        scale=1.0,
        translation=(0.0, 0.0, 0.0),
    )
    dwg = evaluate_method(
        method="DWG",
        configuration=configuration,
        reference_json=reference_json,
        mesh=dwg_mesh,
        asset_root=asset_root,
        test_samples=test_samples,
        reconstruction_samples=reconstruction_samples,
        scale=dwg_scale,
        translation=dwg_translation,
    )
    row: dict[str, Any] = {
        "dataset": dataset,
        **{f"sgr_{key}": value for key, value in sgr.items()},
        **{f"dwg_{key}": value for key, value in dwg.items()},
        "rmse_winner": "SGR" if sgr["rmse"] <= dwg["rmse"] else "DWG",
        "chamfer_winner": (
            "SGR" if sgr["chamfer_l2"] <= dwg["chamfer_l2"] else "DWG"
        ),
        "rmse_ratio_dwg_over_sgr": dwg["rmse"] / sgr["rmse"],
        "chamfer_ratio_dwg_over_sgr": dwg["chamfer_l2"] / sgr["chamfer_l2"],
    }
    print(
        f"  SGR rmse={sgr['rmse']:.8e} chamfer={sgr['chamfer_l2']:.8e}; "
        f"DWG rmse={dwg['rmse']:.8e} chamfer={dwg['chamfer_l2']:.8e}",
        flush=True,
    )
    return row


def evaluate_examples(
    args: argparse.Namespace,
    repository_root: Path,
    asset_root: Path,
    sgr_root: Path,
    dwg_root: Path,
) -> list[dict[str, Any]]:
    examples = discover_examples(repository_root)
    rows: list[dict[str, Any]] = []
    for index, (example, reference) in enumerate(examples, start=1):
        dataset = str(reference["dataset"])
        print(f"[{index:02d}/27] {dataset}", flush=True)
        rows.append(
            comparison_row(
                example,
                reference,
                asset_root=asset_root,
                sgr_root=sgr_root,
                dwg_root=dwg_root,
                test_samples=int(args.test_samples),
                reconstruction_samples=int(args.reconstruction_samples),
            )
        )
    return rows


def summary_metadata(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema": "lowdim-unified-bbox10-comparison/v2",
        "datasets": len(rows),
        "grid_resolution": 360,
        "bbox_pad_fraction_each_side": 0.10,
        "component_policy": "all_components",
        "normalization": "released_training_data_mean_and_data_scale",
        "reference": "one_million_independent_source_mesh_samples",
        "reconstruction_samples": int(args.reconstruction_samples),
        "test_samples": int(args.test_samples),
        "ghost_spacing_reference": "complete_released_input_cloud",
        "sgr_rmse_wins": sum(row["rmse_winner"] == "SGR" for row in rows),
        "dwg_rmse_wins": sum(row["rmse_winner"] == "DWG" for row in rows),
        "sgr_chamfer_wins": sum(
            row["chamfer_winner"] == "SGR" for row in rows
        ),
        "dwg_chamfer_wins": sum(
            row["chamfer_winner"] == "DWG" for row in rows
        ),
        "sgr_both_metric_wins": sum(
            row["rmse_winner"] == "SGR" and row["chamfer_winner"] == "SGR"
            for row in rows
        ),
        "dwg_both_metric_wins": sum(
            row["rmse_winner"] == "DWG" and row["chamfer_winner"] == "DWG"
            for row in rows
        ),
        "mixed_metric_datasets": sum(
            row["rmse_winner"] != row["chamfer_winner"] for row in rows
        ),
        "median_dwg_over_sgr_rmse": statistics.median(
            row["rmse_ratio_dwg_over_sgr"] for row in rows
        ),
        "median_dwg_over_sgr_chamfer": statistics.median(
            row["chamfer_ratio_dwg_over_sgr"] for row in rows
        ),
        "sgr_datasets_with_ghosts": sum(
            row["sgr_ghost_component_count"] > 0 for row in rows
        ),
        "dwg_datasets_with_ghosts": sum(
            row["dwg_ghost_component_count"] > 0 for row in rows
        ),
    }


def write_comparison_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    write_json(output_dir / "metrics.json", {"metadata": metadata, "rows": rows})
    with (output_dir / "metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "table.md").write_text(markdown(rows), encoding="utf-8")
    (output_dir / "table_rmse.tex").write_text(
        latex_table(
            rows,
            sgr_key="sgr_rmse",
            dwg_key="dwg_rmse",
            caption=(
                "Point-to-surface RMSE on independent reference-mesh samples. "
                "Both methods use $360^3$ final extraction over their input "
                "bounding box with 10\\% padding per side, followed by the "
                "same training normalization (lower is better)."
            ),
            label="tab:dwg-rmse",
        ),
        encoding="utf-8",
    )
    (output_dir / "table_chamfer.tex").write_text(
        latex_table(
            rows,
            sgr_key="sgr_chamfer_l2",
            dwg_key="dwg_chamfer_l2",
            caption=(
                "Squared bidirectional Chamfer distance on independent "
                "reference-mesh samples. Both methods use $360^3$ final "
                "extraction over their input bounding box with 10\\% padding "
                "per side, followed by the same training normalization "
                "(lower is better)."
            ),
            label="tab:dwg-chamfer",
        ),
        encoding="utf-8",
    )
    (output_dir / "table_ghost.tex").write_text(
        latex_ghost_table(rows),
        encoding="utf-8",
    )


def main() -> None:
    args = build_arg_parser().parse_args()

    repository_root = Path(args.repository_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    sgr_root = Path(args.sgr_extraction_root).expanduser().resolve()
    dwg_root = Path(args.dwg_extraction_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows = evaluate_examples(args, repository_root, asset_root, sgr_root, dwg_root)
    metadata = summary_metadata(rows, args)
    write_comparison_outputs(output_dir, rows, metadata)
    print(f"complete output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
