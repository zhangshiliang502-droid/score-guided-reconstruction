#!/usr/bin/env python3
"""Run the configuration-driven high-dimensional paper pipeline.

The driver does not reimplement training, reconstruction, or evaluation.  It
reads one or more released ``training.json`` files and invokes the public stage
entry points with the recorded paper parameters.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = ROOT / "example" / "highdim"
STAGE_ORDER = ("generate", "train", "reconstruct", "evaluate")


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    artifacts: tuple[Path, ...]


@dataclass(frozen=True)
class PipelinePaths:
    """Resolved inputs and outputs shared by the four reproduction stages."""

    run: Path
    point_cloud: Path
    data_metadata: Path
    checkpoint: Path
    reconstruction_dir: Path
    reconstruction_points: Path
    reconstruction_metadata: Path
    evaluation: Path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def resolve_experiment(configuration: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (configuration.parent.parent / path).resolve()


def positive_number(mapping: dict[str, Any], key: str) -> float:
    value = float(mapping[key])
    if value <= 0.0:
        raise ValueError(f"{key} must be positive")
    return value


def validate_dimensions_and_splits(
    configuration: Path,
    config: dict[str, Any],
) -> None:
    data = config["data"]
    energy = config["energy_training"]
    sgr = config["sgr_training"]

    if int(config["ambient_dim"]) <= 3:
        raise ValueError(f"{configuration}: ambient_dim must be greater than 3")
    if int(config["intrinsic_dim"]) != int(config["ambient_dim"]) - 1:
        raise ValueError(f"{configuration}: only codimension one is supported")
    if int(data["samples"]) != int(energy["train_size"]) + int(energy["val_size"]):
        raise ValueError(f"{configuration}: Energy split does not use data.samples")
    if int(sgr["train_size"]) + int(sgr["val_size"]) > int(data["samples"]):
        raise ValueError(f"{configuration}: SGR split exceeds data.samples")
    point_count = int(data["samples"])
    expected_energy_val = max(1, int(round(point_count * 0.10)))
    expected_phi_total = min(point_count, 250_000)
    expected_phi_val = max(1, int(round(expected_phi_total * 0.12)))
    expected_splits = {
        "energy_train": point_count - expected_energy_val,
        "energy_val": expected_energy_val,
        "sgr_train": expected_phi_total - expected_phi_val,
        "sgr_val": expected_phi_val,
    }
    observed_splits = {
        "energy_train": int(energy["train_size"]),
        "energy_val": int(energy["val_size"]),
        "sgr_train": int(sgr["train_size"]),
        "sgr_val": int(sgr["val_size"]),
    }
    if observed_splits != expected_splits:
        raise ValueError(
            f"{configuration}: configuration/main.py split mismatch: "
            f"expected={expected_splits}, observed={observed_splits}"
        )
    if int(energy["batch_size"]) != int(sgr["batch_size"]):
        raise ValueError(
            f"{configuration}: main.py requires equal Energy and SGR batch sizes"
        )
    if int(energy["eval_batch_size"]) != int(energy["batch_size"]):
        raise ValueError(
            f"{configuration}: main.py uses the Energy batch size for evaluation"
        )
    if int(sgr["seed"]) != int(energy["seed"]) + 1:
        raise ValueError(f"{configuration}: SGR seed must equal Energy seed + 1")


def validate_fixed_model_contract(
    configuration: Path,
    config: dict[str, Any],
) -> None:
    """Ensure configuration choices match constants fixed in the public code."""
    energy = config["energy_training"]
    sgr = config["sgr_training"]
    evaluation = config["evaluation"]

    if float(energy["sigma"]) != 0.01:
        raise ValueError(f"{configuration}: public Energy pipeline fixes sigma=0.01")
    if int(energy["hidden_dim"]) != 256 or int(energy["num_blocks"]) != 4:
        raise ValueError(f"{configuration}: Energy architecture/code mismatch")
    if str(sgr["model"]) != "SIREN 128x4 omega0=20":
        raise ValueError(f"{configuration}: SIREN architecture/code mismatch")
    if (
        float(sgr["local_shell_std_sigma"]) != 1.0
        or float(sgr["lambda_score_pull"]) != 1000.0
        or str(sgr["local_residual_normalization"]) != "none"
    ):
        raise ValueError(f"{configuration}: fixed SGR loss/code mismatch")
    if str(evaluation["reference_mode"]) != "analytic":
        raise ValueError(f"{configuration}: paper reference mode must be analytic")
    if str(evaluation["metric_space"]) != "normalized":
        raise ValueError(f"{configuration}: paper metrics must be normalized")


def validate_positive_parameters(config: dict[str, Any]) -> None:
    """Reject non-positive iteration counts, batch sizes, steps, and tolerances."""
    energy = config["energy_training"]
    sgr = config["sgr_training"]
    reconstruction = config["reconstruction"]
    evaluation = config["evaluation"]
    for mapping, keys in (
        (
            energy,
            (
                "iterations",
                "batch_size",
                "learning_rate",
                "eval_batch_size",
            ),
        ),
        (
            sgr,
            (
                "iterations",
                "batch_size",
                "global_batch_size",
                "eval_batches",
                "learning_rate",
                "local_shell_center_sigma",
                "lambda_dir",
            ),
        ),
        (
            reconstruction,
            (
                "zero_points",
                "seed_count",
                "projection_batch_size",
                "projection_iterations",
                "projection_max_step",
                "projection_residual",
            ),
        ),
        (
            evaluation,
            (
                "evaluation_points",
                "spacing_reference_points",
                "analytic_projection_iterations",
            ),
        ),
    ):
        for key in keys:
            positive_number(mapping, key)


def validate_configuration(
    configuration: Path,
    config: dict[str, Any],
) -> None:
    validate_dimensions_and_splits(configuration, config)
    validate_fixed_model_contract(configuration, config)
    validate_positive_parameters(config)


def pipeline_paths(
    configuration: Path,
    output_root: Path,
    config: dict[str, Any],
) -> PipelinePaths:
    run = (output_root / str(config["dataset"])).resolve()
    point_cloud = resolve_experiment(configuration, config["data"]["path"])
    reconstruction_dir = run / "reconstruction"
    return PipelinePaths(
        run=run,
        point_cloud=point_cloud,
        data_metadata=resolve_experiment(
            configuration,
            config["data"]["metadata_path"],
        ),
        checkpoint=run / "implicit_function" / "best.pt",
        reconstruction_dir=reconstruction_dir,
        reconstruction_points=reconstruction_dir / "zero_points_physical.npy",
        reconstruction_metadata=reconstruction_dir / "reconstruction.json",
        evaluation=run / "evaluation.json",
    )


def generation_stage(
    configuration: Path,
    paths: PipelinePaths,
    *,
    python: Path,
    overwrite_data: bool,
) -> Stage:
    command = [
        str(python),
        str(ROOT / "02_highdim_sgr" / "code" / "generate_paper_data.py"),
        "--configuration",
        str(configuration),
    ]
    if overwrite_data:
        command.append("--overwrite")
    return Stage("generate", command, (paths.point_cloud, paths.data_metadata))


def training_stage(
    config: dict[str, Any],
    paths: PipelinePaths,
    *,
    python: Path,
    device: str,
) -> Stage:
    energy = config["energy_training"]
    sgr = config["sgr_training"]
    command = [
        str(python),
        str(ROOT / "02_highdim_sgr" / "code" / "main.py"),
        "--point-cloud",
        str(paths.point_cloud),
        "--output-dir",
        str(paths.run),
        "--energy-iterations",
        str(int(energy["iterations"])),
        "--phi-iterations",
        str(int(sgr["iterations"])),
        "--batch-size",
        str(int(energy["batch_size"])),
        "--global-batch-size",
        str(int(sgr["global_batch_size"])),
        "--eval-batches",
        str(int(sgr["eval_batches"])),
        "--energy-learning-rate",
        str(float(energy["learning_rate"])),
        "--phi-learning-rate",
        str(float(sgr["learning_rate"])),
        "--local-shell-center-sigma",
        str(float(sgr["local_shell_center_sigma"])),
        "--lambda-dir",
        str(float(sgr["lambda_dir"])),
        "--seed",
        str(int(energy["seed"])),
        "--device",
        device,
    ]
    return Stage(
        "train",
        command,
        (paths.checkpoint, paths.run / "main_output.json"),
    )


def reconstruction_stage(
    config: dict[str, Any],
    paths: PipelinePaths,
    *,
    python: Path,
    device: str,
) -> Stage:
    reconstruction = config["reconstruction"]
    command = [
        str(python),
        str(ROOT / "02_highdim_sgr" / "code" / "reconstruct.py"),
        "--point-cloud",
        str(paths.point_cloud),
        "--checkpoint",
        str(paths.checkpoint),
        "--output-dir",
        str(paths.reconstruction_dir),
        "--seed",
        str(int(reconstruction["seed"])),
        "--seed-count",
        str(int(reconstruction["seed_count"])),
        "--zero-points",
        str(int(reconstruction["zero_points"])),
        "--projection-batch-size",
        str(int(reconstruction["projection_batch_size"])),
        "--projection-iters",
        str(int(reconstruction["projection_iterations"])),
        "--projection-max-step",
        str(float(reconstruction["projection_max_step"])),
        "--projection-residual",
        str(float(reconstruction["projection_residual"])),
        "--device",
        device,
    ]
    return Stage(
        "reconstruct",
        command,
        (paths.reconstruction_points, paths.reconstruction_metadata),
    )


def evaluation_stage(
    configuration: Path,
    config: dict[str, Any],
    paths: PipelinePaths,
    *,
    python: Path,
    device: str,
) -> Stage:
    reconstruction = config["reconstruction"]
    evaluation = config["evaluation"]
    command = [
        str(python),
        str(ROOT / "03_evaluation" / "evaluate_highdim_reconstruction.py"),
        "--configuration",
        str(configuration),
        "--point-cloud",
        str(paths.point_cloud),
        "--checkpoint",
        str(paths.checkpoint),
        "--reconstruction",
        str(paths.reconstruction_points),
        "--reconstruction-metadata",
        str(paths.reconstruction_metadata),
        "--output",
        str(paths.evaluation),
        "--reference-mode",
        str(evaluation["reference_mode"]),
        "--evaluation-points",
        str(int(evaluation["evaluation_points"])),
        "--spacing-reference-points",
        str(int(evaluation["spacing_reference_points"])),
        "--projection-batch-size",
        str(int(reconstruction["projection_batch_size"])),
        "--projection-iters",
        str(int(reconstruction["projection_iterations"])),
        "--projection-max-step",
        str(float(reconstruction["projection_max_step"])),
        "--projection-residual",
        str(float(reconstruction["projection_residual"])),
        "--analytic-projection-iters",
        str(int(evaluation["analytic_projection_iterations"])),
        "--device",
        device,
    ]
    return Stage("evaluate", command, (paths.evaluation,))


def build_stages(
    configuration: Path,
    output_root: Path,
    *,
    python: Path,
    device: str,
    overwrite_data: bool,
) -> list[Stage]:
    configuration = configuration.resolve()
    config = load_json(configuration)
    validate_configuration(configuration, config)
    paths = pipeline_paths(configuration, output_root, config)
    return [
        generation_stage(
            configuration,
            paths,
            python=python,
            overwrite_data=overwrite_data,
        ),
        training_stage(config, paths, python=python, device=device),
        reconstruction_stage(config, paths, python=python, device=device),
        evaluation_stage(
            configuration,
            config,
            paths,
            python=python,
            device=device,
        ),
    ]


def discover_examples() -> dict[str, Path]:
    return {
        path.parent.parent.name: path
        for path in sorted(EXAMPLE_ROOT.glob("example_*/config/training.json"))
    }


def selected_examples(values: list[str], available: dict[str, Path]) -> list[str]:
    if "all" in values:
        if len(values) != 1:
            raise ValueError("'all' cannot be combined with named examples")
        return sorted(available, key=lambda value: int(value[8:-1]))
    unknown = sorted(set(values) - set(available))
    if unknown:
        raise ValueError(
            f"Unknown examples {unknown}; choices are {sorted(available)} or all"
        )
    return values


def selected_stages(values: list[str]) -> set[str]:
    if "all" in values:
        if len(values) != 1:
            raise ValueError("'all' cannot be combined with named stages")
        return set(STAGE_ORDER)
    return set(values)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples",
        nargs="+",
        required=True,
        help="Example names such as example_4d example_6d, or all.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=(*STAGE_ORDER, "all"),
        default=["all"],
    )
    parser.add_argument("--output-root", default="runs/paper_highdim")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run stages even when all declared output artifacts exist.",
    )
    parser.add_argument(
        "--overwrite-data",
        action="store_true",
        help="Pass --overwrite to the deterministic analytic data generator.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved commands without creating files or running stages.",
    )
    return parser


def reproduction_plan(example_name: str, configuration: Path, output_root: Path) -> dict[str, Any]:
    return {
        "schema": "sgr-highdim-reproduction-run/v1",
        "example": example_name,
        "configuration": str(configuration),
        "output": str(output_root / example_name),
        "stages": {},
    }


def stage_status(stage: Stage, *, force: bool) -> tuple[bool, str]:
    complete = all(path.is_file() for path in stage.artifacts)
    status = "would_skip_complete" if complete and not force else "would_run"
    return complete, status


def print_stage_plan(example_name: str, stage: Stage, status: str) -> None:
    print(
        json.dumps(
            {
                "example": example_name,
                "stage": stage.name,
                "status": status,
                "command": stage.command,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_stage(
    stage: Stage,
    plan: dict[str, Any],
    state_path: Path,
) -> None:
    started = time.time()
    plan["stages"][stage.name] = {
        "status": "running",
        "command": stage.command,
    }
    atomic_json(state_path, plan)
    try:
        subprocess.run(stage.command, cwd=ROOT, check=True)
        missing = [str(path) for path in stage.artifacts if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"stage {stage.name} did not create expected artifacts: {missing}"
            )
        plan["stages"][stage.name] = {
            "status": "complete",
            "elapsed_seconds": time.time() - started,
            "artifacts": [str(path) for path in stage.artifacts],
        }
        atomic_json(state_path, plan)
    except Exception as error:
        plan["stages"][stage.name] = {
            "status": "failed",
            "error": str(error),
        }
        atomic_json(state_path, plan)
        raise


def run_example(
    example_name: str,
    configuration: Path,
    output_root: Path,
    enabled_stages: set[str],
    args: argparse.Namespace,
) -> None:
    stages = build_stages(
        configuration,
        output_root,
        python=Path(args.python).expanduser().resolve(),
        device=str(args.device),
        overwrite_data=bool(args.overwrite_data),
    )
    plan = reproduction_plan(example_name, configuration, output_root)
    state_path = output_root / example_name / "reproduction_state.json"
    for stage in stages:
        if stage.name not in enabled_stages:
            continue
        complete, status = stage_status(stage, force=bool(args.force))
        print_stage_plan(example_name, stage, status)
        if args.dry_run:
            continue
        if complete and not args.force:
            plan["stages"][stage.name] = {
                "status": "skipped_complete",
                "artifacts": [str(path) for path in stage.artifacts],
            }
            atomic_json(state_path, plan)
            continue
        run_stage(stage, plan, state_path)


def main() -> None:
    args = build_arg_parser().parse_args()

    available = discover_examples()
    examples = selected_examples(list(args.examples), available)
    enabled_stages = selected_stages(list(args.stages))
    output_root = Path(args.output_root).expanduser().resolve()

    for example_name in examples:
        run_example(
            example_name,
            available[example_name],
            output_root,
            enabled_stages,
            args,
        )


if __name__ == "__main__":
    main()
