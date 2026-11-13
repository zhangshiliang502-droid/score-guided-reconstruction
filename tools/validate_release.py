#!/usr/bin/env python3
"""Validate the public source tree and optional paper assets.

Profiles:

``source``
    Source files, JSON schemas, example counts, and portable paths.
``results``
    ``source`` plus the 27 retained low-dimensional checkpoints and meshes.
``lowdim-evaluation``
    ``results`` plus the 54 checksummed input clouds and reference meshes.
``highdim-data``
    ``source`` plus the six locally generated analytic training arrays.
``paper``
    ``lowdim-evaluation`` plus ``highdim-data``.

The command uses only the Python standard library so it can run before the
scientific environment is installed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


EXPECTED_LOWDIM = 27
EXPECTED_HIGHDIM = 6
FORBIDDEN_BINARY_PATH_MARKERS = (
    b"/" + b"Users/",
    b"/" + b"home/",
    b"/var/" + b"folders/",
    b"C:\\" + b"Users\\",
)
IGNORED_SOURCE_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "runs",
    "outputs",
}


class Report:
    def __init__(self) -> None:
        self.checks: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def check(self, message: str) -> None:
        self.checks.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def npy_header(stream: Any) -> dict[str, Any]:
    magic = stream.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("invalid NPY magic")
    major, _minor = struct.unpack("BB", stream.read(2))
    if major == 1:
        header_size = struct.unpack("<H", stream.read(2))[0]
    else:
        header_size = struct.unpack("<I", stream.read(4))[0]
    header = ast.literal_eval(stream.read(header_size).decode("latin1").strip())
    if not isinstance(header, dict):
        raise ValueError("invalid NPY header")
    return header


def npz_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue
            with archive.open(name) as stream:
                header = npy_header(stream)
            shapes[Path(name).stem] = tuple(int(value) for value in header["shape"])
    return shapes


def resolve_asset(asset_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (asset_root / path).resolve()


def resolve_experiment(configuration: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (configuration.parent.parent / path).resolve()


def check_required_source(root: Path, report: Report) -> None:
    required = [
        "README.md",
        "LICENSE",
        "ARTIFACTS.md",
        "CITATION.cff",
        "requirements.txt",
        "environment.yml",
        ".gitignore",
        ".gitattributes",
        "01_lowdim_sgr/code/main.py",
        "01_lowdim_sgr/code/plan_gaussian_knn.py",
        "01_lowdim_sgr/code/construct_score.py",
        "01_lowdim_sgr/code/train_phi.py",
        "01_lowdim_sgr/code/reconstruct.py",
        "02_highdim_sgr/code/main.py",
        "02_highdim_sgr/code/construct_score.py",
        "02_highdim_sgr/code/train_phi.py",
        "02_highdim_sgr/code/reconstruct.py",
        "02_highdim_sgr/code/generate_paper_data.py",
        "03_evaluation/evaluate_lowdim_reconstruction.py",
        "03_evaluation/evaluate_highdim_reconstruction.py",
        "example/lowdim/evaluation_assets.json",
        "example/lowdim/summary/dwg_configurations/configurations.csv",
        "example/lowdim/summary/dwg_configurations/README.md",
        "docs/REPRODUCIBILITY.md",
        "docs/DATA_LAYOUT.md",
        "docs/VALIDATION.md",
        "docs/TESTED_ENVIRONMENTS.md",
        "tools/build_lowdim_asset_manifest.py",
        "tools/reproduce_highdim.py",
        "tools/smoke_test_lowdim.py",
        ".github/workflows/ci.yml",
    ]
    missing = [value for value in required if not (root / value).is_file()]
    if missing:
        report.error(f"missing required source files: {missing}")
    else:
        report.check(f"required source files: {len(required)}")


def parse_sources(root: Path, report: Report) -> None:
    def included(path: Path) -> bool:
        return not any(
            part in IGNORED_SOURCE_DIRECTORIES
            for part in path.relative_to(root).parts
        )

    python_files = sorted(path for path in root.rglob("*.py") if included(path))
    json_files = sorted(path for path in root.rglob("*.json") if included(path))
    jsonl_files = sorted(path for path in root.rglob("*.jsonl") if included(path))
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as error:
            report.error(f"Python syntax: {path.relative_to(root)}: {error}")
    for path in json_files:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            report.error(f"JSON syntax: {path.relative_to(root)}: {error}")
    for path in jsonl_files:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    json.loads(line)
        except Exception as error:
            report.error(f"JSONL syntax: {path.relative_to(root)}: {error}")
    report.check(f"parsed Python files: {len(python_files)}")
    report.check(f"parsed JSON files: {len(json_files)}")
    report.check(f"parsed JSONL files: {len(jsonl_files)}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_lfs_pointer(path: Path) -> bool:
    try:
        return path.read_bytes()[:128].startswith(
            b"version https://git-lfs.github.com/spec/v1\n"
        )
    except OSError:
        return False


def forbidden_binary_path_markers(path: Path) -> list[str]:
    payload = path.read_bytes()
    return [
        marker.decode("utf-8", errors="replace")
        for marker in FORBIDDEN_BINARY_PATH_MARKERS
        if marker in payload
    ]


def expected_lowdim_assets(root: Path) -> dict[tuple[str, str], str]:
    expected: dict[tuple[str, str], str] = {}
    for configuration_path in sorted(
        (root / "example" / "lowdim").glob(
            "example_*/config/training.json"
        )
    ):
        example = configuration_path.parent.parent
        config = load_json(configuration_path)
        reference = load_json(example / "dataset" / "reference.json")
        dataset = str(config["dataset"])
        expected[(dataset, "training_point_cloud")] = str(
            config["training_parameters"]["point_cloud"]
        )
        expected[(dataset, "independent_reference_mesh")] = str(
            reference["asset_root_path"]
        )
    return expected


def validate_manifest_sources(value: Any) -> tuple[dict[str, Any], list[str]]:
    """Validate provenance records shared by the low-dimensional assets."""
    if not isinstance(value, dict) or not value:
        return {}, ["manifest has no sources object"]

    sources = value
    invalid: list[str] = []
    required_fields = ("url", "acquisition", "distribution_status", "note")
    for name, source in sources.items():
        if not isinstance(source, dict):
            invalid.append(f"invalid source record: {name}")
            continue
        for field in required_fields:
            if not str(source.get(field, "")).strip():
                invalid.append(f"source {name} has no {field}")
    return sources, invalid


def validate_manifest_entries(
    entries: list[Any],
    sources: dict[str, Any],
) -> tuple[dict[tuple[str, str], str], list[str]]:
    """Return the dataset-role path map and any structural errors."""
    actual: dict[tuple[str, str], str] = {}
    invalid: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            invalid.append("non-object asset entry")
            continue

        key = (str(entry.get("dataset", "")), str(entry.get("role", "")))
        value = str(entry.get("path", ""))
        source_group = str(entry.get("source_group", ""))
        if key in actual:
            invalid.append(f"duplicate entry: {key}")
        actual[key] = value

        path = Path(value)
        if not value or path.is_absolute() or ".." in path.parts:
            invalid.append(f"non-portable path: {key}: {value}")
        digest = str(entry.get("sha256", ""))
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            invalid.append(f"invalid SHA-256: {key}")
        if int(entry.get("bytes", 0)) <= 0:
            invalid.append(f"invalid byte count: {key}")
        if source_group not in sources:
            invalid.append(f"unknown source group: {key}: {source_group}")
    return actual, invalid


def manifest_config_mismatch(
    actual: dict[tuple[str, str], str],
    expected: dict[tuple[str, str], str],
) -> str:
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(
        key
        for key in set(actual) & set(expected)
        if actual[key] != expected[key]
    )
    return (
        f"manifest/config mismatch: missing={missing}, extra={extra}, "
        f"changed={changed}"
    )


def verify_external_asset_identities(
    entries: list[dict[str, Any]],
    asset_root: Path,
    report: Report,
) -> None:
    """Verify the size and SHA-256 identity of each evaluation input."""
    mismatches: list[str] = []
    verified = 0
    for entry in entries:
        path = resolve_asset(asset_root, str(entry["path"]))
        if not path.is_file():
            continue
        label = f"{entry['dataset']} {entry['role']}"
        if path.stat().st_size != int(entry["bytes"]):
            mismatches.append(f"{label}: byte count: {path}")
            continue
        if sha256(path) != str(entry["sha256"]):
            mismatches.append(f"{label}: SHA-256: {path}")
            continue
        verified += 1

    if mismatches:
        report.error(
            "low-dimensional external asset identity mismatch:\n  "
            + "\n  ".join(mismatches)
        )
    else:
        report.check(f"low-dimensional external asset SHA-256 values: {verified}")


def check_lowdim_asset_manifest(
    root: Path,
    report: Report,
    *,
    asset_root: Path | None,
) -> None:
    manifest_path = root / "example" / "lowdim" / "evaluation_assets.json"
    try:
        manifest = load_json(manifest_path)
        if str(manifest.get("schema")) != "sgr-lowdim-evaluation-assets/v1":
            report.error("unexpected low-dimensional evaluation asset schema")
            return
        entries = manifest.get("assets")
        if not isinstance(entries, list):
            report.error("low-dimensional evaluation asset manifest has no assets list")
            return

        expected = expected_lowdim_assets(root)
        sources, invalid = validate_manifest_sources(manifest.get("sources"))
        actual, entry_errors = validate_manifest_entries(entries, sources)
        invalid.extend(entry_errors)
        if int(manifest.get("asset_count", -1)) != len(entries):
            invalid.append("asset_count does not match assets")
        if int(manifest.get("total_bytes", -1)) != sum(
            int(entry["bytes"])
            for entry in entries
            if isinstance(entry, dict)
        ):
            invalid.append("total_bytes does not match assets")
        if actual != expected:
            invalid.append(manifest_config_mismatch(actual, expected))
        if invalid:
            report.error(
                "invalid low-dimensional evaluation asset manifest:\n  "
                + "\n  ".join(invalid)
            )
            return
        report.check(f"low-dimensional evaluation asset manifest: {len(entries)}")

        if asset_root is not None:
            verify_external_asset_identities(entries, asset_root, report)
    except Exception as error:
        report.error(f"low-dimensional evaluation asset manifest: {error}")


def load_lowdim_records(
    example: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Load the five records that jointly describe a retained 3D result."""
    config = load_json(example / "config" / "training.json")
    reference = load_json(example / "dataset" / "reference.json")
    summary = load_json(example / "result" / "summary.json")
    evaluation = load_json(example / "result" / "evaluation.json")
    stats = load_json(example / "result" / "reconstruction" / "stats.json")
    return config, reference, summary, evaluation, stats


def consistent_lowdim_dataset(
    example: Path,
    records: tuple[dict[str, Any], ...],
    report: Report,
) -> str | None:
    names = {str(record["dataset"]) for record in records[:4]}
    if len(names) != 1:
        report.error(f"dataset name mismatch: {example.name}: {sorted(names)}")
        return None
    return names.pop()


def check_lowdim_protocol(
    dataset: str,
    config: dict[str, Any],
    stats: dict[str, Any],
    report: Report,
) -> None:
    """Check the frozen 360^3, 10%-padding, all-component protocol."""
    extraction = config["extraction_parameters"]
    expected_extraction = (
        int(extraction["grid_resolution"]) == 360
        and float(extraction["bbox_pad_fraction_each_side"]) == 0.10
        and str(extraction["component_policy"]) == "all_components"
    )
    if not expected_extraction:
        report.error(f"non-formal extraction protocol: {dataset}")

    retained_protocol = (
        int(stats["grid_n"]) == 360
        and float(stats["bbox_pad_fraction_each_side"]) == 0.10
        and str(stats["component_policy"]) == "all_components"
    )
    if not retained_protocol:
        report.error(f"non-formal retained reconstruction: {dataset}")


def check_retained_checkpoint(path: Path, dataset: str, report: Report) -> bool:
    """Return whether a present checkpoint contains only portable metadata."""
    if not path.is_file() or path.stat().st_size <= 0:
        report.error(f"missing retained checkpoint: {dataset}")
        return False
    if is_lfs_pointer(path):
        report.error(f"retained checkpoint is an unpulled Git LFS pointer: {dataset}")
        return False
    markers = forbidden_binary_path_markers(path)
    if markers:
        report.error(
            f"retained checkpoint contains non-portable path markers: "
            f"{dataset}: {markers}"
        )
        return False
    return True


def check_retained_mesh(
    path: Path,
    dataset: str,
    stats: dict[str, Any],
    evaluation: dict[str, Any],
    report: Report,
) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        report.error(f"missing retained mesh: {dataset}")
        return
    if is_lfs_pointer(path):
        report.error(f"retained mesh is an unpulled Git LFS pointer: {dataset}")
        return

    shapes = npz_shapes(path)
    vertices = shapes.get("vertices", ())
    faces = shapes.get("faces", ())
    if len(vertices) != 2 or vertices[1] != 3 or len(faces) != 2 or faces[1] != 3:
        report.error(f"invalid retained mesh arrays: {dataset}")
        return
    counts_match = (
        vertices[0] == int(stats["vertices"])
        and faces[0] == int(stats["faces"])
        and vertices[0] == int(evaluation["vertices"])
        and faces[0] == int(evaluation["faces"])
    )
    if not counts_match:
        report.error(f"mesh/statistics count mismatch: {dataset}")


def missing_lowdim_assets(
    asset_root: Path,
    dataset: str,
    config: dict[str, Any],
    reference: dict[str, Any],
) -> list[str]:
    required = (
        (
            "point cloud",
            resolve_asset(asset_root, config["training_parameters"]["point_cloud"]),
        ),
        ("reference", resolve_asset(asset_root, reference["asset_root_path"])),
    )
    return [f"{dataset}: {role}: {path}" for role, path in required if not path.is_file()]


def check_lowdim_examples(
    root: Path,
    report: Report,
    *,
    require_results: bool,
    asset_root: Path | None,
) -> None:
    examples = sorted((root / "example" / "lowdim").glob("example_*"))
    if len(examples) != EXPECTED_LOWDIM:
        report.error(
            f"expected {EXPECTED_LOWDIM} low-dimensional examples, found {len(examples)}"
        )
    datasets: set[str] = set()
    external_missing: list[str] = []
    portable_checkpoints = 0
    for example in examples:
        try:
            records = load_lowdim_records(example)
            config, reference, _summary, evaluation, stats = records
            dataset = consistent_lowdim_dataset(example, records, report)
            if dataset is None:
                continue
            if dataset in datasets:
                report.error(f"duplicate low-dimensional dataset: {dataset}")
            datasets.add(dataset)
            check_lowdim_protocol(dataset, config, stats, report)

            if require_results:
                checkpoint = example / "result" / "implicit_function" / "best.pt"
                mesh = example / "result" / "reconstruction" / "zero_mesh.npz"
                portable_checkpoints += int(
                    check_retained_checkpoint(checkpoint, dataset, report)
                )
                check_retained_mesh(mesh, dataset, stats, evaluation, report)

            if asset_root is not None:
                external_missing.extend(
                    missing_lowdim_assets(asset_root, dataset, config, reference)
                )
        except Exception as error:
            report.error(f"low-dimensional example {example.name}: {error}")
    if external_missing:
        report.error(
            "missing low-dimensional external assets:\n  "
            + "\n  ".join(external_missing)
        )
    report.check(f"low-dimensional examples: {len(examples)}")
    if require_results:
        report.check(
            f"portable retained checkpoint metadata: {portable_checkpoints}"
        )


ENERGY_PROTOCOL_FIELDS = (
    "iterations",
    "batch_size",
    "eval_batch_size",
    "learning_rate",
    "seed",
)
SGR_PROTOCOL_FIELDS = (
    "iterations",
    "batch_size",
    "global_batch_size",
    "eval_batches",
    "learning_rate",
    "local_shell_center_sigma",
    "lambda_dir",
    "seed",
)


def missing_training_protocol(config: dict[str, Any]) -> list[str]:
    energy = config.get("energy_training", {})
    sgr = config.get("sgr_training", {})
    return [
        f"energy_training.{key}" for key in ENERGY_PROTOCOL_FIELDS if key not in energy
    ] + [f"sgr_training.{key}" for key in SGR_PROTOCOL_FIELDS if key not in sgr]


def check_analytic_definition(
    example: Path,
    configuration: Path,
    analytic: dict[str, Any],
    report: Report,
) -> None:
    analytic_path = resolve_experiment(configuration, analytic["module"])
    tree = ast.parse(analytic_path.read_text(encoding="utf-8"))
    classes = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    if str(analytic["class"]) not in classes:
        report.error(f"analytic class missing: {example.name}: {analytic['class']}")


def missing_highdim_data(
    example: Path,
    configuration: Path,
    data: dict[str, Any],
) -> list[str]:
    paths = (
        resolve_experiment(configuration, data["path"]),
        resolve_experiment(configuration, data["metadata_path"]),
    )
    return [f"{example.name}: {path}" for path in paths if not path.is_file()]


def check_highdim_example(
    example: Path,
    report: Report,
    *,
    require_data: bool,
) -> tuple[int, list[str]]:
    configuration = example / "config" / "training.json"
    config = load_json(configuration)
    summary = load_json(example / "result" / "summary.json")
    dimension = int(config["ambient_dim"])

    if str(config["dataset"]) != str(summary["dataset"]):
        report.error(f"high-dimensional dataset mismatch: {example.name}")
    if int(config["intrinsic_dim"]) != dimension - 1:
        report.error(f"high-dimensional codimension is not one: {example.name}")

    data = config.get("data", {})
    for key in ("path", "metadata_path", "generator"):
        if not str(data.get(key, "")).strip():
            report.error(f"{example.name} does not define data.{key}")
    check_analytic_definition(
        example,
        configuration,
        config.get("analytic_definition", {}),
        report,
    )
    generator_path = resolve_experiment(configuration, data["generator"])
    if not generator_path.is_file():
        report.error(f"generator missing: {example.name}: {generator_path}")

    missing_protocol = missing_training_protocol(config)
    if missing_protocol:
        report.error(
            f"incomplete high-dimensional training protocol: "
            f"{example.name}: {missing_protocol}"
        )
    missing_data = (
        missing_highdim_data(example, configuration, data) if require_data else []
    )
    return dimension, missing_data


def check_highdim_examples(
    root: Path,
    report: Report,
    *,
    require_data: bool,
) -> None:
    examples = sorted((root / "example" / "highdim").glob("example_*"))
    if len(examples) != EXPECTED_HIGHDIM:
        report.error(
            f"expected {EXPECTED_HIGHDIM} high-dimensional examples, found {len(examples)}"
        )
    dimensions: set[int] = set()
    missing_data: list[str] = []
    for example in examples:
        try:
            dimension, example_missing = check_highdim_example(
                example,
                report,
                require_data=require_data,
            )
            dimensions.add(dimension)
            missing_data.extend(example_missing)
        except Exception as error:
            report.error(f"high-dimensional example {example.name}: {error}")
    if dimensions != {4, 6, 8, 9, 10, 12}:
        report.error(f"unexpected high-dimensional dimensions: {sorted(dimensions)}")
    if missing_data:
        report.error(
            "missing generated high-dimensional datasets; run "
            "'python tools/reproduce_highdim.py --examples all "
            "--stages generate' from the repository root:\n  "
            + "\n  ".join(missing_data)
        )
    report.check(f"high-dimensional examples: {len(examples)}")


def check_lfs_policy(root: Path, report: Report) -> None:
    attributes = (root / ".gitattributes").read_text(encoding="utf-8")
    ignore = (root / ".gitignore").read_text(encoding="utf-8")
    required_attributes = (
        "example/lowdim/example_*/result/implicit_function/best.pt filter=lfs",
        "example/lowdim/example_*/result/reconstruction/zero_mesh.npz filter=lfs",
    )
    required_exceptions = (
        "!example/lowdim/example_*/result/implicit_function/best.pt",
        "!example/lowdim/example_*/result/reconstruction/zero_mesh.npz",
    )
    for value in required_attributes:
        if value not in attributes:
            report.error(f"missing Git LFS rule: {value}")
    for value in required_exceptions:
        if value not in ignore:
            report.error(f"missing .gitignore retained-result exception: {value}")
    if shutil.which("git-lfs") is None:
        report.warn(
            "git-lfs is not installed; install it before staging retained .pt/.npz files"
        )
    else:
        report.check("git-lfs executable is available")


def check_git(root: Path, report: Report) -> None:
    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    inside = run("rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0:
        report.error("release root is not a Git working tree")
        return
    head = run("rev-parse", "--verify", "HEAD")
    if head.returncode != 0:
        report.error("repository has no commit; create and inspect the initial commit")
    status = run("status", "--porcelain")
    if status.stdout.strip():
        report.error("Git working tree/index is not clean")
    else:
        report.check("Git working tree/index is clean")


def emit(report: Report, *, profile: str, as_json: bool) -> int:
    payload = {
        "profile": profile,
        "status": "ok" if not report.errors else "failed",
        "checks": report.checks,
        "warnings": report.warnings,
        "errors": report.errors,
    }
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"release validation: {payload['status']} (profile={profile})")
        for value in report.checks:
            print(f"  OK      {value}")
        for value in report.warnings:
            print(f"  WARNING {value}")
        for value in report.errors:
            print(f"  ERROR   {value}")
    return 0 if not report.errors else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--profile",
        choices=(
            "source",
            "results",
            "lowdim-evaluation",
            "highdim-data",
            "paper",
        ),
        default="source",
    )
    parser.add_argument(
        "--asset-root",
        default="",
        help=(
            "Required by --profile lowdim-evaluation and paper; root resolving "
            "low-dimensional paths."
        ),
    )
    parser.add_argument("--check-git", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(args.repository_root).expanduser().resolve()
    report = Report()
    check_required_source(root, report)
    parse_sources(root, report)
    require_results = args.profile in {
        "results",
        "lowdim-evaluation",
        "paper",
    }
    asset_root: Path | None = None
    if args.profile in {"lowdim-evaluation", "paper"}:
        if not args.asset_root:
            report.error(f"--asset-root is required by --profile {args.profile}")
        else:
            asset_root = Path(args.asset_root).expanduser().resolve()
    check_lowdim_examples(
        root,
        report,
        require_results=require_results,
        asset_root=asset_root,
    )
    check_lowdim_asset_manifest(
        root,
        report,
        asset_root=asset_root,
    )
    check_highdim_examples(
        root,
        report,
        require_data=args.profile in {"highdim-data", "paper"},
    )
    check_lfs_policy(root, report)
    if args.check_git:
        check_git(root, report)
    raise SystemExit(emit(report, profile=args.profile, as_json=bool(args.json)))


if __name__ == "__main__":
    main()
