#!/usr/bin/env python3
"""Package the retained low-dimensional results as a checksummed archive.

The archive contains checkpoints, reconstructions and evaluation records.
The input clouds and reference meshes are downloaded separately through Git
LFS under datasets/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


BUNDLE_SCHEMA = "sgr-retained-paper-results/v1"
RESULT_PATTERNS = (
    "example/lowdim/evaluation_assets.json",
    "example/lowdim/example_*/config/training.json",
    "example/lowdim/example_*/dataset/reference.json",
    "example/lowdim/example_*/result/summary.json",
    "example/lowdim/example_*/result/evaluation.json",
    "example/lowdim/example_*/result/implicit_function/best.pt",
    "example/lowdim/example_*/result/reconstruction/zero_mesh.npz",
    "example/lowdim/example_*/result/reconstruction/stats.json",
    "example/lowdim/summary/**/*",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover(root: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in RESULT_PATTERNS:
        files.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--output",
        default="",
        help="Default: <repository>/dist/score-guided-reconstruction-results.zip",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_output(args: argparse.Namespace, root: Path) -> Path:
    if args.output:
        return Path(args.output).expanduser().resolve()
    return root / "dist" / "score-guided-reconstruction-results.zip"


def validate_result_set(files: list[Path]) -> tuple[list[Path], list[Path]]:
    checkpoints = [path for path in files if path.name == "best.pt"]
    meshes = [path for path in files if path.name == "zero_mesh.npz"]
    if len(checkpoints) != 27 or len(meshes) != 27:
        raise RuntimeError(
            "A complete bundle requires 27 best.pt and 27 zero_mesh.npz files; "
            f"found {len(checkpoints)} and {len(meshes)}"
        )
    return checkpoints, meshes


def manifest_entries(root: Path, files: list[Path]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(root).as_posix()
        entries.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": sha256(path),
            }
        )
        print(f"[{index:03d}/{len(files):03d}] {relative}", flush=True)
    return entries


def build_manifest(
    entries: list[dict[str, Any]],
    checkpoint_count: int,
    mesh_count: int,
) -> dict[str, Any]:
    return {
        "schema": BUNDLE_SCHEMA,
        "purpose": (
            "Selected low-dimensional SGR checkpoints, complete 360^3 zero "
            "meshes, evaluation records, and aggregate tables."
        ),
        "contains_third_party_training_data": False,
        "checkpoints": checkpoint_count,
        "meshes": mesh_count,
        "files": len(entries),
        "bytes_uncompressed": sum(entry["bytes"] for entry in entries),
        "entries": entries,
    }


def write_archive(
    root: Path,
    output: Path,
    files: list[Path],
    manifest: dict[str, Any],
) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr(
            "paper-results-manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
    temporary.replace(output)
    digest = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n",
        encoding="utf-8",
    )
    return digest


def main() -> None:
    args = build_arg_parser().parse_args()

    root = Path(args.repository_root).expanduser().resolve()
    output = resolve_output(args, root)
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output}; pass --overwrite explicitly"
        )
    files = discover(root)
    checkpoints, meshes = validate_result_set(files)
    entries = manifest_entries(root, files)
    manifest = build_manifest(entries, len(checkpoints), len(meshes))
    digest = write_archive(
        root,
        output,
        files,
        manifest,
    )
    print(
        json.dumps(
            {
                "event": "done",
                "archive": str(output),
                "sha256": digest,
                "files": len(entries),
                "bytes": int(output.stat().st_size),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
