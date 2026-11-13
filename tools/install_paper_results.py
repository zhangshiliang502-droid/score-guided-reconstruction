#!/usr/bin/env python3
"""Verify and install a retained-results release archive into a source clone."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


EXPECTED_SCHEMA = "sgr-retained-paper-results/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe archive path: {value!r}")
    return Path(*path.parts)


def load_manifest(archive: zipfile.ZipFile) -> dict[str, Any]:
    value = json.loads(
        archive.read("paper-results-manifest.json").decode("utf-8")
    )
    if (
        not isinstance(value, dict)
        or value.get("schema") != EXPECTED_SCHEMA
        or not isinstance(value.get("entries"), list)
    ):
        raise ValueError("Invalid paper-results manifest")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--sha256",
        default="",
        help="Optional expected SHA-256 for the whole archive.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing result files after verification.",
    )
    return parser


def verify_archive_digest(archive_path: Path, expected: str) -> None:
    if not expected:
        return
    measured = sha256(archive_path)
    if measured.lower() != expected.strip().lower():
        raise ValueError(
            f"Archive SHA-256 mismatch: expected={expected}, got={measured}"
        )


def stage_entry(
    archive: zipfile.ZipFile,
    names: set[str],
    entry: dict[str, Any],
    staging: Path,
) -> tuple[Path, Path, str]:
    relative = safe_relative(str(entry["path"]))
    archive_name = relative.as_posix()
    if archive_name not in names:
        raise FileNotFoundError(
            f"Manifest entry is absent from archive: {archive_name}"
        )
    staged_path = staging / relative
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(archive_name) as source, staged_path.open("wb") as target:
        shutil.copyfileobj(source, target)
    if staged_path.stat().st_size != int(entry["bytes"]):
        raise ValueError(f"Size mismatch: {archive_name}")
    measured = sha256(staged_path)
    if measured.lower() != str(entry["sha256"]).lower():
        raise ValueError(f"SHA-256 mismatch: {archive_name}")
    return staged_path, relative, measured


def needs_install(
    destination: Path,
    staged_path: Path,
    measured_digest: str,
    *,
    overwrite: bool,
) -> bool:
    if not destination.exists() or overwrite:
        return True
    already_installed = (
        destination.stat().st_size == staged_path.stat().st_size
        and sha256(destination) == measured_digest
    )
    if already_installed:
        return False
    raise FileExistsError(
        f"Refusing to replace {destination}; pass --overwrite"
    )


def stage_manifest_entries(
    archive: zipfile.ZipFile,
    manifest: dict[str, Any],
    staging: Path,
    root: Path,
    *,
    overwrite: bool,
) -> list[tuple[Path, Path]]:
    names = set(archive.namelist())
    staged: list[tuple[Path, Path]] = []
    for entry in manifest["entries"]:
        staged_path, relative, measured = stage_entry(
            archive,
            names,
            entry,
            staging,
        )
        destination = root / relative
        if needs_install(
            destination,
            staged_path,
            measured,
            overwrite=overwrite,
        ):
            staged.append((staged_path, destination))
    return staged


def install_staged_files(
    staged: list[tuple[Path, Path]],
    root: Path,
) -> None:
    for staged_path, destination in staged:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_destination = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(staged_path, temporary_destination)
        temporary_destination.replace(destination)
        print(f"installed {destination.relative_to(root)}", flush=True)


def main() -> None:
    args = build_arg_parser().parse_args()

    archive_path = Path(args.archive).expanduser().resolve()
    root = Path(args.repository_root).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    verify_archive_digest(archive_path, args.sha256)

    with zipfile.ZipFile(archive_path) as archive:
        manifest = load_manifest(archive)
        with tempfile.TemporaryDirectory(prefix="sgr-paper-results-") as temporary:
            staging = Path(temporary)
            staged = stage_manifest_entries(
                archive,
                manifest,
                staging,
                root,
                overwrite=bool(args.overwrite),
            )
            install_staged_files(staged, root)
    print(
        json.dumps(
            {
                "event": "done",
                "archive": str(archive_path),
                "repository_root": str(root),
                "files_installed": len(staged),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
