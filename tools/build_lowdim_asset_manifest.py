#!/usr/bin/env python3
"""Build the checksummed manifest for exact low-dimensional re-evaluation.

The manifest records portable paths, sizes, SHA-256 values and source
provenance for the exact input clouds and reference meshes distributed
through Git LFS under datasets/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "example" / "lowdim" / "evaluation_assets.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_group(dataset: str) -> str:
    if dataset.startswith("ABC_"):
        return "abc_v00_chunk_0000"
    if dataset in {"Bunny", "Armadillo"}:
        return "stanford_3d_scanning_repository"
    if dataset == "Monkey":
        return "open3d_monkey_model"
    if dataset in {"Sphere", "CV2Radial_a020_f16", "FlowerPetal"}:
        return "repository_analytic_geometry"
    return "author_supplied_geometry"


def asset_entry(
    asset_root: Path,
    *,
    dataset: str,
    role: str,
    relative_path: str,
) -> dict[str, Any]:
    path = (asset_root / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{dataset} {role}: {path}")
    return {
        "dataset": dataset,
        "role": role,
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "source_group": source_group(dataset),
    }


def build_manifest(asset_root: Path) -> dict[str, Any]:
    assets: list[dict[str, Any]] = []
    for configuration in sorted(
        (ROOT / "example" / "lowdim").glob(
            "example_*/config/training.json"
        )
    ):
        example = configuration.parent.parent
        config = load_json(configuration)
        reference = load_json(example / "dataset" / "reference.json")
        dataset = str(config["dataset"])
        if str(reference["dataset"]) != dataset:
            raise ValueError(f"Dataset mismatch in {example}")
        assets.append(
            asset_entry(
                asset_root,
                dataset=dataset,
                role="training_point_cloud",
                relative_path=str(config["training_parameters"]["point_cloud"]),
            )
        )
        assets.append(
            asset_entry(
                asset_root,
                dataset=dataset,
                role="independent_reference_mesh",
                relative_path=str(reference["asset_root_path"]),
            )
        )
    assets.sort(key=lambda value: (value["dataset"], value["role"]))
    return {
        "schema": "sgr-lowdim-evaluation-assets/v1",
        "purpose": (
            "Exact re-evaluation of the 27 retained low-dimensional SGR "
            "checkpoints and meshes"
        ),
        "asset_root_contract": (
            "Each path is resolved under datasets/ relative to --asset-root."
        ),
        "asset_count": len(assets),
        "total_bytes": sum(int(asset["bytes"]) for asset in assets),
        "sources": {
            "abc_v00_chunk_0000": {
                "url": "https://deep-geometry.github.io/abc-dataset/",
                "acquisition": "Run git lfs pull from the repository root. The exact manifest-identified inputs are included under datasets/; verify them with --profile lowdim-evaluation --asset-root .",
                "distribution_status": "included-in-this-repository-via-git-lfs",
                "note": "ABC v00 OBJ chunk 0000 is the upstream geometry source; the exact paper point clouds and selected components are paper preprocessing outputs identified by SHA-256 below."
            },
            "stanford_3d_scanning_repository": {
                "url": "https://graphics.stanford.edu/data/3Dscanrep/",
                "acquisition": "Run git lfs pull from the repository root. The exact manifest-identified inputs are included under datasets/; verify them with --profile lowdim-evaluation --asset-root .",
                "distribution_status": "included-in-this-repository-via-git-lfs",
                "note": "Upstream source for Bunny and Armadillo; exact paper samples are identified by SHA-256 below."
            },
            "open3d_monkey_model": {
                "url": "https://www.open3d.org/docs/release/python_api/open3d.data.MonkeyModel.html",
                "acquisition": "Run git lfs pull from the repository root. The exact manifest-identified inputs are included under datasets/; verify them with --profile lowdim-evaluation --asset-root .",
                "distribution_status": "included-in-this-repository-via-git-lfs",
                "note": "Upstream source for Monkey; exact paper samples are identified by SHA-256 below."
            },
            "repository_analytic_geometry": {
                "url": "https://github.com/zhangshiliang502-droid/score-guided-reconstruction/blob/main/03_evaluation/generate_analytic_reference_meshes.py",
                "acquisition": "Run git lfs pull from the repository root. The exact manifest-identified inputs are included under datasets/; verify them with --profile lowdim-evaluation --asset-root .",
                "distribution_status": "included-in-this-repository-via-git-lfs",
                "note": "The exact analytic reference meshes and author-generated training clouds are included. The reference meshes can also be regenerated with the repository script."
            },
            "author_supplied_geometry": {
                "url": "https://github.com/zhangshiliang502-droid/score-guided-reconstruction/blob/main/docs/DATA_LAYOUT.md",
                "acquisition": "Run git lfs pull from the repository root. The exact manifest-identified inputs are included under datasets/; verify them with --profile lowdim-evaluation --asset-root .",
                "distribution_status": "included-in-this-repository-via-git-lfs",
                "note": "The exact author-supplied Knot point cloud and reference mesh are included for reproducibility; their sizes and SHA-256 identities are recorded below."
            }
        },
        "assets": assets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        required=True,
        help="Root against which public configuration paths such as datasets/... resolve.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    asset_root = Path(args.asset_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    manifest = build_manifest(asset_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(output),
                "assets": manifest["asset_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
