from __future__ import annotations

import ast
import json
import struct
import unittest
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "runs",
    "outputs",
}


def public_file(path: Path) -> bool:
    return not any(
        part in IGNORED_DIRECTORIES for part in path.relative_to(ROOT).parts
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def npy_header(stream: Any) -> dict[str, Any]:
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("invalid NPY magic")
    major, _minor = struct.unpack("BB", stream.read(2))
    size = (
        struct.unpack("<H", stream.read(2))[0]
        if major == 1
        else struct.unpack("<I", stream.read(4))[0]
    )
    return ast.literal_eval(stream.read(size).decode("latin1").strip())


def npz_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    output: dict[str, tuple[int, ...]] = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue
            with archive.open(name) as stream:
                output[Path(name).stem] = tuple(
                    int(value) for value in npy_header(stream)["shape"]
                )
    return output


def is_lfs_pointer(path: Path) -> bool:
    return path.read_bytes()[:128].startswith(
        b"version https://git-lfs.github.com/spec/v1\n"
    )


class ReleaseLayoutTests(unittest.TestCase):
    def test_public_gaussian_knn_planner_is_present(self) -> None:
        self.assertTrue(
            (
                ROOT
                / "01_lowdim_sgr"
                / "code"
                / "plan_gaussian_knn.py"
            ).is_file()
        )

    def test_python_and_json_syntax(self) -> None:
        for path in ROOT.rglob("*.py"):
            if not public_file(path):
                continue
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in ROOT.rglob("*.json"):
            if not public_file(path):
                continue
            json.loads(path.read_text(encoding="utf-8"))
        for path in ROOT.rglob("*.jsonl"):
            if not public_file(path):
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    json.loads(line)

    def test_example_counts_and_names(self) -> None:
        lowdim = sorted((ROOT / "example" / "lowdim").glob("example_*"))
        highdim = sorted((ROOT / "example" / "highdim").glob("example_*"))
        self.assertEqual(len(lowdim), 27)
        self.assertEqual(len(highdim), 6)
        low_names = {
            load_json(path / "config" / "training.json")["dataset"]
            for path in lowdim
        }
        self.assertEqual(len(low_names), 27)
        dimensions = {
            int(load_json(path / "config" / "training.json")["ambient_dim"])
            for path in highdim
        }
        self.assertEqual(dimensions, {4, 6, 8, 9, 10, 12})

    def test_highdim_configs_resolve_generator_and_analytic_module(self) -> None:
        for example in sorted((ROOT / "example" / "highdim").glob("example_*")):
            configuration = example / "config" / "training.json"
            config = load_json(configuration)
            data = config["data"]
            self.assertTrue(str(data["path"]).startswith("dataset/"))
            self.assertTrue(str(data["metadata_path"]).startswith("dataset/"))
            generator = (example / data["generator"]).resolve()
            analytic = (example / config["analytic_definition"]["module"]).resolve()
            self.assertEqual(
                generator,
                (ROOT / "02_highdim_sgr" / "code" / "generate_paper_data.py").resolve(),
            )
            self.assertTrue(analytic.is_file())

    def test_lowdim_evaluation_asset_manifest_matches_configurations(self) -> None:
        manifest = load_json(
            ROOT / "example" / "lowdim" / "evaluation_assets.json"
        )
        self.assertEqual(
            manifest["schema"],
            "sgr-lowdim-evaluation-assets/v1",
        )
        entries = {
            (str(entry["dataset"]), str(entry["role"])): str(entry["path"])
            for entry in manifest["assets"]
        }
        self.assertEqual(len(entries), 54)
        self.assertEqual(
            int(manifest["total_bytes"]),
            sum(int(entry["bytes"]) for entry in manifest["assets"]),
        )
        self.assertTrue(manifest["sources"])
        for name, source in manifest["sources"].items():
            for field in ("url", "acquisition", "distribution_status", "note"):
                self.assertTrue(str(source[field]).strip(), f"{name}.{field}")
        expected: dict[tuple[str, str], str] = {}
        for example in sorted((ROOT / "example" / "lowdim").glob("example_*")):
            config = load_json(example / "config" / "training.json")
            reference = load_json(example / "dataset" / "reference.json")
            dataset = str(config["dataset"])
            expected[(dataset, "training_point_cloud")] = str(
                config["training_parameters"]["point_cloud"]
            )
            expected[(dataset, "independent_reference_mesh")] = str(
                reference["asset_root_path"]
            )
        self.assertEqual(entries, expected)
        for entry in manifest["assets"]:
            self.assertEqual(len(str(entry["sha256"])), 64)
            self.assertGreater(int(entry["bytes"]), 0)
            self.assertIn(entry["source_group"], manifest["sources"])
            self.assertTrue(str(entry["path"]).startswith("datasets/"))

    def test_retained_mesh_metadata_when_results_are_installed(self) -> None:
        examples = sorted((ROOT / "example" / "lowdim").glob("example_*"))
        installed = [
            example
            for example in examples
            if zipfile.is_zipfile(
                example / "result" / "reconstruction" / "zero_mesh.npz"
            )
        ]
        if not installed:
            self.skipTest("retained result bundle is not installed")
        self.assertEqual(len(installed), 27)
        for example in installed:
            stats = load_json(
                example / "result" / "reconstruction" / "stats.json"
            )
            evaluation = load_json(example / "result" / "evaluation.json")
            shapes = npz_shapes(
                example / "result" / "reconstruction" / "zero_mesh.npz"
            )
            self.assertEqual(shapes["vertices"], (int(stats["vertices"]), 3))
            self.assertEqual(shapes["faces"], (int(stats["faces"]), 3))
            self.assertEqual(int(evaluation["vertices"]), int(stats["vertices"]))
            self.assertEqual(int(evaluation["faces"]), int(stats["faces"]))

    def test_no_absolute_user_paths_in_public_text(self) -> None:
        suffixes = {".py", ".md", ".json", ".txt", ".yml", ".yaml", ".tex"}
        for path in ROOT.rglob("*"):
            if (
                not path.is_file()
                or path.suffix.lower() not in suffixes
                or not public_file(path)
            ):
                continue
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/" + "Users/", text, str(path.relative_to(ROOT)))
            self.assertNotIn("/" + "home/", text, str(path.relative_to(ROOT)))

    def test_no_absolute_user_paths_in_retained_checkpoints(self) -> None:
        markers = (
            b"/" + b"Users/",
            b"/" + b"home/",
            b"/var/" + b"folders/",
            b"C:\\" + b"Users\\",
        )
        checkpoints = sorted(
            (ROOT / "example" / "lowdim").glob(
                "example_*/result/implicit_function/best.pt"
            )
        )
        self.assertEqual(len(checkpoints), 27)
        for path in checkpoints:
            if is_lfs_pointer(path):
                continue
            payload = path.read_bytes()
            for marker in markers:
                self.assertNotIn(marker, payload, str(path.relative_to(ROOT)))


if __name__ == "__main__":
    unittest.main()
