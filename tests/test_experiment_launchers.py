from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "empirical_vs_trained_small_sigma"


class ExperimentLauncherTests(unittest.TestCase):
    def run_launcher(
        self, name: str, *, fail_training: bool = False, datasets: str = "bunny"
    ) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
        with tempfile.TemporaryDirectory(prefix="sgr-launcher-test-") as directory:
            workspace = Path(directory)
            calls = workspace / "calls.jsonl"
            executable = workspace / "python-stub"
            stub = workspace / "python-stub.py"
            stub.write_text(
                "import json, os, sys\n"
                "record = {'argv': sys.argv[1:], 'cwd': os.getcwd(), "
                "'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}\n"
                "with open(os.environ['LAUNCHER_TEST_CALLS'], 'a') as stream:\n"
                "    stream.write(json.dumps(record) + '\\n')\n"
                "if os.environ.get('LAUNCHER_TEST_FAIL') == '1' and "
                "len(sys.argv) > 2 and sys.argv[2] == 'train':\n"
                "    sys.exit(23)\n",
                encoding="utf-8",
            )
            executable.write_text(
                "#!/bin/sh\nexec "
                + shlex.quote(sys.executable)
                + " "
                + shlex.quote(str(stub))
                + ' "$@"\n',
                encoding="utf-8",
            )
            executable.chmod(0o755)
            environment = os.environ.copy()
            environment.update(
                PYTHON_BIN=str(executable),
                OUTPUT_ROOT=str(workspace / "outputs"),
                GPU_IDS="0",
                DATASETS=datasets,
                ASSET_ROOT=str(ROOT),
                GARGOYLE_POINT_CLOUD=str(workspace / "missing-gargoyle.csv"),
                LAUNCHER_TEST_CALLS=str(calls),
                LAUNCHER_TEST_FAIL="1" if fail_training else "0",
            )
            completed = subprocess.run(
                ["bash", str(EXPERIMENT / name)],
                cwd=workspace,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
                check=False,
            )
            records = (
                [json.loads(line) for line in calls.read_text().splitlines()]
                if calls.exists()
                else []
            )
            return completed, records

    def test_launchers_run_from_an_unrelated_working_directory(self) -> None:
        for name, expected_training_jobs in (
            ("launch_real_shapes.sh", 18),
            ("launch_server.sh", 24),
        ):
            with self.subTest(launcher=name):
                completed, records = self.run_launcher(name)
                self.assertEqual(completed.returncode, 0, completed.stdout)
                training = [r for r in records if r["argv"][1] == "train"]
                self.assertEqual(len(training), expected_training_jobs)
                self.assertTrue(all(r["gpu"] == "0" for r in training))
                self.assertTrue(all(Path(r["cwd"]) == EXPERIMENT for r in records))
                self.assertEqual(records[-1]["argv"][1], "evaluate")
                if name == "launch_real_shapes.sh":
                    args = records[0]["argv"]
                    point_cloud = args[args.index("--point-cloud") + 1]
                    self.assertEqual(
                        Path(point_cloud),
                        ROOT / "datasets/Bunny/bunny_surface_points_250k_20260526.csv",
                    )

    def test_training_failure_stops_before_evaluation(self) -> None:
        for name in ("launch_real_shapes.sh", "launch_server.sh"):
            with self.subTest(launcher=name):
                completed, records = self.run_launcher(name, fail_training=True)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("Training failed", completed.stdout)
                self.assertFalse(any(r["argv"][1] == "evaluate" for r in records))

    def test_missing_gargoyle_input_fails_before_starting_jobs(self) -> None:
        completed, records = self.run_launcher(
            "launch_real_shapes.sh", datasets="gargoyle"
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Missing gargoyle input", completed.stdout)
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
