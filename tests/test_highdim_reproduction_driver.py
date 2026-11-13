from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "tools" / "reproduce_highdim.py"


def load_driver():
    module_name = "sgr_test_highdim_reproduction_driver"
    spec = importlib.util.spec_from_file_location(module_name, DRIVER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class HighDimensionalReproductionDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.driver = load_driver()

    def test_all_released_configurations_build_four_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            for configuration in sorted(
                (ROOT / "example" / "highdim").glob(
                    "example_*/config/training.json"
                )
            ):
                with self.subTest(example=configuration.parent.parent.name):
                    stages = self.driver.build_stages(
                        configuration,
                        output_root,
                        python=Path("/usr/bin/python3"),
                        device="cuda",
                        overwrite_data=False,
                    )
                    self.assertEqual(
                        [stage.name for stage in stages],
                        ["generate", "train", "reconstruct", "evaluate"],
                    )
                    for stage in stages:
                        self.assertGreaterEqual(len(stage.command), 2)
                        self.assertTrue(stage.artifacts)

    def test_6d_commands_use_the_recorded_seeds_and_protocol(self) -> None:
        configuration = (
            ROOT
            / "example"
            / "highdim"
            / "example_6d"
            / "config"
            / "training.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            stages = self.driver.build_stages(
                configuration,
                Path(directory),
                python=Path("/usr/bin/python3"),
                device="cuda",
                overwrite_data=False,
            )
        commands = {stage.name: stage.command for stage in stages}
        self.assertIn("2026071832", commands["train"])
        self.assertIn("2026071834", commands["reconstruct"])
        self.assertIn("1000000", commands["reconstruct"])
        self.assertIn("250000", commands["reconstruct"])
        self.assertIn("100000", commands["evaluate"])
        self.assertIn("analytic", commands["evaluate"])

    def test_example_selection_is_stable(self) -> None:
        available = self.driver.discover_examples()
        self.assertEqual(
            self.driver.selected_examples(["all"], available),
            [
                "example_4d",
                "example_6d",
                "example_8d",
                "example_9d",
                "example_10d",
                "example_12d",
            ],
        )


if __name__ == "__main__":
    unittest.main()
