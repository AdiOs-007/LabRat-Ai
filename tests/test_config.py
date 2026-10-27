"""Tests for LabRat AI configuration validation."""

import json
import tempfile
import unittest
from pathlib import Path

from labrat.core.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline.json"


class TestConfig(unittest.TestCase):
    def test_accepts_random_forest_model(self):
        with BASELINE_CONFIG.open("r", encoding="utf-8") as file:
            config = json.load(file)

        config["model"] = "random_forest"
        config["model_parameters"] = {
            "n_estimators": 100,
            "random_state": 42,
        }

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "random_forest.json"
            config_path.write_text(
                json.dumps(config),
                encoding="utf-8",
            )

            loaded_config = load_config(config_path)

        self.assertEqual(loaded_config["model"], "random_forest")


if __name__ == "__main__":
    unittest.main()
