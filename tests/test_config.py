"""Tests for LabRat AI configuration validation."""

import json
import tempfile
import unittest
from pathlib import Path

from labrat.core.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline.json"


class TestConfig(unittest.TestCase):
    def setUp(self):
        with BASELINE_CONFIG.open("r", encoding="utf-8") as file:
            self.config = json.load(file)

    def load_modified_config(self, updates):
        config = self.config.copy()
        config.update(updates)

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "experiment.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            return load_config(config_path)

    def test_accepts_logistic_regression_campaign_parameters(self):
        loaded = self.load_modified_config(
            {"model_parameters": {"max_iter": 1000, "C": 0.1}}
        )
        self.assertEqual(loaded["model_parameters"]["C"], 0.1)

    def test_accepts_random_forest_model(self):
        loaded = self.load_modified_config(
            {
                "model": "random_forest",
                "model_parameters": {
                    "n_estimators": 200,
                    "max_depth": 20,
                    "min_samples_split": 5,
                    "random_state": 42,
                },
            }
        )
        self.assertEqual(loaded["model"], "random_forest")

    def test_accepts_random_forest_unlimited_depth(self):
        loaded = self.load_modified_config(
            {
                "model": "random_forest",
                "model_parameters": {
                    "n_estimators": 100,
                    "max_depth": None,
                    "min_samples_split": 2,
                },
            }
        )
        self.assertIsNone(loaded["model_parameters"]["max_depth"])

    def test_rejects_unknown_logistic_regression_parameter(self):
        with self.assertRaisesRegex(ValueError, "Unsupported parameters"):
            self.load_modified_config(
                {
                    "model_parameters": {
                        "max_iter": 1000,
                        "learning_rate": 0.01,
                    }
                }
            )

    def test_rejects_unknown_random_forest_parameter(self):
        with self.assertRaisesRegex(ValueError, "Unsupported parameters"):
            self.load_modified_config(
                {
                    "model": "random_forest",
                    "model_parameters": {
                        "n_estimators": 100,
                        "learning_rate": 0.01,
                    },
                }
            )

    def test_rejects_invalid_logistic_regression_parameters(self):
        invalid_parameters = [
            {"max_iter": 0},
            {"max_iter": -1},
            {"max_iter": True},
            {"max_iter": 1000, "C": 0},
            {"max_iter": 1000, "C": -1},
            {"max_iter": 1000, "C": True},
            {"max_iter": 1000, "C": float("nan")},
            {"max_iter": 1000, "C": float("inf")},
            {"max_iter": 1000, "random_state": -1},
            {"max_iter": 1000, "random_state": True},
        ]

        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    self.load_modified_config({"model_parameters": parameters})

    def test_rejects_invalid_random_forest_parameters(self):
        invalid_parameters = [
            {"n_estimators": 0},
            {"n_estimators": -1},
            {"n_estimators": True},
            {"n_estimators": 100, "max_depth": 0},
            {"n_estimators": 100, "max_depth": -1},
            {"n_estimators": 100, "max_depth": True},
            {"n_estimators": 100, "min_samples_split": 1},
            {"n_estimators": 100, "min_samples_split": 1.5},
            {"n_estimators": 100, "min_samples_split": True},
            {"n_estimators": 100, "random_state": -1},
            {"n_estimators": 100, "random_state": True},
        ]

        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    self.load_modified_config(
                        {
                            "model": "random_forest",
                            "model_parameters": parameters,
                        }
                    )


if __name__ == "__main__":
    unittest.main()
