"""Tests for LabRat AI's experiment runner."""

import json
import tempfile
import unittest
from pathlib import Path

from labrat.runners.experiment_runner import run_experiment
from labrat.storage.sqlite_backend import SQLiteBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline.json"


class TestExperimentRunner(unittest.TestCase):
    """Verify baseline execution and configuration error handling."""

    def test_baseline_experiment_returns_expected_results(self):
        result = run_experiment(str(BASELINE_CONFIG))

        self.assertEqual(result["dataset"], "breast_cancer")
        self.assertEqual(result["model"], "logistic_regression")
        self.assertEqual(result["train_samples"], 455)
        self.assertEqual(result["test_samples"], 114)
        self.assertEqual(result["positive_class"], "malignant")

        self.assertEqual(
            set(result["metrics"]),
            {"accuracy", "precision", "recall", "f1"},
        )

        for score in result["metrics"].values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

        self.assertGreaterEqual(result["training_seconds"], 0.0)

    def test_repeated_runs_produce_same_metrics(self):
        first = run_experiment(str(BASELINE_CONFIG))
        second = run_experiment(str(BASELINE_CONFIG))

        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["train_samples"], second["train_samples"])
        self.assertEqual(first["test_samples"], second["test_samples"])

    def test_experiment_result_is_persisted_to_sqlite(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "experiments.sqlite3"

            result = run_experiment(
                str(BASELINE_CONFIG),
                database_path=database_path,
            )

            self.assertIn("id", result)

            storage = SQLiteBackend(database_path)
            try:
                saved_result = storage.get_experiment(result["id"])

                self.assertIsNotNone(saved_result)
                self.assertEqual(
                    saved_result["experiment_name"],
                    result["experiment_name"],
                )
                self.assertEqual(saved_result["metrics"], result["metrics"])
                self.assertEqual(saved_result["dataset"], result["dataset"])
            finally:
                storage.close()

    def test_invalid_configuration_is_rejected(self):
        with BASELINE_CONFIG.open("r", encoding="utf-8") as file:
            config = json.load(file)

        config["test_size"] = 1.5

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "invalid_config.json"
            config_path.write_text(
                json.dumps(config),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "test_size"):
                run_experiment(str(config_path))


if __name__ == "__main__":
    unittest.main()
