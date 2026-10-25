"""Tests for validation-based experiment execution."""

import tempfile
import unittest
from pathlib import Path

from labrat.runners.validation_runner import run_validation_experiment
from labrat.storage.sqlite_backend import SQLiteBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline.json"


class TestValidationRunner(unittest.TestCase):
    def test_validation_run_preserves_a_test_holdout(self):
        result = run_validation_experiment(BASELINE_CONFIG)

        self.assertEqual(result["train_samples"], 341)
        self.assertEqual(result["test_samples"], 114)
        self.assertEqual(result["evaluation"]["validation_samples"], 114)
        self.assertEqual(result["evaluation"]["metric_source"], "validation")
        self.assertEqual(result["evaluation"]["protocol_version"], 2)

        self.assertEqual(
            set(result["metrics"]),
            {"accuracy", "precision", "recall", "f1"},
        )
        for score in result["metrics"].values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_repeated_runs_have_identical_metrics(self):
        first = run_validation_experiment(BASELINE_CONFIG)
        second = run_validation_experiment(BASELINE_CONFIG)

        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(
            first["dataset_fingerprint"],
            second["dataset_fingerprint"],
        )

    def test_invalid_validation_size_is_rejected(self):
        for value in (0, 1, -0.1, float("inf"), True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "validation_size"):
                    run_validation_experiment(
                        BASELINE_CONFIG,
                        validation_size=value,
                    )

    def test_validation_result_can_be_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "experiments.sqlite3"
            result = run_validation_experiment(
                BASELINE_CONFIG,
                database_path=database,
            )

            storage = SQLiteBackend(database)
            try:
                saved = storage.get_experiment(result["id"])
                self.assertEqual(
                    saved["evaluation"]["metric_source"],
                    "validation",
                )
                self.assertEqual(saved["metrics"], result["metrics"])
            finally:
                storage.close()


if __name__ == "__main__":
    unittest.main()
