"""Tests for LabRat AI's SQLite storage backend."""

import tempfile
import unittest
from pathlib import Path

from labrat.storage.sqlite_backend import SQLiteBackend


def make_result(name: str = "test_experiment") -> dict:
    """Build a sample experiment result for testing."""
    return {
        "experiment_name": name,
        "dataset": "breast_cancer",
        "model": "logistic_regression",
        "random_state": 42,
        "train_samples": 455,
        "test_samples": 114,
        "positive_class": "malignant",
        "metrics": {
            "accuracy": 0.9825,
            "precision": 0.9762,
            "recall": 0.9762,
            "f1": 0.9762,
        },
        "training_seconds": 0.0034,
    }


class TestSQLiteBackend(unittest.TestCase):
    """Verify experiment storage and retrieval."""

    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_directory.name) / "history.sqlite3"
        self.storage = SQLiteBackend(self.database_path)

    def tearDown(self):
        self.storage.close()
        self.temp_directory.cleanup()

    def test_save_and_retrieve_experiment(self):
        result = make_result()
        experiment_id = self.storage.save_experiment(result)

        retrieved = self.storage.get_experiment(experiment_id)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["experiment_name"], result["experiment_name"])
        self.assertEqual(retrieved["metrics"], result["metrics"])
        self.assertEqual(retrieved["dataset"], result["dataset"])
        self.assertIn("created_at", retrieved)

    def test_result_survives_reopening_database(self):
        experiment_id = self.storage.save_experiment(make_result())
        self.storage.close()

        self.storage = SQLiteBackend(self.database_path)
        retrieved = self.storage.get_experiment(experiment_id)

        self.assertIsNotNone(retrieved)
        self.assertEqual(
            retrieved["experiment_name"],
            "test_experiment",
        )

    def test_list_returns_newest_first_and_respects_limit(self):
        first_id = self.storage.save_experiment(make_result("first"))
        second_id = self.storage.save_experiment(make_result("second"))
        self.storage.save_experiment(make_result("third"))

        recent = self.storage.list_experiments(limit=2)

        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["experiment_name"], "third")
        self.assertEqual(recent[1]["experiment_name"], "second")
        self.assertGreater(second_id, first_id)

    def test_missing_experiment_returns_none(self):
        self.assertIsNone(self.storage.get_experiment(-1))

    def test_invalid_limit_is_rejected(self):
        for invalid_limit in (0, -1, True, 1.5):
            with self.subTest(limit=invalid_limit):
                with self.assertRaises(ValueError):
                    self.storage.list_experiments(limit=invalid_limit)

    def test_non_finite_metric_is_rejected(self):
        result = make_result()
        result["metrics"]["accuracy"] = float("nan")

        with self.assertRaises(ValueError):
            self.storage.save_experiment(result)

        self.assertEqual(self.storage.list_experiments(), [])


if __name__ == "__main__":
    unittest.main()
