"""Tests for LabRat AI's SQLite storage backend."""

import sqlite3
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

    def test_reproducibility_metadata_survives_round_trip(self):
        result = make_result()
        result["dataset_fingerprint"] = "a" * 64
        result["model_parameters"] = {"max_iter": 1000}
        result["evaluation"] = {
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True,
            "protocol_version": 1,
        }

        experiment_id = self.storage.save_experiment(result)
        self.storage.close()
        self.storage = SQLiteBackend(self.database_path)

        retrieved = self.storage.get_experiment(experiment_id)

        self.assertIsNotNone(retrieved)
        self.assertEqual(
            retrieved["dataset_fingerprint"],
            result["dataset_fingerprint"],
        )
        self.assertEqual(
            retrieved["model_parameters"],
            result["model_parameters"],
        )
        self.assertEqual(retrieved["evaluation"], result["evaluation"])

    def test_legacy_database_migrates_without_losing_records(self):
        self.storage.close()
        self.database_path.unlink(missing_ok=True)

        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    model TEXT NOT NULL,
                    random_state INTEGER NOT NULL,
                    train_samples INTEGER NOT NULL,
                    test_samples INTEGER NOT NULL,
                    positive_class TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    training_seconds REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO experiments (
                    experiment_name, dataset, model, random_state,
                    train_samples, test_samples, positive_class,
                    metrics_json, training_seconds, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "legacy_experiment",
                    "breast_cancer",
                    "logistic_regression",
                    42,
                    455,
                    114,
                    "malignant",
                    '{"accuracy": 0.98}',
                    0.01,
                    "2026-01-01T00:00:00+00:00",
                ),
            )

        self.storage = SQLiteBackend(self.database_path)
        retrieved = self.storage.get_experiment(1)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["experiment_name"], "legacy_experiment")
        self.assertEqual(retrieved["metrics"], {"accuracy": 0.98})
        self.assertNotIn("dataset_fingerprint", retrieved)
        self.assertNotIn("model_parameters", retrieved)
        self.assertNotIn("evaluation", retrieved)

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
