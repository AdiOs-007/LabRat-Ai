"""Tests for bounded experiment campaigns."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from labrat.runners.campaign_runner import run_campaign

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline.json"


class TestCampaignRunner(unittest.TestCase):
    def test_rejects_invalid_trial_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "experiments.sqlite3"

            for value in (0, 6, -1, True, 1.5):
                with self.subTest(max_trials=value):
                    with self.assertRaises(ValueError):
                        run_campaign(
                            BASELINE_CONFIG,
                            database_path,
                            max_trials=value,
                        )

    def test_rejects_invalid_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_campaign(
                    BASELINE_CONFIG,
                    Path(directory) / "experiments.sqlite3",
                    direction="sideways",
                )

    def test_rejects_metric_missing_from_config(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_campaign(
                    BASELINE_CONFIG,
                    Path(directory) / "experiments.sqlite3",
                    metric="roc_auc",
                )

    def test_runs_and_ranks_requested_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "experiments.sqlite3"

            result = run_campaign(
                BASELINE_CONFIG,
                database_path,
                max_trials=2,
                metric="f1",
                direction="maximize",
            )

            self.assertEqual(result["trials_requested"], 2)
            self.assertEqual(result["trials_completed"], 2)
            self.assertEqual(len(result["candidate_results"]), 2)
            self.assertEqual(len(result["ranking"]["ranked"]), 2)
            self.assertEqual(result["ranking"]["excluded"], [])
            self.assertEqual(result["metric"], "f1")
            self.assertEqual(result["direction"], "maximize")
            self.assertTrue(database_path.exists())

    def test_uses_selector_for_each_trial(self):
        first_result = {
            "experiment_name": "trial_1",
            "dataset": "breast_cancer",
            "dataset_fingerprint": "same-data",
            "model": "logistic_regression",
            "model_parameters": {"C": 1.0},
            "random_state": 42,
            "metrics": {"f1": 0.90},
            "evaluation": {
                "test_size": 0.2,
                "validation_size": 0.25,
                "random_state": 42,
                "stratify": True,
                "protocol_version": 2,
            },
        }
        second_result = {
            **first_result,
            "experiment_name": "trial_2",
            "model_parameters": {"C": 10.0},
            "metrics": {"f1": 0.95},
        }

        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "experiments.sqlite3"

            selector_inputs = []

            def choose_candidate(completed_trials, remaining, metric, direction):
                selector_inputs.append(
                    (list(completed_trials), list(remaining), metric, direction)
                )
                return [1.0, 10.0][len(selector_inputs) - 1]

            with (
                patch(
                    "labrat.runners.campaign_runner.select_next_candidate",
                    side_effect=choose_candidate,
                ) as selector,
                patch(
                    "labrat.runners.campaign_runner.run_validation_experiment",
                    side_effect=[first_result, second_result],
                ),
            ):
                result = run_campaign(
                    BASELINE_CONFIG,
                    database_path,
                    max_trials=2,
                )

            self.assertEqual(selector.call_count, 2)
            self.assertEqual(selector_inputs[0][0], [])
            self.assertEqual(
                selector_inputs[1][0],
                [first_result],
            )
            self.assertEqual(result["trials_completed"], 2)
            self.assertEqual(
                [item["model_parameters"]["C"] for item in result["candidate_results"]],
                [1.0, 10.0],
            )


if __name__ == "__main__":
    unittest.main()
