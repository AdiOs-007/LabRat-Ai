"""Tests for bounded experiment campaigns."""

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
