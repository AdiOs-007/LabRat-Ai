"""Tests for Markdown experiment reporting."""

import unittest

from labrat.reporting import generate_markdown_report


def make_experiment(
    experiment_id: int,
    f1: float,
    parameters: dict | None = None,
) -> dict:
    """Build an experiment record for report tests."""
    return {
        "id": experiment_id,
        "experiment_name": f"trial_{experiment_id}",
        "dataset": "breast_cancer",
        "model": "logistic_regression",
        "model_parameters": parameters or {"C": 1.0},
        "metrics": {"accuracy": 0.95, "f1": f1},
        "evaluation": {
            "metric_source": "validation",
            "validation_size": 0.25,
            "test_size": 0.2,
        },
    }


class TestGenerateMarkdownReport(unittest.TestCase):
    def test_empty_report_does_not_claim_a_winner(self):
        report = generate_markdown_report(
            [],
            {
                "metric": "f1",
                "direction": "maximize",
                "ranked": [],
                "excluded": [],
            },
        )

        self.assertIn("No experiments could be ranked.", report)
        self.assertNotIn("## Best Configuration", report)

    def test_report_includes_ranked_results_and_best_configuration(self):
        experiments = [
            make_experiment(1, 0.9, {"C": 1.0}),
            make_experiment(2, 0.8, {"C": 0.1}),
        ]
        ranking = {
            "metric": "f1",
            "direction": "maximize",
            "ranked": [
                {
                    "rank": 1,
                    "id": 1,
                    "experiment_name": "trial_1",
                    "model": "logistic_regression",
                    "metric": "f1",
                    "value": 0.9,
                },
                {
                    "rank": 2,
                    "id": 2,
                    "experiment_name": "trial_2",
                    "model": "logistic_regression",
                    "metric": "f1",
                    "value": 0.8,
                },
            ],
            "excluded": [],
        }

        report = generate_markdown_report(experiments, ranking)

        self.assertIn("| 1 | 1 | trial_1 |", report)
        self.assertIn("| 2 | 2 | trial_2 |", report)
        self.assertIn("## Best Configuration", report)
        self.assertIn("'C': 1.0", report)
        self.assertIn("0.900000", report)

    def test_report_discloses_validation_metric_source(self):
        experiment = make_experiment(1, 0.9)
        ranking = {
            "metric": "f1",
            "direction": "maximize",
            "ranked": [
                {
                    "rank": 1,
                    "id": 1,
                    "experiment_name": "trial_1",
                    "model": "logistic_regression",
                    "metric": "f1",
                    "value": 0.9,
                }
            ],
            "excluded": [],
        }

        report = generate_markdown_report([experiment], ranking)

        self.assertIn("**Metric source:** `validation`", report)
        self.assertIn("not be interpreted as final test performance", report)

    def test_report_includes_excluded_experiments(self):
        ranking = {
            "metric": "f1",
            "direction": "maximize",
            "ranked": [],
            "excluded": [{"id": 7, "reason": "Missing evaluation metadata."}],
        }

        report = generate_markdown_report([], ranking)

        self.assertIn("Experiment ID 7", report)
        self.assertIn("Missing evaluation metadata.", report)


if __name__ == "__main__":
    unittest.main()
