"""Tests for experiment comparison and ranking."""

import math
import unittest

from labrat.experiments.comparator import rank_experiments


def make_experiment(
    experiment_id: int,
    metric_value: object = 0.8,
    dataset: str = "breast_cancer",
) -> dict:
    """Build a minimal experiment result for testing."""
    return {
        "id": experiment_id,
        "experiment_name": f"experiment_{experiment_id}",
        "dataset": dataset,
        "model": "logistic_regression",
        "metrics": {"f1": metric_value},
    }


class TestRankExperiments(unittest.TestCase):
    def test_maximize_ranks_highest_value_first(self):
        experiments = [
            make_experiment(1, 0.7),
            make_experiment(2, 0.9),
            make_experiment(3, 0.8),
        ]

        result = rank_experiments(experiments, "f1")

        self.assertEqual([item["id"] for item in result["ranked"]], [2, 3, 1])
        self.assertEqual(
            [item["rank"] for item in result["ranked"]],
            [1, 2, 3],
        )

    def test_minimize_ranks_lowest_value_first(self):
        experiments = [
            make_experiment(1, 0.7),
            make_experiment(2, 0.9),
            make_experiment(3, 0.8),
        ]

        result = rank_experiments(experiments, "f1", "minimize")

        self.assertEqual([item["id"] for item in result["ranked"]], [1, 3, 2])

    def test_equal_values_receive_competition_ranks(self):
        experiments = [
            make_experiment(1, 0.9),
            make_experiment(2, 0.9),
            make_experiment(3, 0.7),
        ]

        result = rank_experiments(experiments, "f1")

        self.assertEqual(
            [item["rank"] for item in result["ranked"]],
            [1, 1, 3],
        )

    def test_missing_and_invalid_metrics_are_excluded(self):
        experiments = [
            make_experiment(1, 0.9),
            make_experiment(2, None),
            make_experiment(3, True),
            make_experiment(4, "0.8"),
        ]

        result = rank_experiments(experiments, "f1")

        self.assertEqual([item["id"] for item in result["ranked"]], [1])
        self.assertEqual(len(result["excluded"]), 3)
        self.assertEqual(
            {item["id"] for item in result["excluded"]},
            {2, 3, 4},
        )

    def test_non_finite_metrics_are_excluded(self):
        experiments = [
            make_experiment(1, 0.9),
            make_experiment(2, math.nan),
            make_experiment(3, math.inf),
            make_experiment(4, -math.inf),
        ]

        result = rank_experiments(experiments, "f1")

        self.assertEqual([item["id"] for item in result["ranked"]], [1])
        self.assertEqual(len(result["excluded"]), 3)
        self.assertTrue(all("finite" in item["reason"] for item in result["excluded"]))

    def test_missing_metric_name_excludes_all_experiments(self):
        result = rank_experiments([make_experiment(1)], "accuracy")

        self.assertEqual(result["ranked"], [])
        self.assertEqual(len(result["excluded"]), 1)

    def test_empty_input_returns_empty_results(self):
        result = rank_experiments([], "f1")

        self.assertEqual(result["ranked"], [])
        self.assertEqual(result["excluded"], [])

    def test_empty_metric_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "metric must not be empty"):
            rank_experiments([], " ")

    def test_invalid_direction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "direction"):
            rank_experiments([], "f1", "sideways")

    def test_mixed_datasets_are_rejected(self):
        experiments = [
            make_experiment(1, 0.9, "breast_cancer"),
            make_experiment(2, 0.8, "iris"),
        ]

        with self.assertRaisesRegex(ValueError, "different datasets"):
            rank_experiments(experiments, "f1")


if __name__ == "__main__":
    unittest.main()
