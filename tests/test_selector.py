"""Tests for adaptive candidate selection."""

import unittest

from labrat.planner.selector import select_next_candidate


class TestCandidateSelector(unittest.TestCase):
    def setUp(self):
        self.candidates = [0.01, 0.1, 1.0, 10.0, 100.0]

    def test_starts_at_center_candidate(self):
        self.assertEqual(
            select_next_candidate([], self.candidates, "f1"),
            1.0,
        )

    def test_selects_nearest_candidate_to_best_result(self):
        trials = [
            {
                "metrics": {"f1": 0.98},
                "model_parameters": {"C": 1.0},
            }
        ]
        self.assertEqual(
            select_next_candidate(trials, [0.01, 0.1, 10.0, 100.0], "f1"),
            0.1,
        )

    def test_supports_minimize_direction(self):
        trials = [
            {"metrics": {"f1": 0.20}, "model_parameters": {"C": 1.0}},
            {"metrics": {"f1": 0.10}, "model_parameters": {"C": 10.0}},
        ]
        self.assertEqual(
            select_next_candidate(
                trials,
                [0.01, 0.1, 100.0],
                "f1",
                direction="minimize",
            ),
            100.0,
        )

    def test_rejects_invalid_direction(self):
        with self.assertRaises(ValueError):
            select_next_candidate([], self.candidates, "f1", direction="sideways")

    def test_rejects_empty_candidates(self):
        with self.assertRaises(ValueError):
            select_next_candidate([], [], "f1")

    def test_rejects_nonpositive_candidates(self):
        with self.assertRaises(ValueError):
            select_next_candidate([], [0.0, 1.0], "f1")

    def test_ignores_trials_missing_requested_metric(self):
        trials = [
            {
                "metrics": {"accuracy": 0.99},
                "model_parameters": {"C": 100.0},
            },
            {
                "metrics": {"f1": 0.95},
                "model_parameters": {"C": 10.0},
            },
        ]

        selected = select_next_candidate(
            trials,
            [0.1, 1.0, 100.0],
            "f1",
        )

        self.assertEqual(selected, 1.0)

    def test_falls_back_when_all_trials_are_unusable(self):
        trials = [
            {"metrics": {"accuracy": 0.99}, "model_parameters": {"C": 10.0}},
            {"metrics": {"f1": "invalid"}, "model_parameters": {"C": 100.0}},
        ]

        selected = select_next_candidate(
            trials,
            [0.1, 1.0, 10.0],
            "f1",
        )

        self.assertEqual(selected, 1.0)

    def test_ignores_trials_with_missing_model_parameters(self):
        trials = [
            {"metrics": {"f1": 0.99}},
            {"metrics": {"f1": 0.90}, "model_parameters": {"C": 10.0}},
        ]

        selected = select_next_candidate(
            trials,
            [0.1, 1.0, 100.0],
            "f1",
        )

        self.assertEqual(selected, 1.0)


if __name__ == "__main__":
    unittest.main()
