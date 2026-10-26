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


if __name__ == "__main__":
    unittest.main()
