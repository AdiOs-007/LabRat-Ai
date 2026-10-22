"""Tests for the LabRat AI command-line interface."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from labrat.cli import main


class TestCLI(unittest.TestCase):
    """Test CLI commands using an isolated SQLite database."""

    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "experiments.sqlite3"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_run_and_show_experiment(self) -> None:
        exit_code = main(
            [
                "run",
                "configs/baseline.json",
                "--database",
                str(self.database),
            ]
        )
        self.assertEqual(exit_code, 0)

        exit_code = main(["show", "1", "--database", str(self.database)])
        self.assertEqual(exit_code, 0)

    def test_history_with_no_experiments(self) -> None:
        exit_code = main(["history", "--database", str(self.database)])
        self.assertEqual(exit_code, 0)

    def test_show_missing_experiment(self) -> None:
        exit_code = main(["show", "999", "--database", str(self.database)])
        self.assertEqual(exit_code, 1)

    def test_invalid_history_limit(self) -> None:
        with self.assertRaises(SystemExit) as context:
            main(["history", "--limit", "0"])
        self.assertEqual(context.exception.code, 2)

    def test_compare_with_no_experiments(self) -> None:
        exit_code = main(
            [
                "compare",
                "--metric",
                "f1",
                "--database",
                str(self.database),
            ]
        )
        self.assertEqual(exit_code, 0)

    def test_compare_after_running_experiment(self) -> None:
        main(
            [
                "run",
                "configs/baseline.json",
                "--database",
                str(self.database),
            ]
        )

        exit_code = main(
            [
                "compare",
                "--metric",
                "f1",
                "--direction",
                "maximize",
                "--database",
                str(self.database),
            ]
        )
        self.assertEqual(exit_code, 0)

    def test_invalid_compare_limit(self) -> None:
        with self.assertRaises(SystemExit) as context:
            main(
                [
                    "compare",
                    "--metric",
                    "f1",
                    "--limit",
                    "0",
                    "--database",
                    str(self.database),
                ]
            )
        self.assertEqual(context.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
