"""Command-line interface for LabRat AI."""

import argparse
import json
from pathlib import Path
from typing import Any

from labrat.runners.experiment_runner import run_experiment
from labrat.storage.sqlite_backend import SQLiteBackend

DEFAULT_DATABASE = Path("data/experiments.sqlite3")


def print_json(data: dict[str, Any]) -> None:
    """Print a dictionary as readable JSON."""
    print(json.dumps(data, indent=2))


def handle_run(args: argparse.Namespace) -> int:
    """Run an experiment and persist its results."""
    result = run_experiment(
        args.config,
        database_path=args.database,
    )
    print_json(result)
    return 0


def handle_history(args: argparse.Namespace) -> int:
    """Display recent experiment history."""
    storage = SQLiteBackend(args.database)
    try:
        experiments = storage.list_experiments(limit=args.limit)
    finally:
        storage.close()

    if not experiments:
        print("No experiments found.")
        return 0

    for experiment in experiments:
        metrics = json.dumps(experiment["metrics"], sort_keys=True)
        print(
            f"ID: {experiment['id']} | "
            f"{experiment['experiment_name']} | "
            f"Model: {experiment['model']} | "
            f"Metrics: {metrics}"
        )

    return 0


def handle_show(args: argparse.Namespace) -> int:
    """Display the details of one experiment."""
    storage = SQLiteBackend(args.database)
    try:
        experiment = storage.get_experiment(args.experiment_id)
    finally:
        storage.close()

    if experiment is None:
        print(f"Experiment {args.experiment_id} not found.")
        return 1

    print_json(experiment)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the LabRat AI command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="labrat",
        description="Run and inspect LabRat AI experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run an experiment from a JSON configuration.",
    )
    run_parser.add_argument("config", help="Path to the experiment JSON file.")
    run_parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help="SQLite database path (default: data/experiments.sqlite3).",
    )
    run_parser.set_defaults(handler=handle_run)

    history_parser = subparsers.add_parser(
        "history",
        help="List recent experiments.",
    )
    history_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of experiments to display (default: 20).",
    )
    history_parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help="SQLite database path (default: data/experiments.sqlite3).",
    )
    history_parser.set_defaults(handler=handle_history)

    show_parser = subparsers.add_parser(
        "show",
        help="Display one experiment by its database ID.",
    )
    show_parser.add_argument(
        "experiment_id",
        type=int,
        help="Database ID of the experiment.",
    )
    show_parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help="SQLite database path (default: data/experiments.sqlite3).",
    )
    show_parser.set_defaults(handler=handle_show)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return its exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "history" and args.limit < 1:
        parser.error("--limit must be at least 1")

    try:
        return args.handler(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
