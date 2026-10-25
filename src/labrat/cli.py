"""Command-line interface for LabRat AI."""

import argparse
import json
from pathlib import Path
from typing import Any

from labrat.experiments.comparator import rank_experiments
from labrat.runners.campaign_runner import run_campaign
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


def handle_campaign(args: argparse.Namespace) -> int:
    """Run a bounded campaign and display its ranking."""
    result = run_campaign(
        args.config,
        database_path=args.database,
        max_trials=args.max_trials,
        metric=args.metric,
        direction=args.direction,
        validation_size=args.validation_size,
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


def handle_compare(args: argparse.Namespace) -> int:
    """Rank stored experiments by a selected metric."""
    storage = SQLiteBackend(args.database)
    try:
        experiments = storage.list_experiments(limit=args.limit)
    finally:
        storage.close()

    result = rank_experiments(
        experiments,
        metric=args.metric,
        direction=args.direction,
    )

    print_json(result)
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

    compare_parser = subparsers.add_parser(
        "compare",
        help="Rank stored experiments by a selected metric.",
    )
    compare_parser.add_argument(
        "--metric",
        required=True,
        help="Metric to rank by, such as f1 or accuracy.",
    )
    compare_parser.add_argument(
        "--direction",
        choices=["maximize", "minimize"],
        default="maximize",
        help="Whether higher or lower metric values are better.",
    )
    compare_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of recent experiments to compare.",
    )
    compare_parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help="SQLite database path (default: data/experiments.sqlite3).",
    )
    compare_parser.set_defaults(handler=handle_compare)

    campaign_parser = subparsers.add_parser(
        "campaign",
        help="Run and rank a bounded set of model experiments.",
    )
    campaign_parser.add_argument(
        "config",
        help="Path to the experiment JSON configuration.",
    )
    campaign_parser.add_argument(
        "--max-trials",
        type=int,
        default=5,
        help="Maximum number of candidate experiments (1-5; default: 5).",
    )
    campaign_parser.add_argument(
        "--metric",
        default="f1",
        help="Metric used to rank candidates (default: f1).",
    )
    campaign_parser.add_argument(
        "--direction",
        choices=["maximize", "minimize"],
        default="maximize",
        help="Whether higher or lower metric values are better.",
    )
    campaign_parser.add_argument(
        "--validation-size",
        type=float,
        default=0.25,
        help="Fraction of development data used for validation (default: 0.25).",
    )
    campaign_parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help="SQLite database path (default: data/experiments.sqlite3).",
    )
    campaign_parser.set_defaults(handler=handle_campaign)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return its exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {"history", "compare"} and args.limit < 1:
        parser.error("--limit must be at least 1")

    try:
        return args.handler(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
