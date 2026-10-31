"""Command-line interface for LabRat AI."""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from labrat.experiments.comparator import rank_experiments
from labrat.reporting import generate_markdown_report
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


def handle_report(args: argparse.Namespace) -> int:
    """Export a Markdown report with separate rankings per evaluation group."""
    storage = SQLiteBackend(args.database)
    try:
        experiments = storage.list_experiments(limit=args.limit)
    finally:
        storage.close()

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for experiment in experiments:
        evaluation = experiment.get("evaluation")
        group_key = (
            str(experiment.get("dataset")),
            str(experiment.get("dataset_fingerprint")),
            json.dumps(evaluation, sort_keys=True),
        )
        groups.setdefault(group_key, []).append(experiment)

    if not groups:
        ranking = rank_experiments(
            [],
            metric=args.metric,
            direction=args.direction,
        )
        report = generate_markdown_report([], ranking)
    else:
        report_lines = [
            "# LabRat AI — Experiment Report",
            "",
            f"- **Experiments available:** {len(experiments)}",
            f"- **Evaluation groups:** {len(groups)}",
            f"- **Ranking metric:** `{args.metric}`",
            f"- **Optimization direction:** {args.direction}",
            "",
        ]

        for index, group in enumerate(groups.values(), start=1):
            ranking = rank_experiments(
                group,
                metric=args.metric,
                direction=args.direction,
            )
            group_report = generate_markdown_report(group, ranking)
            evaluation = group[0].get("evaluation")
            evaluation = evaluation if isinstance(evaluation, dict) else {}

            report_lines.extend(
                [
                    f"## Evaluation group {index}",
                    "",
                    f"- **Dataset:** `{group[0].get('dataset')}`",
                    f"- **Metric source:** "
                    f"`{evaluation.get('metric_source', 'unspecified')}`",
                    f"- **Protocol version:** "
                    f"`{evaluation.get('protocol_version', 'unspecified')}`",
                    "",
                ]
            )

            for line in group_report.splitlines()[1:]:
                report_lines.append(
                    line.replace("## ", "### ", 1) if line.startswith("## ") else line
                )

            report_lines.append("")

        report = "\n".join(report_lines)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Report saved to: {args.output}")
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

    report_parser = subparsers.add_parser(
        "report",
        help="Export a Markdown report of stored experiments.",
    )
    report_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/labrat-report.md"),
        help="Markdown report path.",
    )
    report_parser.add_argument(
        "--metric",
        default="f1",
        help="Metric used to rank experiments (default: f1).",
    )
    report_parser.add_argument(
        "--direction",
        choices=["maximize", "minimize"],
        default="maximize",
        help="Whether higher or lower metric values are better.",
    )
    report_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of recent experiments to include.",
    )
    report_parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help="SQLite database path.",
    )
    report_parser.set_defaults(handler=handle_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return its exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {"history", "compare", "report"} and args.limit < 1:
        parser.error("--limit must be at least 1")

    try:
        return args.handler(args)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
