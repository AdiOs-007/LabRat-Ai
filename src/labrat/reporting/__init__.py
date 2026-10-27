"""Generate Markdown reports from stored experiment results."""

from datetime import UTC, datetime
from typing import Any


def generate_markdown_report(
    experiments: list[dict[str, Any]],
    ranking: dict[str, Any],
) -> str:
    """Create a Markdown summary of ranked experiments."""
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    ranked = ranking["ranked"]
    excluded = ranking["excluded"]
    metric = ranking["metric"]
    direction = ranking["direction"]

    lines = [
        "# LabRat AI — Experiment Report",
        "",
        f"- **Generated:** {generated_at}",
        f"- **Experiments available:** {len(experiments)}",
        f"- **Ranking metric:** `{metric}`",
        f"- **Optimization direction:** {direction}",
        "",
        "## Ranked Results",
        "",
    ]

    if not ranked:
        lines.append("No experiments could be ranked.")
    else:
        lines.extend(
            [
                "| Rank | ID | Experiment | Model | Metric | Score |",
                "|---:|---:|---|---|---|---:|",
            ]
        )

        for item in ranked:
            lines.append(
                f"| {item['rank']} | {item['id']} | "
                f"{item['experiment_name']} | {item['model']} | "
                f"{item['metric']} | {item['value']:.6f} |"
            )

        winner_id = ranked[0]["id"]
        winner = next(
            (item for item in experiments if item.get("id") == winner_id),
            None,
        )

        if winner is not None:
            evaluation = winner.get("evaluation", {})
            metrics = winner.get("metrics", {})
            parameters = winner.get("model_parameters", {})

            lines.extend(
                [
                    "",
                    "## Best Configuration",
                    "",
                    f"- **Experiment:** {winner['experiment_name']}",
                    f"- **Model:** `{winner['model']}`",
                    f"- **Dataset:** `{winner['dataset']}`",
                    f"- **Parameters:** `{parameters}`",
                    "",
                    "### Metrics",
                    "",
                    "| Metric | Value |",
                    "|---|---:|",
                ]
            )

            for name, value in sorted(metrics.items()):
                if isinstance(value, int | float) and not isinstance(value, bool):
                    lines.append(f"| {name} | {value:.6f} |")
                else:
                    lines.append(f"| {name} | {value} |")

            metric_source = evaluation.get("metric_source", "unspecified")
            lines.extend(
                [
                    "",
                    "### Evaluation Protocol",
                    "",
                    f"- **Metric source:** `{metric_source}`",
                    f"- **Evaluation settings:** `{evaluation}`",
                    "- Scores describe the recorded evaluation split; "
                    "they should not be interpreted as final test performance "
                    "unless the recorded metric source confirms that.",
                ]
            )

    lines.extend(["", "## Excluded Experiments", ""])

    if excluded:
        for item in excluded:
            lines.append(f"- Experiment ID {item.get('id')}: {item['reason']}")
    else:
        lines.append("None.")

    lines.append("")
    return "\n".join(lines)
