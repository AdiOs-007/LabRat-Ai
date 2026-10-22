"""Compare and rank stored machine learning experiment results."""

import math
from typing import Any


def rank_experiments(
    experiments: list[dict[str, Any]],
    metric: str,
    direction: str = "maximize",
) -> dict[str, Any]:
    """Rank comparable experiments by a selected metric."""
    if not metric.strip():
        raise ValueError("metric must not be empty.")

    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be 'maximize' or 'minimize'.")

    if not experiments:
        return {
            "metric": metric,
            "direction": direction,
            "ranked": [],
            "excluded": [],
        }

    datasets = {experiment["dataset"] for experiment in experiments}
    if len(datasets) > 1:
        raise ValueError(
            "Experiments from different datasets cannot be compared together."
        )

    ranked_candidates = []
    excluded = []

    for experiment in experiments:
        metrics = experiment.get("metrics", {})
        value = metrics.get(metric)

        if isinstance(value, bool) or not isinstance(value, int | float):
            excluded.append(
                {
                    "id": experiment.get("id"),
                    "reason": f"Metric '{metric}' is missing or not numeric.",
                }
            )
            continue

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            excluded.append(
                {
                    "id": experiment.get("id"),
                    "reason": f"Metric '{metric}' is not a finite number.",
                }
            )
            continue

        ranked_candidates.append((experiment, numeric_value))

    ranked_candidates.sort(
        key=lambda item: item[1],
        reverse=direction == "maximize",
    )

    ranked = []
    previous_value = None
    current_rank = 0

    for position, (experiment, value) in enumerate(ranked_candidates, start=1):
        if previous_value is None or value != previous_value:
            current_rank = position

        ranked.append(
            {
                "rank": current_rank,
                "id": experiment.get("id"),
                "experiment_name": experiment.get("experiment_name"),
                "dataset": experiment.get("dataset"),
                "model": experiment.get("model"),
                "metric": metric,
                "value": value,
            }
        )
        previous_value = value

    return {
        "metric": metric,
        "direction": direction,
        "ranked": ranked,
        "excluded": excluded,
    }
