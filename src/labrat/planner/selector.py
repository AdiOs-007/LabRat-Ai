"""Select candidates for a bounded model-tuning campaign."""

from math import isfinite, log10
from typing import Any


def select_next_candidate(
    completed_trials: list[dict[str, Any]],
    remaining_candidates: list[float],
    metric: str,
    direction: str = "maximize",
) -> float:
    """Choose a candidate using observed results and logarithmic proximity."""
    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be 'maximize' or 'minimize'.")

    if not remaining_candidates:
        raise ValueError("No remaining candidates.")

    if any(candidate <= 0 for candidate in remaining_candidates):
        raise ValueError("Candidates must be positive.")

    if not completed_trials:
        center = min(
            remaining_candidates,
            key=lambda candidate: abs(candidate - 1.0),
        )
        return center

    valid_trials = []
    for trial in completed_trials:
        metrics = trial.get("metrics")
        parameters = trial.get("model_parameters")

        if not isinstance(metrics, dict) or not isinstance(parameters, dict):
            continue

        metric_value = metrics.get(metric)
        candidate = parameters.get("C")

        if (
            isinstance(metric_value, bool)
            or not isinstance(metric_value, (int, float))
            or not isfinite(metric_value)
        ):
            continue

        if (
            isinstance(candidate, bool)
            or not isinstance(candidate, (int, float))
            or not isfinite(candidate)
            or candidate <= 0
        ):
            continue

        valid_trials.append(trial)

    if not valid_trials:
        return min(
            remaining_candidates,
            key=lambda candidate: abs(candidate - 1.0),
        )

    best_trial = (max if direction == "maximize" else min)(
        valid_trials,
        key=lambda trial: trial["metrics"][metric],
    )
    best_c = best_trial["model_parameters"]["C"]

    return min(
        remaining_candidates,
        key=lambda candidate: abs(log10(candidate) - log10(best_c)),
    )
