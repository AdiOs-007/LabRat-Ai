"""Select candidates for a bounded model-tuning campaign."""

from math import log10
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

    valid_trials = [
        trial
        for trial in completed_trials
        if metric in trial.get("metrics", {})
        and isinstance(trial.get("model_parameters"), dict)
        and "C" in trial["model_parameters"]
    ]

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
