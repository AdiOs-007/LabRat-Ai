"""Run bounded campaigns of validation-based ML experiments."""

import json
import tempfile
from pathlib import Path
from typing import Any

from labrat.core.config import load_config
from labrat.experiments.comparator import rank_experiments
from labrat.runners.validation_runner import run_validation_experiment

DEFAULT_CANDIDATE_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)


def run_campaign(
    config_path: str | Path,
    database_path: str | Path,
    max_trials: int = 5,
    metric: str = "f1",
    direction: str = "maximize",
    validation_size: float = 0.25,
) -> dict[str, Any]:
    """Evaluate a bounded set of logistic-regression C values."""
    if (
        isinstance(max_trials, bool)
        or not isinstance(max_trials, int)
        or not 1 <= max_trials <= len(DEFAULT_CANDIDATE_VALUES)
    ):
        raise ValueError(
            f"max_trials must be between 1 and {len(DEFAULT_CANDIDATE_VALUES)}."
        )

    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be 'maximize' or 'minimize'.")

    base_config = load_config(config_path)

    if metric not in base_config["metrics"]:
        raise ValueError(
            f"Metric '{metric}' must be included in the configuration's metrics."
        )

    candidates = DEFAULT_CANDIDATE_VALUES[:max_trials]
    results = []

    with tempfile.TemporaryDirectory(prefix="labrat-campaign-") as directory:
        config_directory = Path(directory)

        for trial_number, c_value in enumerate(candidates, start=1):
            candidate_config = dict(base_config)
            candidate_config["experiment_name"] = (
                f"{base_config['experiment_name']}_campaign_trial_{trial_number}"
            )
            candidate_config["model_parameters"] = {
                **base_config["model_parameters"],
                "C": c_value,
            }

            candidate_path = config_directory / f"trial_{trial_number}.json"
            candidate_path.write_text(
                json.dumps(candidate_config, indent=2),
                encoding="utf-8",
            )

            result = run_validation_experiment(
                candidate_path,
                database_path=database_path,
                validation_size=validation_size,
            )
            results.append(result)

    ranking = rank_experiments(
        results,
        metric=metric,
        direction=direction,
    )

    return {
        "campaign_name": base_config["experiment_name"],
        "trials_requested": max_trials,
        "trials_completed": len(results),
        "metric": metric,
        "direction": direction,
        "candidate_results": results,
        "ranking": ranking,
    }
