"""Run bounded campaigns of validation-based ML experiments."""

import json
import tempfile
from pathlib import Path
from typing import Any

from labrat.core.config import load_config
from labrat.experiments.comparator import rank_experiments
from labrat.planner.selector import select_next_candidate
from labrat.runners.validation_runner import run_validation_experiment

DEFAULT_CANDIDATE_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)

RANDOM_FOREST_CANDIDATES = (
    {"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
    {"n_estimators": 200, "max_depth": None, "min_samples_split": 2},
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2},
    {"n_estimators": 200, "max_depth": 20, "min_samples_split": 5},
    {"n_estimators": 300, "max_depth": 20, "min_samples_split": 10},
)


def run_campaign(
    config_path: str | Path,
    database_path: str | Path,
    max_trials: int = 5,
    metric: str = "f1",
    direction: str = "maximize",
    validation_size: float = 0.25,
) -> dict[str, Any]:
    """Run a bounded, model-specific hyperparameter campaign."""
    base_config = load_config(config_path)

    if base_config["model"] == "logistic_regression":
        candidate_limit = len(DEFAULT_CANDIDATE_VALUES)
    elif base_config["model"] == "random_forest":
        candidate_limit = len(RANDOM_FOREST_CANDIDATES)
    else:
        raise ValueError(f"Unsupported campaign model: {base_config['model']}")

    if (
        isinstance(max_trials, bool)
        or not isinstance(max_trials, int)
        or not 1 <= max_trials <= candidate_limit
    ):
        raise ValueError(f"max_trials must be between 1 and {candidate_limit}.")

    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be 'maximize' or 'minimize'.")

    if metric not in base_config["metrics"]:
        raise ValueError(
            f"Metric '{metric}' must be included in the configuration's metrics."
        )

    results = []

    with tempfile.TemporaryDirectory(prefix="labrat-campaign-") as directory:
        config_directory = Path(directory)

        for trial_number in range(1, max_trials + 1):
            candidate_config = dict(base_config)

            if base_config["model"] == "logistic_regression":
                remaining_candidates = [
                    value
                    for value in DEFAULT_CANDIDATE_VALUES
                    if value
                    not in [result["model_parameters"]["C"] for result in results]
                ]
                candidate_value = select_next_candidate(
                    results,
                    remaining_candidates,
                    metric,
                    direction,
                )
                candidate_config["model_parameters"] = {
                    **base_config["model_parameters"],
                    "C": candidate_value,
                }
            else:
                remaining_candidates = [
                    candidate
                    for candidate in RANDOM_FOREST_CANDIDATES
                    if candidate
                    not in [
                        {key: result["model_parameters"].get(key) for key in candidate}
                        for result in results
                    ]
                ]
                candidate = remaining_candidates[0]
                candidate_config["model_parameters"] = {
                    **base_config["model_parameters"],
                    **candidate,
                }

            candidate_config["experiment_name"] = (
                f"{base_config['experiment_name']}_campaign_trial_{trial_number}"
            )

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
