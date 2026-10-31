"""Load and validate LabRat AI experiment configurations."""

import json
from math import isfinite
from pathlib import Path
from typing import Any

SUPPORTED_DATASETS = {"breast_cancer"}
SUPPORTED_MODELS = {"logistic_regression", "random_forest"}
SUPPORTED_METRICS = {"accuracy", "precision", "recall", "f1"}

MODEL_PARAMETERS = {
    "logistic_regression": {"C", "max_iter", "random_state"},
    "random_forest": {
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "random_state",
    },
}


def _validate_model_parameters(model: str, parameters: dict[str, Any]) -> None:
    """Validate supported model-specific hyperparameters."""
    unsupported = set(parameters) - MODEL_PARAMETERS[model]
    if unsupported:
        raise ValueError(f"Unsupported parameters for {model}: {sorted(unsupported)}")

    if model == "logistic_regression":
        max_iter = parameters.get("max_iter")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("model_parameters.max_iter must be a positive integer.")

        if "C" in parameters:
            c_value = parameters["C"]
            if (
                isinstance(c_value, bool)
                or not isinstance(c_value, (int, float))
                or not isfinite(c_value)
                or c_value <= 0
            ):
                raise ValueError("model_parameters.C must be a finite positive number.")

    elif model == "random_forest":
        n_estimators = parameters.get("n_estimators", 100)
        if (
            isinstance(n_estimators, bool)
            or not isinstance(n_estimators, int)
            or n_estimators < 1
        ):
            raise ValueError(
                "model_parameters.n_estimators must be a positive integer."
            )

        max_depth = parameters.get("max_depth")
        if max_depth is not None and (
            isinstance(max_depth, bool)
            or not isinstance(max_depth, int)
            or max_depth < 1
        ):
            raise ValueError(
                "model_parameters.max_depth must be a positive integer or null."
            )

        min_samples_split = parameters.get("min_samples_split", 2)
        if (
            isinstance(min_samples_split, bool)
            or not isinstance(min_samples_split, int)
            or min_samples_split < 2
        ):
            raise ValueError(
                "model_parameters.min_samples_split must be an integer >= 2."
            )

    if "random_state" in parameters:
        random_state = parameters["random_state"]
        if random_state is not None and (
            isinstance(random_state, bool)
            or not isinstance(random_state, int)
            or random_state < 0
        ):
            raise ValueError(
                "model_parameters.random_state must be a non-negative integer or null."
            )


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON experiment configuration and validate its fields."""
    config_path = Path(path)

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")

    required_fields = {
        "experiment_name",
        "dataset",
        "model",
        "test_size",
        "random_state",
        "model_parameters",
        "metrics",
    }
    missing_fields = required_fields - config.keys()
    if missing_fields:
        raise ValueError(f"Missing required fields: {sorted(missing_fields)}")

    if (
        not isinstance(config["experiment_name"], str)
        or not config["experiment_name"].strip()
    ):
        raise ValueError("experiment_name must be a non-empty string.")

    if config["dataset"] not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")

    if config["model"] not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {config['model']}")

    test_size = config["test_size"]
    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, (int, float))
        or not isfinite(test_size)
        or not 0 < test_size < 1
    ):
        raise ValueError("test_size must be a number strictly between 0 and 1.")

    random_state = config["random_state"]
    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, int)
        or random_state < 0
    ):
        raise ValueError("random_state must be a non-negative integer.")

    parameters = config["model_parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("model_parameters must be a JSON object.")

    _validate_model_parameters(config["model"], parameters)

    metrics = config["metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("metrics must be a non-empty list.")

    if any(not isinstance(metric, str) for metric in metrics):
        raise ValueError("Every metric must be a string.")

    if len(metrics) != len(set(metrics)):
        raise ValueError("metrics must not contain duplicates.")

    unsupported_metrics = set(metrics) - SUPPORTED_METRICS
    if unsupported_metrics:
        raise ValueError(f"Unsupported metrics: {sorted(unsupported_metrics)}")

    return config
