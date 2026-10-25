"""Run experiments using a validation split for model selection."""

from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from labrat.core.config import load_config
from labrat.experiments.builder import build_model, load_dataset
from labrat.runners.experiment_runner import fingerprint_dataset
from labrat.storage.sqlite_backend import SQLiteBackend


def run_validation_experiment(
    config_path: str | Path,
    database_path: str | Path | None = None,
    validation_size: float = 0.25,
) -> dict[str, Any]:
    """Train on one split and score on validation data, preserving a test holdout."""
    if (
        isinstance(validation_size, bool)
        or not isinstance(validation_size, (int, float))
        or not isfinite(validation_size)
        or not 0 < validation_size < 1
    ):
        raise ValueError("validation_size must be between 0 and 1.")

    config = load_config(config_path)
    X, y, class_names = load_dataset(config["dataset"])

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=validation_size,
        random_state=config["random_state"],
        stratify=y_development,
    )

    model = build_model(config["model"], config["model_parameters"])

    start_time = perf_counter()
    model.fit(X_train, y_train)
    training_seconds = perf_counter() - start_time

    predictions = model.predict(X_validation)

    metric_functions = {
        "accuracy": lambda: accuracy_score(y_validation, predictions),
        "precision": lambda: precision_score(
            y_validation, predictions, pos_label=0, zero_division=0
        ),
        "recall": lambda: recall_score(
            y_validation, predictions, pos_label=0, zero_division=0
        ),
        "f1": lambda: f1_score(y_validation, predictions, pos_label=0, zero_division=0),
    }

    metrics = {name: float(metric_functions[name]()) for name in config["metrics"]}

    result = {
        "experiment_name": config["experiment_name"],
        "dataset": config["dataset"],
        "dataset_fingerprint": fingerprint_dataset(X, y),
        "model": config["model"],
        "model_parameters": config["model_parameters"],
        "random_state": config["random_state"],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "positive_class": class_names[0],
        "metrics": metrics,
        "training_seconds": training_seconds,
        "evaluation": {
            "test_size": config["test_size"],
            "validation_size": validation_size,
            "validation_samples": len(X_validation),
            "random_state": config["random_state"],
            "stratify": True,
            "protocol_version": 2,
            "metric_source": "validation",
        },
    }

    if database_path is not None:
        storage = SQLiteBackend(database_path)
        try:
            result["id"] = storage.save_experiment(result)
        finally:
            storage.close()

    return result
