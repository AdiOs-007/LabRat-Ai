"""Execute configured machine learning experiments."""

from pathlib import Path
from time import perf_counter
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from labrat.core.config import load_config
from labrat.experiments.builder import build_model, load_dataset
from labrat.storage.sqlite_backend import SQLiteBackend


def run_experiment(
    config_path: str | Path,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train, evaluate, and optionally persist a configured experiment."""
    config = load_config(config_path)

    X, y, class_names = load_dataset(config["dataset"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )

    model = build_model(config["model"], config["model_parameters"])

    start_time = perf_counter()
    model.fit(X_train, y_train)
    training_seconds = perf_counter() - start_time

    predictions = model.predict(X_test)

    metric_functions = {
        "accuracy": lambda: accuracy_score(y_test, predictions),
        "precision": lambda: precision_score(
            y_test, predictions, pos_label=0, zero_division=0
        ),
        "recall": lambda: recall_score(
            y_test, predictions, pos_label=0, zero_division=0
        ),
        "f1": lambda: f1_score(y_test, predictions, pos_label=0, zero_division=0),
    }

    metrics = {name: float(metric_functions[name]()) for name in config["metrics"]}

    result = {
        "experiment_name": config["experiment_name"],
        "dataset": config["dataset"],
        "model": config["model"],
        "random_state": config["random_state"],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "positive_class": class_names[0],
        "metrics": metrics,
        "training_seconds": training_seconds,
    }

    if database_path is not None:
        storage = SQLiteBackend(database_path)
        try:
            result["id"] = storage.save_experiment(result)
        finally:
            storage.close()

    return result
