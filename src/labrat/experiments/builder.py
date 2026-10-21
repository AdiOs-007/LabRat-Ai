"""Build datasets and machine learning pipelines for experiments."""

from typing import Any

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset(name: str) -> tuple[Any, Any, list[str]]:
    """Load a supported dataset and return features, targets, and class names."""
    if name == "breast_cancer":
        dataset = load_breast_cancer()
        return dataset.data, dataset.target, list(dataset.target_names)

    raise ValueError(f"Unsupported dataset: {name}")


def build_model(model_name: str, parameters: dict[str, Any]) -> Pipeline:
    """Build a supported model with preprocessing inside its pipeline."""
    if model_name == "logistic_regression":
        model = LogisticRegression(**parameters)
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", model),
            ]
        )

    raise ValueError(f"Unsupported model: {model_name}")
