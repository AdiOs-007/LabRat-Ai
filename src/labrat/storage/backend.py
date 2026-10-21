"""Common interface for LabRat AI experiment storage backends."""

from typing import Any, Protocol

ExperimentResult = dict[str, Any]


class StorageBackend(Protocol):
    """Operations required from every experiment storage backend."""

    def save_experiment(self, result: ExperimentResult) -> int:
        """Save an experiment result and return its unique ID."""
        ...

    def get_experiment(self, experiment_id: int) -> ExperimentResult | None:
        """Retrieve one experiment result, or None if it does not exist."""
        ...

    def list_experiments(self, limit: int = 20) -> list[ExperimentResult]:
        """Return recent experiment results, newest first."""
        ...

    def close(self) -> None:
        """Release resources held by the backend."""
        ...
