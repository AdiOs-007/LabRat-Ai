"""SQLite implementation of LabRat AI experiment storage."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from labrat.storage.backend import ExperimentResult


class SQLiteBackend:
    """Persist experiment results in a local SQLite database."""

    def __init__(self, database_path: str | Path) -> None:
        """Open or create the database and initialize its schema."""
        self.database_path = str(database_path)

        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self.connection = sqlite3.connect(self.database_path)
        self.connection.row_factory = sqlite3.Row
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Create the experiment history table if it does not exist."""
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    model TEXT NOT NULL,
                    random_state INTEGER NOT NULL,
                    train_samples INTEGER NOT NULL,
                    test_samples INTEGER NOT NULL,
                    positive_class TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    training_seconds REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    dataset_fingerprint TEXT,
                    model_parameters_json TEXT,
                    evaluation_json TEXT
                )
                """
            )

            existing_columns = {
                row["name"]
                for row in self.connection.execute(
                    "PRAGMA table_info(experiments)"
                ).fetchall()
            }
            migrations = {
                "dataset_fingerprint": "TEXT",
                "model_parameters_json": "TEXT",
                "evaluation_json": "TEXT",
            }

            for column, column_type in migrations.items():
                if column not in existing_columns:
                    self.connection.execute(
                        f"ALTER TABLE experiments ADD COLUMN {column} {column_type}"
                    )

    def save_experiment(self, result: ExperimentResult) -> int:
        """Save a completed experiment and return its database ID."""
        required_fields = {
            "experiment_name",
            "dataset",
            "model",
            "random_state",
            "train_samples",
            "test_samples",
            "positive_class",
            "metrics",
            "training_seconds",
        }

        missing_fields = required_fields - result.keys()
        if missing_fields:
            raise ValueError(
                f"Missing experiment result fields: {sorted(missing_fields)}"
            )

        if not isinstance(result["metrics"], dict):
            raise ValueError("metrics must be a dictionary.")

        serialized_metadata = {}
        for field in ("model_parameters", "evaluation"):
            if field in result:
                if not isinstance(result[field], dict):
                    raise ValueError(f"{field} must be a dictionary.")
                serialized_metadata[field] = json.dumps(
                    result[field],
                    allow_nan=False,
                )

        metrics_json = json.dumps(
            result["metrics"],
            allow_nan=False,
        )
        created_at = datetime.now(UTC).isoformat()

        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO experiments (
                    experiment_name,
                    dataset,
                    model,
                    random_state,
                    train_samples,
                    test_samples,
                    positive_class,
                    metrics_json,
                    training_seconds,
                    created_at,
                    dataset_fingerprint,
                    model_parameters_json,
                    evaluation_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result["experiment_name"],
                    result["dataset"],
                    result["model"],
                    result["random_state"],
                    result["train_samples"],
                    result["test_samples"],
                    result["positive_class"],
                    metrics_json,
                    result["training_seconds"],
                    created_at,
                    result.get("dataset_fingerprint"),
                    serialized_metadata.get("model_parameters"),
                    serialized_metadata.get("evaluation"),
                ),
            )

        return int(cursor.lastrowid)

    def get_experiment(self, experiment_id: int) -> ExperimentResult | None:
        """Retrieve one experiment by ID, or return None if not found."""
        row = self.connection.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()

        if row is None:
            return None

        return self._row_to_result(row)

    def list_experiments(self, limit: int = 20) -> list[ExperimentResult]:
        """Return the most recent experiments first."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer.")

        rows = self.connection.execute(
            """
            SELECT * FROM experiments
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [self._row_to_result(row) for row in rows]

    @staticmethod
    def _row_to_result(row: sqlite3.Row) -> ExperimentResult:
        """Convert a database row to the public result format."""
        result = {
            "id": row["id"],
            "experiment_name": row["experiment_name"],
            "dataset": row["dataset"],
            "model": row["model"],
            "random_state": row["random_state"],
            "train_samples": row["train_samples"],
            "test_samples": row["test_samples"],
            "positive_class": row["positive_class"],
            "metrics": json.loads(row["metrics_json"]),
            "training_seconds": row["training_seconds"],
            "created_at": row["created_at"],
        }

        if row["dataset_fingerprint"] is not None:
            result["dataset_fingerprint"] = row["dataset_fingerprint"]
        if row["model_parameters_json"] is not None:
            result["model_parameters"] = json.loads(row["model_parameters_json"])
        if row["evaluation_json"] is not None:
            result["evaluation"] = json.loads(row["evaluation_json"])

        return result

    def close(self) -> None:
        """Close the database connection."""
        self.connection.close()
