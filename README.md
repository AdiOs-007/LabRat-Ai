# LabRat AI

LabRat AI is an autonomous machine learning experimentation agent in development.

## Vision

The goal is to build a system that can inspect datasets, establish baseline
models, diagnose performance, propose controlled experiments, execute them,
record results, and use observed outcomes to guide future experiments.

## Planned capabilities

- Dataset inspection and task configuration
- Baseline model training and evaluation
- Diagnosis-guided experiment selection
- Budget-aware experiment execution
- Experiment history and reproducibility
- Results comparison and report generation
- Future deep learning and remote compute support

## Design principles

- Base decisions on measured experiment results.
- Keep configurations and evaluations reproducible.
- Validate configurations and enforce execution limits.
- Separate planning, execution, storage, and reporting.
- Compare against appropriate baseline search strategies.

## Current status

Early setup phase. The package structure and development environment
are being prepared. The experimentation engine is not implemented yet.

## Development environment

- Python 3.12
- Project-local virtual environment: `.venv`
- Git for version control

## Development

Activate the environment from the project root:

    source .venv/bin/activate

Project packaging configuration is maintained in `pyproject.toml`.
