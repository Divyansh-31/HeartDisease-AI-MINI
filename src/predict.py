"""Artifact loading and prediction helpers.

Training writes these artifacts under ``artifacts/``:

- ``models/registry.json``
- ``models/<model_key>.joblib``

The same loading path is used by evaluation and by any future UI, so saved
pipelines are exercised consistently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from src.data_loader import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIRNAME = "models"
REGISTRY_FILENAME = "registry.json"


def _resolve_artifacts_dir(artifacts_dir: str | Path | None) -> Path:
    return Path(artifacts_dir) if artifacts_dir is not None else DEFAULT_ARTIFACTS_DIR


def registry_path(artifacts_dir: str | Path | None = None) -> Path:
    """Return the expected path of the model registry."""
    return _resolve_artifacts_dir(artifacts_dir) / MODELS_DIRNAME / REGISTRY_FILENAME


def load_registry(artifacts_dir: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the model registry written by training."""
    path = registry_path(artifacts_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Model registry not found at {path}. "
            "Run `python -m src.train_models` first."
        )

    with path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)

    models = registry.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"Invalid model registry at {path}: no model entries found.")

    keys = [entry.get("key") for entry in models]
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError(f"Invalid model registry at {path}: every entry needs a key.")
    if len(keys) != len(set(keys)):
        raise ValueError(f"Invalid model registry at {path}: duplicate model keys.")

    feature_columns = registry.get("feature_columns")
    if feature_columns is not None and list(feature_columns) != FEATURE_COLUMNS:
        raise ValueError(
            f"Model registry at {path} was created for different feature columns. "
            "Retrain the models with the current dataset schema."
        )

    return registry


def _model_entries(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["key"]: entry for entry in registry["models"]}


def load_model(
    model_key: str,
    artifacts_dir: str | Path | None = None,
):
    """Load one fitted pipeline by registry key."""
    registry = load_registry(artifacts_dir)
    entries = _model_entries(registry)
    if model_key not in entries:
        available = ", ".join(entries)
        raise ValueError(
            f"Unknown model '{model_key}'. Available models: {available}."
        )

    entry = entries[model_key]
    artifact_name = entry.get("artifact")
    if not isinstance(artifact_name, str) or not artifact_name:
        raise ValueError(
            f"Registry entry for '{model_key}' does not name an artifact file."
        )

    artifact_path = (
        _resolve_artifacts_dir(artifacts_dir) / MODELS_DIRNAME / artifact_name
    )
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Artifact for model '{model_key}' not found at {artifact_path}. "
            "Re-run `python -m src.train_models`."
        )

    model = joblib.load(artifact_path)
    if not hasattr(model, "predict"):
        raise ValueError(
            f"Artifact at {artifact_path} does not contain a fitted model."
        )
    if "preprocessor" not in getattr(model, "named_steps", {}):
        raise ValueError(
            f"Artifact at {artifact_path} does not contain the expected "
            "preprocessing pipeline."
        )
    return model


def load_all_models(
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load every fitted pipeline in registry order."""
    registry = load_registry(artifacts_dir)
    return {
        entry["key"]: load_model(entry["key"], artifacts_dir)
        for entry in registry["models"]
    }


def predict_scores(model, X) -> tuple[np.ndarray | None, str | None]:
    """Return continuous positive-class scores and their source.

    ``predict_proba`` is used when the fitted estimator provides it. SVC is
    intentionally left uncalibrated (scikit-learn 1.9 deprecates
    ``SVC(probability=True)``), so its ``decision_function`` is used instead.
    Hard class predictions are never used as ROC-AUC scores.
    """
    row_count = len(X)

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        if probabilities.ndim != 2:
            raise ValueError("predict_proba must return a 2D array.")
        classes = np.asarray(getattr(model, "classes_", []))
        positive = np.flatnonzero(classes == 1)
        if positive.size != 1:
            raise ValueError(
                "Expected a binary model exposing exactly one positive class (1)."
            )
        scores = probabilities[:, int(positive[0])]
        score_type = "probability"
    elif hasattr(model, "decision_function"):
        raw_scores = np.asarray(model.decision_function(X), dtype=float)
        if raw_scores.ndim == 2:
            classes = np.asarray(getattr(model, "classes_", []))
            positive = np.flatnonzero(classes == 1)
            if raw_scores.shape[1] == 2 and positive.size == 1:
                raw_scores = raw_scores[:, int(positive[0])]
            else:
                raw_scores = raw_scores.ravel()
        scores = np.asarray(raw_scores, dtype=float).ravel()
        score_type = "decision_function"
    else:
        return None, None

    if scores.shape[0] != row_count:
        raise ValueError(
            "The model returned a different number of scores than input rows."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("The model produced non-finite prediction scores.")
    return scores, score_type


def predict_heart_disease(
    patient: Mapping[str, Any] | pd.DataFrame,
    model_key: str,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Predict one patient record with a saved pipeline.

    Parameters
    ----------
    patient:
        Mapping or one-row DataFrame containing exactly the 13 feature
        columns. Missing feature values may be ``NaN`` and are imputed by the
        fitted preprocessing pipeline.
    model_key:
        Registry key such as ``logistic_regression``.
    artifacts_dir:
        Artifacts directory. Defaults to ``<project>/artifacts``.

    Returns
    -------
    dict
        Prediction, human-readable label, continuous score (probability for
        probabilistic models, decision-function value for SVC), and score type.
    """
    if isinstance(patient, Mapping):
        frame = pd.DataFrame([dict(patient)])
    elif isinstance(patient, pd.DataFrame):
        frame = patient.copy()
    else:
        raise TypeError("patient must be a mapping or a pandas DataFrame.")

    if len(frame) != 1:
        raise ValueError("Exactly one patient record is required.")

    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in FEATURE_COLUMNS]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}.")
    if extra:
        raise ValueError(f"Unexpected columns: {extra}.")

    frame = frame.loc[:, FEATURE_COLUMNS].copy()
    non_numeric = [
        column
        for column in FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise ValueError(
            f"Feature columns must be numeric; received non-numeric values in: "
            f"{non_numeric}."
        )

    model = load_model(model_key, artifacts_dir)
    registry = load_registry(artifacts_dir)
    entry = _model_entries(registry)[model_key]

    prediction = int(np.asarray(model.predict(frame))[0])
    if prediction not in (0, 1):
        raise ValueError(f"Unexpected prediction value: {prediction}.")

    scores, score_type = predict_scores(model, frame)
    score = float(scores[0]) if scores is not None else None

    return {
        "model_key": model_key,
        "model_name": entry.get("display_name", model_key),
        "prediction": prediction,
        "label": "heart disease" if prediction == 1 else "no heart disease",
        "score": score,
        "score_type": score_type,
    }
