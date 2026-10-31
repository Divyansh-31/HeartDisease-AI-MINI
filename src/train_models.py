"""Reproducible training for the seven model configurations.

Protocol
--------
- One stratified train/test split, fixed seed (default 42), 20% test.
- Preprocessing is fitted inside every pipeline on training data only.
- Supervised models use all training labels.
- The self-training MLP uses only a reproducible subset of training labels;
  the remaining training rows are marked ``-1`` (unlabeled) as required by
  scikit-learn's ``SelfTrainingClassifier``. Test labels are never touched.
- Every model is evaluated on the same untouched test rows.

Seven configurations, in registry order:
1. Logistic Regression (supervised)
2. Decision Tree Classifier (supervised)
3. Support Vector Machine (supervised)
4. K-Nearest Neighbors (supervised)
5. Random Forest Classifier (supervised + ensemble)
6. Gradient Boosting Classifier (supervised + ensemble)
7. Self-Training Neural Network with MLPClassifier (semi-supervised)
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import (
    ARCHIVE_SHA256,
    BINARY_TARGET_COLUMN,
    FEATURE_COLUMNS,
    load_heart_data,
)
from src.evaluate_models import evaluate_pipelines, save_evaluation_results
from src.predict import (
    DEFAULT_ARTIFACTS_DIR,
    MODELS_DIRNAME,
    REGISTRY_FILENAME,
)
from src.preprocessing import build_preprocessor, split_features_target

RANDOM_STATE = 42
TEST_SIZE = 0.2
SEMI_SUPERVISED_LABELED_FRACTION = 0.3
SEMI_SUPERVISED_RANDOM_STATE = 202


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata and estimator factory for one model configuration."""

    key: str
    display_name: str
    categories: tuple[str, ...]
    estimator_factory: Callable[[int], BaseEstimator]
    semi_supervised: bool = False


def _self_training_mlp(seed: int) -> SelfTrainingClassifier:
    # SelfTrainingClassifier uses the modern `estimator` argument in sklearn
    # 1.9 (the old `base_estimator` name is deprecated).
    return SelfTrainingClassifier(
        estimator=MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=2000,
            early_stopping=False,
            n_iter_no_change=30,
            random_state=seed,
        ),
        threshold=0.75,
        criterion="threshold",
        max_iter=20,
        verbose=False,
    )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        key="logistic_regression",
        display_name="Logistic Regression",
        categories=("supervised",),
        estimator_factory=lambda seed: LogisticRegression(
            max_iter=2000, random_state=seed
        ),
    ),
    "decision_tree": ModelSpec(
        key="decision_tree",
        display_name="Decision Tree Classifier",
        categories=("supervised",),
        estimator_factory=lambda seed: DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=5, random_state=seed
        ),
    ),
    "svm": ModelSpec(
        key="svm",
        display_name="Support Vector Machine (SVM)",
        categories=("supervised",),
        # probability=False is intentional: sklearn 1.9 deprecates
        # SVC(probability=True), and decision_function is a valid continuous
        # score for ROC-AUC.
        estimator_factory=lambda seed: SVC(
            kernel="rbf", C=1.0, gamma="scale", random_state=seed
        ),
    ),
    "knn": ModelSpec(
        key="knn",
        display_name="K-Nearest Neighbors (KNN)",
        categories=("supervised",),
        estimator_factory=lambda seed: KNeighborsClassifier(
            n_neighbors=7, weights="distance"
        ),
    ),
    "random_forest": ModelSpec(
        key="random_forest",
        display_name="Random Forest Classifier",
        categories=("supervised", "ensemble"),
        estimator_factory=lambda seed: RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=1,
        ),
    ),
    "gradient_boosting": ModelSpec(
        key="gradient_boosting",
        display_name="Gradient Boosting Classifier",
        categories=("supervised", "ensemble"),
        estimator_factory=lambda seed: GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=seed,
        ),
    ),
    "self_training_mlp": ModelSpec(
        key="self_training_mlp",
        display_name="Self-Training Neural Network (MLP)",
        categories=("semi-supervised",),
        estimator_factory=_self_training_mlp,
        semi_supervised=True,
    ),
}

if len(MODEL_REGISTRY) != 7:
    raise RuntimeError("The model registry must contain exactly seven models.")


@dataclass(frozen=True)
class SplitData:
    """The single stratified train/test split used by every model."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    random_state: int
    test_size: float


def build_model_pipeline(
    model_key: str,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build an unfitted pipeline with preprocessing followed by one model."""
    if model_key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY)
        raise ValueError(
            f"Unknown model '{model_key}'. Available models: {available}."
        )
    spec = MODEL_REGISTRY[model_key]
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", spec.estimator_factory(random_state)),
        ]
    )


def make_train_test_split(
    frame: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> SplitData:
    """Create the one reproducible stratified split used by all models."""
    if not isinstance(random_state, int) or random_state < 0:
        raise ValueError("random_state must be a non-negative integer.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1.")

    features, target = split_features_target(frame)
    if set(target.unique().tolist()) != {0, 1}:
        raise ValueError("The target must contain both binary classes 0 and 1.")

    indices = np.arange(len(features))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    train_indices = tuple(sorted(int(index) for index in train_indices))
    test_indices = tuple(sorted(int(index) for index in test_indices))

    X_train = features.iloc[list(train_indices)].reset_index(drop=True)
    X_test = features.iloc[list(test_indices)].reset_index(drop=True)
    y_train = target.iloc[list(train_indices)].reset_index(drop=True)
    y_test = target.iloc[list(test_indices)].reset_index(drop=True)

    if set(y_train.unique().tolist()) != {0, 1}:
        raise ValueError("Stratified training split lost a target class.")
    if set(y_test.unique().tolist()) != {0, 1}:
        raise ValueError("Stratified test split lost a target class.")

    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_indices=train_indices,
        test_indices=test_indices,
        random_state=random_state,
        test_size=test_size,
    )


def make_partially_labeled_target(
    y_train,
    *,
    labeled_fraction: float = SEMI_SUPERVISED_LABELED_FRACTION,
    random_state: int = SEMI_SUPERVISED_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Retain a stratified subset of *training* labels and mark the rest -1.

    This function accepts only the training labels. The test set is neither
    accepted nor referenced. ``-1`` is the unlabeled marker required by
    scikit-learn's SelfTrainingClassifier.
    """
    if not 0 < labeled_fraction < 1:
        raise ValueError("labeled_fraction must be strictly between 0 and 1.")
    if not isinstance(random_state, int) or random_state < 0:
        raise ValueError("random_state must be a non-negative integer.")

    y_array = np.asarray(y_train, dtype=int)
    if y_array.ndim != 1:
        raise ValueError("y_train must be one-dimensional.")
    if set(np.unique(y_array).tolist()) != {0, 1}:
        raise ValueError("y_train must contain both binary classes 0 and 1.")

    indices = np.arange(len(y_array))
    labeled_indices, unlabeled_indices = train_test_split(
        indices,
        train_size=labeled_fraction,
        random_state=random_state,
        stratify=y_array,
    )
    labeled_indices = np.sort(labeled_indices)

    if len(labeled_indices) == 0:
        raise ValueError("No training labels were retained.")
    if len(unlabeled_indices) == 0:
        raise ValueError("No training rows remain unlabeled.")

    y_semi = np.full(len(y_array), -1, dtype=int)
    y_semi[labeled_indices] = y_array[labeled_indices]

    if set(np.unique(y_semi[y_semi != -1]).tolist()) != {0, 1}:
        raise ValueError("The retained training labels lost a target class.")

    return y_semi, labeled_indices


def split_metadata(split: SplitData) -> dict[str, Any]:
    """Serialize the split for reproducibility and later evaluation."""
    return {
        "random_state": split.random_state,
        "test_size": split.test_size,
        "n_total": len(split.X_train) + len(split.X_test),
        "n_train": len(split.X_train),
        "n_test": len(split.X_test),
        "train_indices": list(split.train_indices),
        "test_indices": list(split.test_indices),
        "train_target_counts": {
            str(int(label)): int(count)
            for label, count in split.y_train.value_counts().items()
        },
        "test_target_counts": {
            str(int(label)): int(count)
            for label, count in split.y_test.value_counts().items()
        },
        "feature_columns": FEATURE_COLUMNS,
        "target_column": BINARY_TARGET_COLUMN,
        "dataset_archive_sha256": ARCHIVE_SHA256,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return repr(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _classifier_metadata(spec: ModelSpec, pipeline: Pipeline) -> dict[str, Any]:
    classifier = pipeline.named_steps["classifier"]

    # scikit-learn 1.9 exposes "deprecated" sentinels for a few default
    # parameters (for example SVC.probability and LogisticRegression.penalty).
    # They describe an API deprecation, not a setting used by this project.
    def is_deprecated(value: Any) -> bool:
        return isinstance(value, str) and value == "deprecated"

    params = {
        key: _jsonable(value)
        for key, value in classifier.get_params(deep=False).items()
        if key != "estimator" and not is_deprecated(value)
    }
    metadata: dict[str, Any] = {
        "classifier": type(classifier).__name__,
        "params": params,
    }
    if spec.semi_supervised:
        base_estimator = classifier.estimator
        metadata["base_estimator"] = type(base_estimator).__name__
        metadata["base_estimator_params"] = {
            key: _jsonable(value)
            for key, value in base_estimator.get_params(deep=False).items()
            if not is_deprecated(value)
        }
    return metadata


def _prepare_artifacts_dir(artifacts_dir: str | Path | None) -> Path:
    artifacts_path = (
        Path(artifacts_dir) if artifacts_dir is not None else DEFAULT_ARTIFACTS_DIR
    )
    models_dir = artifacts_path / MODELS_DIRNAME
    models_dir.mkdir(parents=True, exist_ok=True)

    # Remove obsolete generated model files (for example artifacts from an
    # older model list) so the directory matches the current registry exactly.
    expected_artifacts = {f"{key}.joblib" for key in MODEL_REGISTRY}
    for stale_path in models_dir.glob("*.joblib"):
        if stale_path.name not in expected_artifacts:
            stale_path.unlink()

    return artifacts_path


def train_and_evaluate_models(
    artifacts_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    *,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    labeled_fraction: float = SEMI_SUPERVISED_LABELED_FRACTION,
    semi_random_state: int = SEMI_SUPERVISED_RANDOM_STATE,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train all seven pipelines, save artifacts, and evaluate on the test set.

    Returns a dictionary with the results table, artifact paths, split, and
    semi-supervised label metadata.
    """
    artifacts_path = _prepare_artifacts_dir(artifacts_dir)
    models_dir = artifacts_path / MODELS_DIRNAME

    frame = load_heart_data(data_dir=data_dir)
    split = make_train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
    )
    y_semi, labeled_indices = make_partially_labeled_target(
        split.y_train,
        labeled_fraction=labeled_fraction,
        random_state=semi_random_state,
    )

    if verbose:
        print(
            f"Dataset: {len(frame)} records | "
            f"train={len(split.X_train)} | test={len(split.X_test)}"
        )
        print(
            f"Semi-supervised labels: {len(labeled_indices)} of "
            f"{len(split.y_train)} training rows retained; "
            f"{len(y_semi) - len(labeled_indices)} rows marked unlabeled (-1)."
        )

    pipelines: dict[str, Pipeline] = {}
    fitted_entries: list[dict[str, Any]] = []
    total_models = len(MODEL_REGISTRY)

    for position, spec in enumerate(MODEL_REGISTRY.values(), start=1):
        pipeline = build_model_pipeline(spec.key, random_state=random_state)
        fit_target = y_semi if spec.semi_supervised else split.y_train

        started = time.perf_counter()
        try:
            pipeline.fit(split.X_train, fit_target)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to train {spec.display_name}: {exc}"
            ) from exc
        fit_seconds = time.perf_counter() - started

        predictions = np.asarray(pipeline.predict(split.X_test))
        if len(predictions) != len(split.X_test):
            raise RuntimeError(
                f"{spec.display_name} returned an unexpected prediction count."
            )
        if not set(np.unique(predictions).tolist()).issubset({0, 1}):
            raise RuntimeError(
                f"{spec.display_name} produced non-binary predictions."
            )

        artifact_name = f"{spec.key}.joblib"
        joblib.dump(pipeline, models_dir / artifact_name)
        pipelines[spec.key] = pipeline

        entry: dict[str, Any] = {
            "key": spec.key,
            "display_name": spec.display_name,
            "categories": list(spec.categories),
            "semi_supervised": spec.semi_supervised,
            "artifact": artifact_name,
            "fit_seconds": round(fit_seconds, 4),
            **_classifier_metadata(spec, pipeline),
        }

        if spec.semi_supervised:
            classifier = pipeline.named_steps["classifier"]
            labeled_iter = np.asarray(
                getattr(classifier, "labeled_iter_", np.array([]))
            )
            entry["self_training"] = {
                "n_iter": int(getattr(classifier, "n_iter_", 0)),
                "termination_condition": getattr(
                    classifier, "termination_condition_", "unknown"
                ),
                "n_initial_labeled": int(np.sum(labeled_iter == 0)),
                "n_pseudo_labeled": int(np.sum(labeled_iter > 0)),
                "n_still_unlabeled": int(np.sum(labeled_iter == -1)),
            }

        fitted_entries.append(entry)
        if verbose:
            print(
                f"[{position}/{total_models}] {spec.display_name} "
                f"({fit_seconds:.2f}s)"
            )

    if len(pipelines) != total_models:
        raise RuntimeError("Not all registered models were trained.")

    split_manifest = split_metadata(split)
    _write_json(artifacts_path / "data_split.json", split_manifest)

    registry_payload = {
        "artifact_format": "joblib",
        "random_state": random_state,
        "test_size": test_size,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": BINARY_TARGET_COLUMN,
        "models": fitted_entries,
    }
    _write_json(models_dir / REGISTRY_FILENAME, registry_payload)

    semi_metadata = {
        "labeled_fraction": labeled_fraction,
        "random_state": semi_random_state,
        "labeled_representation": -1,
        "n_labeled": int(len(labeled_indices)),
        "n_unlabeled": int(len(y_semi) - len(labeled_indices)),
        "labeled_target_counts": {
            str(int(label)): int(count)
            for label, count in pd.Series(
                y_semi[y_semi != -1], name=BINARY_TARGET_COLUMN
            ).value_counts().items()
        },
        "labeled_indices": [int(index) for index in labeled_indices],
        "note": (
            "Only training-set labels were partially hidden. The test labels "
            "were never passed to the semi-supervised pipeline."
        ),
    }

    training_metadata = {
        "dataset": {
            "archive_sha256": ARCHIVE_SHA256,
            "n_records": len(frame),
        },
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "split": split_manifest,
        "semi_supervised": semi_metadata,
        "models": fitted_entries,
        "test_set_used_for_training": False,
        "test_set_used_for_preprocessing": False,
        "test_set_used_for_hyperparameter_selection": False,
        "test_set_used_for_threshold_tuning": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(artifacts_path / "training_metadata.json", training_metadata)

    model_info = {
        spec.key: {
            "display_name": spec.display_name,
            "categories": spec.categories,
        }
        for spec in MODEL_REGISTRY.values()
    }
    results = evaluate_pipelines(
        pipelines,
        split.X_test,
        split.y_test,
        model_info=model_info,
    )

    evaluation_metadata = {
        "random_state": random_state,
        "test_size": test_size,
        "n_train": len(split.X_train),
        "n_test": len(split.X_test),
        "test_target_counts": split_manifest["test_target_counts"],
        "dataset_archive_sha256": ARCHIVE_SHA256,
        "evaluation_protocol": (
            "All pipelines were evaluated on the same untouched stratified "
            "test set. Test data was not used for training, preprocessing "
            "fitting, hyperparameter selection, or threshold tuning."
        ),
        "roc_auc_protocol": (
            "ROC-AUC uses positive-class predict_proba where available and "
            "SVC decision_function otherwise; it is never computed from hard "
            "class predictions."
        ),
    }
    paths = save_evaluation_results(
        results,
        artifacts_dir=artifacts_path,
        metadata=evaluation_metadata,
    )

    if verbose:
        print("\nMetrics on the untouched test set:")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(
                results.to_string(
                    index=False,
                    float_format=lambda value: f"{value:.4f}",
                )
            )

    return {
        "results": results,
        "paths": {
            **paths,
            "models_dir": models_dir,
            "registry": models_dir / REGISTRY_FILENAME,
            "split_manifest": artifacts_path / "data_split.json",
            "training_metadata": artifacts_path / "training_metadata.json",
        },
        "split": split,
        "semi_supervised": semi_metadata,
    }


def main() -> int:
    try:
        run = train_and_evaluate_models()
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    print("\nSaved artifacts:")
    for label, path in run["paths"].items():
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
