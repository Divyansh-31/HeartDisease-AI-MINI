"""Evaluation of saved heart-disease pipelines on one untouched test set.

The training step saves the split manifest and fitted pipelines. This module
reloads them so evaluation always uses exactly the same held-out records and
the exact serialized models used later for prediction.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_loader import ARCHIVE_SHA256, load_heart_data
from src.predict import (
    DEFAULT_ARTIFACTS_DIR,
    load_all_models,
    load_registry,
    predict_scores,
)
from src.preprocessing import split_features_target

METRIC_COLUMNS = ("accuracy", "precision", "recall", "f1", "roc_auc")
RESULT_COLUMNS = (
    "model_key",
    "model_name",
    "categories",
    *METRIC_COLUMNS,
    "score_type",
)


def _artifacts_path(artifacts_dir: str | Path | None) -> Path:
    return Path(artifacts_dir) if artifacts_dir is not None else DEFAULT_ARTIFACTS_DIR


def evaluate_pipeline(
    model,
    X_test,
    y_test,
    *,
    model_key: str,
    display_name: str | None = None,
    categories: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Compute all five metrics for one fitted pipeline."""
    if len(X_test) == 0:
        raise ValueError("Cannot evaluate on an empty test set.")
    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must contain the same number of rows.")

    y_true = np.asarray(y_test, dtype=int)
    labels = set(np.unique(y_true).tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Expected binary test labels 0/1, got {sorted(labels)}.")
    if not labels:
        raise ValueError("Test labels are empty.")

    y_pred = np.asarray(model.predict(X_test), dtype=int)
    if len(y_pred) != len(y_true):
        raise ValueError("Model predictions do not match the test-set length.")
    if not set(np.unique(y_pred).tolist()).issubset({0, 1}):
        raise ValueError("Model produced non-binary predictions.")

    scores, score_type = predict_scores(model, X_test)
    roc_auc = None
    if scores is not None and labels == {0, 1}:
        roc_auc = float(roc_auc_score(y_true, scores))

    return {
        "model_key": model_key,
        "model_name": display_name or model_key,
        "categories": ", ".join(categories),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0, pos_label=1)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0, pos_label=1)
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0, pos_label=1)),
        "roc_auc": roc_auc,
        "score_type": score_type,
    }


def evaluate_pipelines(
    models: Mapping[str, Any],
    X_test,
    y_test,
    *,
    model_info: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Evaluate every model in registry order on the same test rows."""
    if not models:
        raise ValueError("No models were supplied for evaluation.")

    model_info = model_info or {}
    rows = []
    for model_key, model in models.items():
        info = model_info.get(model_key, {})
        rows.append(
            evaluate_pipeline(
                model,
                X_test,
                y_test,
                model_key=model_key,
                display_name=info.get("display_name", model_key),
                categories=tuple(info.get("categories", ())),
            )
        )
    return pd.DataFrame(rows, columns=list(RESULT_COLUMNS))


def _format_metric(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.4f}"


def results_to_markdown(results: pd.DataFrame) -> str:
    """Render the comparison results as a clean GitHub-flavored table."""
    required = {"model_name", "categories", *METRIC_COLUMNS}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Results table is missing columns: {sorted(missing)}")

    lines = [
        "| Model | Category | Accuracy | Precision | Recall | F1-score | ROC-AUC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.model_name),
                    str(row.categories),
                    _format_metric(row.accuracy),
                    _format_metric(row.precision),
                    _format_metric(row.recall),
                    _format_metric(row.f1),
                    _format_metric(row.roc_auc),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def create_comparison_chart(results: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a grouped bar chart of the five metrics for all models."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
        "roc_auc": "ROC-AUC",
    }
    x_positions = np.arange(len(results))
    width = 0.15

    fig, ax = plt.subplots(figsize=(13, 6))
    for index, metric in enumerate(METRIC_COLUMNS):
        values = pd.to_numeric(results[metric], errors="coerce").to_numpy(
            dtype=float
        )
        ax.bar(
            x_positions + (index - 2) * width,
            values,
            width,
            label=metric_labels[metric],
        )

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Heart Disease Models - Untouched Test Set")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(results["model_name"], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=5, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_evaluation_results(
    results: pd.DataFrame,
    *,
    artifacts_dir: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Persist CSV/JSON/Markdown results, metadata, and the comparison chart."""
    output_dir = _artifacts_path(artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = set(RESULT_COLUMNS) - set(results.columns)
    if missing:
        raise ValueError(f"Results table is missing columns: {sorted(missing)}")
    if results.empty:
        raise ValueError("Refusing to save an empty results table.")

    csv_path = output_dir / "comparison_results.csv"
    json_path = output_dir / "comparison_results.json"
    markdown_path = output_dir / "comparison_results.md"
    chart_path = output_dir / "comparison_chart.png"
    metadata_path = output_dir / "evaluation_metadata.json"

    results.to_csv(csv_path, index=False, float_format="%.6f")

    results_json = json.loads(results.to_json(orient="records"))
    results_payload = {
        "metadata": dict(metadata or {}),
        "results": results_json,
    }
    json_path.write_text(
        json.dumps(results_payload, indent=2, default=str),
        encoding="utf-8",
    )

    markdown_path.write_text(results_to_markdown(results), encoding="utf-8")
    create_comparison_chart(results, chart_path)

    evaluation_metadata = {
        **dict(metadata or {}),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metric_columns": list(METRIC_COLUMNS),
        "score_type_by_model": {
            row.model_key: row.score_type
            for row in results.itertuples(index=False)
        },
        "versions": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "results_file": csv_path.name,
    }
    metadata_path.write_text(
        json.dumps(evaluation_metadata, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "results_csv": csv_path,
        "results_json": json_path,
        "results_markdown": markdown_path,
        "comparison_chart": chart_path,
        "evaluation_metadata": metadata_path,
    }


def load_split_manifest(
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load the split manifest saved by training."""
    path = _artifacts_path(artifacts_dir) / "data_split.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Split manifest not found at {path}. "
            "Run `python -m src.train_models` first."
        )

    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    required = {
        "random_state",
        "test_size",
        "n_total",
        "n_train",
        "n_test",
        "train_indices",
        "test_indices",
        "test_target_counts",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Split manifest at {path} is missing: {sorted(missing)}.")
    return manifest


def _validate_indices(name: str, values: Any, n_total: int) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"Split manifest field '{name}' must be a non-empty list.")
    if any(not isinstance(value, int) for value in values):
        raise ValueError(f"Split manifest field '{name}' must contain integers.")
    if len(values) != len(set(values)):
        raise ValueError(f"Split manifest field '{name}' contains duplicates.")
    if min(values) < 0 or max(values) >= n_total:
        raise ValueError(f"Split manifest field '{name}' has out-of-range indices.")
    return values


def load_test_data(
    artifacts_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Reconstruct the exact saved test set from the split manifest."""
    manifest = load_split_manifest(artifacts_dir)
    frame = load_heart_data(data_dir=data_dir)
    features, target = split_features_target(frame)

    n_total = int(manifest["n_total"])
    if n_total != len(frame):
        raise ValueError(
            "The dataset row count does not match the saved split manifest "
            f"({len(frame)} vs {n_total}). Re-run training."
        )

    train_indices = _validate_indices(
        "train_indices", manifest["train_indices"], n_total
    )
    test_indices = _validate_indices(
        "test_indices", manifest["test_indices"], n_total
    )
    if set(train_indices) & set(test_indices):
        raise ValueError("Saved train and test indices overlap.")
    if set(train_indices) | set(test_indices) != set(range(n_total)):
        raise ValueError("Saved train and test indices do not cover the dataset.")
    if len(train_indices) != int(manifest["n_train"]):
        raise ValueError("Saved training-set size does not match its index list.")
    if len(test_indices) != int(manifest["n_test"]):
        raise ValueError("Saved test-set size does not match its index list.")

    X_test = features.iloc[test_indices].reset_index(drop=True)
    y_test = target.iloc[test_indices].reset_index(drop=True)

    expected_counts = {
        str(key): int(value)
        for key, value in manifest["test_target_counts"].items()
    }
    actual_counts = {
        str(int(label)): int(count)
        for label, count in y_test.value_counts().items()
    }
    if actual_counts != expected_counts:
        raise ValueError(
            "Saved test-set target distribution does not match the manifest."
        )

    return X_test, y_test, manifest


def evaluate_saved_models(
    artifacts_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Load saved pipelines and evaluate them on the saved untouched test set."""
    artifacts_path = _artifacts_path(artifacts_dir)
    registry = load_registry(artifacts_path)
    X_test, y_test, manifest = load_test_data(artifacts_path, data_dir)
    models = load_all_models(artifacts_path)

    registry_keys = [entry["key"] for entry in registry["models"]]
    if list(models) != registry_keys:
        raise ValueError("Loaded model order does not match the registry.")

    model_info = {
        entry["key"]: {
            "display_name": entry.get("display_name", entry["key"]),
            "categories": tuple(entry.get("categories", ())),
        }
        for entry in registry["models"]
    }
    results = evaluate_pipelines(
        models, X_test, y_test, model_info=model_info
    )

    metadata = {
        "random_state": manifest["random_state"],
        "test_size": manifest["test_size"],
        "n_train": manifest["n_train"],
        "n_test": manifest["n_test"],
        "test_target_counts": manifest["test_target_counts"],
        "dataset_archive_sha256": manifest.get(
            "dataset_archive_sha256", ARCHIVE_SHA256
        ),
        "evaluation_protocol": (
            "Every saved pipeline was evaluated on the same saved stratified "
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
        metadata=metadata,
    )
    return results, paths


def main() -> int:
    try:
        results, paths = evaluate_saved_models()
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\nSaved evaluation artifacts:")
    for label, path in paths.items():
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
