"""Shared Streamlit UI helpers for the two prediction pages.

This module is the single source of truth for the 13 patient input controls.
It validates its feature definitions against the existing preprocessing
constants, reuses the saved model registry and prediction functions, and reads
the saved evaluation table from ``artifacts/comparison_results.csv``.

Nothing here trains models, recomputes official metrics, or writes artifacts.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data_loader import FEATURE_COLUMNS
from src.evaluate_models import METRIC_COLUMNS
from src.predict import (
    DEFAULT_ARTIFACTS_DIR,
    load_registry,
    predict_heart_disease,
)
from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

DISCLAIMER = (
    "This is an educational college mini-project. Predictions come from "
    "models trained on a small public dataset and are not a medical diagnosis, "
    "risk assessment, or substitute for advice from a qualified clinician."
)

# A small stylesheet keeps the Streamlit defaults consistent with the white
# theme: white surfaces, charcoal text, light borders, and readable tables.
# The selectors are limited to stable Streamlit test ids and plain HTML tags.
_PAGE_CSS = """
<style>
  .block-container {
      padding-top: 2.5rem;
      padding-bottom: 3rem;
      max-width: 1180px;
  }
  h1, h2, h3, h4, h5, h6 {
      color: #111827;
      letter-spacing: -0.01em;
  }
  h1 {
      font-weight: 700;
  }
  [data-testid="stSidebar"] {
      border-right: 1px solid #E5E7EB;
  }
  [data-testid="stMetric"] {
      background-color: #FFFFFF;
      border: 1px solid #E5E7EB;
      border-radius: 0.5rem;
      padding: 0.75rem 1rem;
  }
  [data-testid="stMetricLabel"] {
      color: #4B5563;
  }
  table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
  }
  th, td {
      border-bottom: 1px solid #E5E7EB;
      padding: 0.5rem 0.7rem;
      color: #1F2937;
      vertical-align: top;
  }
  th {
      background-color: #F8FAFC;
      color: #111827;
      font-weight: 600;
      white-space: nowrap;
  }
  tr:last-child td {
      border-bottom: none;
  }
  hr {
      border-color: #E5E7EB;
  }
</style>
"""


def apply_page_style() -> None:
    """Apply the small, stable stylesheet shared by both pages."""
    import streamlit as st

    st.markdown(_PAGE_CSS, unsafe_allow_html=True)

# Ranges and category codes verified from the downloaded UCI Cleveland data.
# Numeric entries: (minimum, maximum, step).
VERIFIED_NUMERIC_RANGES: dict[str, tuple[int | float, int | float, int | float]] = {
    "age": (29, 77, 1),
    "trestbps": (94, 200, 1),
    "chol": (126, 564, 1),
    "thalach": (71, 202, 1),
    "oldpeak": (0.0, 6.2, 0.1),
}

VERIFIED_CATEGORICAL_VALUES: dict[str, tuple[int, ...]] = {
    "sex": (0, 1),
    "cp": (1, 2, 3, 4),
    "fbs": (0, 1),
    "restecg": (0, 1, 2),
    "exang": (0, 1),
    "slope": (1, 2, 3),
    "ca": (0, 1, 2, 3),
    "thal": (3, 6, 7),
}

# Defaults are the dataset medians (numeric) and modes (categorical), so the
# initial form is a realistic but explicitly illustrative record.
_DEFAULTS: dict[str, int | float] = {
    "age": 56,
    "sex": 1,
    "cp": 4,
    "trestbps": 130,
    "chol": 241,
    "fbs": 0,
    "restecg": 0,
    "thalach": 153,
    "exang": 0,
    "oldpeak": 0.8,
    "slope": 1,
    "ca": 0,
    "thal": 3,
}

_NUMERIC_COPY: dict[str, tuple[str, str]] = {
    "age": ("Age (years)", "Patient age in years."),
    "trestbps": (
        "Resting blood pressure (mm Hg)",
        "Resting blood pressure in mm Hg.",
    ),
    "chol": ("Serum cholesterol (mg/dl)", "Serum cholesterol in mg/dl."),
    "thalach": (
        "Maximum heart rate achieved",
        "Maximum heart rate achieved during the exercise test.",
    ),
    "oldpeak": (
        "ST depression (oldpeak)",
        "ST depression induced by exercise relative to rest.",
    ),
}

# Categorical meanings follow the dataset documentation summarised in README.
_CATEGORICAL_COPY: dict[str, tuple[str, tuple[tuple[int, str], ...], str]] = {
    "sex": (
        "Sex",
        ((0, "0 - female"), (1, "1 - male")),
        "Dataset coding: 0 = female, 1 = male.",
    ),
    "cp": (
        "Chest pain type (cp)",
        (
            (1, "1 - typical angina"),
            (2, "2 - atypical angina"),
            (3, "3 - non-anginal pain"),
            (4, "4 - asymptomatic"),
        ),
        "Chest pain type code used by the dataset.",
    ),
    "fbs": (
        "Fasting blood sugar > 120 mg/dl (fbs)",
        ((0, "0 - false"), (1, "1 - true (> 120 mg/dl)")),
        "Whether fasting blood sugar exceeds 120 mg/dl.",
    ),
    "restecg": (
        "Resting ECG (restecg)",
        (
            (0, "0 - normal"),
            (1, "1 - ST-T wave abnormality"),
            (2, "2 - probable/definite left ventricular hypertrophy"),
        ),
        "Resting electrocardiographic result code.",
    ),
    "exang": (
        "Exercise-induced angina (exang)",
        ((0, "0 - no"), (1, "1 - yes")),
        "Whether exercise induced angina.",
    ),
    "slope": (
        "ST segment slope (slope)",
        (
            (1, "1 - upsloping"),
            (2, "2 - flat"),
            (3, "3 - downsloping"),
        ),
        "Slope of the peak exercise ST segment.",
    ),
    "ca": (
        "Major vessels coloured by fluoroscopy (ca)",
        (
            (0, "0 vessels"),
            (1, "1 vessel"),
            (2, "2 vessels"),
            (3, "3 vessels"),
        ),
        "Number of major vessels (0-3) coloured by fluoroscopy.",
    ),
    "thal": (
        "Thalassemia (thal)",
        (
            (3, "3 - normal"),
            (6, "6 - fixed defect"),
            (7, "7 - reversible defect"),
        ),
        "Thalassemia code used by the dataset.",
    ),
}


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for one patient input widget."""

    name: str
    label: str
    kind: str
    minimum: int | float | None
    maximum: int | float | None
    step: int | float | None
    default: int | float
    options: tuple[int, ...] = ()
    option_labels: tuple[tuple[int, str], ...] = ()
    help_text: str = ""

    def option_label(self, value: int) -> str:
        return dict(self.option_labels)[value]

    def widget_key(self) -> str:
        return f"heart_input_{self.name}"


def _build_feature_specs() -> tuple[FeatureSpec, ...]:
    specs: list[FeatureSpec] = []
    for name in FEATURE_COLUMNS:
        if name in NUMERIC_FEATURES:
            minimum, maximum, step = VERIFIED_NUMERIC_RANGES[name]
            label, help_text = _NUMERIC_COPY[name]
            specs.append(
                FeatureSpec(
                    name=name,
                    label=label,
                    kind="numeric",
                    minimum=minimum,
                    maximum=maximum,
                    step=step,
                    default=_DEFAULTS[name],
                    help_text=help_text,
                )
            )
        else:
            label, option_labels, help_text = _CATEGORICAL_COPY[name]
            specs.append(
                FeatureSpec(
                    name=name,
                    label=label,
                    kind="categorical",
                    minimum=None,
                    maximum=None,
                    step=None,
                    default=_DEFAULTS[name],
                    options=VERIFIED_CATEGORICAL_VALUES[name],
                    option_labels=option_labels,
                    help_text=help_text,
                )
            )
    return tuple(specs)


FEATURE_SPECS = _build_feature_specs()


def _validate_feature_specs() -> None:
    names = tuple(spec.name for spec in FEATURE_SPECS)
    if names != tuple(FEATURE_COLUMNS):
        raise RuntimeError(
            "UI feature order does not match FEATURE_COLUMNS: "
            f"{names} vs {tuple(FEATURE_COLUMNS)}"
        )

    numeric_names = tuple(spec.name for spec in FEATURE_SPECS if spec.kind == "numeric")
    if numeric_names != tuple(NUMERIC_FEATURES):
        raise RuntimeError(
            "UI numeric features do not match NUMERIC_FEATURES: "
            f"{numeric_names} vs {tuple(NUMERIC_FEATURES)}"
        )

    categorical_names = tuple(
        spec.name for spec in FEATURE_SPECS if spec.kind == "categorical"
    )
    if categorical_names != tuple(CATEGORICAL_FEATURES):
        raise RuntimeError(
            "UI categorical features do not match CATEGORICAL_FEATURES: "
            f"{categorical_names} vs {tuple(CATEGORICAL_FEATURES)}"
        )

    for spec in FEATURE_SPECS:
        if spec.kind == "categorical":
            if spec.options != VERIFIED_CATEGORICAL_VALUES[spec.name]:
                raise RuntimeError(
                    f"Verified options for '{spec.name}' are inconsistent."
                )
            if spec.default not in spec.options:
                raise RuntimeError(
                    f"Default for '{spec.name}' is not one of its options."
                )
        else:
            assert spec.minimum is not None and spec.maximum is not None
            if not spec.minimum <= spec.default <= spec.maximum:
                raise RuntimeError(
                    f"Default for '{spec.name}' is outside its verified range."
                )


_validate_feature_specs()


def default_patient() -> dict[str, int | float]:
    """Return a complete, valid illustrative patient record."""
    return {spec.name: spec.default for spec in FEATURE_SPECS}


def validate_patient_record(patient: Mapping[str, Any]) -> dict[str, int | float]:
    """Validate and normalise one patient record in feature order."""
    if not isinstance(patient, Mapping):
        raise TypeError("patient must be a mapping of feature names to values.")

    missing = [spec.name for spec in FEATURE_SPECS if spec.name not in patient]
    extra = [name for name in patient if name not in FEATURE_COLUMNS]
    if missing:
        raise ValueError(f"Missing required feature values: {missing}.")
    if extra:
        raise ValueError(f"Unexpected feature values: {extra}.")

    record: dict[str, int | float] = {}
    for spec in FEATURE_SPECS:
        raw_value = patient[spec.name]
        if isinstance(raw_value, bool) or not isinstance(raw_value, numbers.Real):
            raise ValueError(f"'{spec.name}' must be a numeric value.")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"'{spec.name}' must be a finite number.")

        if spec.kind == "numeric":
            assert spec.minimum is not None and spec.maximum is not None
            if not spec.minimum <= value <= spec.maximum:
                raise ValueError(
                    f"'{spec.name}' must be between {spec.minimum} and "
                    f"{spec.maximum}."
                )
            if spec.step == 1:
                if not value.is_integer():
                    raise ValueError(f"'{spec.name}' must be a whole number.")
                record[spec.name] = int(value)
            else:
                record[spec.name] = value
        else:
            if not value.is_integer():
                raise ValueError(f"'{spec.name}' must be a whole-number code.")
            code = int(value)
            if code not in spec.options:
                raise ValueError(
                    f"'{spec.name}' must be one of {list(spec.options)}."
                )
            record[spec.name] = code
    return record


def get_model_options(artifacts_dir: str | Path | None = None) -> dict[str, str]:
    """Return registry keys mapped to display names, in registry order."""
    registry = load_registry(artifacts_dir)
    return {
        entry["key"]: entry.get("display_name", entry["key"])
        for entry in registry["models"]
    }


def predict_one(
    patient: Mapping[str, Any],
    model_key: str,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one record and predict with one saved pipeline."""
    record = validate_patient_record(patient)
    return predict_heart_disease(record, model_key, artifacts_dir)


def predict_all(
    patient: Mapping[str, Any],
    artifacts_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Predict one record with every saved model, in registry order."""
    record = validate_patient_record(patient)
    keys = list(get_model_options(artifacts_dir))
    return [
        predict_heart_disease(record, model_key, artifacts_dir)
        for model_key in keys
    ]


def _artifacts_dir(artifacts_dir: str | Path | None) -> Path:
    return Path(artifacts_dir) if artifacts_dir is not None else DEFAULT_ARTIFACTS_DIR


REQUIRED_METRIC_COLUMNS = ("model_key", "model_name", *METRIC_COLUMNS)


def load_evaluation_metrics(
    artifacts_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load the saved held-out metrics and align them to the registry.

    The official metrics are only read; they are never recomputed or written.
    """
    path = _artifacts_dir(artifacts_dir) / "comparison_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Saved evaluation results not found at {path}. "
            "Run `python -m src.train_models` first."
        )

    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        raise ValueError(f"Could not read evaluation results at {path}: {exc}") from exc

    missing = [column for column in REQUIRED_METRIC_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Evaluation results at {path} are missing columns: {missing}."
        )
    if frame.empty:
        raise ValueError(f"Evaluation results at {path} contain no rows.")
    if frame["model_key"].duplicated().any():
        raise ValueError(f"Evaluation results at {path} contain duplicate model keys.")

    for metric in METRIC_COLUMNS:
        values = pd.to_numeric(frame[metric], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"Evaluation results at {path} contain non-numeric '{metric}' values."
            )
        if ((values < 0) | (values > 1)).any():
            raise ValueError(
                f"Evaluation results at {path} contain '{metric}' values outside [0, 1]."
            )
        frame[metric] = values.astype(float)

    registry = load_registry(artifacts_dir)
    registry_keys = [entry["key"] for entry in registry["models"]]
    present = set(frame["model_key"].astype(str))
    missing_keys = [key for key in registry_keys if key not in present]
    extra_keys = sorted(present - set(registry_keys))
    if missing_keys or extra_keys:
        raise ValueError(
            "Evaluation results do not match the current model registry. "
            f"Missing: {missing_keys}; unexpected: {extra_keys}. "
            "Re-run `python -m src.train_models`."
        )

    return (
        frame.set_index("model_key")
        .loc[registry_keys]
        .reset_index()
        .loc[:, [*REQUIRED_METRIC_COLUMNS, *(
            ["score_type"] if "score_type" in frame.columns else []
        )]]
    )


def compare_predictions(
    predictions: Sequence[Mapping[str, Any]],
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Join one prediction per model to its saved metrics by ``model_key``."""
    if not predictions:
        raise ValueError("No prediction results were supplied.")

    prediction_frame = pd.DataFrame([dict(result) for result in predictions])
    required_prediction_columns = {
        "model_key",
        "model_name",
        "prediction",
        "label",
        "score",
        "score_type",
    }
    missing_prediction = required_prediction_columns - set(prediction_frame.columns)
    if missing_prediction:
        raise ValueError(
            f"Prediction results are missing columns: {sorted(missing_prediction)}."
        )
    if prediction_frame["model_key"].duplicated().any():
        raise ValueError("Prediction results contain duplicate model keys.")

    metric_columns = ["model_key", *METRIC_COLUMNS]
    if "score_type" in metrics.columns:
        metric_columns.append("score_type")
    missing_metrics = set(metric_columns) - set(metrics.columns)
    if missing_metrics:
        raise ValueError(
            f"Metrics table is missing columns: {sorted(missing_metrics)}."
        )

    metric_subset = metrics.loc[:, metric_columns].copy()
    if "score_type" in metric_subset.columns:
        metric_subset = metric_subset.rename(
            columns={"score_type": "metrics_score_type"}
        )

    merged = prediction_frame.merge(
        metric_subset,
        on="model_key",
        how="left",
        validate="one_to_one",
    )
    if merged[list(METRIC_COLUMNS)].isna().any().any():
        raise ValueError(
            "At least one prediction has no saved metrics. Re-run "
            "`python -m src.train_models`."
        )

    if "metrics_score_type" in merged.columns:
        mismatched = merged[
            merged["score_type"].fillna("") != merged["metrics_score_type"].fillna("")
        ]
        if not mismatched.empty:
            keys = ", ".join(mismatched["model_key"].astype(str))
            raise ValueError(
                "Saved metrics use a different score type for: "
                f"{keys}. Re-run `python -m src.train_models`."
            )
        merged = merged.drop(columns=["metrics_score_type"])

    return merged


def format_score(score: float | None, score_type: str | None) -> str:
    """Format a continuous score without implying false calibration."""
    if score is None:
        return "Not available"
    value = float(score)
    if score_type == "probability":
        return f"{value:.3f} (probability of class 1)"
    if score_type == "decision_function":
        return f"{value:+.3f} (decision score, not a probability)"
    return f"{value:.3f} ({score_type or 'score'})"


def format_score_value(score: float | None) -> str:
    """Format just the numeric score for a compact comparison table."""
    if score is None:
        return "Not available"
    return f"{float(score):.3f}"


def score_type_label(score_type: str | None) -> str:
    """Return a short, explicit score-type label."""
    if score_type == "probability":
        return "probability"
    if score_type == "decision_function":
        return "decision score (not a probability)"
    return score_type or "not available"


def predictions_to_markdown(predictions: Sequence[Mapping[str, Any]]) -> str:
    """Render the seven-model prediction comparison as a GFM table."""
    lines = [
        "| Model | Predicted class | Target label | Score | Score type |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for result in predictions:
        lines.append(
            "| {model} | {prediction} | {label} | {score} | {score_type} |".format(
                model=result["model_name"],
                prediction=result["prediction"],
                label=result["label"],
                score=format_score_value(result.get("score")),
                score_type=score_type_label(result.get("score_type")),
            )
        )
    return "\n".join(lines)


def metrics_to_markdown(metrics: pd.DataFrame) -> str:
    """Render the saved held-out evaluation metrics as a GFM table."""
    header = (
        "| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Score type |"
    )
    lines = [header, "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in metrics.itertuples(index=False):
        lines.append(
            "| {model} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | "
            "{f1:.4f} | {roc_auc:.4f} | {score_type} |".format(
                model=row.model_name,
                accuracy=float(row.accuracy),
                precision=float(row.precision),
                recall=float(row.recall),
                f1=float(row.f1),
                roc_auc=float(row.roc_auc),
                score_type=getattr(row, "score_type", "not available"),
            )
        )
    return "\n".join(lines)


def build_metrics_figure(metrics: pd.DataFrame):
    """Build an in-memory grouped bar chart of the saved metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
        "roc_auc": "ROC-AUC",
    }
    palette = ("#1E3A8A", "#1D4ED8", "#2563EB", "#3B82F6", "#60A5FA")
    positions = np.arange(len(metrics))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    for index, metric in enumerate(METRIC_COLUMNS):
        ax.bar(
            positions + (index - 2) * width,
            metrics[metric].to_numpy(dtype=float),
            width,
            label=labels[metric],
            color=palette[index % len(palette)],
            edgecolor="none",
        )
    ax.set_ylabel("Score", color="#4B5563")
    ax.set_ylim(0, 1.02)
    ax.set_title(
        "Saved held-out test-set metrics",
        color="#111827",
        fontsize=13,
        pad=14,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(metrics["model_name"], rotation=20, ha="right")
    ax.tick_params(axis="both", colors="#4B5563", labelsize=9)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=5,
        frameon=False,
        labelcolor="#1F2937",
    )
    fig.tight_layout()
    return fig


def reset_patient_inputs() -> None:
    """Clear the shared widget keys so defaults are restored on the next run."""
    import streamlit as st

    for spec in FEATURE_SPECS:
        st.session_state.pop(spec.widget_key(), None)


_SPEC_BY_NAME = {spec.name: spec for spec in FEATURE_SPECS}


def _render_widget(spec: FeatureSpec):
    """Render one numeric or categorical input widget."""
    import streamlit as st

    if spec.kind == "numeric":
        return st.number_input(
            spec.label,
            min_value=spec.minimum,
            max_value=spec.maximum,
            value=spec.default,
            step=spec.step,
            key=spec.widget_key(),
            help=spec.help_text,
        )
    return st.selectbox(
        spec.label,
        options=list(spec.options),
        index=list(spec.options).index(int(spec.default)),
        format_func=spec.option_label,
        key=spec.widget_key(),
        help=spec.help_text,
    )


def render_patient_inputs() -> dict[str, int | float]:
    """Render all 13 input widgets and return their values in feature order."""
    import streamlit as st

    values: dict[str, int | float] = {}

    def render_section(title: str, names: tuple[str, ...]) -> None:
        st.markdown(f"#### {title}")
        columns = st.columns(len(names))
        for column, name in zip(columns, names):
            with column:
                values[name] = _render_widget(_SPEC_BY_NAME[name])

    render_section("Patient characteristics", ("age", "sex", "cp"))
    render_section(
        "Vital signs and laboratory values",
        ("trestbps", "chol", "fbs"),
    )
    render_section(
        "Resting ECG and exercise test",
        ("restecg", "thalach", "exang"),
    )
    render_section("Exercise response", ("oldpeak", "slope"))
    render_section("Imaging findings", ("ca", "thal"))

    missing = [spec.name for spec in FEATURE_SPECS if spec.name not in values]
    if missing:
        raise RuntimeError(f"UI failed to render input widgets for: {missing}")

    # Return values in the exact training feature order.
    return {spec.name: values[spec.name] for spec in FEATURE_SPECS}


def render_disclaimer() -> None:
    """Display the shared educational-use disclaimer as a subtle note."""
    import streamlit as st

    st.caption(DISCLAIMER)
