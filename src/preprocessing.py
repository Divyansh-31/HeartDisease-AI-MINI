"""Feature groups and leakage-safe preprocessing for the heart dataset.

Imputation and encoding are wrapped in a scikit-learn ``Pipeline`` so they can
later be fitted on training folds only. The preprocessor is built here but is
not fitted on the full dataset during training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import BINARY_TARGET_COLUMN, FEATURE_COLUMNS

# Continuous measurements: median imputation and standardization.
NUMERIC_FEATURES = (
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
)

# Coded clinical categories: most-frequent imputation and one-hot encoding.
# `ca` and `thal` are the two columns that contain missing values.
CATEGORICAL_FEATURES = (
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
)

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = BINARY_TARGET_COLUMN

if set(FEATURES) != set(FEATURE_COLUMNS):
    raise RuntimeError(
        "Preprocessing feature groups do not match the dataset schema: "
        f"{sorted(set(FEATURES) ^ set(FEATURE_COLUMNS))}"
    )


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted preprocessing transformer for the 13 features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_features_target(
    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the 13 features from a target column with validation."""
    missing_features = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing_features:
        raise ValueError(
            "Dataset is missing required feature columns: "
            f"{sorted(missing_features)}"
        )
    if target_column not in frame.columns:
        raise ValueError(
            f"Target column '{target_column}' is not present in the dataset."
        )

    features = frame.loc[:, FEATURE_COLUMNS].copy()
    target = frame[target_column].astype("int64").copy()

    if target.isna().any():
        raise ValueError(f"Target column '{target_column}' contains missing values.")

    return features, target
