#!/usr/bin/env python
"""Verify the official UCI Heart Disease download and data contract.

Run from the project root:

    python scripts/verify_data.py

The script downloads the archive if it is not cached, verifies its SHA-256
checksum, checks the inspected schema/target assumptions, and runs a
preprocessing smoke test. It exits non-zero on any failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (  # noqa: E402
    ARCHIVE_SHA256,
    BINARY_TARGET_COLUMN,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    UCI_DATASET_DOI,
    UCI_DATASET_ID,
    ensure_dataset,
    load_heart_data,
)
from src.preprocessing import (  # noqa: E402
    build_preprocessor,
    split_features_target,
)


def main() -> int:
    source = ensure_dataset()
    frame = load_heart_data()

    print("UCI Heart Disease - data verification")
    print(f"Dataset: ID {UCI_DATASET_ID}, DOI {UCI_DATASET_DOI}")
    print(f"Source:  {source}")
    print(f"Archive SHA-256: {ARCHIVE_SHA256}")
    print(f"Shape: {frame.shape[0]} rows x {frame.shape[1]} columns")
    print(f"Features: {', '.join(FEATURE_COLUMNS)}")

    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN, BINARY_TARGET_COLUMN]
    assert list(frame.columns) == expected_columns, (
        f"Unexpected columns: {list(frame.columns)}"
    )
    assert len(frame) == 303, f"Expected 303 records, found {len(frame)}"
    assert frame[TARGET_COLUMN].notna().all(), "Raw target contains missing values"
    assert sorted(frame[TARGET_COLUMN].unique().tolist()) == [0, 1, 2, 3, 4], (
        "Unexpected raw target labels"
    )

    missing = {
        column: int(count)
        for column, count in frame[FEATURE_COLUMNS].isna().sum().items()
        if count
    }
    rows_with_missing = int(
        frame[FEATURE_COLUMNS].isna().any(axis=1).sum()
    )
    assert missing == {"ca": 4, "thal": 2}, (
        f"Unexpected missing-value pattern: {missing}"
    )
    assert rows_with_missing == 6, (
        f"Expected 6 rows with missing values, found {rows_with_missing}"
    )

    raw_counts = frame[TARGET_COLUMN].value_counts().sort_index()
    binary_counts = frame[BINARY_TARGET_COLUMN].value_counts().sort_index()
    assert set(binary_counts.index) == {0, 1}, "Binary target must contain 0 and 1"
    assert (
        frame.loc[
            frame[TARGET_COLUMN] == 0, BINARY_TARGET_COLUMN
        ].eq(0).all()
    ), "Raw label 0 must map to target 0"
    assert (
        frame.loc[
            frame[TARGET_COLUMN].isin([1, 2, 3, 4]), BINARY_TARGET_COLUMN
        ].eq(1).all()
    ), "Raw labels 1-4 must map to target 1"
    assert int(binary_counts.sum()) == len(frame)

    print(
        "Missing values: "
        + ", ".join(f"{column}={count}" for column, count in missing.items())
        + f" ({rows_with_missing} rows; retained)"
    )
    print(
        "Raw target 'num': "
        + ", ".join(f"{label}={count}" for label, count in raw_counts.items())
    )
    print(
        "Binary target 'target': "
        + ", ".join(
            f"{label}={count}" for label, count in binary_counts.items()
        )
    )

    features, target = split_features_target(frame)
    assert list(features.columns) == FEATURE_COLUMNS
    assert len(features) == len(target) == len(frame)

    # Smoke test only: the real training code must fit this transformer on
    # training folds, not on the full dataset, to avoid leakage.
    preprocessor = build_preprocessor()
    transformed = np.asarray(preprocessor.fit_transform(features))
    assert transformed.shape[0] == len(frame), "Preprocessor changed row count"
    assert transformed.shape[1] > 0, "Preprocessor produced no features"
    assert np.isfinite(transformed).all(), (
        "Preprocessing left non-finite values (missing data not imputed)"
    )
    print(f"Preprocessor output: {transformed.shape[0]} rows x {transformed.shape[1]} features")
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, RuntimeError, ValueError) as exc:
        print(f"\nVERIFICATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
