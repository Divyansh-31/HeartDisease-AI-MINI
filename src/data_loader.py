"""Official UCI Heart Disease data retrieval and loading.

Dataset: Heart Disease, UCI Machine Learning Repository ID 45,
DOI 10.24432/C52P4X (Janosi et al., 1989).

This module downloads the official ZIP archive, verifies its pinned SHA-256
checksum, extracts ``processed.cleveland.data`` (the only "processed" file in
the archive and the subset used in the published ML experiments), and loads
it as a typed pandas DataFrame.

No records are dropped and no data are synthesized. The raw target column
``num`` is preserved; :func:`load_heart_data` additionally creates a binary
``target`` column using the documented 0 vs. 1-4 mapping.
"""

from __future__ import annotations

import hashlib
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Official dataset metadata.
UCI_DATASET_ID = 45
UCI_DATASET_DOI = "10.24432/C52P4X"
UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/45/heart+disease"
UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

# Snapshot retrieved 2026-10-27 from the URL above. The checksum makes the
# download reproducible and detects accidental or unexpected changes.
ARCHIVE_SHA256 = (
    "B17CD273DA9CE1CAA4710FCE80227EA454D4DBF9FCBC8E6A9121672751563ADC"
)
ARCHIVE_FILENAME = "heart+disease.zip"
EXTRACTED_DIRNAME = "heart_disease"
MEMBER_FILENAME = "processed.cleveland.data"

# The 13 published features, in the order used by the data file.
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_COLUMN = "num"
BINARY_TARGET_COLUMN = "target"
COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


class DatasetIntegrityError(RuntimeError):
    """Raised when the downloaded archive is missing or has changed."""


def sha256_file(path: str | Path) -> str:
    """Return the uppercase SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _download_archive(archive_path: Path) -> None:
    """Download the official archive atomically, with a manual fallback hint."""
    partial_path = archive_path.with_name(archive_path.name + ".part")
    try:
        with requests.get(
            UCI_ARCHIVE_URL, stream=True, timeout=(10, 120)
        ) as response:
            response.raise_for_status()
            with partial_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        handle.write(chunk)
        partial_path.replace(archive_path)
    except requests.RequestException as exc:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the official UCI Heart Disease archive from "
            f"{UCI_ARCHIVE_URL}.\n"
            "Check your network connection, or download the ZIP manually in a "
            f"browser from {UCI_DATASET_PAGE} and save it as:\n"
            f"  {archive_path}\n"
            "Then re-run the verification script. The loader will verify the "
            "archive checksum before extracting it.\n"
            f"Original error: {exc}"
        ) from exc
    except OSError:
        partial_path.unlink(missing_ok=True)
        raise


def _verify_archive(archive_path: Path) -> None:
    actual = sha256_file(archive_path)
    if actual != ARCHIVE_SHA256:
        raise DatasetIntegrityError(
            f"Checksum mismatch for {archive_path}.\n"
            f"Expected: {ARCHIVE_SHA256}\n"
            f"Actual:   {actual}\n"
            "The file is not the expected UCI snapshot. Delete it and retry the "
            "download. If UCI has legitimately republished the dataset, inspect "
            "the new file and update ARCHIVE_SHA256 in src/data_loader.py."
        )


def _extract_member(archive_path: Path, extracted_path: Path) -> None:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            try:
                payload = archive.read(MEMBER_FILENAME)
            except KeyError as exc:
                raise DatasetIntegrityError(
                    f"{archive_path} does not contain {MEMBER_FILENAME}."
                ) from exc
    except zipfile.BadZipFile as exc:
        raise DatasetIntegrityError(
            f"{archive_path} is not a valid ZIP archive. Delete it and retry."
        ) from exc

    if not payload:
        raise DatasetIntegrityError(
            f"{MEMBER_FILENAME} in {archive_path} is empty."
        )

    extracted_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=extracted_path.name + ".",
        suffix=".part",
        dir=extracted_path.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(payload)

    try:
        temp_path.replace(extracted_path)
    except OSError:
        temp_path.unlink(missing_ok=True)
        raise


def ensure_dataset(
    data_dir: str | Path | None = None,
    *,
    force_download: bool = False,
    verify_checksum: bool = True,
) -> Path:
    """Return the path to the verified Cleveland CSV, downloading if needed.

    Parameters
    ----------
    data_dir:
        Cache directory. Defaults to ``<project>/data/raw``.
    force_download:
        Download a fresh archive even if a cached copy exists.
    verify_checksum:
        Enforce the pinned SHA-256 checksum. Keep this enabled for
        reproducibility.
    """
    target_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    archive_path = target_dir / ARCHIVE_FILENAME
    extracted_path = target_dir / EXTRACTED_DIRNAME / MEMBER_FILENAME

    downloaded = False
    if force_download or not archive_path.exists():
        _download_archive(archive_path)
        downloaded = True

    if verify_checksum:
        try:
            _verify_archive(archive_path)
        except DatasetIntegrityError:
            if downloaded:
                archive_path.unlink(missing_ok=True)
            raise

    # The verified archive is the source of truth. Re-extracting the small
    # member on each load prevents a stale or manually edited cached CSV from
    # being used silently.
    _extract_member(archive_path, extracted_path)

    return extracted_path


def _validate_raw_data(frame: pd.DataFrame, source: Path | None = None) -> None:
    if list(frame.columns) != COLUMNS:
        raise DatasetIntegrityError(
            "Unexpected dataset schema. Expected columns "
            f"{COLUMNS}, got {list(frame.columns)}."
        )
    if frame.empty:
        raise DatasetIntegrityError(
            f"The dataset at {source or 'the configured source'} has no rows."
        )
    if frame[TARGET_COLUMN].isna().any():
        raise DatasetIntegrityError(
            f"The raw target column '{TARGET_COLUMN}' contains missing values."
        )

    raw_labels = frame[TARGET_COLUMN].unique()
    if not all(float(value).is_integer() for value in raw_labels):
        raise DatasetIntegrityError(
            f"The raw target column '{TARGET_COLUMN}' must contain integers."
        )

    labels = {int(value) for value in raw_labels}
    unexpected = labels - set(range(5))
    if unexpected:
        raise DatasetIntegrityError(
            "Unexpected raw target labels: "
            f"{sorted(unexpected)}. Expected integer values 0-4."
        )


def load_raw_heart_data(
    data_dir: str | Path | None = None,
    *,
    force_download: bool = False,
    verify_checksum: bool = True,
) -> pd.DataFrame:
    """Load the raw 14-column Cleveland data without target transformation.

    Missing values encoded as ``?`` in the source file become ``NaN``. The six
    affected records are retained; imputation happens inside model pipelines.
    """
    source = ensure_dataset(
        data_dir,
        force_download=force_download,
        verify_checksum=verify_checksum,
    )
    frame = pd.read_csv(
        source,
        header=None,
        names=COLUMNS,
        na_values=["?"],
    )
    _validate_raw_data(frame, source)
    return frame


def add_binary_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``target``: 0 for ``num == 0``, 1 for ``num`` in 1-4.

    The mapping is applied only after validating that the actual labels are a
    subset of {0, 1, 2, 3, 4}.
    """
    _validate_raw_data(frame)
    result = frame.copy()
    result[BINARY_TARGET_COLUMN] = (result[TARGET_COLUMN] != 0).astype("int8")
    return result


def load_heart_data(
    data_dir: str | Path | None = None,
    *,
    binarize_target: bool = True,
    force_download: bool = False,
    verify_checksum: bool = True,
) -> pd.DataFrame:
    """Load the verified Cleveland dataset, optionally adding ``target``.

    Returns 15 columns by default: the 13 features, the raw ``num`` label, and
    the derived binary ``target`` label.
    """
    frame = load_raw_heart_data(
        data_dir,
        force_download=force_download,
        verify_checksum=verify_checksum,
    )
    if binarize_target:
        frame = add_binary_target(frame)
    return frame
