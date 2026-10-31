# Heart Disease Prediction Using Comparative Machine Learning Algorithms

A college AI/ML mini-project comparing seven supervised, ensemble, and
semi-supervised classifiers on the UCI Heart Disease dataset.

**Current milestone:** Milestone 3 is complete. The project includes exactly
two Streamlit pages that load the saved pipelines and evaluation table without
retraining. Milestone 2's seven trained models, fixed split, and metrics remain
unchanged.

> This project is for educational purposes only and must not be used as a
> medical diagnostic tool.

## Dataset

| Item | Value |
| --- | --- |
| Dataset | Heart Disease |
| UCI repository ID | 45 |
| DOI | [10.24432/C52P4X](https://doi.org/10.24432/C52P4X) |
| Official page | <https://archive.ics.uci.edu/dataset/45/heart+disease> |
| Official archive | <https://archive.ics.uci.edu/static/public/45/heart+disease.zip> |
| License | CC BY 4.0 |
| Retrieved | 2026-10-27 |
| Archive SHA-256 | `B17CD273DA9CE1CAA4710FCE80227EA454D4DBF9FCBC8E6A9121672751563ADC` |
| Archive internal timestamp | 2023-05-22 |
| File used | `processed.cleveland.data` |
| Records | 303 |
| Columns | 14 (13 features + raw target) |

The UCI archive contains data from four hospitals, but only the Cleveland
subset is available as an official *processed* file. Cleveland is also the
subset used in the published machine-learning experiments and is the standard
choice for this dataset, so this project uses **only**
`processed.cleveland.data`. The other centers are not merged, and no records
are filtered out.

UCI does not publish a semantic version number for this dataset. The archive
SHA-256 above acts as the version pin. The loader verifies it on every run.

### Features

| # | Column | Type | Description | Missing |
| --- | --- | --- | --- | --- |
| 1 | `age` | numeric | Age in years | 0 |
| 2 | `sex` | categorical | 1 = male, 0 = female | 0 |
| 3 | `cp` | categorical | Chest pain type: 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic | 0 |
| 4 | `trestbps` | numeric | Resting blood pressure (mm Hg) | 0 |
| 5 | `chol` | numeric | Serum cholesterol (mg/dl) | 0 |
| 6 | `fbs` | categorical | Fasting blood sugar > 120 mg/dl: 1 = true, 0 = false | 0 |
| 7 | `restecg` | categorical | Resting ECG: 0 = normal, 1 = ST-T wave abnormality, 2 = probable/definite left ventricular hypertrophy | 0 |
| 8 | `thalach` | numeric | Maximum heart rate achieved | 0 |
| 9 | `exang` | categorical | Exercise-induced angina: 1 = yes, 0 = no | 0 |
| 10 | `oldpeak` | numeric | ST depression induced by exercise relative to rest | 0 |
| 11 | `slope` | categorical | Slope of the peak exercise ST segment: 1 = upsloping, 2 = flat, 3 = downsloping | 0 |
| 12 | `ca` | categorical | Number of major vessels (0-3) colored by fluoroscopy | 4 |
| 13 | `thal` | categorical | Thalassemia: 3 = normal, 6 = fixed defect, 7 = reversible defect | 2 |
| 14 | `num` | target | Raw diagnosis label (0-4) | 0 |

The counts and value sets were verified from the downloaded archive, not
assumed in advance.

### Target definition

The raw `num` column is integer-valued from 0 to 4:

- `0` = no angiographic disease (< 50% diameter narrowing)
- `1`, `2`, `3`, `4` = presence of disease (> 50% diameter narrowing)

The downloaded file actually contains all five labels
(`{0, 1, 2, 3, 4}`), with this distribution:

| Raw `num` | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| Records | 164 | 55 | 36 | 35 | 13 |

`src/data_loader.py` validates those labels before deriving the binary target:

```text
target = 0  if num == 0
target = 1  if num in {1, 2, 3, 4}
```

Result: **164 negative** and **139 positive** records. The raw `num` column is
retained alongside `target` for auditability.

### Missing values

Six records contain missing values encoded as `?` in the source file:

- `ca`: 4 missing values
- `thal`: 2 missing values
- Total: 6 missing cells across 6 records

**No rows are dropped.** The preprocessing transformer imputes numeric
features with the median and categorical features with the most frequent
value. Because the imputer is inside each scikit-learn pipeline, it is fitted
on training data only, preventing test-set leakage.

## The seven model configurations

The registry in `src/train_models.py` contains exactly these models, in this
order:

| # | Registry key | Model | Conceptual category | Implementation notes |
| --- | --- | --- | --- | --- |
| 1 | `logistic_regression` | Logistic Regression | Supervised | L2-regularized linear classifier |
| 2 | `decision_tree` | Decision Tree Classifier | Supervised | Fixed depth/leaf-size controls; no test-set tuning |
| 3 | `svm` | Support Vector Machine (SVM) | Supervised | RBF kernel; uncalibrated, ROC-AUC from `decision_function` |
| 4 | `knn` | K-Nearest Neighbors (KNN) | Supervised | Distance-weighted, `k=7` |
| 5 | `random_forest` | Random Forest Classifier | Supervised + ensemble | 300 trees |
| 6 | `gradient_boosting` | Gradient Boosting Classifier | Supervised + ensemble | 200 estimators, learning rate 0.05 |
| 7 | `self_training_mlp` | Self-Training Neural Network (MLP) | Semi-supervised | `SelfTrainingClassifier` wrapping `MLPClassifier` |

Random Forest and Gradient Boosting are both supervised and ensemble methods.
The seventh model is the only semi-supervised configuration.

## Training and evaluation protocol

### One fixed, stratified split

- Dataset: 303 records.
- Split: 80% training / 20% test.
- Random seed: `42`.
- Training rows: **242** (131 negative, 111 positive).
- Test rows: **61** (33 negative, 28 positive).
- Split strategy: `train_test_split(..., stratify=target, random_state=42)`.
- Every model is evaluated on the same 61 test rows.

The exact train/test indices are saved in `artifacts/data_split.json`. The
standalone evaluation command reloads those indices, so it cannot accidentally
evaluate on a different test set.

### Leakage-safe preprocessing

Every model is a scikit-learn `Pipeline`:

```text
Pipeline(
    preprocessor=ColumnTransformer(
        numeric: SimpleImputer(median) -> StandardScaler,
        categorical: SimpleImputer(most_frequent) -> OneHotEncoder,
    ),
    classifier=...
)
```

The preprocessor is fitted only on `X_train` when `pipeline.fit` is called.
Test features are passed to the already-fitted pipeline for prediction only.
No test data is used for imputation fitting, scaling, encoding, hyperparameter
selection, or threshold tuning.

The semi-supervised pipeline also fits its preprocessing on training features
only. Its unlabeled rows are training rows, never test rows.

### Semi-supervised learning and its limitation

The seventh model is a genuine `SelfTrainingClassifier` with an
`MLPClassifier` base estimator. To simulate limited labels **inside the
training set only**:

- 30% of the 242 training labels are retained: **72 labels**
  (39 negative, 33 positive).
- The remaining **170 training rows are marked `-1`**, the unlabeled marker
  required by scikit-learn.
- The test labels are never passed to the semi-supervised pipeline and are not
  used to choose pseudo-labels.
- The self-training run used `threshold=0.75`, `criterion="threshold"`, and
  `max_iter=20`. It stopped after 3 iterations having pseudo-labeled 169 of the
  170 unlabeled training rows; 1 row remained unlabeled.

**Important limitation:** this is not a real external unlabeled patient cohort.
It is a controlled simulation in which labels are hidden only within the
training split. The supervised models use all 242 training labels, while the
semi-supervised model initially uses only 72. Therefore the comparison measures
the effect of the *different label budgets and algorithms together*, not the
algorithm alone. The semi-supervised result must not be presented as proof that
self-training is superior or inferior to fully supervised training.

### Metrics

All models are scored with:

- Accuracy
- Precision for the positive class (disease = 1)
- Recall for the positive class
- F1-score for the positive class
- ROC-AUC

Classification metrics use `zero_division=0`. ROC-AUC is computed from
continuous scores: positive-class `predict_proba` probabilities for
probabilistic models, and SVC `decision_function` output for the uncalibrated
SVM. Hard class predictions are never used as ROC-AUC scores.

## Actual results

These values were produced by the command below on the pinned dataset and
saved to `artifacts/comparison_results.csv`. The test set contains only 61
records, so small differences should not be over-interpreted.

| Model | Category | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | supervised | 0.8852 | 0.8387 | 0.9286 | 0.8814 | 0.9665 |
| Decision Tree Classifier | supervised | 0.7377 | 0.6765 | 0.8214 | 0.7419 | 0.8555 |
| Support Vector Machine (SVM) | supervised | 0.8852 | 0.8387 | 0.9286 | 0.8814 | 0.9643 |
| K-Nearest Neighbors (KNN) | supervised | 0.9016 | 0.8667 | 0.9286 | 0.8966 | 0.9643 |
| Random Forest Classifier | supervised, ensemble | 0.9016 | 0.8438 | 0.9643 | 0.9000 | 0.9470 |
| Gradient Boosting Classifier | supervised, ensemble | 0.8852 | 0.8387 | 0.9286 | 0.8814 | 0.9524 |
| Self-Training Neural Network (MLP) | semi-supervised | 0.8525 | 0.8519 | 0.8214 | 0.8364 | 0.9307 |

The chart is saved as `artifacts/comparison_chart.png`. The exact same table is
also available as CSV, JSON, and Markdown under `artifacts/`.

## Project structure

```text
AIL-mini-project/
├── app.py                     # two-page navigation entry point
├── pages/
│   ├── 1_Predict.py           # individual prediction page
│   └── 2_Compare_Models.py    # all-seven comparison page
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # official download, checksum, schema, target
│   ├── preprocessing.py       # feature groups and sklearn transformer
│   ├── train_models.py        # 7-model training, artifacts, evaluation run
│   ├── evaluate_models.py     # metrics, results table, chart, saved-model eval
│   ├── predict.py             # artifact loading and single-record prediction
│   └── prediction_ui.py       # shared input metadata and UI helper functions
├── scripts/
│   └── verify_data.py         # data-loading verification script
├── tests/
│   ├── test_models.py         # training/evaluation/loading/reproducibility tests
│   └── test_prediction_ui.py  # UI helper, prediction, metrics, and AppTest tests
├── artifacts/                 # generated artifacts, ignored by Git
│   ├── .gitkeep
│   ├── comparison_chart.png
│   ├── comparison_results.csv
│   ├── comparison_results.json
│   ├── comparison_results.md
│   ├── data_split.json
│   ├── evaluation_metadata.json
│   ├── training_metadata.json
│   └── models/
│       ├── registry.json
│       └── <model_key>.joblib
├── data/                      # downloaded dataset, ignored by Git
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python **3.13** (tested with 3.13.0)
- scikit-learn **1.9.0**
- pandas 3.0.3, NumPy 2.4.6
- joblib 1.6.0 for pipeline persistence
- matplotlib 3.11.0 for the comparison chart and the UI chart
- streamlit 1.66.1 for the two-page UI
- TensorFlow 2.21.0 remains pinned from the initial environment setup and
  imports successfully on Python 3.13. The current seven-model comparison uses
  scikit-learn's `MLPClassifier`, not TensorFlow.

Two scikit-learn 1.9 API details are handled deliberately:

- `SelfTrainingClassifier` is constructed with the current `estimator=`
  argument (not the older `base_estimator=` name).
- `SVC(probability=True)` is deprecated in 1.9. The SVM is left uncalibrated
  and evaluated with its continuous `decision_function` scores.

The dataset and generated model artifacts are **not** committed to Git:

- `data/` is ignored because it is downloaded reproducibly.
- `artifacts/*` is ignored except for `.gitkeep`.

## Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, call the interpreter
directly:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Verify the data pipeline

```powershell
python scripts/verify_data.py
```

On first run, the script downloads the official ZIP archive, verifies its
SHA-256 checksum, extracts `processed.cleveland.data`, and checks:

- the 14-column schema and 303 records,
- the raw target labels `{0, 1, 2, 3, 4}`,
- the 0 vs. 1-4 binary target mapping,
- the missing-value pattern,
- a preprocessing smoke test with no remaining non-finite values.

Expected summary:

```text
UCI Heart Disease - data verification
Shape: 303 rows x 15 columns
Missing values: ca=4, thal=2 (6 rows; retained)
Raw target 'num': 0=164, 1=55, 2=36, 3=35, 4=13
Binary target 'target': 0=164, 1=139
All checks passed.
```

### Manual download fallback

UCI does not publish this dataset as a standalone CSV; the official archive
contains the `.data` file used here. If automated downloading is blocked:

1. Open <https://archive.ics.uci.edu/dataset/45/heart-disease> in a browser.
2. Download the official ZIP archive (`heart+disease.zip`, about 126 KB).
3. Save it exactly as `data/raw/heart+disease.zip`.
4. Run `python scripts/verify_data.py`.

The loader will verify the pinned checksum and extract
`processed.cleveland.data` itself, so a manually downloaded ZIP and an
automated download produce identical data. Do not substitute a Kaggle mirror
or hand-edited CSV. If the checksum differs, inspect the new official file
deliberately and only then update `ARCHIVE_SHA256` in `src/data_loader.py`.

## Train and evaluate the seven models

Run the complete pipeline from the project root:

```powershell
python -m src.train_models
```

This command:

1. loads and verifies the dataset,
2. creates the fixed stratified 80/20 split,
3. creates the partially labeled training target for the semi-supervised model,
4. fits all seven leakage-safe pipelines,
5. saves every pipeline and the split/registry metadata under `artifacts/`,
6. evaluates all seven models on the untouched test set, and
7. writes the comparison table and chart under `artifacts/`.

To independently re-evaluate the saved artifacts without retraining:

```powershell
python -m src.evaluate_models
```

This reloads the saved pipelines and the saved test indices, recomputes the
five metrics, and overwrites the comparison outputs. It is the quickest check
that persisted models and evaluation are consistent.

## Saved artifacts

| Artifact | Contents |
| --- | --- |
| `artifacts/models/registry.json` | Model keys, display names, categories, artifact filenames, params |
| `artifacts/models/<key>.joblib` | Fitted preprocessing + classifier pipeline |
| `artifacts/data_split.json` | Seed, sizes, exact train/test indices, target counts |
| `artifacts/training_metadata.json` | Versions, split, semi-supervised label mask summary, leakage flags |
| `artifacts/comparison_results.csv` | Machine-readable metric table |
| `artifacts/comparison_results.json` | Metric table plus evaluation metadata |
| `artifacts/comparison_results.md` | Markdown metric table |
| `artifacts/comparison_chart.png` | Grouped metric bar chart |
| `artifacts/evaluation_metadata.json` | Protocol, metric definitions, score sources, versions |

`artifacts/*` is gitignored, so these files are generated locally rather than
committed.

## Prediction API

The saved pipelines can be loaded consistently without touching the UI:

```python
from src.predict import predict_heart_disease

# Illustrative values only; not a real patient.
patient = {
    "age": 58,
    "sex": 1,
    "cp": 4,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 3,
}

result = predict_heart_disease(patient, "logistic_regression")
print(result["prediction"], result["label"])
print(result["score"], result["score_type"])
```

For probabilistic models, `score_type` is `"probability"`. For the
uncalibrated SVM, it is `"decision_function"`; that value is not a probability
and must not be described as one.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The suite covers:

- the exact seven-model registry,
- stratified split reproducibility and disjointness,
- partial-label construction using training rows only,
- genuine self-training pseudo-labeling,
- preprocessing fitted on training data only,
- determinism of a repeated fit,
- saved artifact existence and metric validity,
- loading all seven serialized pipelines,
- continuous-score ROC-AUC for the SVM,
- standalone saved-model evaluation matching the training run,
- single-record prediction and validation errors,
- reconstruction of the saved test set,
- exactly 13 UI inputs in `FEATURE_COLUMNS` order, validated against the
  preprocessing groups,
- the verified numeric ranges and categorical codes,
- predictions from all seven saved models and correct score types, including
  the SVM decision score,
- metrics joined to predictions by `model_key`, not row order,
- missing and malformed artifact error handling,
- both pages rendered and exercised with Streamlit `AppTest`, including the
  prediction, SVM score-label, reset, comparison, metrics, and navigation paths.

Actual result from the completed run:

```text
Ran 35 tests

OK
```

The exit code was 0. Exact timings vary between machines.

## Streamlit UI

Launch the two-page app from the project root:

```powershell
streamlit run app.py
```

`app.py` uses `st.navigation` with exactly two entries:

| Page | Purpose |
| --- | --- |
| **Predict** (`pages/1_Predict.py`) | Enter one patient record, choose one of the seven saved models, and predict |
| **Compare Models** (`pages/2_Compare_Models.py`) | Enter one record once, predict with all seven models, and compare the saved evaluation metrics |

Both pages:

- load only the existing `artifacts/models/registry.json` and `.joblib`
  pipelines; they never train or retrain a model,
- collect the same 13 features in the exact training order,
- display the predicted class (`0` or `1`) and the verified target label
  (`no heart disease` or `heart disease`),
- show probabilities only for probabilistic models, and label the SVM output
  as a decision score that is **not** a probability,
- display the educational-use disclaimer.

The comparison page reads the saved held-out metrics from
`artifacts/comparison_results.csv` and matches them to predictions by
`model_key`. The chart is rendered in memory from that saved table. The page
does not recompute or overwrite the official metrics. Its majority-vote line is
labelled as an informational count, not a separately trained model.

The shared input metadata and helper logic live in `src/prediction_ui.py`.
Feature names, order, and groups are checked against `FEATURE_COLUMNS`,
`NUMERIC_FEATURES`, and `CATEGORICAL_FEATURES` at import time.

## Status and next steps

Completed:

- official, checksum-pinned dataset retrieval and validation,
- leakage-safe preprocessing,
- seven fixed model configurations with saved pipelines,
- semi-supervised training using only partially labeled training data,
- shared untouched test set and five-metric comparison,
- comparison table/chart and reproducibility tests,
- exactly two Streamlit pages using the saved models and saved metrics.

Possible next steps:

1. Add cross-validated uncertainty estimates on the training set.
2. Evaluate on an external cohort before making any broader claims.
3. Add probability calibration if calibrated probabilities are needed; the SVM
   is currently uncalibrated and reports a decision score.

## Citation

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989).
*Heart Disease* [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C52P4X>
