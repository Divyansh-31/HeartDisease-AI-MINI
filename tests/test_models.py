"""Tests for training, evaluation, artifact loading, and reproducibility."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import SelfTrainingClassifier

from src.data_loader import load_heart_data
from src.evaluate_models import (
    METRIC_COLUMNS,
    evaluate_pipeline,
    evaluate_saved_models,
    load_test_data,
    results_to_markdown,
)
from src.predict import (
    load_all_models,
    load_model,
    predict_heart_disease,
    predict_scores,
)
from src.preprocessing import NUMERIC_FEATURES
from src.train_models import (
    MODEL_REGISTRY,
    SEMI_SUPERVISED_LABELED_FRACTION,
    SEMI_SUPERVISED_RANDOM_STATE,
    build_model_pipeline,
    make_partially_labeled_target,
    make_train_test_split,
    train_and_evaluate_models,
)

EXPECTED_KEYS = [
    "logistic_regression",
    "decision_tree",
    "svm",
    "knn",
    "random_forest",
    "gradient_boosting",
    "self_training_mlp",
]


class ModelPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.artifacts_dir = Path(cls.temp_dir.name) / "artifacts"
        cls.frame = load_heart_data()
        cls.split = make_train_test_split(cls.frame)
        cls.training_run = train_and_evaluate_models(
            artifacts_dir=cls.artifacts_dir,
            verbose=False,
        )
        cls.results = cls.training_run["results"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_registry_has_exactly_seven_configurations(self) -> None:
        self.assertEqual(list(MODEL_REGISTRY), EXPECTED_KEYS)
        self.assertEqual(len(MODEL_REGISTRY), 7)
        self.assertFalse(
            any("voting" in key.lower() for key in MODEL_REGISTRY),
        )
        self.assertFalse(
            any("adaboost" in key.lower() for key in MODEL_REGISTRY),
        )

        for key in ("random_forest", "gradient_boosting"):
            self.assertIn("ensemble", MODEL_REGISTRY[key].categories)
            self.assertIn("supervised", MODEL_REGISTRY[key].categories)
        self.assertEqual(
            MODEL_REGISTRY["self_training_mlp"].categories, ("semi-supervised",)
        )
        for key in ("logistic_regression", "decision_tree", "svm", "knn"):
            self.assertEqual(MODEL_REGISTRY[key].categories, ("supervised",))

    def test_split_is_reproducible_stratified_and_disjoint(self) -> None:
        repeat = make_train_test_split(self.frame, random_state=42)
        self.assertEqual(self.split.train_indices, repeat.train_indices)
        self.assertEqual(self.split.test_indices, repeat.test_indices)
        self.assertEqual(len(self.split.X_train), 242)
        self.assertEqual(len(self.split.X_test), 61)

        train_set = set(self.split.train_indices)
        test_set = set(self.split.test_indices)
        self.assertFalse(train_set & test_set)
        self.assertEqual(train_set | test_set, set(range(len(self.frame))))
        self.assertEqual(set(self.split.y_train.unique().tolist()), {0, 1})
        self.assertEqual(set(self.split.y_test.unique().tolist()), {0, 1})

    def test_semi_supervised_uses_only_training_labels(self) -> None:
        y_semi, labeled_indices = make_partially_labeled_target(
            self.split.y_train,
            labeled_fraction=SEMI_SUPERVISED_LABELED_FRACTION,
            random_state=SEMI_SUPERVISED_RANDOM_STATE,
        )
        repeat_y_semi, repeat_indices = make_partially_labeled_target(
            self.split.y_train,
            labeled_fraction=SEMI_SUPERVISED_LABELED_FRACTION,
            random_state=SEMI_SUPERVISED_RANDOM_STATE,
        )

        np.testing.assert_array_equal(y_semi, repeat_y_semi)
        np.testing.assert_array_equal(labeled_indices, repeat_indices)
        self.assertEqual(len(y_semi), len(self.split.y_train))
        self.assertTrue(np.all(labeled_indices < len(self.split.y_train)))

        unlabeled_mask = np.ones(len(y_semi), dtype=bool)
        unlabeled_mask[labeled_indices] = False
        self.assertTrue(np.all(y_semi[unlabeled_mask] == -1))
        self.assertEqual(set(y_semi[labeled_indices].tolist()), {0, 1})
        self.assertGreater(int(unlabeled_mask.sum()), 0)

        with (self.artifacts_dir / "training_metadata.json").open(
            "r", encoding="utf-8"
        ) as handle:
            metadata = json.load(handle)
        semi = metadata["semi_supervised"]
        self.assertEqual(semi["n_labeled"], len(labeled_indices))
        self.assertEqual(
            semi["n_unlabeled"], len(y_semi) - len(labeled_indices)
        )
        self.assertFalse(metadata["test_set_used_for_training"])
        self.assertFalse(metadata["test_set_used_for_preprocessing"])
        self.assertFalse(metadata["test_set_used_for_hyperparameter_selection"])
        self.assertFalse(metadata["test_set_used_for_threshold_tuning"])

    def test_self_training_artifact_is_genuinely_semi_supervised(self) -> None:
        model = load_model("self_training_mlp", self.artifacts_dir)
        classifier = model.named_steps["classifier"]
        self.assertIsInstance(classifier, SelfTrainingClassifier)
        self.assertIsInstance(classifier.estimator, MLPClassifier)
        self.assertGreaterEqual(classifier.n_iter_, 1)
        self.assertEqual(
            int(np.sum(classifier.labeled_iter_ == 0)),
            len(self.training_run["semi_supervised"]["labeled_indices"]),
        )
        self.assertGreater(int(np.sum(classifier.labeled_iter_ > 0)), 0)
        self.assertGreaterEqual(int(np.sum(classifier.labeled_iter_ == -1)), 0)

    def test_preprocessing_is_fitted_on_training_data_only(self) -> None:
        model = load_model("logistic_regression", self.artifacts_dir)
        preprocessor = model.named_steps["preprocessor"]
        numeric_pipeline = preprocessor.named_transformers_["numeric"]
        imputer = numeric_pipeline.named_steps["imputer"]
        scaler = numeric_pipeline.named_steps["scaler"]

        numeric_columns = list(NUMERIC_FEATURES)
        imputed_train = imputer.transform(self.split.X_train[numeric_columns])
        np.testing.assert_allclose(
            scaler.mean_,
            imputed_train.mean(axis=0),
            rtol=0,
            atol=1e-12,
        )

    def test_training_is_reproducible(self) -> None:
        first = build_model_pipeline("logistic_regression", random_state=42)
        second = build_model_pipeline("logistic_regression", random_state=42)
        first.fit(self.split.X_train, self.split.y_train)
        second.fit(self.split.X_train, self.split.y_train)

        np.testing.assert_array_equal(
            first.predict(self.split.X_test),
            second.predict(self.split.X_test),
        )
        np.testing.assert_allclose(
            first.predict_proba(self.split.X_test),
            second.predict_proba(self.split.X_test),
            rtol=0,
            atol=1e-15,
        )

    def test_artifacts_and_comparison_outputs_exist(self) -> None:
        registry_path = self.artifacts_dir / "models" / "registry.json"
        with registry_path.open("r", encoding="utf-8") as handle:
            registry = json.load(handle)
        self.assertEqual(
            [entry["key"] for entry in registry["models"]], EXPECTED_KEYS
        )

        model_files = sorted(
            path.name
            for path in (self.artifacts_dir / "models").glob("*.joblib")
        )
        self.assertEqual(
            model_files, sorted(f"{key}.joblib" for key in EXPECTED_KEYS)
        )

        for filename in (
            "comparison_results.csv",
            "comparison_results.json",
            "comparison_results.md",
            "comparison_chart.png",
            "evaluation_metadata.json",
            "data_split.json",
            "training_metadata.json",
        ):
            path = self.artifacts_dir / filename
            self.assertTrue(path.exists(), f"Missing artifact: {path}")
            self.assertGreater(path.stat().st_size, 0)

        self.assertEqual(len(self.results), 7)
        self.assertEqual(self.results["model_key"].tolist(), EXPECTED_KEYS)
        for metric in METRIC_COLUMNS:
            self.assertTrue(self.results[metric].notna().all(), metric)
            self.assertTrue(
                self.results[metric].between(0, 1, inclusive="both").all(),
                metric,
            )

        score_types = dict(
            zip(self.results["model_key"], self.results["score_type"])
        )
        self.assertEqual(score_types["svm"], "decision_function")
        for key in EXPECTED_KEYS:
            if key != "svm":
                self.assertEqual(score_types[key], "probability")

        markdown = results_to_markdown(self.results)
        for display_name in self.results["model_name"]:
            self.assertIn(display_name, markdown)

    def test_saved_models_load_and_return_continuous_scores(self) -> None:
        models = load_all_models(self.artifacts_dir)
        self.assertEqual(list(models), EXPECTED_KEYS)

        for key, model in models.items():
            predictions = model.predict(self.split.X_test)
            self.assertEqual(len(predictions), len(self.split.X_test))
            self.assertTrue(set(np.unique(predictions).tolist()) <= {0, 1})

            scores, score_type = predict_scores(model, self.split.X_test)
            self.assertIsNotNone(scores, key)
            self.assertIsNotNone(score_type, key)
            self.assertEqual(len(scores), len(self.split.X_test))
            self.assertTrue(np.all(np.isfinite(scores)), key)
            self.assertGreater(np.unique(scores).size, 2, key)

    def test_svm_roc_auc_uses_decision_function_scores(self) -> None:
        model = load_model("svm", self.artifacts_dir)
        scores, score_type = predict_scores(model, self.split.X_test)
        self.assertEqual(score_type, "decision_function")
        self.assertFalse(np.all(np.isin(scores, [0, 1])))

        expected = roc_auc_score(self.split.y_test, scores)
        actual = float(
            self.results.loc[
                self.results["model_key"] == "svm", "roc_auc"
            ].iloc[0]
        )
        self.assertAlmostEqual(actual, expected, places=12)

    def test_saved_evaluation_matches_training_results(self) -> None:
        saved_results, paths = evaluate_saved_models(
            artifacts_dir=self.artifacts_dir
        )
        self.assertEqual(saved_results["model_key"].tolist(), EXPECTED_KEYS)
        for metric in METRIC_COLUMNS:
            np.testing.assert_allclose(
                saved_results[metric].to_numpy(dtype=float),
                self.results[metric].to_numpy(dtype=float),
                rtol=0,
                atol=1e-12,
            )
        self.assertTrue(paths["results_csv"].exists())
        self.assertTrue(paths["comparison_chart"].exists())

    def test_direct_evaluation_matches_saved_results(self) -> None:
        model = load_model("random_forest", self.artifacts_dir)
        row = evaluate_pipeline(
            model,
            self.split.X_test,
            self.split.y_test,
            model_key="random_forest",
            display_name="Random Forest Classifier",
            categories=("supervised", "ensemble"),
        )
        expected = self.results.loc[
            self.results["model_key"] == "random_forest"
        ].iloc[0]
        for metric in METRIC_COLUMNS:
            self.assertAlmostEqual(
                float(row[metric]), float(expected[metric]), places=12
            )

    def test_prediction_helper_and_errors(self) -> None:
        patient = self.split.X_test.iloc[0].to_dict()
        result = predict_heart_disease(
            patient, "logistic_regression", self.artifacts_dir
        )
        self.assertEqual(result["model_key"], "logistic_regression")
        self.assertIn(result["prediction"], (0, 1))
        self.assertEqual(result["score_type"], "probability")
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)

        svm_result = predict_heart_disease(patient, "svm", self.artifacts_dir)
        self.assertEqual(svm_result["score_type"], "decision_function")

        missing = dict(patient)
        missing.pop("age")
        with self.assertRaises(ValueError):
            predict_heart_disease(
                missing, "logistic_regression", self.artifacts_dir
            )

        extra = dict(patient)
        extra["unexpected_field"] = 1.0
        with self.assertRaises(ValueError):
            predict_heart_disease(
                extra, "logistic_regression", self.artifacts_dir
            )

        with self.assertRaises(ValueError):
            load_model("not_a_registered_model", self.artifacts_dir)

        with self.assertRaises(ValueError):
            predict_heart_disease(
                self.split.X_test.iloc[:2], "logistic_regression", self.artifacts_dir
            )

    def test_saved_test_set_is_reconstructed_correctly(self) -> None:
        X_test, y_test, manifest = load_test_data(self.artifacts_dir)
        self.assertEqual(len(X_test), 61)
        self.assertEqual(len(y_test), 61)
        self.assertEqual(
            X_test["age"].tolist(), self.split.X_test["age"].tolist()
        )
        self.assertEqual(y_test.tolist(), self.split.y_test.tolist())
        self.assertEqual(manifest["random_state"], 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)

