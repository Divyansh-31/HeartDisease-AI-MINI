"""Tests for the shared UI helpers and the two Streamlit pages."""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
from streamlit.testing.v1 import AppTest

from src.data_loader import FEATURE_COLUMNS, load_heart_data
from src.prediction_ui import (
    DISCLAIMER,
    FEATURE_SPECS,
    build_metrics_figure,
    compare_predictions,
    default_patient,
    format_score,
    format_score_value,
    get_model_options,
    load_evaluation_metrics,
    metrics_to_markdown,
    predict_all,
    predict_one,
    predictions_to_markdown,
    score_type_label,
    validate_patient_record,
)
from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICT_PAGE = PROJECT_ROOT / "pages" / "1_Predict.py"
COMPARE_PAGE = PROJECT_ROOT / "pages" / "2_Compare_Models.py"
APP_SCRIPT = PROJECT_ROOT / "app.py"

# AppTest emits a bare-mode context warning that is not relevant to these tests.
logging.getLogger(
    "streamlit.runtime.scriptrunner_utils.script_run_context"
).setLevel(logging.ERROR)


class FeatureMetadataTests(unittest.TestCase):
    def test_exactly_thirteen_features_in_training_order(self) -> None:
        self.assertEqual(len(FEATURE_SPECS), 13)
        self.assertEqual(
            tuple(spec.name for spec in FEATURE_SPECS),
            tuple(FEATURE_COLUMNS),
        )

    def test_feature_kinds_match_preprocessing(self) -> None:
        self.assertEqual(
            tuple(spec.name for spec in FEATURE_SPECS if spec.kind == "numeric"),
            tuple(NUMERIC_FEATURES),
        )
        self.assertEqual(
            tuple(spec.name for spec in FEATURE_SPECS if spec.kind == "categorical"),
            tuple(CATEGORICAL_FEATURES),
        )

    def test_categorical_options_match_verified_dataset_values(self) -> None:
        expected = {
            "sex": (0, 1),
            "cp": (1, 2, 3, 4),
            "fbs": (0, 1),
            "restecg": (0, 1, 2),
            "exang": (0, 1),
            "slope": (1, 2, 3),
            "ca": (0, 1, 2, 3),
            "thal": (3, 6, 7),
        }
        actual = {
            spec.name: spec.options
            for spec in FEATURE_SPECS
            if spec.kind == "categorical"
        }
        self.assertEqual(actual, expected)

    def test_numeric_ranges_match_verified_dataset_values(self) -> None:
        expected = {
            "age": (29, 77),
            "trestbps": (94, 200),
            "chol": (126, 564),
            "thalach": (71, 202),
            "oldpeak": (0.0, 6.2),
        }
        actual = {
            spec.name: (spec.minimum, spec.maximum)
            for spec in FEATURE_SPECS
            if spec.kind == "numeric"
        }
        self.assertEqual(actual, expected)

    def test_default_patient_is_complete_and_valid(self) -> None:
        patient = default_patient()
        self.assertEqual(tuple(patient), tuple(FEATURE_COLUMNS))
        self.assertEqual(validate_patient_record(patient), patient)

    def test_validation_rejects_missing_extra_and_out_of_range_values(self) -> None:
        patient = default_patient()

        missing = dict(patient)
        missing.pop("age")
        with self.assertRaises(ValueError):
            validate_patient_record(missing)

        extra = dict(patient)
        extra["unexpected"] = 1
        with self.assertRaises(ValueError):
            validate_patient_record(extra)

        too_young = dict(patient)
        too_young["age"] = 28
        with self.assertRaises(ValueError):
            validate_patient_record(too_young)

        bad_code = dict(patient)
        bad_code["sex"] = 2
        with self.assertRaises(ValueError):
            validate_patient_record(bad_code)

        non_numeric = dict(patient)
        non_numeric["oldpeak"] = "high"
        with self.assertRaises(ValueError):
            validate_patient_record(non_numeric)


class PredictionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        frame = load_heart_data()
        cls.patient = (
            frame.dropna(subset=FEATURE_COLUMNS)
            .iloc[0][FEATURE_COLUMNS]
            .to_dict()
        )
        cls.results = predict_all(cls.patient)
        cls.metrics = load_evaluation_metrics()

    def test_predictions_from_all_seven_saved_models(self) -> None:
        self.assertEqual(
            [result["model_key"] for result in self.results],
            list(get_model_options()),
        )
        self.assertEqual(len(self.results), 7)

        for result in self.results:
            self.assertIn(result["prediction"], (0, 1))
            expected_label = (
                "heart disease"
                if result["prediction"] == 1
                else "no heart disease"
            )
            self.assertEqual(result["label"], expected_label)
            self.assertIsNotNone(result["score"])

    def test_svm_uses_decision_score_and_others_use_probabilities(self) -> None:
        by_key = {result["model_key"]: result for result in self.results}
        self.assertEqual(by_key["svm"]["score_type"], "decision_function")
        self.assertIn("decision score", format_score(
            by_key["svm"]["score"], by_key["svm"]["score_type"]
        ))
        self.assertIn("not a probability", format_score(
            by_key["svm"]["score"], by_key["svm"]["score_type"]
        ))

        for key, result in by_key.items():
            if key == "svm":
                continue
            self.assertEqual(result["score_type"], "probability", key)
            self.assertGreaterEqual(result["score"], 0.0, key)
            self.assertLessEqual(result["score"], 1.0, key)
            self.assertIn("probability", format_score(
                result["score"], result["score_type"]
            ))

    def test_predict_one_matches_batch_prediction(self) -> None:
        single = predict_one(self.patient, "svm")
        batch = next(r for r in self.results if r["model_key"] == "svm")
        self.assertEqual(single, batch)

    def test_format_score_handles_missing_scores(self) -> None:
        self.assertEqual(format_score(None, None), "Not available")
        self.assertIn("probability", format_score(0.75, "probability"))
        self.assertIn("decision score", format_score(-0.5, "decision_function"))
        self.assertEqual(format_score_value(None), "Not available")
        self.assertEqual(format_score_value(0.1234), "0.123")
        self.assertEqual(score_type_label("probability"), "probability")
        self.assertEqual(
            score_type_label("decision_function"),
            "decision score (not a probability)",
        )

    def test_metrics_match_registry_keys_and_columns(self) -> None:
        self.assertEqual(
            self.metrics["model_key"].tolist(),
            list(get_model_options()),
        )
        for column in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            self.assertIn(column, self.metrics.columns)
            self.assertTrue(
                self.metrics[column].between(0, 1, inclusive="both").all(),
                column,
            )

    def test_predictions_and_metrics_join_by_model_key(self) -> None:
        reversed_metrics = self.metrics.iloc[::-1].reset_index(drop=True)
        merged = compare_predictions(self.results, reversed_metrics)
        lookup = self.metrics.set_index("model_key")

        for row in merged.itertuples(index=False):
            self.assertAlmostEqual(
                float(row.accuracy),
                float(lookup.loc[row.model_key, "accuracy"]),
                places=12,
            )
            expected_name = next(
                result["model_name"]
                for result in self.results
                if result["model_key"] == row.model_key
            )
            self.assertEqual(row.model_name, expected_name)

    def test_compare_predictions_rejects_missing_or_mismatched_metrics(self) -> None:
        missing_row = self.metrics[
            self.metrics["model_key"] != "self_training_mlp"
        ]
        with self.assertRaises(ValueError):
            compare_predictions(self.results, missing_row)

        mismatched = self.metrics.copy()
        mismatched.loc[mismatched["model_key"] == "svm", "score_type"] = "probability"
        with self.assertRaises(ValueError):
            compare_predictions(self.results, mismatched)

    def test_missing_artifacts_raise_actionable_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError) as context:
                predict_one(self.patient, "svm", temp_dir)
            self.assertIn("train_models", str(context.exception))

            with self.assertRaises(FileNotFoundError) as context:
                load_evaluation_metrics(temp_dir)
            self.assertIn("train_models", str(context.exception))

    def test_malformed_registry_and_metrics_raise_clear_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = Path(temp_dir)
            models_dir = artifacts / "models"
            models_dir.mkdir()
            (models_dir / "registry.json").write_text(
                '{"models": []}', encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                get_model_options(artifacts)

            (artifacts / "comparison_results.csv").write_text(
                "model_key,accuracy\nlogistic_regression,0.9\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_evaluation_metrics(artifacts)

    def test_markdown_tables_include_all_models_and_metrics(self) -> None:
        predictions = predictions_to_markdown(self.results)
        metrics = metrics_to_markdown(self.metrics)
        for result in self.results:
            self.assertIn(result["model_name"], predictions)
            self.assertIn(result["model_name"], metrics)
        self.assertIn("Score type", predictions)
        self.assertIn("ROC-AUC", metrics)
        self.assertEqual(len(metrics.splitlines()), 9)  # header + separator + 7 rows

    def test_metrics_chart_has_one_bar_per_metric_and_model(self) -> None:
        figure = build_metrics_figure(self.metrics)
        try:
            self.assertEqual(len(figure.axes), 1)
            self.assertEqual(len(figure.axes[0].patches), 5 * 7)
        finally:
            plt.close(figure)


class StreamlitPageTests(unittest.TestCase):
    def test_predict_page_renders_and_predicts(self) -> None:
        at = AppTest.from_file(
            str(PREDICT_PAGE), default_timeout=180
        ).run()
        self.assertFalse(at.exception)
        self.assertEqual(at.title[0].value, "Heart Disease Prediction")
        self.assertEqual(len(at.number_input), 5)
        self.assertEqual(len(at.selectbox), 9)  # 8 features + model
        self.assertEqual(len(at.form_submit_button), 1)
        self.assertTrue(
            any(DISCLAIMER[:40] in caption.value for caption in at.caption)
        )

        at.form_submit_button[0].click().run()
        self.assertFalse(at.exception)
        displayed = {metric.label: metric.value for metric in at.metric}
        self.assertIn(displayed["Predicted class"], {"0", "1"})
        self.assertIn(
            displayed["Target label"], {"heart disease", "no heart disease"}
        )
        self.assertIn("Probability of class 1", displayed)

    def test_predict_page_labels_svm_score_as_decision_score(self) -> None:
        at = AppTest.from_file(
            str(PREDICT_PAGE), default_timeout=180
        ).run()
        at.selectbox[-1].set_value("svm")
        at.form_submit_button[0].click().run()
        self.assertFalse(at.exception)
        displayed = {metric.label: metric.value for metric in at.metric}
        self.assertIn("Decision score", displayed)
        self.assertNotIn("Probability of class 1", displayed)
        self.assertTrue(
            any(
                "not a probability" in caption.value
                for caption in at.caption
            )
        )

    def test_predict_page_reset_restores_defaults_and_clears_result(self) -> None:
        at = AppTest.from_file(
            str(PREDICT_PAGE), default_timeout=180
        ).run()
        at.number_input[0].set_value(70)
        at.form_submit_button[0].click().run()
        self.assertEqual(at.number_input[0].value, 70)
        self.assertTrue(at.metric)

        reset = next(button for button in at.button if button.label == "Reset inputs")
        reset.click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.number_input[0].value, 56)
        self.assertEqual(list(at.metric), [])

    def test_compare_page_renders_metrics_and_all_predictions(self) -> None:
        at = AppTest.from_file(
            str(COMPARE_PAGE), default_timeout=180
        ).run()
        self.assertFalse(at.exception)
        self.assertEqual(at.title[0].value, "Compare Models")
        self.assertEqual(len(at.number_input), 5)
        self.assertEqual(len(at.selectbox), 8)  # no model selector
        self.assertEqual(len(at.form_submit_button), 1)

        metrics = load_evaluation_metrics()
        metrics_text = "\n".join(element.value for element in at.markdown)
        for row in metrics.itertuples(index=False):
            line = next(
                line
                for line in metrics_text.splitlines()
                if line.startswith(f"| {row.model_name} |")
            )
            self.assertIn(f"{float(row.accuracy):.4f}", line)

        at.form_submit_button[0].click().run()
        self.assertFalse(at.exception)
        prediction_text = "\n".join(element.value for element in at.markdown)
        for result in predict_all(default_patient()):
            self.assertIn(result["model_name"], prediction_text)
        self.assertIn("Predicted class", prediction_text)
        self.assertIn("decision score", prediction_text.lower())
        self.assertTrue(
            any(
                "Informational majority vote" in caption.value
                for caption in at.caption
            )
        )

    def test_app_navigation_contains_exactly_two_pages(self) -> None:
        source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("st.Page("), 2)
        self.assertIn("pages/1_Predict.py", source)
        self.assertIn("pages/2_Compare_Models.py", source)
        self.assertNotIn("pages/3_", source)

        at = AppTest.from_file(str(APP_SCRIPT), default_timeout=180).run()
        self.assertFalse(at.exception)
        self.assertEqual(at.title[0].value, "Heart Disease Prediction")
        at.switch_page("pages/2_Compare_Models.py").run()
        self.assertFalse(at.exception)
        self.assertEqual(at.title[0].value, "Compare Models")


if __name__ == "__main__":
    unittest.main(verbosity=2)
