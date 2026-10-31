"""Compare the seven saved models on one patient record and show metrics."""

from __future__ import annotations

import streamlit as st

from src.prediction_ui import (
    apply_page_style,
    build_metrics_figure,
    compare_predictions,
    get_model_options,
    load_evaluation_metrics,
    metrics_to_markdown,
    predict_all,
    predictions_to_markdown,
    render_disclaimer,
    render_patient_inputs,
    reset_patient_inputs,
)

RESULT_KEY = "compare_page_results"


def _reset_page() -> None:
    reset_patient_inputs()
    st.session_state.pop(RESULT_KEY, None)


def _majority_summary(results: list[dict]) -> str:
    positive = sum(int(result["prediction"]) for result in results)
    total = len(results)
    negative = total - positive
    if positive > negative:
        verdict = f"{positive} of {total} models predict class 1 (heart disease)."
    elif negative > positive:
        verdict = f"{negative} of {total} models predict class 0 (no heart disease)."
    else:
        verdict = f"The models are tied: {positive} predict class 1 and {negative} predict class 0."
    return (
        "Informational majority vote (not a separately trained model): "
        f"{verdict} This is a simple count of the seven outputs, not a new "
        "prediction model, and it is not weighted by test-set performance."
    )


apply_page_style()

st.title("Compare Models")
st.write(
    "Enter one patient record once and run all seven saved pipelines on it. "
    "The page also displays the existing held-out evaluation results; it never "
    "retrains models or recomputes the official metrics."
)
render_disclaimer()

try:
    model_options = get_model_options()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"The saved models are not available: {exc}")
    st.code("python -m src.train_models", language="powershell")
    st.stop()

metrics = None
metrics_error: str | None = None
try:
    metrics = load_evaluation_metrics()
except (FileNotFoundError, ValueError, OSError) as exc:
    metrics_error = str(exc)

with st.form("comparison_form"):
    patient = render_patient_inputs()
    submitted = st.form_submit_button("Predict with all seven models", type="primary")

st.button(
    "Reset inputs",
    on_click=_reset_page,
    help="Restore the illustrative defaults and clear the last comparison.",
)

if submitted:
    try:
        results = predict_all(patient)
    except (FileNotFoundError, ValueError, TypeError, OSError, RuntimeError) as exc:
        st.session_state.pop(RESULT_KEY, None)
        st.error(f"Prediction failed: {exc}")
    else:
        st.session_state[RESULT_KEY] = results

results = st.session_state.get(RESULT_KEY)
if results is not None:
    st.subheader("Predictions for this input")
    if len(results) != len(model_options):
        st.error(
            "The comparison did not produce one result per registered model. "
            "Re-run `python -m src.train_models` if this persists."
        )

    if metrics is not None:
        merged = compare_predictions(results, metrics).to_dict("records")
    else:
        merged = results

    st.markdown(predictions_to_markdown(merged))
    st.caption(
        "Score type is shown for every model. Probability outputs and the SVM "
        "decision score are on different scales and must not be ranked or "
        "compared as if both were probabilities."
    )
    st.caption(_majority_summary(results))

st.divider()
st.subheader("Saved held-out evaluation metrics")
if metrics is None:
    st.error(f"Evaluation metrics are unavailable: {metrics_error}")
    st.code("python -m src.train_models", language="powershell")
else:
    st.caption(
        "Read from `artifacts/comparison_results.csv`. These are the official "
        "metrics on the same untouched 61-record test set saved during "
        "training; they are not results for the patient entered above."
    )
    st.markdown(metrics_to_markdown(metrics))

    figure = build_metrics_figure(metrics)
    st.pyplot(figure)
    import matplotlib.pyplot as plt

    plt.close(figure)
    st.caption(
        "The chart visualises the saved metrics only. It is not recomputed on "
        "page load."
    )
