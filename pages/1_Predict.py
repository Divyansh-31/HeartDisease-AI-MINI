"""Individual heart-disease prediction with one saved model."""

from __future__ import annotations

import streamlit as st

from src.prediction_ui import (
    apply_page_style,
    get_model_options,
    predict_one,
    render_disclaimer,
    render_patient_inputs,
    reset_patient_inputs,
)

RESULT_KEY = "predict_page_result"


def _reset_page() -> None:
    reset_patient_inputs()
    st.session_state.pop(RESULT_KEY, None)


apply_page_style()

st.title("Heart Disease Prediction")
st.write(
    "Enter the 13 clinical features used during training, choose one of the "
    "seven saved models, and generate a single prediction. Every value is "
    "used exactly as the fitted preprocessing pipeline expects it."
)
render_disclaimer()

try:
    model_options = get_model_options()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"The saved models are not available: {exc}")
    st.code("python -m src.train_models", language="powershell")
    st.stop()

with st.form("single_prediction_form"):
    patient = render_patient_inputs()
    selected_key = st.selectbox(
        "Model",
        options=list(model_options),
        format_func=lambda key: model_options[key],
        help="Only the seven pipelines saved under artifacts/models are listed.",
    )
    submitted = st.form_submit_button("Predict", type="primary")

st.button(
    "Reset inputs",
    on_click=_reset_page,
    help="Restore the illustrative defaults and clear the last result.",
)

if submitted:
    try:
        result = predict_one(patient, selected_key)
    except (FileNotFoundError, ValueError, TypeError, OSError, RuntimeError) as exc:
        st.session_state.pop(RESULT_KEY, None)
        st.error(f"Prediction failed: {exc}")
    else:
        st.session_state[RESULT_KEY] = result

result = st.session_state.get(RESULT_KEY)
if result is not None:
    st.subheader("Prediction result")
    with st.container(border=True):
        st.markdown(f"**Model:** {result['model_name']}")
        left, middle, right = st.columns(3)
        left.metric("Predicted class", str(result["prediction"]))
        middle.metric("Target label", result["label"])
        if result["score"] is None:
            right.metric("Continuous score", "Not available")
        elif result["score_type"] == "probability":
            right.metric("Probability of class 1", f"{result['score']:.3f}")
        else:
            right.metric("Decision score", f"{result['score']:+.3f}")

        if result["score_type"] == "probability":
            st.progress(
                min(max(float(result["score"]), 0.0), 1.0),
                text="Probability that the model assigns to class 1",
            )
            st.caption(
                "This number is the saved model's probability output for class 1. "
                "It is not a calibrated clinical risk score."
            )
        elif result["score_type"] == "decision_function":
            st.caption(
                "The SVM returns a decision-function score, not a probability. "
                "It must not be read as a confidence percentage."
            )
        else:
            st.caption("This model does not expose a continuous score.")

        st.caption(
            "Target mapping: class 1 = heart disease (raw num 1-4), "
            "class 0 = no heart disease (raw num 0)."
        )

st.divider()
st.caption(
    "The form is illustrative. All inputs are user-entered and are not stored "
    "by this page."
)
