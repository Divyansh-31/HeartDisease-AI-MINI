"""Streamlit entry point with exactly two pages.

The two pages are declared explicitly so the sidebar contains no separate
home/landing page. Model training and evaluation are not performed here.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide",
)

navigation = st.navigation(
    [
        st.Page(
            "pages/1_Predict.py",
            title="Predict",
            default=True,
        ),
        st.Page(
            "pages/2_Compare_Models.py",
            title="Compare Models",
        ),
    ]
)
navigation.run()
