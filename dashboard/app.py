"""Streamlit dashboard for Glimpse."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from glimpse.models.trainer import load_model
from glimpse.explain.shap_explainer import explain_sample

st.set_page_config(page_title="Glimpse", page_icon="🧬", layout="wide")

st.title("🧬 Glimpse")
st.caption("Explainable early detection of rare metabolic diseases from LC-HR-MS spectra")

MODEL_DIR = Path("models")

col1, col2, col3 = st.columns(3)

metrics_path = MODEL_DIR / "metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    col1.metric("AUC (CV)", f"{metrics['auc_mean']:.3f}")
    col2.metric("95% CI", f"[{metrics['auc_ci95'][0]:.3f}, {metrics['auc_ci95'][1]:.3f}]")
    col3.metric("Features", metrics["n_features"])
else:
    st.warning("No trained model found. Run `python -m glimpse.cli train` first.")
    st.stop()

st.divider()

st.subheader("Single sample prediction")

model, features = load_model()

if "sample" not in st.session_state:
    st.session_state.sample = np.zeros(len(features))

with st.form("sample_form"):
    st.write("Enter feature values (or leave zeros for a quick demo):")
    cols = st.columns(4)
    values = []
    for i, f in enumerate(features):
        with cols[i % 4]:
            values.append(st.number_input(f, value=0.0, key=f"f_{i}"))
    submitted = st.form_submit_button("Predict")

if submitted:
    x = np.array(values, dtype=float)
    result = explain_sample(model, features, x)
    st.metric("Risk score", f"{result['risk_score']:.3f}")
    st.write(f"Label: **{result['risk_label']}**")
    st.subheader("Top contributing features")
    df = pd.DataFrame(result["top_contributions"])
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(df["feature"], df["shap_value"])
    ax.set_xlabel("SHAP value (impact on risk)")
    st.pyplot(fig)
    st.dataframe(df, use_container_width=True)