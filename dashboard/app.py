"""Streamlit dashboard for Glimpse (Real Data Edition)."""

import sys
from pathlib import Path

# --- اضافه کردن مسیر src به sys.path برای Streamlit Cloud ---
_src_dir = str(Path(__file__).resolve().parents[1] / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from glimpse.models.trainer import load_model
from glimpse.features.engineering import build_features
from glimpse.preprocessing.pipeline import preprocess
from glimpse.explain.shap_explainer import explain_sample
from glimpse.db import init_db, log_prediction, get_predictions

st.set_page_config(page_title="Glimpse", page_icon="🧬", layout="wide")

# مقداردهی اولیه دیتابیس
init_db()

st.title("🧬 Glimpse (Real-Data Engine)")
st.caption("Explainable early detection of rare metabolic diseases from LC-HR-MS spectra")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.joblib"

# بررسی وجود مدل
if not MODEL_PATH.exists():
    st.error("🚨 Model not found! Please ensure `model.joblib` is available in the `models/` directory.")
    st.info("If you are running this locally, run: `python -m glimpse.cli train --input data/real/real_dataset.csv`")
    st.stop()

# --- نمایش متریک‌ها ---
metrics_path = MODEL_DIR / "metrics.json"
col1, col2, col3 = st.columns(3)
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    col1.metric("AUC (CV)", f"{metrics.get('auc_mean', 0.0):.3f}")
    ci = metrics.get('auc_ci95', [0, 0])
    col2.metric("95% CI", f"[{ci[0]:.3f}, {ci[1]:.3f}]")
    col3.metric("Features", metrics.get("n_features", 0))
else:
    st.warning("Model metrics not found. Run training first.")

st.divider()

# --- ساخت تب‌ها ---
tab1, tab2 = st.tabs(["🔮 Predict Sample", "📜 Prediction History"])

try:
    model, features = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

with tab1:
    st.subheader("Enter patient sample data")
    
    input_method = st.radio("Choose input method:", ["📝 Manual Input (Paste Numbers)", "📁 Upload CSV File"])
    
    sample_data = None
    
    if input_method == "📝 Manual Input (Paste Numbers)":
        st.info(f"Please enter exactly {len(features)} comma-separated numbers.")
        text_input = st.text_area("Paste feature values here:", height=150)
        
        if st.button("Predict from Text"):
            try:
                values = [float(x.strip()) for x in text_input.split(",") if x.strip()]
                if len(values) != len(features):
                    st.error(f"Error: Expected {len(features)} features, but got {len(values)}.")
                else:
                    sample_data = np.array(values, dtype=float)
            except ValueError:
                st.error("Error: Please make sure you only enter numbers separated by commas.")
                
    elif input_method == "📁 Upload CSV File":
        st.info("Upload a CSV file with the same feature columns as the model.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                # اگر ستون‌های مورد نیاز موجود باشند
                missing_feats = [f for f in features if f not in df_upload.columns]
                if missing_feats:
                    st.error(f"Error: Missing {len(missing_feats)} required features. Example: {missing_feats[:5]}")
                else:
                    sample_data = df_upload[features].iloc[0].to_numpy(dtype=float)
                    st.success("File loaded successfully! Ready to predict.")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    # اگر داده آماده بود، پیش‌بینی کن
    if sample_data is not None:
        with st.spinner("Analyzing sample..."):
            result = explain_sample(model, features, sample_data)
            
            # ذخیره در دیتابیس
            log_prediction(result['risk_score'], result['risk_label'], result['top_contributions'])
            
            st.divider()
            st.subheader("Prediction Result")
            st.metric("Risk score", f"{result['risk_score']:.3f}")
            
            if result['risk_label'] == 'high':
                st.error(f"Label: **{result['risk_label'].upper()} RISK** 🚨")
            else:
                st.success(f"Label: **{result['risk_label'].upper()} RISK** ✅")
            
            st.subheader("Top contributing features (SHAP)")
            df_res = pd.DataFrame(result["top_contributions"])
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.barh(df_res["feature"], df_res["shap_value"], color='teal')
            ax.set_xlabel("SHAP value (impact on risk)")
            st.pyplot(fig)
            st.dataframe(df_res, use_container_width=True)

with tab2:
    st.subheader("Recent Predictions")
    history = get_predictions(limit=20)
    if history:
        df_history = pd.DataFrame(history)
        df_history = df_history[["timestamp", "risk_score", "risk_label", "top_feature"]]
        st.dataframe(df_history, use_container_width=True)
    else:
        st.info("No predictions recorded yet. Go to the first tab and make a prediction!")