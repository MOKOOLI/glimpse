"""Glimpse — Streamlit dashboard for explainable IEM detection."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# اضافه کردن src به sys.path برای Streamlit Cloud
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from xgboost import XGBClassifier

from glimpse.db import init_db, log_prediction, get_history

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
XGB_JSON_PATH = MODELS_DIR / "model.json"
FEATURES_PATH = MODELS_DIR / "features.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

st.set_page_config(page_title="Glimpse", page_icon="🧬", layout="wide")


@st.cache_resource
def load_model():
    """بارگذاری مدل با سازگاری بین نسخه‌های XGBoost."""
    if not MODEL_PATH.exists():
        return None, None

    pipeline = joblib.load(MODEL_PATH)

    if XGB_JSON_PATH.exists():
        clf = XGBClassifier()
        clf.load_model(str(XGB_JSON_PATH))
        pipeline.named_steps["clf"] = clf

    features = json.loads(FEATURES_PATH.read_text()) if FEATURES_PATH.exists() else []
    return pipeline, features


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text())


def build_input_vector(
    user_values: dict[str, float], features: list[str]
) -> tuple[np.ndarray, list[str]]:
    """
    ساخت بردار ورودی از دیکشنری {نام_متابولیت: مقدار}.
    اگر نامی با ویژگی‌های مدل نخواند، آن را نادیده می‌گیرد.
    بقیه با 0 پر می‌شوند.
    """
    unknown = [k for k in user_values if k not in features]
    known = [k for k in user_values if k in features]

    if not known:
        raise ValueError(
            "هیچ‌کدام از نام‌های وارد شده با ویژگی‌های مدل مطابقت ندارد. "
            f"مثلاً یکی از این نام‌ها را امتحان کن: {features[:5]}"
        )

    vec = np.zeros(len(features), dtype=float)
    idx = {name: i for i, name in enumerate(features)}
    for k in known:
        vec[idx[k]] = user_values[k]
    return vec.reshape(1, -1), unknown


def main():
    init_db()

    st.title("🧬 Glimpse")
    st.caption("Explainable early detection of rare metabolic diseases from LC-HR-MS spectra")

    model, features = load_model()
    metrics = load_metrics()

    if model is None or not features:
        st.error("مدل یافت نشد. ابتدا `python -m glimpse.cli train` را اجرا کنید.")
        st.stop()

    # کارت‌های بالای صفحه
    col1, col2, col3 = st.columns(3)
    col1.metric("AUC (CV)", f"{metrics.get('auc_mean', 0):.3f}")
    col2.metric(
        "95% CI",
        f"[{metrics.get('auc_ci95', [0, 0])[0]:.3f}, {metrics.get('auc_ci95', [0, 0])[1]:.3f}]",
    )
    col3.metric("Features", metrics.get("n_features", len(features)))

    st.divider()

    tab1, tab2 = st.tabs(["🔬 Predict Sample", "📜 Prediction History"])

    with tab1:
        st.subheader("ورود داده نمونه")
        st.caption(
            "می‌توانی نام متابولیت و مقدار آن را وارد کنی. "
            "مثلاً: `leucine=3.5, isoleucine=2.1`"
        )

        text = st.text_area(
            "مقادیر به شکل name=value (با کاما جدا شوند):",
            height=120,
            placeholder="leucine=3.5, isoleucine=2.1, valine=1.8",
        )

        user_values = {}
        if text.strip():
            try:
                for pair in text.replace("\n", ",").split(","):
                    pair = pair.strip()
                    if not pair:
                        continue
                    if "=" not in pair:
                        raise ValueError(f"جفت نامعتبر: '{pair}'. باید name=value باشد.")
                    name, val = pair.split("=", 1)
                    user_values[name.strip()] = float(val.strip())
            except ValueError as e:
                st.error(str(e))
                user_values = {}

        if user_values:
            try:
                input_vector, unknown = build_input_vector(user_values, features)
                if unknown:
                    st.warning(
                        f"این نام‌ها با مدل مطابقت نداشتند و نادیده گرفته شدند: {unknown[:5]}"
                    )

                st.success(f"✅ {len(user_values) - len(unknown)} مقدار معتبر دریافت شد.")

                if st.button("پیش‌بینی", type="primary"):
                    proba = model.predict_proba(input_vector)[:, 1][0]
                    st.metric("احتمال بیماری", f"{proba:.3f}")
                    log_prediction(risk_score=float(proba))

                    # SHAP برای توضیح
                    st.subheader("🔍 چرا مدل این‌طور تصمیم گرفت؟")
                    with st.spinner("محاسبه SHAP..."):
                        # XGBoost داخل Pipeline آموزش دیده، ولی SHAP را روی
                        # imputer+scaler+clf می‌زنیم تا مقادیر نهایی را ببینیم.
                        X_scaled = model[:-1].transform(input_vector)
                        explainer = shap.TreeExplainer(model.named_steps["clf"])
                        shap_vals = explainer.shap_values(X_scaled)[0]

                    # مهم‌ترین ویژگی‌ها بر اساس |SHAP|
                    order = np.argsort(np.abs(shap_vals))[::-1][:10]
                    rows = []
                    for i in order:
                        rows.append(
                            {
                                "متابولیت": features[i],
                                "مقدار ورودی": float(input_vector[0, i]),
                                "SHAP (تأثیر)": float(shap_vals[i]),
                            }
                        )
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

            except ValueError as e:
                st.error(str(e))

    with tab2:
        st.subheader("تاریخچه پیش‌بینی‌ها")
        history = get_history()
        if history:
            st.dataframe(pd.DataFrame(history))
        else:
            st.info("هنوز پیش‌بینی‌ای ثبت نشده است.")


if __name__ == "__main__":
    main()