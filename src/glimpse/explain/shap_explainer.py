"""Per-sample explanations using SHAP."""

from __future__ import annotations

import numpy as np
import shap


def explain_sample(model, features: list[str], x: np.ndarray, top_k: int = 5) -> dict:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x.reshape(1, -1))
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    values = shap_values[0]
    order = np.argsort(np.abs(values))[::-1][:top_k]
    contributions = [
        {
            "feature": features[i],
            "shap_value": round(float(values[i]), 4),
            "feature_value": round(float(x[i]), 4),
        }
        for i in order
    ]
    prob = float(model.predict_proba(x.reshape(1, -1))[0, 1])
    return {
        "risk_score": round(prob, 4),
        "risk_label": "high" if prob >= 0.5 else "low",
        "top_contributions": contributions,
    }