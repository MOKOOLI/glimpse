"""Model training with honest, patient-level evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("mz_", "ratio_", "stat_"))]


def train(
    df: pd.DataFrame,
    label_col: str = "label",
    group_col: str = "patient_id",
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    feats = _feature_columns(df)
    X = df[feats].to_numpy(dtype=float)
    y = df[label_col].to_numpy(dtype=int)
    groups = df[group_col].to_numpy()

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs, oof_preds = [], np.zeros(len(y))

    for train_idx, val_idx in cv.split(X, y, groups):
        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=random_state, n_jobs=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict_proba(X[val_idx])[:, 1]
        oof_preds[val_idx] = preds
        aucs.append(roc_auc_score(y[val_idx], preds))

    auc_mean = float(np.mean(aucs))
    auc_std = float(np.std(aucs))
    ci_low = auc_mean - 1.96 * auc_std / np.sqrt(n_splits)
    ci_high = auc_mean + 1.96 * auc_std / np.sqrt(n_splits)

    final_model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=random_state, n_jobs=-1,
    )
    final_model.fit(X, y)

    joblib.dump(final_model, MODEL_DIR / "model.joblib")
    with open(MODEL_DIR / "features.json", "w") as f:
        json.dump(feats, f)

    metrics = {
        "auc_mean": round(auc_mean, 4),
        "auc_std": round(auc_std, 4),
        "auc_ci95": [round(ci_low, 4), round(ci_high, 4)],
        "n_samples": len(y),
        "n_features": len(feats),
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def load_model():
    model = joblib.load(MODEL_DIR / "model.joblib")
    with open(MODEL_DIR / "features.json") as f:
        feats = json.load(f)
    return model, feats