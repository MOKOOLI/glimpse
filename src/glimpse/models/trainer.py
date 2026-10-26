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
    """Return all numeric columns except label and patient_id."""
    exclude = {"label", "patient_id", "filename"}
    # انتخاب تمام ستون‌هایی که عددی هستند و در لیست استثناها نیستند
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def train(
    df: pd.DataFrame,
    label_col: str = "label",
    group_col: str = "patient_id",
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    
    # جایگزینی مقادیر بی‌نهایت با NaN و سپس پر کردن NaN با صفر
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    feats = _feature_columns(df)
    
    if len(feats) == 0:
        raise ValueError("No numeric feature columns found in the dataset!")

    X = df[feats].to_numpy(dtype=float)
    y = df[label_col].to_numpy(dtype=int)
    
    # اگر ستون patient_id وجود نداشت، هر سطر را یک بیمار در نظر بگیر
    if group_col in df.columns:
        groups = df[group_col].to_numpy()
    else:
        groups = np.arange(len(y))

    # تعیین تعداد Foldها بر اساس کمترین تعداد نمونه در کلاس‌ها
    unique, counts = np.unique(y, return_counts=True)
    min_class_count = counts.min()
    actual_splits = min(n_splits, min_class_count)
    
    if actual_splits < 2:
        raise ValueError(f"Not enough samples in the minority class ({min_class_count}) to perform cross-validation.")

    cv = StratifiedGroupKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    
    aucs, oof_preds = [], np.zeros(len(y))

    for train_idx, val_idx in cv.split(X, y, groups):
        # اگر در یک Fold خاص فقط یک کلاس وجود داشت، از آن صرف‌نظر می‌کنیم
        if len(np.unique(y[train_idx])) < 2:
            continue

        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=random_state, n_jobs=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict_proba(X[val_idx])[:, 1]
        oof_preds[val_idx] = preds
        
        # محاسبه AUC فقط اگر در داده اعتبارسنجی هر دو کلاس وجود داشته باشد
        if len(np.unique(y[val_idx])) > 1:
            aucs.append(roc_auc_score(y[val_idx], preds))

    if not aucs:
        auc_mean = 0.5
        auc_std = 0.0
    else:
        auc_mean = float(np.mean(aucs))
        auc_std = float(np.std(aucs))
        
    ci_low = max(0.0, auc_mean - 1.96 * auc_std / np.sqrt(max(1, len(aucs))))
    ci_high = min(1.0, auc_mean + 1.96 * auc_std / np.sqrt(max(1, len(aucs))))

    # آموزش مدل نهایی روی تمام داده‌ها
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
        "n_samples": int(len(y)),
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