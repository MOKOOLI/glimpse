"""Model training pipeline for Glimpse."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "model.joblib"
FEATURES_PATH = MODELS_DIR / "features.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

LABEL_COL = "label"
AUX_COLS = ("patient_id", "sample_id", "filename")


def _build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _clean_features(df) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, feature_names) with all non-numeric/aux/empty columns removed."""
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing '{LABEL_COL}' column")

    y = df[LABEL_COL].astype(int).values

    # فقط ستون‌های عددی
    X = df.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number])

    # حذف ستون‌های کمکی (patient_id و مانند آن)
    drop_cols = [c for c in X.columns if c in AUX_COLS]
    if drop_cols:
        X = X.drop(columns=drop_cols, errors="ignore")

    # حذف ستون‌های کاملاً خالی (همه NaN)
    X = X.loc[:, X.notna().any(axis=0)]

    feature_names = X.columns.tolist()
    return X.values, y, feature_names


def train(df, n_splits: int = 5) -> dict:
    """Train model on dataframe with 'label' column. Returns metrics dict."""
    X, y, feature_names = _clean_features(df)

    n_splits = min(n_splits, int(np.bincount(y).min()))
    if n_splits < 2:
        raise ValueError("Not enough samples per class for cross-validation")

    pipeline = _build_pipeline()

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    # fit نهایی روی کل داده برای ذخیره مدل
    pipeline.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(feature_names, indent=2))

    metrics = {
        "auc_mean": float(np.mean(scores)),
        "auc_std": float(np.std(scores)),
        "auc_ci95": [
            float(np.percentile(scores, 2.5)),
            float(np.percentile(scores, 97.5)),
        ],
        "n_samples": int(len(df)),
        "n_features": int(len(feature_names)),
        "n_features_selected": int(len(feature_names)),
        "cv_scores": [float(s) for s in scores],
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


def load_model(path: str | Path = MODEL_PATH):
    """Load a trained pipeline from disk."""
    return joblib.load(path)