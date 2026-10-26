"""Signal preprocessing for LC-HR-MS spectra."""

from __future__ import annotations

import numpy as np
import pandas as pd

def baseline_correction(X: np.ndarray, quantile: float = 0.05) -> np.ndarray:
    baseline = np.nanquantile(X, quantile, axis=1, keepdims=True)
    return np.clip(X - baseline, a_min=0, a_max=None)

def total_ion_normalize(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    row_sums = np.nansum(X, axis=1, keepdims=True) + eps
    return X / row_sums

def log_transform(X: np.ndarray, offset: float = 1.0) -> np.ndarray:
    return np.log1p(X + offset)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c.startswith("mz_")]
    X = df[feature_cols].to_numpy(dtype=float)

    # مدیریت مقادیر گمشده (NaN) با جایگزینی با صفر
    X = np.nan_to_num(X, nan=0.0)

    X = baseline_correction(X)
    X = total_ion_normalize(X)
    X = log_transform(X)

    out = df.copy()
    out[feature_cols] = X
    return out