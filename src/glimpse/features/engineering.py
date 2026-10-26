"""Feature engineering for metabolic spectra."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_ratios(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    out = df.copy()
    for a, b in pairs:
        if a in out.columns and b in out.columns:
            out[f"ratio_{a}_{b}"] = np.log1p(out[a]) - np.log1p(out[b])
    return out


def add_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c.startswith("mz_")]
    X = df[feature_cols]
    out = df.copy()
    out["stat_mean"] = X.mean(axis=1)
    out["stat_std"] = X.std(axis=1)
    out["stat_max"] = X.max(axis=1)
    out["stat_nonzero"] = (X > 0).sum(axis=1)
    return out


def build_features(df: pd.DataFrame, pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    out = add_summary_stats(df)
    if pairs:
        out = add_ratios(out, pairs)
    return out