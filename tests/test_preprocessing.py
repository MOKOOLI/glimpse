import numpy as np
import pandas as pd

from glimpse.preprocessing.pipeline import baseline_correction, preprocess, total_ion_normalize


def test_baseline_correction_removes_low_quantile():
    X = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    out = baseline_correction(X, quantile=0.0)
    assert np.all(out >= 0)


def test_total_ion_normalize_sums_to_one():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = total_ion_normalize(X)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_preprocess_preserves_metadata():
    df = pd.DataFrame({
        "mz_0001": [1.0, 2.0],
        "mz_0002": [3.0, 4.0],
        "label": [0, 1],
    })
    out = preprocess(df)
    assert "label" in out.columns
    assert list(out["label"]) == [0, 1]