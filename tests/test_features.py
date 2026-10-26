import pandas as pd

from glimpse.features.engineering import add_summary_stats, build_features


def test_summary_stats_are_added():
    df = pd.DataFrame({"mz_0001": [1.0, 2.0], "mz_0002": [3.0, 4.0]})
    out = add_summary_stats(df)
    for col in ["stat_mean", "stat_std", "stat_max", "stat_nonzero"]:
        assert col in out.columns


def test_build_features_returns_dataframe():
    df = pd.DataFrame({"mz_0001": [1.0], "mz_0002": [2.0]})
    out = build_features(df)
    assert isinstance(out, pd.DataFrame)