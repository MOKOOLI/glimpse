"""Command-line interface for Glimpse."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from glimpse.features.engineering import build_features
from glimpse.models.trainer import load_model, train
from glimpse.preprocessing.pipeline import preprocess


def _cmd_generate(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    n = args.n
    n_features = args.n_features
    mz_names = [f"mz_{i:04d}" for i in range(n_features)]

    y = rng.integers(0, 2, size=n)
    X = rng.gamma(shape=2.0, scale=1.0, size=(n, n_features))
    signal_bins = [3, 17, 42, 88]
    for i, label in enumerate(y):
        if label == 1:
            for b in signal_bins:
                X[i, b] *= rng.uniform(2.5, 4.0)

    df = pd.DataFrame(X, columns=mz_names)
    df["label"] = y
    df["patient_id"] = np.arange(n) // 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"✅ Generated {n} samples → {out}")


def _prepare_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = preprocess(df)
    df = build_features(df)
    return df


def _cmd_train(args: argparse.Namespace) -> None:
    df = _prepare_dataframe(Path(args.input))
    metrics = train(df)
    print("📊 Training complete")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


def _cmd_predict(args: argparse.Namespace) -> None:
    df = _prepare_dataframe(Path(args.input))
    model, features = load_model()
    X = df[features].to_numpy(dtype=float)
    preds = model.predict_proba(X)[:, 1]
    out = df[["patient_id"]].copy() if "patient_id" in df.columns else pd.DataFrame()
    out["risk_score"] = np.round(preds, 4)
    out.to_csv(args.out, index=False)
    print(f"✅ Predictions written → {args.out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="glimpse", description="Glimpse CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="Generate synthetic demo data")
    p_gen.add_argument("--n", type=int, default=500)
    p_gen.add_argument("--n-features", type=int, default=120)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--out", type=str, default="data/synthetic/demo.csv")
    p_gen.set_defaults(func=_cmd_generate)

    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--input", type=str, default="data/synthetic/demo.csv")
    p_train.set_defaults(func=_cmd_train)

    p_pred = sub.add_parser("predict", help="Run batch predictions")
    p_pred.add_argument("--input", type=str, required=True)
    p_pred.add_argument("--out", type=str, default="data/synthetic/predictions.csv")
    p_pred.set_defaults(func=_cmd_predict)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())