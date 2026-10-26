"""Command-line interface for Glimpse."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from glimpse.preprocessing.pipeline import preprocess
from glimpse.features.engineering import build_features
from glimpse.models.trainer import train, load_model
from glimpse.preprocessing.extractor import build_dataset_from_directory

def _cmd_generate(args: argparse.Namespace) -> None:
    """Generate synthetic demo data (kept for quick testing)."""
    rng = np.random.default_rng(args.seed)
    n, n_features = args.n, args.n_features
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
    print(f"✅ Generated {n} synthetic samples → {out}")

def _cmd_extract(args: argparse.Namespace) -> None:
    """Extract features from raw mzML files into a CSV dataset."""
    data_dir = Path(args.data_dir)
    labels_file = Path(args.labels_file)
    
    if not data_dir.exists() or not labels_file.exists():
        print(f"Error: Ensure {data_dir} and {labels_file} exist.")
        return

    # خواندن فایل لیبل‌ها (فرض بر این است که یک فایل CSV با ستون‌های filename و label است)
    labels_df = pd.read_csv(labels_file)
    labels_dict = dict(zip(labels_df['filename'], labels_df['label']))
    
    print(f"Found {len(labels_dict)} labels. Starting extraction...")
    df = build_dataset_from_directory(data_dir, labels_dict)
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Extracted real data from mzML files → {out_path}")

def _prepare_dataframe(path: Path) -> pd.DataFrame:
    """Load and preprocess a dataset."""
    df = pd.read_csv(path)
    
    # اطمینان از وجود ستون label
    if "label" not in df.columns:
        raise ValueError(f"Dataset {path} must contain a 'label' column (0 or 1).")
        
    # اگر ستون patient_id وجود نداشت، یک ستون موقت بساز (برای GroupKFold)
    if "patient_id" not in df.columns:
        df["patient_id"] = df.index

    df = preprocess(df)
    df = build_features(df)
    return df

def _cmd_train(args: argparse.Namespace) -> None:
    """Train the model on a real dataset."""
    df = _prepare_dataframe(Path(args.input))
    metrics = train(df)
    print(f"📊 Training complete on {args.input}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

def _cmd_predict(args: argparse.Namespace) -> None:
    """Run batch predictions on a dataset."""
    df = _prepare_dataframe(Path(args.input))
    model, features = load_model()
    X = df[features].to_numpy(dtype=float)
    preds = model.predict_proba(X)[:, 1]
    
    out = df[["patient_id"]].copy() if "patient_id" in df.columns else pd.DataFrame()
    out["risk_score"] = np.round(preds, 4)
    
    # اضافه کردن لیبل واقعی اگر وجود داشته باشد (برای مقایسه)
    if "label" in df.columns:
        out["true_label"] = df["label"]
        
    out.to_csv(args.out, index=False)
    print(f"✅ Predictions written → {args.out}")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="glimpse", description="Glimpse CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # دستور تولید داده شبیه‌سازی‌شده (برای تست‌های سریع)
    p_gen = sub.add_parser("generate", help="Generate synthetic demo data")
    p_gen.add_argument("--n", type=int, default=500)
    p_gen.add_argument("--n-features", type=int, default=120)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--out", type=str, default="data/synthetic/demo.csv")
    p_gen.set_defaults(func=_cmd_generate)

    # دستور استخراج ویژگی از داده‌های خام
    p_ext = sub.add_parser("extract", help="Extract features from raw mzML files")
    p_ext.add_argument("--data-dir", type=str, required=True, help="Directory containing .mzML files")
    p_ext.add_argument("--labels-file", type=str, required=True, help="CSV file mapping filenames to labels")
    p_ext.add_argument("--out", type=str, default="data/real/real_dataset.csv")
    p_ext.set_defaults(func=_cmd_extract)

    # دستور آموزش مدل روی داده واقعی
    p_train = sub.add_parser("train", help="Train the model on a dataset")
    p_train.add_argument("--input", type=str, required=True, help="Path to the processed CSV dataset")
    p_train.set_defaults(func=_cmd_train)

    # دستور پیش‌بینی
    p_pred = sub.add_parser("predict", help="Run batch predictions")
    p_pred.add_argument("--input", type=str, required=True)
    p_pred.add_argument("--out", type=str, default="data/synthetic/predictions.csv")
    p_pred.set_defaults(func=_cmd_predict)

    args = parser.parse_args(argv)
    args.func(args)
    return 0

if __name__ == "__main__":
    sys.exit(main())