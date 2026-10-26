"""Generate a realistic dataset mimicking LC-MS data for training."""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

n_samples = 200
n_features = 120
mz_names = [f"mz_{i:04d}" for i in range(n_features)]

# ساخت داده پایه با توزیع گاما (شبیه به شدت پیک‌های LC-MS)
X = np.random.gamma(shape=2.0, scale=1.0, size=(n_samples, n_features))

# تزریق سیگنال قوی‌تر به ۴ پیک مشخص (شبیه به بیماری متابولیک)
signal_bins = [3, 17, 42, 88]
y = np.random.binomial(1, 0.3, size=n_samples)  # ۳۰٪ بیمار، ۷۰٪ سالم

for i, label in enumerate(y):
    if label == 1:
        for b in signal_bins:
            # سیگنال بسیار قوی‌تر (۵ تا ۸ برابر) برای یادگیری بهتر
            X[i, b] *= np.random.uniform(5.0, 8.0)

# اضافه کردن نویز واقعی (مقادیر گمشده و خطای اندازه‌گیری)
mask = np.random.rand(n_samples, n_features) < 0.05  # ۵٪ مقادیر گمشده
X[mask] = np.nan

# ساخت DataFrame
df = pd.DataFrame(X, columns=mz_names)
df["label"] = y
df["patient_id"] = np.arange(n_samples) // 2  # هر بیمار ۲ نمونه

# ذخیره
out_path = Path("data/real/real_dataset.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"✅ Generated realistic dataset → {out_path}")
print(f"   Disease prevalence: {y.mean():.1%}")