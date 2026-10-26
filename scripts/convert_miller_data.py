import pandas as pd
import numpy as np
from pathlib import Path

patient_path = Path("data/real/jimd1029-sup-0001.xls")
control_path = Path("data/real/MSdata_ST003794_AN006236.txt")
out_path = Path("data/real/real_dataset.csv")

print("=" * 60)
print("READING MILLER PATIENT DATA")
print("=" * 60)

df = pd.read_excel(patient_path, sheet_name=0, header=None)
print("Raw shape:", df.shape)

# ردیف 0 = Diagnosis per column
# ردیف 1 = sample IDs (IEM_xxxx)
diag_row = df.iloc[0].tolist()
sample_row = df.iloc[1].tolist()

# پیدا کردن ستون‌هایی که در ردیف 1 اسم IEM_ دارن
iem_idx = [i for i, c in enumerate(sample_row) if str(c).startswith("IEM_")]
print(f"Found {len(iem_idx)} IEM columns")
print("First 3 IEM:", [sample_row[i] for i in iem_idx[:3]])
print("First 3 Diagnoses:", [diag_row[i] for i in iem_idx[:3]])

if len(iem_idx) == 0:
    print("ERROR: No IEM columns found in row 1!")
    print("Row 1 first 20:", sample_row[:20])
    raise SystemExit(1)

iem_names = [str(sample_row[i]) for i in iem_idx]
diagnoses = [str(diag_row[i]).strip() for i in iem_idx]

# متابولیت‌ها از ردیف 2 به بعد، ستون 0
met_names = df.iloc[2:, 0].astype(str).tolist()
print(f"Metabolite count: {len(met_names)}")
print("First 3 metabolites:", met_names[:3])

# داده = ردیف 2 به بعد، ستون‌های IEM
data_block = df.iloc[2:, iem_idx].apply(pd.to_numeric, errors="coerce")
data_block.index = met_names
data_block.columns = iem_names

# ترنسپوز: ردیف = نمونه
pat_samples = data_block.T
print("Patient matrix shape:", pat_samples.shape)

# ========== تعیین label از روی Diagnosis ==========
# MSUD = 1
# No biochemical genetic diagnosis = 0
# بقیه = حذف (NaN)
labels = []
kept_diags = []
for d in diagnoses:
    d_clean = d.strip().lower()
    if "maple syrup urine disease" in d_clean:
        labels.append(1)
        kept_diags.append(d)
    elif d_clean.startswith("no biochemical genetic diagnosis"):
        labels.append(0)
        kept_diags.append(d)
    else:
        labels.append(np.nan)
        kept_diags.append(d)

pat_samples["label"] = labels

# حذف نمونه‌هایی که label ندارن
before = len(pat_samples)
pat_samples = pat_samples.dropna(subset=["label"])
print(f"Kept {len(pat_samples)} of {before} patient samples")

# توزیع label
label_counts = pat_samples["label"].value_counts().to_dict()
print("Patient label distribution:", label_counts)
print(f"  MSUD (1): {label_counts.get(1.0, 0)}")
print(f"  No-diagnosis (0): {label_counts.get(0.0, 0)}")

# ========== خواندن کنترل‌ها (اختیاری - برای augmentation) ==========
print("=" * 60)
print("READING CONTROL DATA (for potential augmentation)")
print("=" * 60)

try:
    ctrl_raw = pd.read_csv(control_path, sep="\t")
    print("Raw control shape:", ctrl_raw.shape)

    sample_names = ctrl_raw.columns[1:].tolist()
    ctrl_raw = ctrl_raw.rename(columns={ctrl_raw.columns[0]: "metabolite"})
    factors_row = ctrl_raw.iloc[0]
    ctrl_data = ctrl_raw.iloc[1:].copy()
    ctrl_data = ctrl_data.set_index("metabolite")
    ctrl_data = ctrl_data.apply(pd.to_numeric, errors="coerce")
    ctrl_samples = ctrl_data.T
    print("Control matrix shape:", ctrl_samples.shape)
except Exception as e:
    print(f"WARNING: Could not read controls: {e}")
    ctrl_samples = None

# ========== ذخیره داده Miller با label واقعی ==========
print("=" * 60)
print("SAVING")
print("=" * 60)

# فقط ستون‌های متابولیت + label
meta_cols = ["label"]
feature_cols = [c for c in pat_samples.columns if c not in meta_cols]
final = pat_samples[feature_cols + meta_cols].copy()
final = final.fillna(0)

print("Final shape:", final.shape)
print("Label distribution:", final["label"].value_counts().to_dict())
print("Columns:", final.columns.tolist()[:5], "...", final.columns.tolist()[-3:])

final.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
print("\n🎯 Ready for training:")
print(f"  python -m glimpse.cli train --input {out_path}")