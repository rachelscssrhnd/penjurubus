import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from penjurubus.data_paths import resolve_path

ROOT = r"D:\penjurubus"
PROC_DIR = os.path.join(ROOT, "data", "processed")
SPLIT_DIR = os.path.join(ROOT, "data", "split")
OUT_DIR = os.path.join(ROOT, "data", "audit")
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "train": os.path.join(PROC_DIR, "train_processed.parquet"),
    "val": os.path.join(PROC_DIR, "val_processed.parquet"),
    "test": os.path.join(PROC_DIR, "test_processed.parquet"),
}

RAW_FILES = {
    "train": os.path.join(SPLIT_DIR, "train", "features.parquet"),
    "val": os.path.join(SPLIT_DIR, "val", "features.parquet"),
    "test": os.path.join(SPLIT_DIR, "test", "features.parquet"),
}

TARGET_COL = "is_candidate_stop"

def read_df(path):
    p = resolve_path(path)
    return pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()

def inspect_df(df, name):
    info = {
        "name": name,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_count": df.isna().sum().to_dict() if len(df) else {},
        "missing_pct": ((df.isna().sum() / len(df)) * 100).round(2).to_dict() if len(df) else {},
    }
    return info

def summarize_numeric(df):
    num = df.select_dtypes(include=[np.number]).copy()
    if len(num) == 0:
        return {}
    return {
        c: {
            "min": float(num[c].min()) if pd.notna(num[c].min()) else None,
            "max": float(num[c].max()) if pd.notna(num[c].max()) else None,
            "mean": float(num[c].mean()) if pd.notna(num[c].mean()) else None,
            "std": float(num[c].std()) if pd.notna(num[c].std()) else None,
            "n_unique": int(num[c].nunique(dropna=True)),
        }
        for c in num.columns
    }

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

report = {
    "raw": {},
    "processed": {},
    "diff": {},
}

print_section("STEP 3 AUDIT — RAW FILES")
for split, path in RAW_FILES.items():
    df = read_df(path)
    report["raw"][split] = inspect_df(df, f"raw_{split}")
    print(f"[{split.upper()} RAW] shape={df.shape}")
    print("Columns:")
    print(list(df.columns))
    print("Dtypes:")
    print(df.dtypes.astype(str).to_dict())

print_section("STEP 3 AUDIT — PROCESSED FILES")
for split, path in FILES.items():
    df = read_df(path)
    report["processed"][split] = inspect_df(df, f"processed_{split}")
    print(f"[{split.upper()} PROCESSED] shape={df.shape}")
    print("Columns:")
    print(list(df.columns))
    print("Dtypes:")
    print(df.dtypes.astype(str).to_dict())
    print("Numeric summary:")
    print(summarize_numeric(df))

print_section("STEP 3 AUDIT — COLUMN DIFF")
for split in ["train", "val", "test"]:
    raw_df = read_df(RAW_FILES[split])
    proc_df = read_df(FILES[split])

    raw_cols = set(raw_df.columns)
    proc_cols = set(proc_df.columns)

    dropped = sorted(list(raw_cols - proc_cols))
    added = sorted(list(proc_cols - raw_cols))

    report["diff"][split] = {
        "dropped": dropped,
        "added": added,
        "raw_count": len(raw_cols),
        "processed_count": len(proc_cols),
    }

    print(f"\n[{split.upper()}]")
    print(f"Raw columns       : {len(raw_cols)}")
    print(f"Processed columns : {len(proc_cols)}")
    print(f"Dropped columns   : {dropped}")
    print(f"Added columns     : {added}")

print_section("STEP 3 AUDIT — TARGET CHECK")
for split in ["train", "val", "test"]:
    df = read_df(FILES[split])
    if len(df) and TARGET_COL in df.columns:
        pos = int(df[TARGET_COL].sum())
        neg = int(len(df) - pos)
        print(f"[{split.upper()}] target={TARGET_COL}, positive={pos}, negative={neg}")
    else:
        print(f"[{split.upper()}] target column missing or empty")

with open(os.path.join(OUT_DIR, "step3_audit_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# Save compact text report too
txt_path = os.path.join(OUT_DIR, "step3_audit_report.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    for split in ["train", "val", "test"]:
        raw_info = report["raw"][split]
        proc_info = report["processed"][split]
        diff_info = report["diff"][split]
        f.write(f"[{split.upper()}]\n")
        f.write(f"Raw shape: {raw_info['shape']}\n")
        f.write(f"Processed shape: {proc_info['shape']}\n")
        f.write(f"Dropped: {diff_info['dropped']}\n")
        f.write(f"Added: {diff_info['added']}\n\n")

print_section("DONE")
print(f"Report JSON saved to: {os.path.join(OUT_DIR, 'step3_audit_report.json')}")
print(f"Report TXT  saved to: {txt_path}")