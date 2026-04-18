import pandas as pd
import numpy as np
import os
import json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (SPLIT_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
                    RANDOM_SEED, VALIDATION_CITIES, TEST_CITIES)

def split_and_save(features_dict: dict) -> dict:
    """
    Split data PER KOTA — bukan random across cities.
    
    Strategi split:
    - TEST   → Malang (kota tanpa BRT aktual)
    - VAL    → Surabaya
    - TRAIN  → Surabaya
    
    Kenapa per kota, bukan random?
    → Mencegah spatial leakage: grid bertetangga dari kota sama
      memiliki fitur yang sangat mirip. Kalau split random, model
      bisa "menghafal" pola spasial kota tertentu.
    
    PENTING: fungsi ini WAJIB dipanggil SEBELUM preprocessing apapun.
    Setelah split, folder test/ TIDAK boleh dibuka sampai evaluasi akhir.
    """
    
    print("\n" + "="*50)
    print("SPLITTING DATA — ANTI DATA LEAKAGE")
    print("="*50)
    
    train_frames = []
    val_frames   = []
    test_frames  = []
    
    split_log = {
        "random_seed":   RANDOM_SEED,
        "split_strategy": "per_city_spatial",
        "train_cities":  [],
        "val_cities":    [],
        "test_cities":   [],
        "grid_counts":   {}
    }
    
    for city, df in features_dict.items():
        df = df.copy()
        df["city"] = city
        
        if city in TEST_CITIES:
            test_frames.append(df)
            split_log["test_cities"].append(city)
            print(f"[SPLIT] {city:12s} → TEST  ({len(df)} grids)")
            
        elif city in VALIDATION_CITIES[:1]:
            val_frames.append(df)
            split_log["val_cities"].append(city)
            print(f"[SPLIT] {city:12s} → VAL   ({len(df)} grids)")
            
        else:
            train_frames.append(df)
            split_log["train_cities"].append(city)
            print(f"[SPLIT] {city:12s} → TRAIN ({len(df)} grids)")
        
        split_log["grid_counts"][city] = len(df)
    
    # Gabungkan per split
    train_df = pd.concat(train_frames, ignore_index=True)
    val_df   = pd.concat(val_frames,   ignore_index=True)
    test_df  = pd.concat(test_frames,  ignore_index=True)
    
    # Simpan — TEST disimpan tapi JANGAN dibaca sampai evaluasi akhir
    os.makedirs(SPLIT_DIR, exist_ok=True)
    for split_name, df in [("train", train_df),
                            ("val",   val_df),
                            ("test",  test_df)]:
        out_dir = os.path.join(SPLIT_DIR, split_name)
        os.makedirs(out_dir, exist_ok=True)
        df.to_parquet(os.path.join(out_dir, "features.parquet"), index=False)
    
    # Simpan split log sebagai bukti anti-leakage
    log_path = os.path.join(SPLIT_DIR, "split_log.json")
    with open(log_path, "w") as f:
        json.dump(split_log, f, indent=2)
    
    print(f"\n[SPLIT] Train: {len(train_df)} | "
          f"Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"[SPLIT] Log saved: {log_path}")
    print("[SPLIT] ⚠️  test/ folder LOCKED — buka hanya saat evaluasi akhir!")
    
    return {
        "train": train_df,
        "val":   val_df,
        "test":  test_df,
        "log":   split_log
    }


def verify_no_leakage(split_dict: dict) -> bool:
    """
    Verifikasi tidak ada overlap grid_id antara train/val/test.
    Wajib dipanggil setelah split.
    """
    train_ids = set(split_dict["train"]["grid_id"])
    val_ids   = set(split_dict["val"]["grid_id"])
    test_ids  = set(split_dict["test"]["grid_id"])
    
    tv_overlap  = train_ids & val_ids
    tt_overlap  = train_ids & test_ids
    vt_overlap  = val_ids   & test_ids
    
    print("\n[VERIFY] Data Leakage Check:")
    print(f"  Train ∩ Val  : {len(tv_overlap)} overlap")
    print(f"  Train ∩ Test : {len(tt_overlap)} overlap")
    print(f"  Val ∩ Test   : {len(vt_overlap)} overlap")
    
    no_leakage = len(tv_overlap) == 0 and \
                 len(tt_overlap) == 0 and \
                 len(vt_overlap) == 0
    
    if no_leakage:
        print("  ✓ NO DATA LEAKAGE DETECTED")
    else:
        print("  ✗ WARNING: DATA LEAKAGE DETECTED!")
    
    return no_leakage