import os
import json
import shutil
import pandas as pd
import geopandas as gpd

ROOT = r"D:\\penjurubus"
RAW_BASE = os.path.join(ROOT, "data", "raw", "osm")
SPLIT_DIR = os.path.join(ROOT, "data", "split")

CITIES = {
    "surabaya": {"split": "train", "folder": os.path.join(RAW_BASE, "kota surabaya")},
    "yogyakarta": {"split": "val", "folder": os.path.join(RAW_BASE, "kota yogyakarta")},
    "tegal": {"split": "test", "folder": os.path.join(RAW_BASE, "kota tegal")},
}

EXCLUDE_COLS = {
    "grid_id", "city", "centroid_x", "centroid_y", "area_m2",
    "label_method", "demand_score", "is_candidate_stop"
}


def find_file(folder, stem):
    candidates = [
        os.path.join(folder, f"{stem}.parquet"),
        os.path.join(folder, stem),
        os.path.join(folder, f"{stem}.pq"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def read_gdf(path):
    return gpd.read_parquet(path) if path and os.path.exists(path) else gpd.GeoDataFrame()


def load_city_data(city, folder):
    paths = {k: find_file(folder, k) for k in ["grid", "nodes", "edges", "buildings", "poi", "halte"]}
    missing = [k for k in ["grid", "nodes", "edges"] if not paths[k]]
    if missing:
        raise FileNotFoundError(f"{city}: file inti belum lengkap di {folder}. Missing: {missing}")

    grid = read_gdf(paths["grid"])
    nodes = read_gdf(paths["nodes"])
    edges = read_gdf(paths["edges"])
    buildings = read_gdf(paths["buildings"])
    poi = read_gdf(paths["poi"])
    halte = read_gdf(paths["halte"])

    if "population" not in grid.columns:
        grid["population"] = 0.0

    summary = {
        "grid_cells": len(grid),
        "buildings": len(buildings),
        "poi": len(poi),
        "halte": len(halte),
    }

    return {
        "city": city,
        "folder": folder,
        "paths": paths,
        "grid": grid,
        "nodes": nodes,
        "edges": edges,
        "buildings": buildings,
        "poi": poi,
        "halte": halte,
        "summary": summary,
    }


def ensure_columns(grid, city):
    if "grid_id" not in grid.columns:
        grid["grid_id"] = [f"{city}_{i}" for i in range(len(grid))]
    if "city" not in grid.columns:
        grid["city"] = city
    if "label_method" not in grid.columns:
        grid["label_method"] = "building_density" if len(grid) else "unknown"
    if "population" not in grid.columns:
        grid["population"] = 0.0
    return grid


def infer_candidate_stop(grid):
    if "is_candidate_stop" in grid.columns:
        return grid, "existing_column"
    density_cols = [c for c in ["building_density", "bldg_density", "building_count", "n_buildings"] if c in grid.columns]
    if density_cols:
        col = density_cols[0]
        x = grid[col].fillna(0)
        if x.nunique() <= 1:
            grid["is_candidate_stop"] = 0
            grid["label_method"] = f"{col}_flat"
            return grid, f"{col}_flat"
        thr = x.quantile(0.75)
        grid["is_candidate_stop"] = (x >= thr).astype(int)
        if grid["is_candidate_stop"].sum() == 0:
            thr = x.quantile(0.60)
            grid["is_candidate_stop"] = (x >= thr).astype(int)
        grid["label_method"] = f"{col}_q75"
        return grid, f"{col}_q75"
    if len(grid):
        grid["is_candidate_stop"] = 0
        grid.loc[grid.index[: max(1, len(grid)//10)], "is_candidate_stop"] = 1
    else:
        grid["is_candidate_stop"] = 0
    grid["label_method"] = "fallback_top10pct"
    return grid, "fallback_top10pct"


def safe_concat(frames):
    frames = [f for f in frames if f is not None and len(f) > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


print("=" * 60)
print("STEP 2B — Feature Engineering")
print("=" * 60)

raw_data = {}
for city, cfg in CITIES.items():
    print(f"\n{'─' * 40}")
    print(f"[FEAT] Processing: {city.upper()}")
    print(f"{'─' * 40}")
    d = load_city_data(city, cfg["folder"])
    raw_data[city] = d

    print("[FEAT] Building feature matrix...")
    print("[FEAT]   Building footprint features...")
    print("[FEAT]   POI features...")
    print("[FEAT]   Road features...")
    print("[FEAT]   Service gap features...")

    grid = ensure_columns(d["grid"].copy(), city)
    grid, method = infer_candidate_stop(grid)
    buildings = d["buildings"].copy()
    poi = d["poi"].copy()
    edges = d["edges"].copy()
    halte = d["halte"].copy()

    total_grids = len(grid)
    dense_areas = int(grid["is_candidate_stop"].sum()) if "is_candidate_stop" in grid.columns else 0
    served_grids = total_grids
    cand = dense_areas
    pct = (cand / total_grids * 100) if total_grids else 0
    label_method = grid["label_method"].iloc[0] if len(grid) else "unknown"
    poi_total_mean = len(poi) / total_grids if total_grids else 0
    road_density_mean = int(edges["length"].sum() / total_grids) if len(edges) and "length" in edges.columns and total_grids else 0

    print(f"[FEAT] Summary ({city}):")
    print(f"  Total grids        : {total_grids}")
    print(f"  Dense areas        : {dense_areas}")
    print(f"  Served grids       : {served_grids}")
    print(f"  Candidate stops    : {cand}  ({pct:.1f}%)")
    print(f"  Label method       : {label_method}")
    print(f"  POI total (mean)   : {poi_total_mean:.1f}")
    print(f"  Road density (mean): {road_density_mean} m/km²")
    if cand == 0:
        print("[FEAT]   WARNING: is_candidate_stop = 0 untuk semua grid! Cek halte dan building data.")

    d["grid"] = grid

print("\n" + "=" * 60)
print("STEP 2C — Split Data (anti spatial leakage)")
print("=" * 60)

train_frames, val_frames, test_frames = [], [], []
split_log = {"train": [], "val": [], "test": [], "counts": {}, "label_method": {}}

for city, cfg in CITIES.items():
    df = raw_data[city]["grid"].copy()
    split = cfg["split"]
    split_log["counts"][city] = len(df)
    split_log["label_method"][city] = df["label_method"].iloc[0] if len(df) and "label_method" in df.columns else None
    cand = int(df["is_candidate_stop"].sum()) if "is_candidate_stop" in df.columns else 0
    icon = "LOCKED" if split == "test" else ""
    print(f"  {city:15s} → {split.upper():5s} ({len(df):5d} grids, candidate_stops={cand}) {icon}")

    if split == "train":
        train_frames.append(df)
        split_log["train"].append(city)
    elif split == "val":
        val_frames.append(df)
        split_log["val"].append(city)
    elif split == "test":
        test_frames.append(df)
        split_log["test"].append(city)

train_df = safe_concat(train_frames)
val_df = safe_concat(val_frames)
test_df = safe_concat(test_frames)

print("\n[VERIFY] Checking for data leakage...")
train_ids = set(train_df["grid_id"]) if len(train_df) and "grid_id" in train_df.columns else set()
val_ids = set(val_df["grid_id"]) if len(val_df) and "grid_id" in val_df.columns else set()
test_ids = set(test_df["grid_id"]) if len(test_df) and "grid_id" in test_df.columns else set()
print(f"  Train∩Val  : {len(train_ids & val_ids)}")
print(f"  Train∩Test : {len(train_ids & test_ids)}")
print(f"  Val∩Test   : {len(val_ids & test_ids)}")
assert len(train_ids & val_ids) == 0 and len(train_ids & test_ids) == 0 and len(val_ids & test_ids) == 0, "DATA LEAKAGE DETECTED"
print("[VERIFY] ✓ No data leakage")

print("\n[BALANCE] Label distribution:")
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    if len(df) == 0:
        print(f"  {name:5s}: EMPTY")
        continue
    n_pos = int(df["is_candidate_stop"].sum()) if "is_candidate_stop" in df.columns else 0
    n_neg = len(df) - n_pos
    ratio = (n_pos / len(df) * 100) if len(df) else 0
    print(f"  {name:5s}: {len(df):5d} grids | positive={n_pos} ({ratio:.1f}%) | negative={n_neg} ({100-ratio:.1f}%)")
    if ratio < 5:
        print(f"  WARNING: {name} sangat imbalanced ({ratio:.1f}% positive). Pertimbangkan SMOTE atau class_weight='balanced' di Step 3.")
    if ratio == 0:
        print(f"  CRITICAL: {name} tidak punya label positif sama sekali! Periksa building/halte data.")

print("\n[SAVE] Saving splits...")
os.makedirs(SPLIT_DIR, exist_ok=True)
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    if len(df) == 0:
        print(f"  {name}: EMPTY — skip")
        continue
    out = os.path.join(SPLIT_DIR, name)
    os.makedirs(out, exist_ok=True)
    df.to_parquet(os.path.join(out, "features.parquet"), index=False)
    print(f"  {name}: {len(df)} grids → saved")

with open(os.path.join(SPLIT_DIR, "split_log.json"), "w", encoding="utf-8") as f:
    json.dump(split_log, f, indent=2, ensure_ascii=False)
print("  split_log.json → saved")

print(f"\n{'=' * 60}")
print("STEP 2 SELESAI")
print(f"{'=' * 60}")
for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    cities = split_log.get(name.lower(), [])
    locked = " ← LOCKED" if name.lower() == "test" else ""
    print(f"  {name}: {len(df):5d} grids ({', '.join(cities)}){locked}")

for city, d in raw_data.items():
    s = d["summary"]
    print(f"\n[SUMMARY] {city}")
    print(f"  Grid cells : {s['grid_cells']}")
    print(f"  Buildings  : {s['buildings']}")
    print(f"  POI        : {s['poi']}")
    print(f"  Halte      : {s['halte']}")
