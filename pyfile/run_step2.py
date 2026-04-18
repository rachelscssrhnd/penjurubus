"""
run_step2.py
Step 2 PenjuruBus — Data Acquisition + Feature Engineering + Split

Split strategy (anti spatial leakage):
  TRAIN → Surabaya  (ground truth geocoded JSON tersedia)
  VAL   → Semarang  (skala mirip Surabaya, bukan provinsi)
  TEST  → Malang    (DIKUNCI sampai evaluasi akhir)

Sebelum run:
  pip install osmnx geopandas pandas numpy scikit-learn joblib pyarrow
  pip install requests tqdm rasterio   # opsional tapi direkomendasikan

File yang dibutuhkan (opsional tapi meningkatkan kualitas):
  data/raw/mitra_darat/surabaya_routes_geocoded.json
  data/raw/mitra_darat/semarang_routes_geocoded.json  (jika ada)
  data/raw/bps/idn_general_2020.tif                   (WorldPop raster)
"""

import sys, os, json, time
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Load modules ──────────────────────────────────────────────────────────────
import importlib.util

def _load(name, rel_path):
    path = os.path.join(ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    m    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

loader   = _load("loader",  "src/data/loader.py")
feat_bld = _load("feat",    "src/data/feature_builder.py")

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Key = nama internal (dipakai sebagai prefix grid_id dan nama folder)
# Nilai:
#   geocoded_json → path ke file geocoded (opsional, None = fallback ke OSM)
#   pop_raster    → path ke WorldPop raster (opsional)
#   split         → "train" | "val" | "test"

POP_RASTER = os.path.join(ROOT, "data", "raw", "bps", "idn_general_2020.tif")

# Path ke file PBF Jawa yang sudah kamu download
# Dipakai untuk ekstrak building footprints — jauh lebih andal dari Overpass
# Ganti path ini sesuai lokasi file di komputermu
PBF_PATH = r"D:/penjurubus/data/raw/osm/java.osm.pbf"   # Windows
# PBF_PATH = "/home/user/data/java.osm.pbf"                  # Linux/Mac
# Set None jika belum punya PBF → fallback ke Overpass (bisa gagal kota besar)
if not os.path.exists(PBF_PATH):
    print(f"[WARN] PBF tidak ditemukan: {PBF_PATH}")
    print(f"[WARN] Building footprints akan coba download dari Overpass")
    print(f"[WARN] (ini sering gagal untuk Surabaya/Yogyakarta — kota besar)")
    PBF_PATH = None

CITIES = {
    "tegal": {
        "geocoded_json": None,    # tidak ada ground truth → halte dari OSM
        "pop_raster": POP_RASTER,
        "pbf_path":   PBF_PATH,
        "split": "test",          # DIKUNCI sampai evaluasi akhir
    },
}

SPLIT_DIR = os.path.join(ROOT, "data", "split")


# ── STEP 2A: Data Acquisition ─────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2A — Data Acquisition")
print("="*60)

raw_data = {}
for city, cfg in CITIES.items():
    pop_path = cfg["pop_raster"] if os.path.exists(
        cfg.get("pop_raster", "")
    ) else None

    raw_data[city] = loader.load_city_data(
        city_name              = city,
        population_raster_path = pop_path,
        geocoded_json_path     = cfg.get("geocoded_json"),
        pbf_path               = cfg.get("pbf_path"),
    )

    # Jeda antar kota untuk rate limit Overpass
    time.sleep(5)


# ── STEP 2B: Feature Engineering ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2B — Feature Engineering")
print("="*60)

features_dict = {}
for city, data in raw_data.items():
    print(f"\n{'─'*40}")
    print(f"[FEAT] Processing: {city.upper()}")
    print(f"{'─'*40}")

    features_dict[city] = feat_bld.build_features(
        grid_gdf      = data["grid"],
        poi_gdf       = data["poi"],
        edges_gdf     = data["edges"],
        buildings_gdf = data["buildings"],
        halte_gdf     = data["halte"],
    )


# ── STEP 2C: Split per kota ───────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2C — Split Data (anti spatial leakage)")
print("="*60)

train_frames, val_frames, test_frames = [], [], []
split_log = {
    "train": [], "val": [], "test": [],
    "counts": {}, "label_method": {}
}

for city, df in features_dict.items():
    df        = df.copy()
    df["city"] = city
    split     = CITIES[city]["split"]

    split_log["counts"][city]       = len(df)
    split_log["label_method"][city] = df["label_method"].iloc[0]

    icon = "🔒 LOCKED" if split == "test" else ""
    print(f"  {city:15s} → {split.upper():5s} "
          f"({len(df):5d} grids, "
          f"candidate_stops={df['is_candidate_stop'].sum()}) {icon}")

    if split == "train":
        train_frames.append(df)
        split_log["train"].append(city)
    elif split == "val":
        val_frames.append(df)
        split_log["val"].append(city)
    elif split == "test":
        test_frames.append(df)
        split_log["test"].append(city)

# Concat — handle kasus kosong
def safe_concat(frames):
    frames = [f for f in frames if f is not None and len(f) > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

train_df = safe_concat(train_frames)
val_df   = safe_concat(val_frames)
test_df  = safe_concat(test_frames)


# ── STEP 2D: Anti-leakage verification ───────────────────────────────────────
print("\n[VERIFY] Checking for data leakage...")

train_ids = set(train_df["grid_id"]) if len(train_df) else set()
val_ids   = set(val_df["grid_id"])   if len(val_df)   else set()
test_ids  = set(test_df["grid_id"])  if len(test_df)  else set()

tv = len(train_ids & val_ids)
tt = len(train_ids & test_ids)
vt = len(val_ids   & test_ids)

print(f"  Train∩Val  : {tv}")
print(f"  Train∩Test : {tt}")
print(f"  Val∩Test   : {vt}")

assert tv == 0 and tt == 0 and vt == 0, \
    f"DATA LEAKAGE DETECTED! tv={tv} tt={tt} vt={vt}"
print("[VERIFY] ✓ No data leakage")


# ── STEP 2E: Class balance check ──────────────────────────────────────────────
print("\n[BALANCE] Label distribution:")
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    if len(df) == 0:
        continue
    n_pos = df["is_candidate_stop"].sum()
    n_neg = len(df) - n_pos
    ratio = n_pos / len(df) * 100
    print(f"  {name:5s}: {len(df):5d} grids | "
          f"positive={n_pos} ({ratio:.1f}%) | "
          f"negative={n_neg} ({100-ratio:.1f}%)")

    if ratio < 5:
        print(f"  ⚠ WARNING: {name} sangat imbalanced ({ratio:.1f}% positive). "
              "Pertimbangkan SMOTE atau class_weight='balanced' di Step 3.")
    if ratio == 0:
        print(f"  ✗ CRITICAL: {name} tidak punya label positif sama sekali! "
              "Periksa building/halte data.")


# ── STEP 2F: Save splits ──────────────────────────────────────────────────────
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

with open(os.path.join(SPLIT_DIR, "split_log.json"), "w") as f:
    json.dump(split_log, f, indent=2, ensure_ascii=False)
print("  split_log.json → saved")


# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("STEP 2 SELESAI")
print(f"{'='*60}")

for name, df in [("Train", train_df), ("Val  ", val_df), ("Test ", test_df)]:
    cities = split_log.get(name.strip().lower(), [])
    locked = " ← LOCKED" if name.strip() == "test" else ""
    print(f"  {name}: {len(df):5d} grids "
          f"({', '.join(cities)}){locked}")

feature_cols = [c for c in train_df.columns
                if c not in ["grid_id", "city", "centroid_x", "centroid_y",
                              "area_m2", "label_method",
                              "demand_score", "is_candidate_stop"]]

print(f"\n  Feature columns ({len(feature_cols)}):")
for c in sorted(feature_cols):
    print(f"    {c}")

print("\n→ Next: Step 3 — Preprocessing (fit scaler/imputer pada train only)")
print("  Jalankan: python run_step3.py")