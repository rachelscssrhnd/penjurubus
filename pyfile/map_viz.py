import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = r"D:/penjurubus"
PROC_DIR = os.path.join(ROOT, "data", "processed_v3")
PRED_DIR = os.path.join(ROOT, "data", "predictions_v4_search")
OUT_DIR = os.path.join(ROOT, "output_visual")
os.makedirs(OUT_DIR, exist_ok=True)

CITY_FILES = {
    "Surabaya": os.path.join(PROC_DIR, "surabaya", "features_v3.parquet"),
    "Yogyakarta": os.path.join(PROC_DIR, "yogyakarta", "features_v3.parquet"),
    "Tegal": os.path.join(PROC_DIR, "tegal", "features_v3.parquet"),
}

TEGAL_TOP20 = os.path.join(PRED_DIR, "tegal_top20.csv")

def read_any(path):
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.lower().endswith(".pkl") or path.lower().endswith(".pickle"):
        return pd.read_pickle(path)
    elif path.lower().endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file: {path}")

def to_gdf(df):
    if "geometry" not in df.columns:
        return None
    try:
        gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.GeoSeries.from_wkb(df["geometry"]))
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        return gdf
    except Exception:
        try:
            gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.GeoSeries.from_wkt(df["geometry"].astype(str)))
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326", allow_override=True)
            return gdf
        except Exception:
            return None

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for ax, (city, path) in zip(axes, CITY_FILES.items()):
    df = read_any(path)
    gdf = to_gdf(df)

    if gdf is None:
        ax.set_title(f"{city} - geometry tidak terbaca")
        ax.axis("off")
        continue

    gdf = gdf.to_crs("EPSG:3857")
    gdf.plot(ax=ax, color="white", edgecolor="black", linewidth=0.2)

    if "is_candidate_stop" in gdf.columns:
        cand = gdf[gdf["is_candidate_stop"] == 1]
        if len(cand) > 0:
            cand.plot(ax=ax, color="red", edgecolor="darkred", linewidth=0.4)

    ax.set_title(city)
    ax.set_axis_off()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "grid_map_3cities.png"), dpi=300, bbox_inches="tight")
plt.show()