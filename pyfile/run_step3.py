import os
import json
import pandas as pd
import geopandas as gpd
import numpy as np
from penjurubus.data_paths import resolve_path

ROOT = r"D:\penjurubus"
RAW_BASE = os.path.join(ROOT, "data", "raw", "osm")
SPLIT_DIR = os.path.join(ROOT, "data", "split")
OUT_DIR = os.path.join(ROOT, "data", "processed_v3")
os.makedirs(OUT_DIR, exist_ok=True)

CITIES = {
    "surabaya": {"split": "train", "folder": os.path.join(RAW_BASE, "kota surabaya")},
    "yogyakarta": {"split": "val", "folder": os.path.join(RAW_BASE, "kota yogyakarta")},
    "tegal": {"split": "test", "folder": os.path.join(RAW_BASE, "kota tegal")},
}

TARGET_COL = "is_candidate_stop"

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
    p = resolve_path(path)
    return gpd.read_parquet(p) if p and os.path.exists(p) else gpd.GeoDataFrame()

def safe_crs(gdf, target=None):
    if len(gdf) == 0:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    if target is not None:
        gdf = gdf.to_crs(target)
    return gdf

def centroid_of_geom(gdf):
    if len(gdf) == 0:
        return gdf
    gdf = gdf.copy()
    if "geometry" not in gdf.columns:
        return gdf
    gdf = gdf[gdf.geometry.notna()].copy()
    return gdf

def load_city(folder):
    paths = {k: find_file(folder, k) for k in ["grid", "nodes", "edges", "buildings", "poi", "halte"]}
    missing = [k for k in ["grid", "nodes", "edges"] if not paths[k]]
    if missing:
        raise FileNotFoundError(f"Missing file inti: {missing} in {folder}")
    return {k: read_gdf(v) for k, v in paths.items()}

def prep_points(gdf, name):
    gdf = gdf.copy()
    if len(gdf) == 0:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    gdf = gdf.to_crs("EPSG:3857")
    if "geometry" in gdf.columns:
        gdf["x"] = gdf.geometry.x
        gdf["y"] = gdf.geometry.y
    return gdf

def prep_polygons(gdf):
    gdf = gdf.copy()
    if len(gdf) == 0:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf.to_crs("EPSG:3857")

def prep_lines(gdf):
    gdf = gdf.copy()
    if len(gdf) == 0:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf.to_crs("EPSG:3857")

def make_grid_features(grid, buildings, poi, halte, edges, city):
    grid = grid.copy()
    if len(grid) == 0:
        return grid

    if grid.crs is None:
        grid = grid.set_crs("EPSG:4326", allow_override=True)
    grid = grid.to_crs("EPSG:3857")

    if "grid_id" not in grid.columns:
        grid["grid_id"] = [f"{city}_{i}" for i in range(len(grid))]
    grid["city"] = city

    if "population" not in grid.columns:
        grid["population"] = 0.0

    grid["area_m2"] = grid.geometry.area if "geometry" in grid.columns else 0.0
    grid["centroid_x"] = grid.geometry.centroid.x if "geometry" in grid.columns else 0.0
    grid["centroid_y"] = grid.geometry.centroid.y if "geometry" in grid.columns else 0.0

    grid["poi_count"] = 0
    grid["building_count"] = 0
    grid["halte_count"] = 0
    grid["road_length"] = 0.0
    grid["road_density"] = 0.0
    grid["dist_to_nearest_halte"] = np.nan

    if len(poi):
        poi = prep_points(poi, "poi")
        join_poi = gpd.sjoin(poi, grid[["grid_id", "geometry"]], predicate="intersects", how="left")
        poi_counts = join_poi.groupby("grid_id").size()
        grid["poi_count"] = grid["grid_id"].map(poi_counts).fillna(0).astype(int)

    if len(buildings):
        buildings = prep_polygons(buildings)
        join_bld = gpd.sjoin(buildings, grid[["grid_id", "geometry"]], predicate="intersects", how="left")
        bld_counts = join_bld.groupby("grid_id").size()
        grid["building_count"] = grid["grid_id"].map(bld_counts).fillna(0).astype(int)

    if len(halte):
        halte = prep_points(halte, "halte")
        join_h = gpd.sjoin(halte, grid[["grid_id", "geometry"]], predicate="intersects", how="left")
        h_counts = join_h.groupby("grid_id").size()
        grid["halte_count"] = grid["grid_id"].map(h_counts).fillna(0).astype(int)

        grid_pts = grid[["grid_id", "geometry"]].copy()
        nearest = gpd.sjoin_nearest(grid_pts, halte[["geometry"]], how="left", distance_col="dist_to_nearest_halte")
        nearest = nearest[["grid_id", "dist_to_nearest_halte"]]
        grid = grid.drop(columns=["dist_to_nearest_halte"]).merge(nearest, on="grid_id", how="left")

    if len(edges):
        edges = prep_lines(edges)
        edges["length_m"] = edges.geometry.length
        join_e = gpd.sjoin(edges, grid[["grid_id", "geometry"]], predicate="intersects", how="left")
        e_len = join_e.groupby("grid_id")["length_m"].sum()
        grid["road_length"] = grid["grid_id"].map(e_len).fillna(0.0)
        grid["road_density"] = grid["road_length"] / grid["area_m2"].replace(0, np.nan)

    if "population" in grid.columns:
        grid["pop_density"] = grid["population"] / grid["area_m2"].replace(0, np.nan)
    else:
        grid["pop_density"] = 0.0

    if "area_m2" in grid.columns and "population" in grid.columns:
        grid["pop_x_area"] = grid["population"] * grid["area_m2"]
    else:
        grid["pop_x_area"] = 0.0

    if "is_candidate_stop" not in grid.columns:
        score = (
            grid["poi_count"].fillna(0)
            + grid["building_count"].fillna(0)
            + grid["halte_count"].fillna(0)
        )
        thr = score.quantile(0.90) if len(score) else 0
        grid["is_candidate_stop"] = (score >= thr).astype(int)
        grid["label_method"] = "poi_building_halte_q90"
    else:
        if "label_method" not in grid.columns:
            grid["label_method"] = "existing"

    grid = grid.to_crs("EPSG:4326")
    return grid

raw_data = {}
for city, cfg in CITIES.items():
    print(f"[LOAD] {city}")
    d = load_city(cfg["folder"])
    raw_data[city] = d

    grid = d["grid"]
    buildings = d["buildings"]
    poi = d["poi"]
    halte = d["halte"]
    edges = d["edges"]

    print(f"  raw grid      : {len(grid)}")
    print(f"  raw buildings : {len(buildings)}")
    print(f"  raw poi       : {len(poi)}")
    print(f"  raw halte     : {len(halte)}")

    feat = make_grid_features(grid, buildings, poi, halte, edges, city)

    city_out = os.path.join(OUT_DIR, city)
    os.makedirs(city_out, exist_ok=True)
    feat.to_parquet(os.path.join(city_out, "features_v3.parquet"), index=False)

    summary = {
        "city": city,
        "grid_cells": len(feat),
        "poi_total": int(feat["poi_count"].sum()) if "poi_count" in feat.columns else 0,
        "building_total": int(feat["building_count"].sum()) if "building_count" in feat.columns else 0,
        "halte_total": int(feat["halte_count"].sum()) if "halte_count" in feat.columns else 0,
        "positive": int(feat["is_candidate_stop"].sum()) if "is_candidate_stop" in feat.columns else 0,
        "label_method": feat["label_method"].iloc[0] if len(feat) and "label_method" in feat.columns else None
    }
    with open(os.path.join(city_out, "summary_v3.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  saved: {city_out}")

print("\nSTEP 3 V3 selesai")