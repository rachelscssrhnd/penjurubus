"""
src/data/feature_builder.py
Feature matrix builder untuk PenjuruBus — Step 2
"""

import geopandas as gpd
import pandas as pd
import numpy as np

WALK_RADIUS_M          = 500
BUILDING_SUM_AREA_MIN  = 20000.0   # m² total luas per grid → kawasan padat
BUILDING_MEAN_AREA_MAX = 200.0     # m² rata-rata per unit → bukan komersial


def build_features(grid_gdf:      gpd.GeoDataFrame,
                   poi_gdf:       gpd.GeoDataFrame,
                   edges_gdf:     gpd.GeoDataFrame,
                   buildings_gdf: gpd.GeoDataFrame,
                   halte_gdf:     gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Feature matrix — SATU BARIS = SATU GRID CELL 500m×500m

    Features:
    ┌─ Populasi ──────────────────────────────────────────────────────┐
    │  population                                                     │
    ├─ Building footprint ────────────────────────────────────────────┤
    │  building_count     jumlah bangunan residential dalam grid      │
    │  building_sum_m2    total luas lantai                           │
    │  building_mean_m2   rata-rata luas per unit                     │
    │  building_density   count / km²                                 │
    │  is_dense_area      flag: sum≥20000 AND mean≤200 AND count>0   │
    ├─ POI (dalam radius 500m dari centroid grid) ────────────────────┤
    │  poi_amenity, poi_shop, poi_railway, poi_office, dll            │
    │  poi_total          total semua POI                             │
    │  poi_diversity      berapa kategori yang hadir                  │
    ├─ Road network ──────────────────────────────────────────────────┤
    │  road_length_m      total panjang jalan                        │
    │  road_count         jumlah segmen jalan                        │
    │  road_density       m/km²                                      │
    ├─ Aksesibilitas ─────────────────────────────────────────────────┤
    │  accessibility_index  composite score 0-1                      │
    ├─ Service gap ───────────────────────────────────────────────────┤
    │  halte_within_500m  jumlah halte dalam 500m                    │
    │  is_served          1 jika sudah ada halte dalam 500m          │
    │  nearest_halte_m    jarak ke halte terdekat                    │
    └─ TARGET ────────────────────────────────────────────────────────┘
       demand_score        proxy demand 0-1
       is_candidate_stop   1 = layak jadi halte baru
                           (padat TAPI belum terlayani)
    """
    print("[FEAT] Building feature matrix...")

    # Pastikan semua dalam CRS yang sama
    grid  = grid_gdf.copy().to_crs(epsg=32749)

    has_poi      = poi_gdf is not None and len(poi_gdf) > 0
    has_edges    = edges_gdf is not None and len(edges_gdf) > 0
    has_buildings = buildings_gdf is not None and len(buildings_gdf) > 0
    has_halte    = halte_gdf is not None and len(halte_gdf) > 0

    poi   = poi_gdf.copy().to_crs(epsg=32749)   if has_poi      else None
    edges = edges_gdf.copy().to_crs(epsg=32749) if has_edges    else None
    bldg  = buildings_gdf.copy().to_crs(epsg=32749) if has_buildings else None
    halte = halte_gdf.copy().to_crs(epsg=32749) if has_halte    else None

    feat = grid[["grid_id", "city", "population",
                  "centroid_x", "centroid_y", "area_m2"]].copy()

    # ── BUILDING FOOTPRINT ────────────────────────────────────────────────────
    if has_buildings:
        print("[FEAT]   Building footprint features...")
        joined_bldg = gpd.sjoin(
            bldg[["geometry", "footprint_area_m2"]],
            grid[["grid_id", "geometry"]],
            how="right",
            predicate="within"
        )
        bldg_stats = (
            joined_bldg
            .groupby("grid_id")["footprint_area_m2"]
            .agg(building_count="count",
                 building_sum_m2="sum",
                 building_mean_m2="mean")
            .reset_index()
        )
        feat = feat.merge(bldg_stats, on="grid_id", how="left")
        feat["building_count"]   = feat["building_count"].fillna(0).astype(int)
        feat["building_sum_m2"]  = feat["building_sum_m2"].fillna(0.0)
        feat["building_mean_m2"] = feat["building_mean_m2"].fillna(0.0)

        feat["building_density"] = (
            feat["building_count"] / (feat["area_m2"] / 1e6)
        )
        feat["is_dense_area"] = (
            (feat["building_sum_m2"]  >= BUILDING_SUM_AREA_MIN) &
            (feat["building_mean_m2"] <= BUILDING_MEAN_AREA_MAX) &
            (feat["building_count"]   > 0)
        ).astype(int)

        n_dense = feat["is_dense_area"].sum()
        print(f"[FEAT]   Dense areas: {n_dense}/{len(feat)} "
              f"({n_dense/len(feat)*100:.1f}%)")
    else:
        print("[FEAT]   WARNING: No building data — all building features = 0")
        feat["building_count"]   = 0
        feat["building_sum_m2"]  = 0.0
        feat["building_mean_m2"] = 0.0
        feat["building_density"] = 0.0
        feat["is_dense_area"]    = 0

    # ── POI FEATURES ──────────────────────────────────────────────────────────
    if has_poi:
        print("[FEAT]   POI features...")
        # Buffer grid centroid 500m untuk catchment area halte
        grid_buf = grid.copy()
        grid_buf["geometry"] = grid.geometry.buffer(WALK_RADIUS_M)

        for cat in poi["category"].unique():
            poi_cat = poi[poi["category"] == cat][["geometry"]].copy()
            joined  = gpd.sjoin(
                poi_cat,
                grid_buf[["grid_id", "geometry"]],
                how="right",
                predicate="within"
            )
            cnt = (joined.groupby("grid_id")
                         .size()
                         .reset_index(name=f"poi_{cat}"))
            feat = feat.merge(cnt, on="grid_id", how="left")
            feat[f"poi_{cat}"] = feat[f"poi_{cat}"].fillna(0).astype(int)

    poi_cols = [c for c in feat.columns if c.startswith("poi_")]
    if poi_cols:
        feat["poi_total"]     = feat[poi_cols].sum(axis=1)
        feat["poi_diversity"] = (feat[poi_cols] > 0).sum(axis=1)
    else:
        feat["poi_total"]     = 0
        feat["poi_diversity"] = 0

    # ── ROAD FEATURES ─────────────────────────────────────────────────────────
    if has_edges:
        print("[FEAT]   Road features...")
        # Pastikan kolom 'length' ada
        if "length" not in edges.columns:
            edges["length"] = edges.geometry.length

        roads_in = gpd.sjoin(
            edges[["geometry", "length"]],
            grid[["grid_id", "geometry"]],
            how="right",
            predicate="intersects"
        )
        road_stats = (
            roads_in.groupby("grid_id")["length"]
                    .agg(road_length_m="sum", road_count="count")
                    .reset_index()
        )
        feat = feat.merge(road_stats, on="grid_id", how="left")
        feat["road_length_m"] = feat["road_length_m"].fillna(0.0)
        feat["road_count"]    = feat["road_count"].fillna(0).astype(int)
        feat["road_density"]  = feat["road_length_m"] / (feat["area_m2"] / 1e6)
    else:
        feat["road_length_m"] = 0.0
        feat["road_count"]    = 0
        feat["road_density"]  = 0.0

    # ── SERVICE GAP ───────────────────────────────────────────────────────────
    if has_halte:
        print("[FEAT]   Service gap features...")

        grid_centroids = grid[["grid_id", "geometry"]].copy()
        grid_centroids["geometry"] = grid.geometry.centroid

        halte_buf = halte.copy()
        halte_buf["geometry"] = halte.geometry.buffer(WALK_RADIUS_M)

        served_join = gpd.sjoin(
            grid_centroids,
            halte_buf[["geometry"]],
            how="left",
            predicate="within"
        )
        served = (served_join.groupby("grid_id")
                             .size()
                             .reset_index(name="halte_within_500m"))
        feat = feat.merge(served, on="grid_id", how="left")
        feat["halte_within_500m"] = feat["halte_within_500m"].fillna(0).astype(int)
        feat["is_served"]         = (feat["halte_within_500m"] > 0).astype(int)

        # Jarak ke halte terdekat
        from shapely.ops import nearest_points
        halte_union   = halte.geometry.unary_union
        nearest_dist  = []
        for _, row in grid_centroids.iterrows():
            try:
                _, p2   = nearest_points(row.geometry, halte_union)
                nearest_dist.append(row.geometry.distance(p2))
            except Exception:
                nearest_dist.append(9999.0)
        feat["nearest_halte_m"] = nearest_dist
    else:
        print("[FEAT]   No halte data — is_served = 0")
        feat["halte_within_500m"] = 0
        feat["is_served"]         = 0
        feat["nearest_halte_m"]   = 9999.0

    # ── ACCESSIBILITY INDEX ───────────────────────────────────────────────────
    def minmax(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-9)

    feat["accessibility_index"] = (
        0.30 * minmax(feat["building_density"]) +
        0.25 * minmax(feat["poi_total"]) +
        0.25 * minmax(feat["road_density"]) +
        0.20 * minmax(feat["population"])
    )

    # ── DEMAND SCORE (continuous target) ──────────────────────────────────────
    feat["demand_score"] = (
        0.30 * minmax(feat["building_sum_m2"]) +
        0.25 * minmax(feat["population"]) +
        0.25 * minmax(feat["poi_total"]) +
        0.20 * minmax(feat["road_density"])
    )

    # ── IS_CANDIDATE_STOP (binary target) ─────────────────────────────────────
    # Strategi:
    # 1. Jika ada building data  → pakai is_dense_area & is_served
    # 2. Jika building gagal     → fallback demand_score quartile 60%
    has_building_data = feat["building_count"].sum() > 0

    if has_building_data:
        feat["is_candidate_stop"] = (
            (feat["is_dense_area"] == 1) &
            (feat["is_served"]     == 0)
        ).astype(int)
        method = "building_density"
    else:
        print("[FEAT]   WARNING: No building data — "
              "using demand_score fallback for is_candidate_stop")
        threshold = feat["demand_score"].quantile(0.60)
        feat["is_candidate_stop"] = (
            (feat["demand_score"] >= threshold) &
            (feat["is_served"]    == 0)
        ).astype(int)
        method = "demand_score_fallback"

    feat["label_method"] = method

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    n_cand   = feat["is_candidate_stop"].sum()
    n_served = feat["is_served"].sum()
    print(f"\n[FEAT] Summary ({feat['city'].iloc[0]}):")
    print(f"  Total grids        : {len(feat)}")
    print(f"  Dense areas        : {feat['is_dense_area'].sum()}")
    print(f"  Served grids       : {n_served}")
    print(f"  Candidate stops    : {n_cand}  ({n_cand/len(feat)*100:.1f}%)")
    print(f"  Label method       : {method}")
    print(f"  POI total (mean)   : {feat['poi_total'].mean():.1f}")
    print(f"  Road density (mean): {feat['road_density'].mean():.0f} m/km²")

    # Validasi: pastikan ada variasi di target
    if n_cand == 0:
        print("[FEAT]   WARNING: is_candidate_stop = 0 untuk semua grid! "
              "Cek halte dan building data.")
    elif n_cand == len(feat):
        print("[FEAT]   WARNING: is_candidate_stop = 1 untuk semua grid! "
              "Semua area dianggap butuh halte — periksa is_served.")

    return feat