import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import os, json, time
from shapely.geometry import Point
from tqdm import tqdm

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "data", "raw")
GRID_SIZE_M = 500
WALK_RADIUS_M = 500

RESIDENTIAL_TAGS = [
    'residential', 'house', 'apartments', 'detached',
    'semidetached_house', 'terrace', 'dormitory',
    'hut', 'cabin', 'bungalow', 'yes'
]

def _osm_place_name(city_name: str) -> str:
    mapping = {
        "yogyakarta": "Kota Yogyakarta",
        "kota yogyakarta": "Kota Yogyakarta",
        "surabaya": "Kota Surabaya",
        "kota surabaya": "Kota Surabaya",
        "kota bandung": "Kota Bandung",
        "tegal": "Kota Tegal",
        "kota tegal": "Kota Tegal",
    }
    key = city_name.lower().strip()
    return mapping.get(key, city_name) + ", Indonesia"


def load_osm_network(city_name: str, save: bool = True) -> tuple:
    place = _osm_place_name(city_name)
    print(f"[OSM] Downloading road network: {place}...")

    G = ox.graph_from_place(place, network_type="drive", simplify=True)
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

    nodes = ox.io._stringify_nonnumeric_cols(nodes)
    edges = ox.io._stringify_nonnumeric_cols(edges)

    drop_cols = [c for c in ["osmid", "oneway"] if c in edges.columns]
    edges = edges.drop(columns=drop_cols, errors="ignore")

    if save:
        out = os.path.join(RAW_DIR, "osm", city_name)
        os.makedirs(out, exist_ok=True)
        nodes.to_parquet(os.path.join(out, "nodes.parquet"))
        edges.to_parquet(os.path.join(out, "edges.parquet"))
        print(f"[OSM] {len(nodes)} nodes, {len(edges)} edges → saved")

    return G, nodes, edges


def generate_city_grid(city_name: str,
                       grid_size_m: int = GRID_SIZE_M,
                       save: bool = True) -> gpd.GeoDataFrame:
    place = _osm_place_name(city_name)
    print(f"[GRID] Generating {grid_size_m}m grid: {place}...")

    boundary = ox.geocode_to_gdf(place).to_crs(epsg=32749)
    bounds = boundary.total_bounds

    from shapely.geometry import box
    cells, ids = [], []
    xs = np.arange(bounds[0], bounds[2], grid_size_m)
    ys = np.arange(bounds[1], bounds[3], grid_size_m)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            cell = box(x, y, x + grid_size_m, y + grid_size_m)
            if boundary.geometry.iloc[0].intersects(cell):
                cells.append(cell)
                ids.append(f"{city_name}_{i:04d}_{j:04d}")

    gdf = gpd.GeoDataFrame(
        {"grid_id": ids, "geometry": cells}, crs="EPSG:32749"
    )
    gdf["centroid_x"] = gdf.geometry.centroid.x
    gdf["centroid_y"] = gdf.geometry.centroid.y
    gdf["city"] = city_name
    gdf["area_m2"] = grid_size_m ** 2

    if save:
        out = os.path.join(RAW_DIR, "grid", city_name)
        os.makedirs(out, exist_ok=True)
        gdf.to_parquet(os.path.join(out, "grid.parquet"))
        print(f"[GRID] {len(gdf)} cells → saved")

    return gdf


def load_building_footprints(city_name: str,
                             pbf_path: str = None,
                             save: bool = True,
                             max_retries: int = 3,
                             retry_wait: int = 15) -> gpd.GeoDataFrame:
    cache_path = os.path.join(RAW_DIR, "buildings", city_name, "buildings.parquet")
    try:
        from penjurubus.data_paths import resolve_path
        p = resolve_path(cache_path)
    except Exception:
        p = cache_path
    if os.path.exists(p):
        print(f"[BLDG] Loading from cache: {city_name}")
        return gpd.read_parquet(p)

    if pbf_path and os.path.exists(pbf_path):
        return _load_buildings_from_pbf(city_name, pbf_path, save)

    raise FileNotFoundError(f"[BLDG] PBF file not found: {pbf_path}")


def _load_buildings_from_pbf(city_name: str,
                             pbf_path: str,
                             save: bool) -> gpd.GeoDataFrame:
    """Ekstrak building footprints dari file PBF lokal pakai pyosmium."""
    print(f"[BLDG] Extracting buildings from PBF: {pbf_path}")
    print(f"[BLDG] (ini bisa 5-15 menit untuk file Jawa — silakan tunggu)")

    try:
        import osmium
        from shapely import wkb as shp_wkb
    except ImportError:
        print("[BLDG] osmium tidak terinstall. Jalankan: pip install osmium")
        return gpd.GeoDataFrame()

    place = _osm_place_name(city_name)
    boundary = ox.geocode_to_gdf(place).to_crs(epsg=32749)
    bounds = boundary.total_bounds

    try:
        import pyproj
        transformer = pyproj.Transformer.from_crs(
            "EPSG:32749", "EPSG:4326", always_xy=True
        )
        minlon, minlat = transformer.transform(bounds[0], bounds[1])
        maxlon, maxlat = transformer.transform(bounds[2], bounds[3])
    except ImportError:
        minlat, maxlat = bounds[1] / 111320, bounds[3] / 111320
        minlon, maxlon = bounds[0] / 111320, bounds[2] / 111320

    bbox_wgs84 = {
        "min_lat": minlat, "max_lat": maxlat,
        "min_lon": minlon, "max_lon": maxlon,
    }

    class BuildingHandler(osmium.SimpleHandler):
        def __init__(self, bbox):
            super().__init__()
            self.bbox = bbox
            self.buildings = []
            self._fab = osmium.geom.WKBFactory()

        def _in_bbox(self, lat, lon):
            return (self.bbox["min_lat"] <= lat <= self.bbox["max_lat"] and
                    self.bbox["min_lon"] <= lon <= self.bbox["max_lon"])

        def area(self, a):
            tags = {t.k: t.v for t in a.tags}
            if "building" not in tags:
                return
            try:
                wkb = self._fab.create_multipolygon(a)
                geom = shp_wkb.loads(wkb, hex=True)
                c = geom.centroid
                if not self._in_bbox(c.y, c.x):
                    return
                self.buildings.append({
                    "building": tags.get("building", "yes"),
                    "geometry": geom,
                })
            except Exception:
                pass

    handler = BuildingHandler(bbox_wgs84)
    try:
        handler.apply_file(pbf_path, locations=True, idx="sparse_mem_array")
    except Exception as e:
        print(f"[BLDG] PBF read error: {e}")
        return gpd.GeoDataFrame()

    if not handler.buildings:
        print("[BLDG] Tidak ada building ditemukan di PBF untuk bbox ini")
        _save_empty_buildings(city_name, save)
        return gpd.GeoDataFrame()

    print(f"[BLDG] {len(handler.buildings)} raw buildings dari PBF")

    gdf = gpd.GeoDataFrame(handler.buildings, crs="EPSG:4326").to_crs(epsg=32749)

    gdf = gdf[gdf["building"].isin(RESIDENTIAL_TAGS)].copy()
    print(f"[BLDG] Setelah filter residential: {len(gdf)}")

    gdf["footprint_area_m2"] = gdf.geometry.area
    gdf = gdf[
        (gdf["footprint_area_m2"] >= 20) &
        (gdf["footprint_area_m2"] <= 2000)
    ].copy()

    gdf = gdf[["geometry", "footprint_area_m2", "building"]].reset_index(drop=True)
    gdf["city"] = city_name

    if save:
        out = os.path.join(RAW_DIR, "buildings", city_name)
        os.makedirs(out, exist_ok=True)
        gdf.to_parquet(os.path.join(out, "buildings.parquet"))
        print(f"[BLDG] {len(gdf)} residential buildings → saved")

    return gdf


def _save_empty_buildings(city_name: str, save: bool):
    if save:
        out = os.path.join(RAW_DIR, "buildings", city_name)
        os.makedirs(out, exist_ok=True)
        empty = gpd.GeoDataFrame(
            columns=["geometry", "footprint_area_m2", "building", "city"],
            geometry="geometry", crs="EPSG:32749"
        )
        empty.to_parquet(os.path.join(out, "buildings.parquet"))


def load_existing_halte(city_name: str,
                        geocoded_json_path: str = None,
                        save: bool = True) -> gpd.GeoDataFrame:
    if geocoded_json_path and os.path.exists(geocoded_json_path):
        print(f"[HALTE] Loading from geocoded JSON: {city_name}...")
        with open(geocoded_json_path) as f:
            data = json.load(f)

        rows = []
        seen = set()
        for r in data.get("routes", []):
            for d in r.get("directions", []):
                for s in d.get("stops", []):
                    nm = s.get("stop_name_normalized", "")
                    if nm not in seen and s.get("lat") and s.get("lon"):
                        rows.append({
                            "stop_name": nm,
                            "route_id": r["route_id"],
                            "lat": s["lat"],
                            "lon": s["lon"],
                            "geometry": Point(s["lon"], s["lat"])
                        })
                        seen.add(nm)

        if not rows:
            print("[HALTE] Geocoded JSON has no valid lat/lon")
            return gpd.GeoDataFrame(
                columns=["geometry", "stop_name", "route_id", "city"],
                geometry="geometry", crs="EPSG:32749"
            )

        halte_gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(epsg=32749)
    else:
        halte_gdf = gpd.GeoDataFrame(
            columns=["geometry", "stop_name", "route_id", "city"],
            geometry="geometry", crs="EPSG:32749"
        )

    halte_gdf["city"] = city_name

    if save:
        out = os.path.join(RAW_DIR, "halte", city_name)
        os.makedirs(out, exist_ok=True)
        halte_gdf.to_parquet(os.path.join(out, "halte.parquet"))
        print(f"[HALTE] {len(halte_gdf)} halte → saved")

    return halte_gdf


def _aggregate_population(city_name: str,
                           raster_path: str,
                           grid_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    try:
        import rasterio
        from rasterio.mask import mask as rio_mask
        from shapely.geometry import mapping
    except ImportError:
        print("[POP] pip install rasterio")
        grid_gdf = grid_gdf.copy()
        grid_gdf["population"] = 0.0
        return grid_gdf

    grid_4326 = grid_gdf.to_crs(epsg=4326)
    pop_values = []

    with rasterio.open(raster_path) as src:
        for _, row in tqdm(grid_4326.iterrows(),
                           total=len(grid_4326),
                           desc="[POP]"):
            try:
                out_img, _ = rio_mask(src, [mapping(row.geometry)], crop=True)
                data = out_img[0]
                nodata = src.nodata
                if nodata is not None:
                    data = data[data != nodata]
                pop_values.append(float(np.nansum(data)))
            except Exception:
                pop_values.append(0.0)

    grid_gdf = grid_gdf.copy()
    grid_gdf["population"] = pop_values
    print(f"[POP] Total population aggregated: {sum(pop_values):,.0f}")
    return grid_gdf