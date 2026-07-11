import os
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
from math import radians, cos, sin, sqrt, atan2
from penjurubus.data_paths import resolve_path

from shapely.geometry import Point, LineString
from typing import List, Tuple, Dict, Any
import random

GRID_SIZE = 500  

def load_candidates():
    pred_path = os.environ.get("TEGAL_PRED_PATH", "data/predictions_v4_search/tegal/predictions_v4_search.parquet")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(pred_path)
    df = pd.read_parquet(pred_path)
    candidates = df[df["pred_label"] == 1].copy()
    candidates["geometry"] = candidates.apply(
        lambda row: Point(row["centroid_x"], row["centroid_y"]), axis=1
    )
    gdf = gpd.GeoDataFrame(candidates, geometry="geometry", crs="EPSG:4326")
    return gdf

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0e3 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def nearest_node_from_point(G, point):
    lat, lon = point.y, point.x
    return ox.nearest_nodes(G, lon, lat)

def obj_func(halte_points: List[Tuple[float, float]], G: nx.Graph, df_pop=None, alpha=1.0, beta=1.0) -> Dict[str, float]:
    """
    :param halte_points: list (lat, lon) dari halte
    :param G: jaringan jalan OSM (networkx)
    :param df_pop: grid populasi (opsional, kolom population)
    :returns: dict dengan cakupan, panjang rute, biaya
    """
    if len(halte_points) == 0:
        return {"coverage": 0.0, "route_km": 0.0, "n_transfers": 0, "cost": np.inf}
    
    coverage = 0.0
    route_length_m = 0.0
    n_transfers = 0

    if df_pop is not None:
        for _, row in df_pop.iterrows():
            g_lat, g_lon = row.get("centroid_y", None), row.get("centroid_x", None)
            if g_lat is None or g_lon is None:
                continue
            dists = [haversine(g_lat, g_lon, h_lat, h_lon) for h_lat, h_lon in halte_points]
            min_dist = min(dists) if dists else np.inf
            if min_dist <= 500:
                coverage += row.get("population", 0.0)

    node_list = [nearest_node_from_point(G, Point(lon, lat)) for lat, lon in halte_points]
    if len(node_list) > 1:
        for i in range(len(node_list) - 1):
            try:
                l = nx.shortest_path_length(G, node_list[i], node_list[i + 1], weight="length")
                route_length_m += l
            except nx.NetworkXNoPath:
                continue
        try:
            n_transfers = len(node_list) - 1
        except:
            pass
    else:
        n_transfers = 0

    route_km = route_length_m / 1000.0
    cost = -coverage + alpha * route_km + beta * n_transfers
    return {
        "coverage": float(coverage),
        "route_km": float(route_km),
        "n_transfers": n_transfers,
        "cost": float(cost)
    }

class IPSO_GA:
    def __init__(
        self,
        candidates: gpd.GeoDataFrame,
        G: nx.Graph,
        df_pop: pd.DataFrame,
        n_particles=20,
        max_iter=50,
        omega=0.7,
        c1=1.49,
        c2=1.49,
        alpha=1.0,
        beta=1.0,
    ):
        self.candidates = candidates
        self.G = G
        self.df_pop = df_pop
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.alpha = alpha
        self.beta = beta
        self.particles = [random.sample(range(len(candidates)), random.randint(3, 8)) for _ in range(n_particles)]
        self.velocities = [[0] * len(ind) for ind in self.particles]  # dummy untuk simplifikasi
        self.p_best = self.particles.copy()
        self.p_cost = [self.evaluate(p) for p in self.particles]
        g_best_idx = np.argmin([v["cost"] for v in self.p_cost])
        self.g_best = self.p_best[g_best_idx].copy()
        self.g_cost = self.p_cost[g_best_idx]

    def evaluate(self, particle: List[int]):
        pts = [self.candidates.iloc[i]["geometry"] for i in particle]
        halte_points = [(p.y, p.x) for p in pts]  # lat, lon
        return obj_func(halte_points, self.G, self.df_pop, self.alpha, self.beta)

    def update_velocity(self, p, idx, it):
        if random.random() < 0.2:
            n = len(p)
            if n > 2:
                crossover_point = random.randint(1, n - 1)
                p = list(set(p[:crossover_point] + self.g_best[crossover_point:]))
        if random.random() < 0.1:
            if len(p) < 10:
                extra = random.choice(range(len(self.candidates)))
                p = list(set(p + [extra]))
        return p

    def run(self) -> Tuple[List[int], Dict[str, float]]:
        for it in range(self.max_iter):
            costs = []
            for i in range(self.n_particles):
                self.particles[i] = self.update_velocity(self.particles[i], i, it)
                cost = self.evaluate(self.particles[i])
                if cost["cost"] < self.p_cost[i]["cost"]:
                    self.p_best[i] = self.particles[i].copy()
                    self.p_cost[i] = cost
                costs.append(cost)

            best_idx = np.argmin([v["cost"] for v in costs])
            if costs[best_idx]["cost"] < self.g_cost["cost"]:
                self.g_best = self.particles[best_idx].copy()
                self.g_cost = costs[best_idx]

        return self.g_best, self.g_cost

def refine_halte_location_in_grid(grid_gdf: pd.Series, G: nx.Graph) -> Tuple[float, float]:
    lon = grid_gdf.get("centroid_x", None)
    lat = grid_gdf.get("centroid_y", None)
    if lon is None or lat is None:
        try:
            center = grid_gdf.geometry.centroid
            lon, lat = center.x, center.y
        except:
            return (0, 0)  # error default
    try:
        node = nearest_node_from_point(G, Point(lon, lat))
        return G.nodes[node]["y"], G.nodes[node]["x"]
    except:
        return lat, lon

def run_ipso_ga(output_dir: str = "output_ipso_ga") -> Dict[str, Any]:
    candidates = load_candidates()
    west, south, east, north = 109.09, -6.90, 109.14, -6.86
    print("Downloading OSM graph...")
    G = ox.graph_from_bbox(north, south, east, west, network_type="drive")

    df_pop = candidates[["grid_id", "population", "geometry"]].copy()

    ipso = IPSO_GA(
        candidates=candidates,
        G=G,
        df_pop=df_pop,
        n_particles=15,
        max_iter=20,
        omega=0.6,
        c1=1.2,
        c2=1.2,
    )
    best_idx, best_cost = ipso.run()
    best_grids = candidates.iloc[best_idx].reset_index(drop=True)

    halte_ideal = []
    for _, row in best_grids.iterrows():
        lat, lon = refine_halte_location_in_grid(row, G)
        halte_ideal.append({"lat": lat, "lon": lon, "grid_id": row["grid_id"]})

    os.makedirs(output_dir, exist_ok=True)
    fp_halte = os.path.join(output_dir, "ipsoga_halte_ideal.geojson")
    gpd.GeoDataFrame(
        halte_ideal,
        geometry=[Point(r["lon"], r["lat"]) for r in halte_ideal],
        crs="EPSG:4326"
    ).to_file(fp_halte, driver="GeoJSON")

    print(f"IPSO‑GA done: {len(best_grids)} grids selected; halte_ideal: {len(halte_ideal)} halte di jalan.")
    print(f"Best cost: {best_cost}")
    return {
        "best_grids": best_grids.to_dict("records"),
        "halte_ideal": halte_ideal,
        "metrics": best_cost,
        "halte_file": fp_halte
    }