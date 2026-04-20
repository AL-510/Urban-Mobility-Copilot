"""Build the multimodal transport graph combining road, transit, and walking networks."""

import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# Edge type constants
EDGE_ROAD = 0
EDGE_TRANSIT = 1
EDGE_WALK = 2
EDGE_TRANSFER = 3


class TransportGraph:
    """Unified multimodal transport graph."""

    def __init__(self):
        self.G = nx.DiGraph()
        self.node_id_map: dict[str, int] = {}
        self.reverse_map: dict[int, str] = {}
        self._next_id = 0

    def _get_or_create_node(
        self, key: str, lat: float, lon: float, node_type: str, **attrs
    ) -> int:
        if key in self.node_id_map:
            return self.node_id_map[key]
        nid = self._next_id
        self._next_id += 1
        self.node_id_map[key] = nid
        self.reverse_map[nid] = key
        self.G.add_node(nid, lat=lat, lon=lon, node_type=node_type, key=key, **attrs)
        return nid

    def add_road_network(self, road_graph: nx.MultiDiGraph) -> None:
        """Add OSM road network nodes and edges."""
        logger.info("Adding road network to transport graph...")
        for node, data in road_graph.nodes(data=True):
            self._get_or_create_node(
                key=f"road_{node}",
                lat=data.get("y", 0),
                lon=data.get("x", 0),
                node_type="road",
            )

        edge_count = 0
        for u, v, data in road_graph.edges(data=True):
            src = self.node_id_map.get(f"road_{u}")
            dst = self.node_id_map.get(f"road_{v}")
            if src is not None and dst is not None and src != dst:
                length = float(data.get("length", 100))
                speed = float(data.get("speed_kph", 40))
                travel_time = float(data.get("travel_time", length / (speed / 3.6)))
                lanes_raw = data.get("lanes", "1")
                if isinstance(lanes_raw, list):
                    lanes_raw = lanes_raw[0]
                try:
                    lanes = int(lanes_raw)
                except (ValueError, TypeError):
                    lanes = 1
                highway = data.get("highway", "residential")
                if isinstance(highway, list):
                    highway = highway[0]

                if not self.G.has_edge(src, dst):
                    self.G.add_edge(
                        src, dst,
                        edge_type=EDGE_ROAD,
                        length_m=length,
                        speed_kph=speed,
                        base_travel_time_s=travel_time,
                        lanes=lanes,
                        highway_class=highway,
                        capacity=_road_capacity(highway, lanes),
                    )
                    edge_count += 1
        logger.info(f"Added {edge_count} road edges")

    def add_transit_network(self, gtfs_data, bounds: dict | None = None) -> None:
        """Add transit stops as nodes and stop-to-stop connections as edges.

        Optimized for large GTFS feeds:
        - Pre-filters stops within geographic bounds
        - Deduplicates stop-to-stop edges across trips using vectorized ops
        - Uses a trip_id -> route_id lookup dict instead of repeated df filtering
        """
        logger.info("Adding transit network to transport graph...")
        stops = gtfs_data.stops
        if stops.empty:
            logger.warning("No GTFS stops available, skipping transit network")
            return

        # Filter stops to geographic bounds if provided
        if bounds:
            stops = stops[
                (stops["stop_lat"] >= bounds["min_lat"])
                & (stops["stop_lat"] <= bounds["max_lat"])
                & (stops["stop_lon"] >= bounds["min_lon"])
                & (stops["stop_lon"] <= bounds["max_lon"])
            ].copy()
            logger.info(f"Filtered to {len(stops)} stops within bounds")

        if stops.empty:
            logger.warning("No stops within bounds")
            return

        valid_stop_ids = set(stops["stop_id"].values)

        # Add transit stop nodes
        for _, stop in stops.iterrows():
            self._get_or_create_node(
                key=f"transit_{stop['stop_id']}",
                lat=stop["stop_lat"],
                lon=stop["stop_lon"],
                node_type="transit",
                stop_name=stop.get("stop_name", ""),
            )
        logger.info(f"Added {len(stops)} transit stop nodes")

        stop_times = gtfs_data.stop_times
        trips = gtfs_data.trips
        if stop_times.empty or trips.empty:
            logger.warning("No stop_times/trips data, skipping transit edges")
            return

        # Build trip_id -> route_id lookup for O(1) access
        trip_route_map = dict(zip(trips["trip_id"], trips["route_id"]))

        # Filter stop_times to only include stops within our bounds
        stop_times = stop_times[stop_times["stop_id"].isin(valid_stop_ids)].copy()
        stop_times["stop_sequence"] = stop_times["stop_sequence"].astype(int)
        logger.info(f"Processing {len(stop_times)} stop_time records across "
                     f"{stop_times['trip_id'].nunique()} trips")

        # Sort by trip and sequence for consecutive pair extraction
        stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])

        # Vectorized consecutive pair extraction:
        # shift rows to get (from_stop, to_stop) for same trip
        st = stop_times[["trip_id", "stop_id", "departure_time", "arrival_time", "stop_sequence"]].copy()
        st_next = st.shift(-1)

        # Mask: same trip_id for consecutive rows
        same_trip = st["trip_id"] == st_next["trip_id"]

        pairs = pd.DataFrame({
            "from_stop": st.loc[same_trip, "stop_id"].values,
            "to_stop": st_next.loc[same_trip, "stop_id"].values,
            "dep_time": st.loc[same_trip, "departure_time"].values,
            "arr_time": st_next.loc[same_trip, "arrival_time"].values,
            "trip_id": st.loc[same_trip, "trip_id"].values,
        })

        # Parse travel times
        pairs["dep_s"] = pairs["dep_time"].apply(lambda x: _parse_gtfs_time(str(x)))
        pairs["arr_s"] = pairs["arr_time"].apply(lambda x: _parse_gtfs_time(str(x)))
        pairs["travel_time_s"] = (pairs["arr_s"] - pairs["dep_s"]).clip(lower=30)

        # Deduplicate: keep median travel time per unique (from_stop, to_stop) pair
        edge_agg = pairs.groupby(["from_stop", "to_stop"]).agg(
            travel_time_s=("travel_time_s", "median"),
            trip_id=("trip_id", "first"),
            trip_count=("trip_id", "count"),
        ).reset_index()

        logger.info(f"Deduplicated to {len(edge_agg)} unique stop-to-stop edges")

        edge_count = 0
        for _, row in edge_agg.iterrows():
            src = self.node_id_map.get(f"transit_{row['from_stop']}")
            dst = self.node_id_map.get(f"transit_{row['to_stop']}")
            if src is not None and dst is not None and src != dst:
                route_id = trip_route_map.get(row["trip_id"], "unknown")
                # Estimate frequency from trip count (trips per day / service hours)
                freq_min = max(60 / max(row["trip_count"] / 18, 1), 2)

                if not self.G.has_edge(src, dst):
                    self.G.add_edge(
                        src, dst,
                        edge_type=EDGE_TRANSIT,
                        base_travel_time_s=float(row["travel_time_s"]),
                        route_id=route_id,
                        length_m=_haversine_m(
                            self.G.nodes[src]["lat"], self.G.nodes[src]["lon"],
                            self.G.nodes[dst]["lat"], self.G.nodes[dst]["lon"],
                        ),
                        frequency_min=round(freq_min, 1),
                    )
                    edge_count += 1

        logger.info(f"Added {edge_count} transit edges")

    def add_transfer_edges(self, max_distance_m: float = 300) -> None:
        """Add walk transfer edges between nearby road and transit nodes.

        Uses scipy KDTree for O(n log n) spatial lookups instead of O(n*m) brute force.
        """
        from scipy.spatial import cKDTree

        logger.info("Adding transfer edges (KDTree-accelerated)...")
        transit_nodes = [
            (n, d) for n, d in self.G.nodes(data=True) if d["node_type"] == "transit"
        ]
        road_nodes = [
            (n, d) for n, d in self.G.nodes(data=True) if d["node_type"] == "road"
        ]

        if not transit_nodes or not road_nodes:
            logger.warning("Missing transit or road nodes, skipping transfers")
            return

        logger.info(f"Building KDTree: {len(transit_nodes)} transit, {len(road_nodes)} road nodes")

        # Convert to approximate meters using lat/lon scaling at graph center latitude
        center_lat = np.mean([d["lat"] for _, d in transit_nodes]) if transit_nodes else 45.52
        cos_lat = np.cos(np.radians(center_lat))
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * cos_lat

        # Build KDTree on road nodes in approximate meter coordinates
        road_ids = [n for n, _ in road_nodes]
        road_coords = np.array([
            [d["lat"] * m_per_deg_lat, d["lon"] * m_per_deg_lon]
            for _, d in road_nodes
        ])
        tree = cKDTree(road_coords)

        # Query for each transit node
        edge_count = 0
        for tn, td in transit_nodes:
            query_point = [td["lat"] * m_per_deg_lat, td["lon"] * m_per_deg_lon]
            nearby_indices = tree.query_ball_point(query_point, max_distance_m)

            for idx in nearby_indices:
                rn = road_ids[idx]
                rd = self.G.nodes[rn]
                dist = _haversine_m(td["lat"], td["lon"], rd["lat"], rd["lon"])
                if dist <= max_distance_m:
                    walk_time = dist / 1.2  # ~4.3 km/h walking
                    for src, dst in [(tn, rn), (rn, tn)]:
                        if not self.G.has_edge(src, dst):
                            self.G.add_edge(
                                src, dst,
                                edge_type=EDGE_TRANSFER,
                                base_travel_time_s=walk_time,
                                length_m=dist,
                            )
                            edge_count += 1
        logger.info(f"Added {edge_count} transfer edges")

    @property
    def num_nodes(self) -> int:
        return self.G.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.G.number_of_edges()

    def get_edge_index(self) -> np.ndarray:
        """Return edge index as (2, E) numpy array."""
        edges = list(self.G.edges())
        if not edges:
            return np.zeros((2, 0), dtype=np.int64)
        return np.array(edges, dtype=np.int64).T

    def get_node_positions(self) -> np.ndarray:
        """Return (N, 2) array of [lat, lon]."""
        positions = np.zeros((self.num_nodes, 2))
        for n, d in self.G.nodes(data=True):
            positions[n] = [d["lat"], d["lon"]]
        return positions

    def nearest_node(self, lat: float, lon: float) -> int:
        """Find nearest node to given coordinates."""
        positions = self.get_node_positions()
        dists = np.sqrt(
            (positions[:, 0] - lat) ** 2 + (positions[:, 1] - lon) ** 2
        )
        return int(np.argmin(dists))

    def nearest_node_within(self, lat: float, lon: float, max_km: float = 2.0) -> int | None:
        """Find nearest node within max_km, or None if too far."""
        positions = self.get_node_positions()
        if positions.shape[0] == 0:
            return None
        # Approximate degree distance
        dists_deg = np.sqrt(
            (positions[:, 0] - lat) ** 2 + (positions[:, 1] - lon) ** 2
        )
        nearest = int(np.argmin(dists_deg))
        # Convert to approximate km
        dist_km = dists_deg[nearest] * 111.0
        if dist_km > max_km:
            return None
        return nearest

    def get_bounds(self) -> dict:
        """Return geographic bounding box of the graph."""
        positions = self.get_node_positions()
        if positions.shape[0] == 0:
            return {"min_lat": 0, "max_lat": 0, "min_lon": 0, "max_lon": 0}
        return {
            "min_lat": float(positions[:, 0].min()),
            "max_lat": float(positions[:, 0].max()),
            "min_lon": float(positions[:, 1].min()),
            "max_lon": float(positions[:, 1].max()),
        }

    def extract_subgraph(self, center_lat: float, center_lon: float,
                         radius_km: float = 3.0) -> "TransportGraph":
        """Extract a spatially bounded subgraph around a center point.

        Returns a new TransportGraph with contiguous node IDs,
        preserving all edge attributes. Essential for making large
        real-world graphs trainable on CPU.
        """
        logger.info(f"Extracting subgraph: center=({center_lat}, {center_lon}), "
                     f"radius={radius_km}km from {self.num_nodes} nodes")

        radius_deg = radius_km / 111.0  # rough conversion

        # Find nodes within radius
        keep_nodes = []
        for n, d in self.G.nodes(data=True):
            dist = np.sqrt((d["lat"] - center_lat) ** 2 +
                           (d["lon"] - center_lon) ** 2)
            if dist <= radius_deg:
                keep_nodes.append(n)

        logger.info(f"Found {len(keep_nodes)} nodes within {radius_km}km radius")

        # Build new graph with contiguous IDs
        sub = TransportGraph()
        old_to_new: dict[int, int] = {}

        for old_id in keep_nodes:
            data = dict(self.G.nodes[old_id])
            old_key = data.pop("key", f"node_{old_id}")
            new_id = sub._get_or_create_node(
                key=old_key,
                lat=data.pop("lat"),
                lon=data.pop("lon"),
                node_type=data.pop("node_type", "road"),
                **data,
            )
            old_to_new[old_id] = new_id

        keep_set = set(keep_nodes)
        edge_count = 0
        for u, v, data in self.G.edges(data=True):
            if u in keep_set and v in keep_set:
                new_u = old_to_new[u]
                new_v = old_to_new[v]
                if new_u != new_v and not sub.G.has_edge(new_u, new_v):
                    sub.G.add_edge(new_u, new_v, **data)
                    edge_count += 1

        logger.info(f"Subgraph: {sub.num_nodes} nodes, {edge_count} edges")
        return sub

    def save(self, path: Path | None = None) -> Path:
        import json
        import pickle

        settings = get_settings()
        out = path or (settings.processed_dir / "transport_graph.pkl")
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "wb") as f:
            pickle.dump(self.G, f, protocol=pickle.HIGHEST_PROTOCOL)

        meta_path = out.with_suffix(".meta.json")
        meta = {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "node_id_map": self.node_id_map,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        logger.info(f"Saved transport graph to {out}")
        return out

    @classmethod
    def load(cls, path: Path | None = None) -> "TransportGraph":
        import json
        import pickle

        settings = get_settings()
        path = path or (settings.processed_dir / "transport_graph.pkl")
        tg = cls()

        with open(path, "rb") as f:
            tg.G = pickle.load(f)

        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            tg.node_id_map = meta.get("node_id_map", {})
            tg.reverse_map = {v: k for k, v in tg.node_id_map.items()}
            tg._next_id = max(tg.reverse_map.keys()) + 1 if tg.reverse_map else 0
        return tg


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in meters."""
    R = 6371000
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _parse_gtfs_time(time_str: str) -> float:
    """Parse GTFS time (HH:MM:SS, may exceed 24h) to seconds since midnight."""
    parts = time_str.strip().split(":")
    if len(parts) != 3:
        return 0.0
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def _road_capacity(highway_class: str, lanes: int) -> float:
    """Estimate road segment capacity (vehicles/hour)."""
    base_capacity = {
        "motorway": 2000, "trunk": 1600, "primary": 1200, "secondary": 900,
        "tertiary": 600, "residential": 300, "unclassified": 300,
    }
    cap = base_capacity.get(highway_class, 300) if isinstance(highway_class, str) else 300
    return cap * max(lanes, 1)
