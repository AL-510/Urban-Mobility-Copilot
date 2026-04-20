"""Build node, edge, and temporal features for the spatio-temporal graph model."""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from src.graph.builder import TransportGraph, EDGE_ROAD, EDGE_TRANSIT, EDGE_TRANSFER

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Constructs feature tensors for ST-GAT model.

    Caches all static computations (node positions, static features, edge features,
    nearby-node mask) so they are computed exactly once per engine instance.
    """

    def __init__(self, graph: TransportGraph):
        self.graph = graph
        self.G = graph.G
        self.num_nodes = graph.num_nodes
        self.num_edges = graph.num_edges

        # Lazy caches (computed on first access, then reused)
        self._cached_static: np.ndarray | None = None
        self._cached_edges: np.ndarray | None = None
        self._cached_positions: np.ndarray | None = None
        self._nearby_mask: np.ndarray | None = None      # (N, N) float32
        self._nearby_counts: np.ndarray | None = None     # (N,) float32
        self._road_mask: np.ndarray | None = None         # (N,) bool

    # ── cached helpers ──────────────────────────────────────────────

    def _get_positions(self) -> np.ndarray:
        """Return (N, 2) positions array, cached."""
        if self._cached_positions is None:
            self._cached_positions = self.graph.get_node_positions()
        return self._cached_positions

    def _get_nearby_mask(self, radius_deg: float = 0.01):
        """Pre-compute (N, N) proximity mask and per-node counts."""
        if self._nearby_mask is None:
            pos = self._get_positions()
            N = pos.shape[0]
            lat = pos[:, 0]
            lon = pos[:, 1]
            # Vectorized pairwise distance
            lat_diff = lat[:, np.newaxis] - lat[np.newaxis, :]  # (N, N)
            lon_diff = lon[:, np.newaxis] - lon[np.newaxis, :]
            dists = np.sqrt(lat_diff ** 2 + lon_diff ** 2)
            self._nearby_mask = (dists < radius_deg).astype(np.float32)
            self._nearby_counts = self._nearby_mask.sum(axis=1).clip(1)
        return self._nearby_mask, self._nearby_counts

    def _get_road_mask(self) -> np.ndarray:
        """Boolean mask of which nodes are road type, cached."""
        if self._road_mask is None:
            self._road_mask = np.array([
                self.G.nodes[n].get("node_type") == "road"
                for n in range(self.num_nodes)
            ], dtype=bool)
        return self._road_mask

    # ── static features (computed once) ─────────────────────────────

    def build_static_node_features(self) -> np.ndarray:
        """Build static node features (N, F_static).

        Features per node:
        - lat (normalized)
        - lon (normalized)
        - node_type one-hot (road=0, transit=1, walk=2) -> 3 dims
        - degree_in (normalized)
        - degree_out (normalized)
        - betweenness_proxy (degree product, normalized)
        Total: 8 features
        """
        if self._cached_static is not None:
            return self._cached_static

        N = self.num_nodes
        features = np.zeros((N, 8), dtype=np.float32)

        # Lat/lon (min-max normalized)
        positions = self._get_positions()
        if N > 0:
            lat_min, lat_max = positions[:, 0].min(), positions[:, 0].max()
            lon_min, lon_max = positions[:, 1].min(), positions[:, 1].max()
            lat_range = max(lat_max - lat_min, 1e-6)
            lon_range = max(lon_max - lon_min, 1e-6)
            features[:, 0] = (positions[:, 0] - lat_min) / lat_range
            features[:, 1] = (positions[:, 1] - lon_min) / lon_range

        # Node type one-hot
        type_map = {"road": 0, "transit": 1, "walk": 2}
        for n, data in self.G.nodes(data=True):
            idx = type_map.get(data.get("node_type", "road"), 0)
            features[n, 2 + idx] = 1.0

        # Degree features
        in_deg = np.array([self.G.in_degree(n) for n in range(N)], dtype=np.float32)
        out_deg = np.array([self.G.out_degree(n) for n in range(N)], dtype=np.float32)
        max_deg = max(in_deg.max(), 1)
        features[:, 5] = in_deg / max_deg
        features[:, 6] = out_deg / max_deg
        features[:, 7] = (in_deg * out_deg) / (max_deg ** 2)

        self._cached_static = features
        return features

    def build_temporal_node_features(
        self,
        timestamp: datetime,
        weather_features: dict | None = None,
        incidents: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Build time-varying node features (N, F_temporal).

        Features per node:
        - hour_sin, hour_cos (2)
        - day_of_week_sin, day_of_week_cos (2)
        - is_weekend (1)
        - is_rush_hour (1)
        - weather_severity (1)
        - precip_mm (1)
        - wind_speed_norm (1)
        - visibility_norm (1)
        - incident_active (1)
        - incident_severity_encoded (1)
        - nearby_disruption_density (1)
        - congestion_proxy (1)
        - is_night (1)
        - month_sin (1)
        Total: 16 features
        """
        N = self.num_nodes
        features = np.zeros((N, 16), dtype=np.float32)

        # Temporal encoding
        hour = timestamp.hour + timestamp.minute / 60
        dow = timestamp.weekday()
        month = timestamp.month

        features[:, 0] = np.sin(2 * np.pi * hour / 24)
        features[:, 1] = np.cos(2 * np.pi * hour / 24)
        features[:, 2] = np.sin(2 * np.pi * dow / 7)
        features[:, 3] = np.cos(2 * np.pi * dow / 7)
        features[:, 4] = float(dow >= 5)
        features[:, 5] = float(7 <= hour <= 9 or 16 <= hour <= 19)
        features[:, 14] = float(hour < 6 or hour > 22)
        features[:, 15] = np.sin(2 * np.pi * month / 12)

        # Weather features (broadcast to all nodes)
        if weather_features:
            def _wx(key, default=0.0):
                v = weather_features.get(key, default)
                return default if v is None else float(v)

            features[:, 6] = _wx("weather_severity", 0.0)
            features[:, 7] = min(_wx("precip_mm", 0.0) / 20, 1.0)
            features[:, 8] = min(_wx("wind_speed_kmh", 0.0) / 80, 1.0)
            features[:, 9] = _wx("visibility_m", 10000) / 10000

        # Incident features
        if incidents is not None and not incidents.empty:
            positions = self._get_positions()
            incident_radius_deg = 0.005  # ~500m

            # Use pre-parsed datetime columns if available (training fast path)
            if "_start_dt" in incidents.columns:
                active = incidents[
                    (incidents["_start_dt"] <= timestamp)
                    & (incidents["_end_dt"] >= timestamp)
                ]
            elif "start_time" in incidents.columns:
                active = incidents[
                    (incidents["start_time"] <= timestamp.isoformat())
                    & (incidents["end_time"] >= timestamp.isoformat())
                ]
            else:
                active = pd.DataFrame()

            severity_map = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}

            for _, inc in active.iterrows():
                dists = np.sqrt(
                    (positions[:, 0] - inc["lat"]) ** 2
                    + (positions[:, 1] - inc["lon"]) ** 2
                )
                affected = dists < incident_radius_deg
                features[affected, 10] = 1.0
                sev = severity_map.get(inc.get("severity", "medium"), 0.5)
                features[affected, 11] = np.maximum(features[affected, 11], sev)

            # Disruption density — vectorized via pre-computed nearby mask
            nearby_mask, nearby_counts = self._get_nearby_mask()
            features[:, 12] = (nearby_mask @ features[:, 10:11]).ravel() / nearby_counts

        # Congestion proxy — vectorized via road mask
        road_mask = self._get_road_mask()
        rush_factor = features[:, 5]  # is_rush_hour (broadcast scalar)
        features[road_mask, 13] = rush_factor[road_mask]

        return features

    def build_edge_features(self) -> np.ndarray:
        """Build edge features (E, F_edge).

        Features per edge:
        - edge_type one-hot (road, transit, transfer, walk) -> 4 dims
        - length_m (normalized, log-scale)
        - base_travel_time_s (normalized, log-scale)
        - speed_kph (normalized)
        - capacity (normalized, log-scale)
        - lanes (normalized)
        - is_highway (1)
        - frequency_min_inv (1)
        - reliability_base (1)
        Total: 12 features
        """
        if self._cached_edges is not None:
            return self._cached_edges

        edges = list(self.G.edges(data=True))
        E = len(edges)
        features = np.zeros((E, 12), dtype=np.float32)

        for i, (u, v, data) in enumerate(edges):
            etype = data.get("edge_type", EDGE_ROAD)
            if etype < 4:
                features[i, etype] = 1.0

            length = data.get("length_m", 100)
            features[i, 4] = np.log1p(length) / 10

            travel_time = data.get("base_travel_time_s", 60)
            features[i, 5] = np.log1p(travel_time) / 10

            speed = data.get("speed_kph", 30)
            features[i, 6] = min(speed / 120, 1.0)

            capacity = data.get("capacity", 300)
            features[i, 7] = np.log1p(capacity) / 10

            lanes = data.get("lanes", 1)
            features[i, 8] = min(lanes / 6, 1.0)

            highway = data.get("highway_class", "residential")
            features[i, 9] = float(highway in ["motorway", "trunk", "primary"])

            freq = data.get("frequency_min", 0)
            features[i, 10] = 1.0 / max(freq, 1) if freq > 0 else 0

            # Base reliability (transit slightly less reliable than road)
            features[i, 11] = 0.95 if etype == EDGE_ROAD else 0.90

        self._cached_edges = features
        return features

    def build_snapshot(
        self,
        timestamp: datetime,
        weather_features: dict | None = None,
        incidents: pd.DataFrame | None = None,
    ) -> dict:
        """Build complete feature snapshot for a timestamp.

        Returns dict with:
        - node_features: (N, F_node) tensor
        - edge_features: (E, F_edge) tensor
        - edge_index: (2, E) tensor
        - timestamp: datetime
        """
        static = self.build_static_node_features()
        temporal = self.build_temporal_node_features(timestamp, weather_features, incidents)
        node_features = np.concatenate([static, temporal], axis=1)

        edge_features = self.build_edge_features()
        edge_index = self.graph.get_edge_index()

        return {
            "node_features": torch.FloatTensor(node_features),
            "edge_features": torch.FloatTensor(edge_features),
            "edge_index": torch.LongTensor(edge_index),
            "timestamp": timestamp,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
        }

    def build_temporal_sequence(
        self,
        timestamps: list[datetime],
        weather_series: list[dict] | None = None,
        incidents: pd.DataFrame | None = None,
    ) -> dict:
        """Build a sequence of snapshots for temporal modeling.

        Optimized: builds static features and edge features ONCE,
        then only recomputes temporal features per timestep.

        Returns dict with:
        - node_features: (T, N, F_node) tensor
        - edge_features: (E, F_edge) tensor (static)
        - edge_index: (2, E) tensor
        """
        static = self.build_static_node_features()  # cached after first call

        node_features_list = []
        for i, ts in enumerate(timestamps):
            wx = weather_series[i] if weather_series and i < len(weather_series) else None
            temporal = self.build_temporal_node_features(ts, wx, incidents)
            combined = np.concatenate([static, temporal], axis=1)
            node_features_list.append(torch.FloatTensor(combined))

        node_seq = torch.stack(node_features_list)
        return {
            "node_features": node_seq,  # (T, N, F)
            "edge_features": torch.FloatTensor(self.build_edge_features()),  # cached
            "edge_index": torch.LongTensor(self.graph.get_edge_index()),
        }
