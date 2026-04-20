"""Preprocessing pipeline: spatial joins, time alignment, and window creation."""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.graph.builder import TransportGraph, _haversine_m

logger = logging.getLogger(__name__)


def spatial_join_incidents_to_nodes(
    graph: TransportGraph,
    incidents: pd.DataFrame,
    radius_m: float = 500,
) -> pd.DataFrame:
    """Assign each incident to the nearest graph node(s) within radius."""
    positions = graph.get_node_positions()
    records = []

    for _, inc in incidents.iterrows():
        inc_lat = inc.get("lat", 0)
        inc_lon = inc.get("lon", 0)

        for nid in range(graph.num_nodes):
            node_lat, node_lon = positions[nid]
            dist = _haversine_m(inc_lat, inc_lon, node_lat, node_lon)
            if dist <= radius_m:
                records.append({
                    "incident_id": inc.get("incident_id", ""),
                    "node_id": nid,
                    "distance_m": round(dist, 1),
                    "start_time": inc.get("start_time", ""),
                    "end_time": inc.get("end_time", ""),
                    "severity": inc.get("severity", ""),
                    "incident_type": inc.get("incident_type", ""),
                    "delay_factor": inc.get("delay_factor", 1.0),
                })

    df = pd.DataFrame(records)
    logger.info(f"Spatial join: {len(df)} incident-node pairs from {len(incidents)} incidents")
    return df


def align_weather_to_timestamps(
    weather: pd.DataFrame,
    timestamps: list[datetime],
) -> list[dict]:
    """Align weather features to a list of timestamps via nearest-hour matching."""
    if weather.empty:
        return [{}] * len(timestamps)

    weather = weather.copy()
    weather["_ts"] = pd.to_datetime(weather["timestamp"])
    results = []

    for ts in timestamps:
        diffs = abs(weather["_ts"] - ts)
        nearest = weather.loc[diffs.idxmin()]
        results.append(nearest.drop("_ts").to_dict())

    return results


def create_temporal_windows(
    timestamps: list[datetime],
    window_size: int = 12,
    step_minutes: int = 5,
) -> list[list[datetime]]:
    """Create sliding windows of timestamps for temporal modeling.

    Each window is a list of `window_size` timestamps spaced `step_minutes` apart,
    ending at the corresponding timestamp in the input list.
    """
    windows = []
    for ts in timestamps:
        window = [
            ts - timedelta(minutes=(window_size - 1 - i) * step_minutes)
            for i in range(window_size)
        ]
        windows.append(window)
    return windows


def compute_graph_segment_mapping(
    graph: TransportGraph,
    route_path: list[int],
) -> list[dict]:
    """Map a route path to graph edge segments with metadata."""
    segments = []
    for i in range(len(route_path) - 1):
        u, v = route_path[i], route_path[i + 1]
        edge_data = graph.G.get_edge_data(u, v, {})
        u_data = graph.G.nodes[u]
        v_data = graph.G.nodes[v]

        segments.append({
            "from_node": u,
            "to_node": v,
            "from_lat": u_data.get("lat", 0),
            "from_lon": u_data.get("lon", 0),
            "to_lat": v_data.get("lat", 0),
            "to_lon": v_data.get("lon", 0),
            "edge_type": edge_data.get("edge_type", 0),
            "length_m": edge_data.get("length_m", 0),
            "base_travel_time_s": edge_data.get("base_travel_time_s", 0),
        })

    return segments
