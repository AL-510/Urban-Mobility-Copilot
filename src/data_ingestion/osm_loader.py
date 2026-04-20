"""Load road and walking network from OpenStreetMap using OSMnx."""

import logging
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def download_road_network(
    center_lat: float | None = None,
    center_lon: float | None = None,
    radius_km: float | None = None,
) -> nx.MultiDiGraph:
    """Download drivable road network from OSM centered on city."""
    settings = get_settings()
    lat = center_lat or settings.city_center_lat
    lon = center_lon or settings.city_center_lon
    dist = int((radius_km or settings.city_radius_km) * 1000)

    logger.info(f"Downloading road network: ({lat}, {lon}), radius={dist}m")
    G = ox.graph_from_point(
        (lat, lon),
        dist=dist,
        network_type="drive",
        simplify=True,
    )
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    logger.info(f"Road network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def download_walk_network(
    center_lat: float | None = None,
    center_lon: float | None = None,
    radius_km: float | None = None,
) -> nx.MultiDiGraph:
    """Download walking network from OSM."""
    settings = get_settings()
    lat = center_lat or settings.city_center_lat
    lon = center_lon or settings.city_center_lon
    dist = int((radius_km or settings.city_radius_km) * 1000)

    logger.info(f"Downloading walk network: ({lat}, {lon}), radius={dist}m")
    G = ox.graph_from_point(
        (lat, lon),
        dist=dist,
        network_type="walk",
        simplify=True,
    )
    logger.info(f"Walk network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def save_graph(G: nx.MultiDiGraph, name: str) -> Path:
    """Save graph to GraphML file."""
    settings = get_settings()
    out_dir = settings.processed_dir / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.graphml"
    ox.save_graphml(G, path)
    logger.info(f"Saved graph to {path}")
    return path


def load_graph(name: str) -> nx.MultiDiGraph:
    """Load graph from GraphML file."""
    settings = get_settings()
    path = settings.processed_dir / "graphs" / f"{name}.graphml"
    if not path.exists():
        raise FileNotFoundError(f"Graph not found: {path}")
    G = ox.load_graphml(path)
    return G


def graph_to_geodataframes(G: nx.MultiDiGraph) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Convert graph to node and edge GeoDataFrames."""
    nodes, edges = ox.graph_to_gdfs(G)
    return nodes, edges
