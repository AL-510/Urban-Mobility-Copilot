"""Generate candidate commute routes using the multimodal transport graph."""

import logging
from typing import Any

import networkx as nx
import numpy as np

from src.graph.builder import TransportGraph, EDGE_ROAD, EDGE_TRANSIT, EDGE_TRANSFER, EDGE_WALK

logger = logging.getLogger(__name__)

MAX_CANDIDATES = 5


class RouteGenerator:
    """Generates candidate multimodal routes between origin and destination."""

    def __init__(self, graph: TransportGraph):
        self.graph = graph
        self.G = graph.G

    def generate_candidates(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        max_routes: int = MAX_CANDIDATES,
    ) -> list[dict]:
        """Generate diverse candidate routes.

        Strategy:
        1. Shortest path by travel time
        2. Shortest path by distance
        3. Transit-preferred path
        4. Road-only path
        5. Penalty-perturbed path for diversity
        """
        src = self.graph.nearest_node(origin_lat, origin_lon)
        dst = self.graph.nearest_node(dest_lat, dest_lon)

        if src == dst:
            return [self._make_walk_route(src, dst, origin_lat, origin_lon, dest_lat, dest_lon)]

        candidates = []

        # Route 1: Shortest by travel time (all modes)
        route = self._shortest_path(src, dst, weight="base_travel_time_s")
        if route:
            candidates.append(self._build_route_info(route, "Fastest Route", "fastest"))

        # Route 2: Shortest by distance
        route = self._shortest_path(src, dst, weight="length_m")
        if route and not self._is_duplicate(route, candidates):
            candidates.append(self._build_route_info(route, "Shortest Distance", "shortest"))

        # Route 3: Transit-preferred
        route = self._transit_preferred_path(src, dst)
        if route and not self._is_duplicate(route, candidates):
            candidates.append(self._build_route_info(route, "Transit Route", "transit"))

        # Route 4: Road-only
        route = self._road_only_path(src, dst)
        if route and not self._is_duplicate(route, candidates):
            candidates.append(self._build_route_info(route, "Driving Route", "drive"))

        # Route 5+: Perturbed paths for diversity
        for attempt in range(3):
            route = self._perturbed_path(src, dst, penalty_factor=1.5 + attempt * 0.5)
            if route and not self._is_duplicate(route, candidates):
                candidates.append(
                    self._build_route_info(route, f"Alternative {attempt + 1}", "alternative")
                )
            if len(candidates) >= max_routes:
                break

        # Ensure at least one route
        if not candidates:
            candidates.append(
                self._make_walk_route(src, dst, origin_lat, origin_lon, dest_lat, dest_lon)
            )

        return candidates[:max_routes]

    def _shortest_path(
        self, src: int, dst: int, weight: str = "base_travel_time_s"
    ) -> list[int] | None:
        try:
            path = nx.shortest_path(self.G, src, dst, weight=weight)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _transit_preferred_path(self, src: int, dst: int) -> list[int] | None:
        """Prefer transit edges by reducing their weight."""
        G_copy = self.G.copy()
        for u, v, data in G_copy.edges(data=True):
            base = data.get("base_travel_time_s", 60)
            if data.get("edge_type") == EDGE_TRANSIT:
                G_copy[u][v]["modified_weight"] = base * 0.5  # Favor transit
            elif data.get("edge_type") == EDGE_TRANSFER:
                G_copy[u][v]["modified_weight"] = base * 0.8
            else:
                G_copy[u][v]["modified_weight"] = base * 1.5  # Penalize driving
        try:
            return nx.shortest_path(G_copy, src, dst, weight="modified_weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _road_only_path(self, src: int, dst: int) -> list[int] | None:
        """Only use road edges (no transit)."""
        road_edges = [
            (u, v) for u, v, d in self.G.edges(data=True)
            if d.get("edge_type") in (EDGE_ROAD, EDGE_TRANSFER)
        ]
        subgraph = self.G.edge_subgraph(road_edges)
        try:
            return nx.shortest_path(subgraph, src, dst, weight="base_travel_time_s")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _perturbed_path(
        self, src: int, dst: int, penalty_factor: float = 1.5
    ) -> list[int] | None:
        """Find path with random perturbation for diversity."""
        G_copy = self.G.copy()
        rng = np.random.default_rng()
        for u, v, data in G_copy.edges(data=True):
            base = data.get("base_travel_time_s", 60)
            noise = rng.uniform(0.8, penalty_factor)
            G_copy[u][v]["perturbed_weight"] = base * noise
        try:
            return nx.shortest_path(G_copy, src, dst, weight="perturbed_weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _build_route_info(self, path: list[int], name: str, strategy: str) -> dict:
        """Build route info from node path."""
        segments = []
        total_time = 0
        total_distance = 0
        modes_used = set()

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = self.G.get_edge_data(u, v, {})
            edge_type = edge_data.get("edge_type", EDGE_ROAD)
            travel_time = edge_data.get("base_travel_time_s", 60)
            length = edge_data.get("length_m", 100)

            mode = {EDGE_ROAD: "drive", EDGE_TRANSIT: "transit", EDGE_TRANSFER: "walk", EDGE_WALK: "walk"}.get(edge_type, "drive")
            modes_used.add(mode)

            segments.append({
                "from_node": u,
                "to_node": v,
                "mode": mode,
                "travel_time_s": travel_time,
                "length_m": length,
                "edge_type": edge_type,
                "route_id": edge_data.get("route_id", None),
            })
            total_time += travel_time
            total_distance += length

        # Build coordinate path
        coordinates = []
        for nid in path:
            node_data = self.G.nodes[nid]
            coordinates.append([node_data["lat"], node_data["lon"]])

        return {
            "name": name,
            "strategy": strategy,
            "path": path,
            "segments": segments,
            "coordinates": coordinates,
            "total_time_s": total_time,
            "total_distance_m": total_distance,
            "modes": list(modes_used),
            "num_transfers": sum(1 for s in segments if s["mode"] == "walk" and len(segments) > 1),
        }

    def _make_walk_route(
        self, src: int, dst: int,
        olat: float, olon: float, dlat: float, dlon: float,
    ) -> dict:
        from src.graph.builder import _haversine_m
        dist = _haversine_m(olat, olon, dlat, dlon)
        walk_time = dist / 1.2
        return {
            "name": "Walking Route",
            "strategy": "walk",
            "path": [src, dst],
            "segments": [{
                "from_node": src, "to_node": dst, "mode": "walk",
                "travel_time_s": walk_time, "length_m": dist, "edge_type": EDGE_WALK,
            }],
            "coordinates": [[olat, olon], [dlat, dlon]],
            "total_time_s": walk_time,
            "total_distance_m": dist,
            "modes": ["walk"],
            "num_transfers": 0,
        }

    def _is_duplicate(self, path: list[int], existing: list[dict], overlap_threshold: float = 0.8) -> bool:
        """Check if a path is too similar to existing candidates."""
        path_set = set(path)
        for candidate in existing:
            existing_set = set(candidate["path"])
            if not existing_set:
                continue
            overlap = len(path_set & existing_set) / max(len(path_set), len(existing_set))
            if overlap > overlap_threshold:
                return True
        return False
