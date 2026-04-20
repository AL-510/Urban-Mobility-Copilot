"""Score and rank candidate routes using model predictions and user preferences.

Supports two scoring modes:
1. ML-enhanced: routes with coordinates sampled against the forecast graph
2. Base: time/distance-only scoring for routes outside forecast regions
"""

import logging
from datetime import datetime

import numpy as np

from src.inference.predictor import DisruptionPredictor

logger = logging.getLogger(__name__)

PREFERENCE_WEIGHTS = {
    "fastest": {"time": 0.6, "reliability": 0.15, "risk": 0.15, "comfort": 0.1},
    "cheapest": {"time": 0.2, "reliability": 0.2, "risk": 0.2, "comfort": 0.4},
    "least_risky": {"time": 0.15, "reliability": 0.35, "risk": 0.4, "comfort": 0.1},
    "balanced": {"time": 0.3, "reliability": 0.25, "risk": 0.25, "comfort": 0.2},
}

# Max number of coordinate points to sample for ML predictions
MAX_SAMPLE_POINTS = 20


class RouteScorer:
    """Scores candidate routes using disruption predictions."""

    def __init__(self, predictor: DisruptionPredictor):
        self.predictor = predictor

    def score_routes_with_coordinates(
        self,
        candidates: list[dict],
        timestamp: datetime,
        preference: str = "balanced",
        weather_features: dict | None = None,
        incidents=None,
        horizon_minutes: int = 30,
        graph=None,
    ) -> list[dict]:
        """Score OSRM routes by sampling coordinates and mapping to graph nodes.

        For each route:
        1. Sample points along the OSRM coordinate path
        2. Find nearest graph nodes for each sample point
        3. Get ML predictions for those nodes
        4. Aggregate to route-level metrics
        5. Compute multi-criteria score
        """
        if not candidates:
            return []

        # Collect graph nodes by sampling route coordinates
        all_nodes = set()
        route_node_maps = []  # Per-route list of sampled node IDs

        for c in candidates:
            coords = c.get("coordinates", [])
            route_nodes = []
            if graph and coords:
                sampled = self._sample_coordinates(coords, MAX_SAMPLE_POINTS)
                for lat, lon in sampled:
                    node_id = graph.nearest_node_within(lat, lon, max_km=2.0)
                    if node_id is not None:
                        route_nodes.append(node_id)
                        all_nodes.add(node_id)
            route_node_maps.append(route_nodes)

        # Get predictions for all unique nodes at once
        all_nodes = sorted(all_nodes)
        node_preds = {}
        if all_nodes:
            preds = self.predictor.predict_for_nodes(
                list(all_nodes), timestamp, weather_features, incidents, horizon_minutes
            )
            for p in preds:
                node_preds[p["node_id"]] = p

        # Score each route
        weights = PREFERENCE_WEIGHTS.get(preference, PREFERENCE_WEIGHTS["balanced"])
        scored = []

        for route, route_nodes in zip(candidates, route_node_maps):
            route_score = self._score_single_route(route, route_nodes, node_preds, weights)
            scored.append(route_score)

        # Rank by composite score (lower is better)
        scored.sort(key=lambda x: x["composite_score"])

        for i, route in enumerate(scored):
            route["rank"] = i + 1
            route["is_recommended"] = (i == 0)

        return scored

    def score_routes(
        self,
        candidates: list[dict],
        timestamp: datetime,
        preference: str = "balanced",
        weather_features: dict | None = None,
        incidents=None,
        horizon_minutes: int = 30,
    ) -> list[dict]:
        """Legacy scoring using path node IDs. Falls back gracefully for empty paths."""
        if not candidates:
            return []

        all_nodes = set()
        for c in candidates:
            all_nodes.update(c.get("path", []))
        all_nodes = sorted(all_nodes)

        node_preds = {}
        if all_nodes:
            preds = self.predictor.predict_for_nodes(
                list(all_nodes), timestamp, weather_features, incidents, horizon_minutes
            )
            for p in preds:
                node_preds[p["node_id"]] = p

        weights = PREFERENCE_WEIGHTS.get(preference, PREFERENCE_WEIGHTS["balanced"])
        scored = []

        for route in candidates:
            path = route.get("path", [])
            route_score = self._score_single_route(route, path, node_preds, weights)
            scored.append(route_score)

        scored.sort(key=lambda x: x["composite_score"])
        for i, route in enumerate(scored):
            route["rank"] = i + 1
            route["is_recommended"] = (i == 0)

        return scored

    def _score_single_route(
        self,
        route: dict,
        node_ids: list[int],
        node_preds: dict[int, dict],
        weights: dict[str, float],
    ) -> dict:
        """Score a single route using node-level predictions.

        node_ids can come from either the graph path or coordinate sampling.
        """
        if not node_ids:
            # No graph coverage — return base time/distance score
            return self._base_score(route, weights)

        # Aggregate node predictions along route
        disruption_probs = []
        delay_medians = []
        delay_q90s = []
        ratio_medians = []
        ratio_q90s = []

        for nid in node_ids:
            pred = node_preds.get(nid, self._default_pred())
            disruption_probs.append(pred["disruption_prob"])
            delay_medians.append(pred["delay_q50"])
            delay_q90s.append(pred["delay_q90"])
            ratio_medians.append(pred["ratio_q50"])
            ratio_q90s.append(pred["ratio_q90"])

        # Route-level aggregation
        max_disruption = max(disruption_probs)
        avg_disruption = np.mean(disruption_probs)
        route_disruption_prob = 0.6 * max_disruption + 0.4 * avg_disruption

        total_delay_median = sum(delay_medians)
        total_delay_q90 = sum(delay_q90s)

        avg_ratio_median = np.mean(ratio_medians)
        avg_ratio_q90 = np.mean(ratio_q90s)

        # Predicted total time
        base_time_s = route["total_time_s"]
        predicted_time_s = base_time_s * avg_ratio_median
        predicted_time_q90_s = base_time_s * avg_ratio_q90

        # Reliability: inverse of uncertainty spread
        time_spread = (predicted_time_q90_s - predicted_time_s) / max(predicted_time_s, 1)
        reliability = max(1.0 - time_spread, 0)

        # Comfort proxy
        transfers = route.get("num_transfers", 0)
        modes = route.get("modes", [])
        comfort = 1.0 - (transfers * 0.15) - (0.1 if "walk" in modes and len(modes) > 1 else 0)
        comfort = max(comfort, 0)

        # Normalize scores to [0, 1] where lower is better
        time_score = min(predicted_time_s / 7200, 1.0)
        risk_score = route_disruption_prob
        reliability_score = 1.0 - reliability
        comfort_score = 1.0 - comfort

        composite = (
            weights["time"] * time_score
            + weights["risk"] * risk_score
            + weights["reliability"] * reliability_score
            + weights["comfort"] * comfort_score
        )

        risk_factors = self._identify_risk_factors(
            node_ids, node_preds, route_disruption_prob, avg_ratio_median
        )

        return {
            **route,
            "predicted_time_s": round(predicted_time_s),
            "predicted_time_q90_s": round(predicted_time_q90_s),
            "disruption_prob": round(route_disruption_prob, 3),
            "total_delay_median_min": round(total_delay_median, 1),
            "total_delay_q90_min": round(total_delay_q90, 1),
            "reliability_score": round(reliability, 3),
            "comfort_score": round(comfort, 3),
            "time_score": round(1 - time_score, 3),
            "risk_score": round(1 - risk_score, 3),
            "composite_score": round(composite, 4),
            "risk_factors": risk_factors,
        }

    def _base_score(self, route: dict, weights: dict[str, float]) -> dict:
        """Base score when no ML predictions are available."""
        time_s = route.get("total_time_s", 0)
        time_score = min(time_s / 7200, 1.0)

        composite = (
            weights["time"] * time_score
            + weights["risk"] * 0.1  # low assumed risk
            + weights["reliability"] * 0.15
            + weights["comfort"] * 0.2
        )

        return {
            **route,
            "predicted_time_s": round(time_s),
            "predicted_time_q90_s": round(time_s * 1.2),
            "disruption_prob": 0.0,
            "total_delay_median_min": 0.0,
            "total_delay_q90_min": 0.0,
            "reliability_score": 0.8,
            "comfort_score": 0.7,
            "time_score": round(1 - time_score, 3),
            "risk_score": 1.0,
            "composite_score": round(composite, 4),
            "risk_factors": [],
        }

    def _sample_coordinates(
        self, coordinates: list[list[float]], max_points: int
    ) -> list[tuple[float, float]]:
        """Evenly sample points along a coordinate path."""
        n = len(coordinates)
        if n <= max_points:
            return [(c[0], c[1]) for c in coordinates]

        step = max(1, n // max_points)
        sampled = [coordinates[i] for i in range(0, n, step)]
        if coordinates[-1] not in sampled:
            sampled.append(coordinates[-1])
        return [(c[0], c[1]) for c in sampled]

    def _identify_risk_factors(
        self,
        node_ids: list[int],
        node_preds: dict[int, dict],
        disruption_prob: float,
        ratio: float,
    ) -> list[dict]:
        """Identify the top contributing risk factors along the route."""
        factors = []

        risk_nodes = sorted(
            [(nid, node_preds.get(nid, self._default_pred())) for nid in node_ids],
            key=lambda x: x[1]["disruption_prob"],
            reverse=True,
        )

        for nid, pred in risk_nodes[:3]:
            if pred["disruption_prob"] > 0.3:
                factors.append({
                    "node_id": nid,
                    "type": "disruption",
                    "severity": "high" if pred["disruption_prob"] > 0.6 else "medium",
                    "description": f"Node {nid}: {pred['disruption_prob']:.0%} disruption risk, "
                                   f"potential delay {pred['delay_q50']:.1f}-{pred['delay_q90']:.1f} min",
                })

        if ratio > 1.3:
            factors.append({
                "type": "congestion",
                "severity": "high" if ratio > 1.6 else "medium",
                "description": f"Route expected {(ratio-1)*100:.0f}% slower than base travel time",
            })

        return factors

    def _default_pred(self) -> dict:
        return {
            "node_id": -1,
            "disruption_prob": 0.1,
            "delay_q10": 0, "delay_q50": 2, "delay_q90": 5,
            "ratio_q10": 1.0, "ratio_q50": 1.1, "ratio_q90": 1.3,
        }
