"""Unit tests for the feature engine."""

from datetime import datetime

import pytest

from src.features.feature_engine import FeatureEngine
from src.graph.builder import TransportGraph, EDGE_ROAD


def _make_small_graph() -> TransportGraph:
    g = TransportGraph()
    n1 = g._get_or_create_node("a", 45.5, -122.6, "road")
    n2 = g._get_or_create_node("b", 45.51, -122.61, "road")
    n3 = g._get_or_create_node("c", 45.52, -122.62, "transit")
    g.G.add_edge(n1, n2, edge_type=EDGE_ROAD, length_m=500, base_travel_time_s=30,
                 speed_kph=40, lanes=2, highway_class="secondary", capacity=900)
    g.G.add_edge(n2, n3, edge_type=EDGE_ROAD, length_m=600, base_travel_time_s=36,
                 speed_kph=40, lanes=1, highway_class="residential", capacity=300)
    return g


class TestFeatureEngine:
    def test_static_node_features_shape(self):
        g = _make_small_graph()
        fe = FeatureEngine(g)
        feat = fe.build_static_node_features()
        assert feat.shape == (3, 8)

    def test_temporal_node_features_shape(self):
        g = _make_small_graph()
        fe = FeatureEngine(g)
        ts = datetime(2024, 6, 15, 8, 30)
        feat = fe.build_temporal_node_features(ts)
        assert feat.shape == (3, 16)

    def test_edge_features_shape(self):
        g = _make_small_graph()
        fe = FeatureEngine(g)
        feat = fe.build_edge_features()
        assert feat.shape == (2, 12)

    def test_snapshot(self):
        g = _make_small_graph()
        fe = FeatureEngine(g)
        ts = datetime(2024, 6, 15, 8, 30)
        snap = fe.build_snapshot(ts)
        assert snap["node_features"].shape[0] == 3
        assert snap["node_features"].shape[1] == 24  # 8 static + 16 temporal
