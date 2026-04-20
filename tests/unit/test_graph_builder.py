"""Unit tests for the transport graph builder."""

import pytest

from src.graph.builder import TransportGraph, EDGE_ROAD, EDGE_TRANSIT, _haversine_m


class TestTransportGraph:
    def test_create_empty_graph(self):
        g = TransportGraph()
        assert g.num_nodes == 0
        assert g.num_edges == 0

    def test_add_nodes(self):
        g = TransportGraph()
        nid = g._get_or_create_node("road_1", 45.5, -122.6, "road")
        assert nid == 0
        assert g.num_nodes == 1

    def test_add_duplicate_node(self):
        g = TransportGraph()
        nid1 = g._get_or_create_node("road_1", 45.5, -122.6, "road")
        nid2 = g._get_or_create_node("road_1", 45.5, -122.6, "road")
        assert nid1 == nid2
        assert g.num_nodes == 1

    def test_nearest_node(self):
        g = TransportGraph()
        g._get_or_create_node("a", 45.50, -122.60, "road")
        g._get_or_create_node("b", 45.55, -122.65, "road")
        g._get_or_create_node("c", 45.60, -122.70, "road")

        nearest = g.nearest_node(45.51, -122.61)
        assert nearest == 0  # Closest to node "a"

    def test_edge_index(self):
        g = TransportGraph()
        n1 = g._get_or_create_node("a", 45.5, -122.6, "road")
        n2 = g._get_or_create_node("b", 45.6, -122.7, "road")
        g.G.add_edge(n1, n2, edge_type=EDGE_ROAD)

        ei = g.get_edge_index()
        assert ei.shape == (2, 1)
        assert ei[0, 0] == n1
        assert ei[1, 0] == n2


class TestHaversine:
    def test_same_point(self):
        d = _haversine_m(45.5, -122.6, 45.5, -122.6)
        assert d < 1.0

    def test_known_distance(self):
        # Portland downtown to Lloyd District ~2km
        d = _haversine_m(45.5152, -122.6784, 45.5311, -122.6590)
        assert 1500 < d < 3000
