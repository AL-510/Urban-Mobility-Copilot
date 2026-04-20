"""Unit tests for ST-GAT model architecture."""

import torch
import pytest

from src.models.stgat import STGAT, LSTMBaseline


class TestSTGAT:
    def test_forward_shape(self):
        model = STGAT(
            node_feat_dim=24,
            edge_feat_dim=12,
            hidden_dim=32,
            num_heads=2,
            num_layers=2,
            temporal_window=4,
            forecast_horizons=[6, 12],
            dropout=0.0,
        )
        T, N, F_node = 4, 10, 24
        E, F_edge = 20, 12

        node_feat = torch.randn(T, N, F_node)
        edge_index = torch.randint(0, N, (2, E))
        edge_feat = torch.randn(E, F_edge)

        outputs = model(node_feat, edge_index, edge_feat)

        assert "disruption_prob_h0" in outputs
        assert "disruption_prob_h1" in outputs
        assert outputs["disruption_prob_h0"].shape == (N,)
        assert outputs["delay_quantiles_h0"].shape == (N, 3)
        assert outputs["travel_time_ratio_h0"].shape == (N, 3)

    def test_disruption_prob_range(self):
        model = STGAT(node_feat_dim=24, edge_feat_dim=12, hidden_dim=32,
                      num_heads=2, num_layers=1, forecast_horizons=[6])
        node_feat = torch.randn(2, 5, 24)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        edge_feat = torch.randn(3, 12)

        outputs = model(node_feat, edge_index, edge_feat)
        probs = outputs["disruption_prob_h0"]
        assert (probs >= 0).all() and (probs <= 1).all()


class TestLSTMBaseline:
    def test_forward_shape(self):
        model = LSTMBaseline(node_feat_dim=24, hidden_dim=32, num_horizons=2)
        node_feat = torch.randn(4, 10, 24)

        outputs = model(node_feat)
        assert "disruption_prob_h0" in outputs
        assert outputs["disruption_prob_h0"].shape == (10,)
