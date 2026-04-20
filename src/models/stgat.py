"""Spatio-Temporal Graph Attention Network (ST-GAT) for disruption forecasting.

Architecture:
1. Node feature projection
2. Stacked spatial GAT layers (message passing over graph)
3. Temporal attention (self-attention over time steps)
4. Multi-horizon quantile prediction heads

Outputs per node per horizon:
- disruption_probability (0-1)
- delay_minutes (quantiles: 10th, 50th, 90th)
- travel_time_ratio (quantiles: 10th, 50th, 90th)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import (
    QuantileHead,
    SpatialGATConv,
    TemporalAttention,
    TemporalPositionalEncoding,
)


class STGAT(nn.Module):
    """Spatio-Temporal Graph Attention Network."""

    def __init__(
        self,
        node_feat_dim: int = 24,
        edge_feat_dim: int = 12,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        temporal_window: int = 12,
        forecast_horizons: list[int] | None = None,
        num_quantiles: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.temporal_window = temporal_window
        self.forecast_horizons = forecast_horizons or [6, 12, 18]
        self.num_horizons = len(self.forecast_horizons)

        # Node feature projection
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Spatial GAT layers
        gat_dim = hidden_dim // num_heads
        self.spatial_layers = nn.ModuleList()
        self.spatial_norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            self.spatial_layers.append(
                SpatialGATConv(
                    in_channels=in_dim,
                    out_channels=gat_dim,
                    edge_dim=edge_feat_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.spatial_norms.append(nn.LayerNorm(hidden_dim))

        # Temporal positional encoding
        self.temporal_pe = TemporalPositionalEncoding(hidden_dim, max_len=temporal_window + 20)

        # Temporal attention layers
        self.temporal_layers = nn.ModuleList([
            TemporalAttention(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(2)
        ])

        # Horizon-specific projection
        self.horizon_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(self.num_horizons)
        ])

        # Prediction heads (per horizon)
        # Target 0: disruption probability (single value via sigmoid)
        # Target 1: delay_minutes (quantiles)
        # Target 2: travel_time_ratio (quantiles)
        self.disruption_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(self.num_horizons)
        ])

        self.quantile_heads = nn.ModuleList([
            QuantileHead(
                in_dim=hidden_dim,
                num_targets=2,  # delay_minutes, travel_time_ratio
                quantiles=(0.1, 0.5, 0.9),
            )
            for _ in range(self.num_horizons)
        ])

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            node_features: (T, N, F_node) temporal sequence of node features
            edge_index: (2, E) graph connectivity
            edge_features: (E, F_edge) edge features

        Returns:
            Dict with keys per horizon h:
                disruption_prob_h{i}: (N,) probabilities
                delay_quantiles_h{i}: (N, 3) quantile predictions
                travel_time_ratio_h{i}: (N, 3) quantile predictions
        """
        T, N, F_in = node_features.shape

        # Project node features
        x = self.node_proj(node_features.reshape(T * N, F_in))
        x = x.view(T, N, self.hidden_dim)

        # Spatial attention at each time step
        spatial_out = []
        for t in range(T):
            h = x[t]  # (N, hidden_dim)
            for layer, norm in zip(self.spatial_layers, self.spatial_norms):
                h_new = layer(h, edge_index, edge_features)
                h = norm(h + F.relu(h_new))
            spatial_out.append(h)
        x = torch.stack(spatial_out)  # (T, N, hidden_dim)

        # Temporal attention
        x = self.temporal_pe(x)
        for temp_layer in self.temporal_layers:
            x = temp_layer(x)

        # Use last time step representation for prediction
        x_last = x[-1]  # (N, hidden_dim)

        # Multi-horizon predictions
        outputs = {}
        for i, horizon in enumerate(self.forecast_horizons):
            h = self.horizon_proj[i](x_last)

            # Disruption probability
            disrupt_logit = self.disruption_heads[i](h).squeeze(-1)
            outputs[f"disruption_prob_h{i}"] = torch.sigmoid(disrupt_logit)

            # Quantile predictions (delay_minutes, travel_time_ratio)
            quantiles = self.quantile_heads[i](h)  # (N, 2, 3)
            outputs[f"delay_quantiles_h{i}"] = F.softplus(quantiles[:, 0, :])
            outputs[f"travel_time_ratio_h{i}"] = 1.0 + F.softplus(quantiles[:, 1, :])

        return outputs

    def predict_single(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Inference for a single snapshot (adds time dimension)."""
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)  # (1, N, F)
        self.eval()
        with torch.no_grad():
            return self.forward(node_features, edge_index, edge_features)


class LSTMBaseline(nn.Module):
    """LSTM baseline for comparison. Per-node temporal model without graph structure."""

    def __init__(
        self,
        node_feat_dim: int = 24,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_horizons: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_horizons = num_horizons

        self.lstm = nn.LSTM(
            input_size=node_feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=False,
        )

        self.disruption_head = nn.Linear(hidden_dim, num_horizons)
        self.delay_head = nn.Linear(hidden_dim, num_horizons * 3)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor = None,
        edge_features: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            node_features: (T, N, F)
        """
        T, N, F_in = node_features.shape
        x = node_features  # (T, N, F_in)

        output, _ = self.lstm(x)
        h = output[-1]  # (N, hidden_dim)

        disrupt = torch.sigmoid(self.disruption_head(h))  # (N, num_horizons)
        delay = F.softplus(self.delay_head(h).view(N, self.num_horizons, 3))

        outputs = {}
        for i in range(self.num_horizons):
            outputs[f"disruption_prob_h{i}"] = disrupt[:, i]
            outputs[f"delay_quantiles_h{i}"] = delay[:, i, :]
            outputs[f"travel_time_ratio_h{i}"] = 1.0 + delay[:, i, :] / 30.0

        return outputs
