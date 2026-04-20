"""Custom layers for the Spatio-Temporal Graph Attention Network."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class SpatialGATConv(MessagePassing):
    """Graph Attention Convolution with edge features.

    Implements multi-head attention over graph neighbors, incorporating
    edge features into the attention computation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.concat = concat

        self.W_q = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.W_k = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.W_v = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.W_e = nn.Linear(edge_dim, heads * out_channels, bias=False)

        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if concat:
            self.out_proj = nn.Linear(heads * out_channels, heads * out_channels)
        else:
            self.out_proj = nn.Linear(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_e.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: (N, in_channels)
            edge_index: (2, E)
            edge_attr: (E, edge_dim)
        Returns:
            (N, heads * out_channels) if concat else (N, out_channels)
        """
        q = self.W_q(x).view(-1, self.heads, self.out_channels)
        k = self.W_k(x).view(-1, self.heads, self.out_channels)
        v = self.W_v(x).view(-1, self.heads, self.out_channels)
        e = self.W_e(edge_attr).view(-1, self.heads, self.out_channels)

        out = self.propagate(edge_index, q=q, k=k, v=v, e=e)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        return self.out_proj(out)

    def message(self, q_i, k_j, v_j, e, index):
        """Compute attention-weighted messages."""
        # Attention scores: query_i * (key_j + edge_feat)
        attn_input = k_j + e
        alpha = (q_i * attn_input * self.att).sum(dim=-1)
        alpha = alpha / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return v_j * alpha.unsqueeze(-1)


class TemporalAttention(nn.Module):
    """Multi-head self-attention over the temporal dimension.

    Given a sequence (T, N, D), attends across time steps for each node.
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, N, D)
        Returns:
            (T, N, D) with temporal attention applied
        """
        # Reshape: treat each node as a separate batch element
        T, N, D = x.shape
        # (T, N, D) -> (T, N, D) — MHA expects (seq_len, batch, dim)
        residual = x
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(residual + self.dropout(attn_out))
        return x


class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2 + d_model % 2])
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: (T, N, D)"""
        return x + self.pe[: x.size(0)]


class QuantileHead(nn.Module):
    """Prediction head that outputs quantile estimates for uncertainty."""

    def __init__(
        self,
        in_dim: int,
        num_targets: int = 3,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ):
        super().__init__()
        self.quantiles = quantiles
        self.num_targets = num_targets
        self.num_quantiles = len(quantiles)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.ReLU(),
                nn.Linear(in_dim // 2, self.num_quantiles),
            )
            for _ in range(num_targets)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) or (B, N, in_dim)
        Returns:
            (N, num_targets, num_quantiles) or (B, N, num_targets, num_quantiles)
        """
        outputs = [head(x) for head in self.heads]
        return torch.stack(outputs, dim=-2)
