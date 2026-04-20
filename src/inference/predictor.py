"""Inference service: loads trained model and produces disruption predictions."""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.config.settings import get_settings
from src.features.feature_engine import FeatureEngine
from src.graph.builder import TransportGraph
from src.models.stgat import STGAT

logger = logging.getLogger(__name__)


class DisruptionPredictor:
    """Loads a trained ST-GAT model and runs inference."""

    def __init__(
        self,
        graph: TransportGraph,
        model: STGAT | None = None,
        checkpoint_path: Path | None = None,
        device: str | None = None,
    ):
        self.settings = get_settings()
        self.device = device or self.settings.device
        self.graph = graph
        self.feature_engine = FeatureEngine(graph)

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = self._load_model(checkpoint_path)

        self.model.eval()

        # Pre-compute static features
        self._edge_features = torch.FloatTensor(
            self.feature_engine.build_edge_features()
        ).to(self.device)
        self._edge_index = torch.LongTensor(
            graph.get_edge_index()
        ).to(self.device)

    def _load_model(self, checkpoint_path: Path | None = None) -> STGAT:
        s = self.settings
        model = STGAT(
            node_feat_dim=s.node_feat_dim,
            edge_feat_dim=s.edge_feat_dim,
            hidden_dim=s.hidden_dim,
            num_heads=s.num_heads,
            num_layers=s.num_layers,
            temporal_window=s.temporal_window,
            forecast_horizons=s.forecast_horizons,
            num_quantiles=s.num_quantiles,
            dropout=s.dropout,
        )

        path = checkpoint_path or (s.checkpoint_dir / f"{s.model_name}_best.pt")
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded model from {path}")
        else:
            logger.warning(f"No checkpoint at {path}, using randomly initialized model")

        return model.to(self.device)

    def predict(
        self,
        timestamp: datetime,
        weather_features: dict | None = None,
        incidents=None,
    ) -> dict:
        """Run disruption prediction for all nodes at given timestamp.

        Returns:
            Dict with per-node predictions for each horizon:
            - disruption_prob: (N,) array of disruption probabilities
            - delay_q10/q50/q90: (N,) arrays of delay quantiles
            - travel_time_ratio_q10/q50/q90: (N,) arrays
        """
        snapshot = self.feature_engine.build_snapshot(
            timestamp, weather_features, incidents
        )
        node_feat = snapshot["node_features"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw = self.model(node_feat, self._edge_index, self._edge_features)

        results = {}
        for hi, horizon_steps in enumerate(self.settings.forecast_horizons):
            horizon_min = horizon_steps * 5
            prefix = f"h{horizon_min}min"

            results[f"{prefix}_disruption_prob"] = (
                raw[f"disruption_prob_h{hi}"].cpu().numpy()
            )

            delay_q = raw[f"delay_quantiles_h{hi}"].cpu().numpy()
            results[f"{prefix}_delay_q10"] = delay_q[:, 0]
            results[f"{prefix}_delay_q50"] = delay_q[:, 1]
            results[f"{prefix}_delay_q90"] = delay_q[:, 2]

            ratio_q = raw[f"travel_time_ratio_h{hi}"].cpu().numpy()
            results[f"{prefix}_ratio_q10"] = ratio_q[:, 0]
            results[f"{prefix}_ratio_q50"] = ratio_q[:, 1]
            results[f"{prefix}_ratio_q90"] = ratio_q[:, 2]

        return results

    def predict_for_nodes(
        self,
        node_ids: list[int],
        timestamp: datetime,
        weather_features: dict | None = None,
        incidents=None,
        horizon_minutes: int = 30,
    ) -> list[dict]:
        """Get predictions for specific nodes."""
        all_preds = self.predict(timestamp, weather_features, incidents)

        horizon_key = f"h{horizon_minutes}min"
        node_results = []
        for nid in node_ids:
            result = {
                "node_id": nid,
                "disruption_prob": float(all_preds.get(f"{horizon_key}_disruption_prob", np.zeros(1))[nid]),
                "delay_q10": float(all_preds.get(f"{horizon_key}_delay_q10", np.zeros(1))[nid]),
                "delay_q50": float(all_preds.get(f"{horizon_key}_delay_q50", np.zeros(1))[nid]),
                "delay_q90": float(all_preds.get(f"{horizon_key}_delay_q90", np.zeros(1))[nid]),
                "ratio_q10": float(all_preds.get(f"{horizon_key}_ratio_q10", np.ones(1))[nid]),
                "ratio_q50": float(all_preds.get(f"{horizon_key}_ratio_q50", np.ones(1))[nid]),
                "ratio_q90": float(all_preds.get(f"{horizon_key}_ratio_q90", np.ones(1))[nid]),
            }
            node_results.append(result)
        return node_results


def create_demo_predictor(graph: TransportGraph) -> DisruptionPredictor:
    """Create a predictor with a randomly initialized model (for demo without training)."""
    settings = get_settings()
    model = STGAT(
        node_feat_dim=settings.node_feat_dim,
        edge_feat_dim=settings.edge_feat_dim,
        hidden_dim=settings.hidden_dim,
        num_heads=settings.num_heads,
        num_layers=settings.num_layers,
        temporal_window=settings.temporal_window,
        forecast_horizons=settings.forecast_horizons,
        num_quantiles=settings.num_quantiles,
        dropout=settings.dropout,
    )
    return DisruptionPredictor(graph=graph, model=model, device=settings.device)
