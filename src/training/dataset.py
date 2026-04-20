"""Training dataset: generates temporal graph snapshots with disruption labels.

Optimized for CPU training:
- Weather data pre-indexed by hour for O(1) lookup (not O(n) scan)
- Incident timestamps pre-parsed to datetime objects
- Node positions cached
- Static features and edge features computed once (via FeatureEngine caching)
"""

import logging
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config.settings import get_settings
from src.features.feature_engine import FeatureEngine
from src.graph.builder import TransportGraph

logger = logging.getLogger(__name__)


class DisruptionDataset(Dataset):
    """Dataset of temporal graph snapshots for disruption forecasting.

    Each sample is a (T_lookback) sequence of graph snapshots with labels
    at future horizons.
    """

    def __init__(
        self,
        graph: TransportGraph,
        incidents: pd.DataFrame,
        weather: pd.DataFrame | None = None,
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        temporal_window: int = 12,
        step_minutes: int = 5,
        forecast_horizons: list[int] | None = None,
        samples_per_day: int = 20,
        seed: int = 42,
    ):
        self.graph = graph
        self.feature_engine = FeatureEngine(graph)
        self.weather = weather
        self.temporal_window = temporal_window
        self.step_minutes = step_minutes
        self.forecast_horizons = forecast_horizons or [6, 12, 18]
        self.num_nodes = graph.num_nodes

        # Pre-compute and cache static tensors
        self.edge_features = torch.FloatTensor(self.feature_engine.build_edge_features())
        self.edge_index = torch.LongTensor(graph.get_edge_index())

        # Pre-index weather by hour for O(1) lookup
        self._weather_index = self._build_weather_index(weather)

        # Pre-parse incident timestamps for fast datetime comparison
        self.incidents = self._prepare_incidents(incidents)

        # Cache node positions for label generation
        self._positions = graph.get_node_positions()

        # Generate sample timestamps
        random.seed(seed)
        np.random.seed(seed)
        self.timestamps = self._generate_sample_times(
            start_date, end_date, samples_per_day
        )
        logger.info(f"Dataset: {len(self.timestamps)} samples, {self.num_nodes} nodes")

    @staticmethod
    def _build_weather_index(weather: pd.DataFrame | None) -> dict:
        """Pre-index weather data by (year, month, day, hour) for O(1) lookup.

        Converts the full DataFrame scan + copy (~0.03s per call) into a dict
        lookup (~0.0001s per call). With 15 lookups per sample, saves ~0.45s.
        """
        if weather is None or weather.empty:
            return {}
        index = {}
        for _, row in weather.iterrows():
            ts = pd.to_datetime(row["timestamp"])
            hour_key = (ts.year, ts.month, ts.day, ts.hour)
            index[hour_key] = row.to_dict()
        return index

    @staticmethod
    def _prepare_incidents(incidents: pd.DataFrame) -> pd.DataFrame:
        """Pre-parse incident timestamps to datetime for fast comparison."""
        if incidents is None or incidents.empty:
            return incidents
        inc = incidents.copy()
        if "start_time" in inc.columns:
            inc["_start_dt"] = pd.to_datetime(inc["start_time"])
            inc["_end_dt"] = pd.to_datetime(inc["end_time"])
        return inc

    def _generate_sample_times(
        self, start_date: str, end_date: str, samples_per_day: int
    ) -> list[datetime]:
        """Generate diverse sample timestamps across the date range."""
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        days = (end - start).days
        timestamps = []

        for day_offset in range(days):
            date = start + timedelta(days=day_offset)
            # Bias toward interesting hours (rush hours, but include off-peak)
            hours = (
                random.choices(range(6, 10), k=samples_per_day // 3)
                + random.choices(range(15, 20), k=samples_per_day // 3)
                + random.choices(range(24), k=samples_per_day - 2 * (samples_per_day // 3))
            )
            for h in hours:
                m = random.randint(0, 59)
                timestamps.append(date.replace(hour=h, minute=m))

        return sorted(timestamps)

    def _get_weather_at(self, timestamp: datetime) -> dict:
        """Get weather features nearest to timestamp — O(1) dict lookup."""
        hour_key = (timestamp.year, timestamp.month, timestamp.day, timestamp.hour)
        return self._weather_index.get(hour_key, {})

    def _generate_labels(self, timestamp: datetime) -> dict[str, torch.Tensor]:
        """Generate disruption labels for future horizons.

        Labels per node per horizon:
        - disruption_prob: binary (is any incident affecting this node at horizon?)
        - delay_minutes: estimated delay (from incident delay_factor)
        - travel_time_ratio: multiplier on base travel time
        """
        labels = {}
        positions = self._positions  # cached

        for hi, horizon_steps in enumerate(self.forecast_horizons):
            horizon_time = timestamp + timedelta(minutes=horizon_steps * self.step_minutes)

            disruption = np.zeros(self.num_nodes, dtype=np.float32)
            delay = np.zeros(self.num_nodes, dtype=np.float32)
            ratio = np.ones(self.num_nodes, dtype=np.float32)

            if not self.incidents.empty:
                # Use pre-parsed datetime columns for fast comparison
                if "_start_dt" in self.incidents.columns:
                    active = self.incidents[
                        (self.incidents["_start_dt"] <= horizon_time)
                        & (self.incidents["_end_dt"] >= horizon_time)
                    ]
                else:
                    active = self.incidents[
                        (self.incidents["start_time"] <= horizon_time.isoformat())
                        & (self.incidents["end_time"] >= horizon_time.isoformat())
                    ]

                for _, inc in active.iterrows():
                    dists = np.sqrt(
                        (positions[:, 0] - inc["lat"]) ** 2
                        + (positions[:, 1] - inc["lon"]) ** 2
                    )
                    radius = 0.005  # ~500m
                    affected = dists < radius
                    proximity = 1.0 - (dists[affected] / radius)

                    disruption[affected] = np.maximum(disruption[affected], proximity)
                    inc_delay = inc.get("delay_factor", 1.5)
                    delay[affected] = np.maximum(delay[affected], (inc_delay - 1) * 10 * proximity)
                    ratio[affected] = np.maximum(ratio[affected], 1.0 + (inc_delay - 1) * proximity)

            # Add weather-based disruption
            wx = self._get_weather_at(horizon_time)
            severity = wx.get("weather_severity", 0)
            if severity is None:
                severity = 0
            severity = float(severity)
            if severity > 0.3:
                weather_boost = severity * 0.3
                disruption = np.clip(disruption + weather_boost, 0, 1)
                delay += severity * 5
                ratio += severity * 0.2

            # Add noise for realism
            disruption += np.random.normal(0, 0.05, self.num_nodes)
            disruption = np.clip(disruption, 0, 1)
            delay = np.clip(delay + np.random.normal(0, 1, self.num_nodes), 0, 60)
            ratio = np.clip(ratio + np.random.normal(0, 0.05, self.num_nodes), 0.8, 5.0)

            labels[f"disruption_prob_h{hi}"] = torch.FloatTensor(disruption)
            labels[f"delay_minutes_h{hi}"] = torch.FloatTensor(delay)
            labels[f"travel_time_ratio_h{hi}"] = torch.FloatTensor(ratio)

        return labels

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, idx: int) -> dict:
        """Return a training sample.

        Returns dict with:
        - node_features: (T, N, F)
        - edge_features: (E, F_edge)
        - edge_index: (2, E)
        - labels: dict of label tensors
        """
        center_time = self.timestamps[idx]

        # Build temporal sequence of snapshots
        timestamps = [
            center_time - timedelta(minutes=(self.temporal_window - 1 - t) * self.step_minutes)
            for t in range(self.temporal_window)
        ]

        weather_series = [self._get_weather_at(ts) for ts in timestamps]
        seq = self.feature_engine.build_temporal_sequence(
            timestamps, weather_series, self.incidents
        )

        labels = self._generate_labels(center_time)

        return {
            "node_features": seq["node_features"],
            "edge_features": self.edge_features,
            "edge_index": self.edge_index,
            **labels,
        }


def collate_graph_batch(batch: list[dict]) -> dict:
    """Custom collate for graph data (all graphs share same topology)."""
    # Since all samples share the same graph structure, we can stack node features
    result = {
        "node_features": torch.stack([b["node_features"] for b in batch]),  # (B, T, N, F)
        "edge_features": batch[0]["edge_features"],  # (E, F_edge) shared
        "edge_index": batch[0]["edge_index"],  # (2, E) shared
    }

    # Stack labels
    label_keys = [k for k in batch[0].keys() if k not in ("node_features", "edge_features", "edge_index")]
    for key in label_keys:
        result[key] = torch.stack([b[key] for b in batch])  # (B, N)

    return result
