"""Evaluation metrics for disruption forecasting models."""

import logging

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.training.dataset import DisruptionDataset, collate_graph_batch

logger = logging.getLogger(__name__)


def evaluate_model(
    model: torch.nn.Module,
    dataset: DisruptionDataset,
    device: str = "cpu",
    batch_size: int = 16,
) -> dict:
    """Evaluate model on dataset and return comprehensive metrics."""
    model = model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graph_batch,
        num_workers=0,
    )

    all_preds = {}
    all_labels = {}

    with torch.no_grad():
        for batch in loader:
            node_feat = batch["node_features"].to(device)
            edge_feat = batch["edge_features"].to(device)
            edge_idx = batch["edge_index"].to(device)

            B = node_feat.shape[0]
            for b in range(B):
                preds = model(node_feat[b], edge_idx, edge_feat)
                labels = {
                    k: v[b] for k, v in batch.items()
                    if k.startswith(("disruption_prob", "delay_minutes", "travel_time_ratio"))
                }

                for key, val in preds.items():
                    if key not in all_preds:
                        all_preds[key] = []
                    all_preds[key].append(val.cpu().numpy())

                for key, val in labels.items():
                    if key not in all_labels:
                        all_labels[key] = []
                    all_labels[key].append(val.cpu().numpy())

    # Concatenate
    for key in all_preds:
        all_preds[key] = np.concatenate(all_preds[key])
    for key in all_labels:
        all_labels[key] = np.concatenate(all_labels[key])

    metrics = {}
    num_horizons = len([k for k in all_preds if k.startswith("disruption_prob")])

    for hi in range(num_horizons):
        prefix = f"h{hi}"

        # Disruption probability metrics
        pred_prob = all_preds[f"disruption_prob_h{hi}"]
        true_prob = all_labels[f"disruption_prob_h{hi}"]
        true_binary = (true_prob > 0.5).astype(float)

        try:
            metrics[f"{prefix}_auroc"] = roc_auc_score(true_binary, pred_prob)
        except ValueError:
            metrics[f"{prefix}_auroc"] = 0.5

        try:
            metrics[f"{prefix}_avg_precision"] = average_precision_score(true_binary, pred_prob)
        except ValueError:
            metrics[f"{prefix}_avg_precision"] = 0.0

        metrics[f"{prefix}_brier"] = brier_score_loss(true_binary, pred_prob)

        # Delay metrics (use median quantile)
        pred_delay = all_preds[f"delay_quantiles_h{hi}"]
        true_delay = all_labels[f"delay_minutes_h{hi}"]
        if pred_delay.ndim == 2:
            pred_median = pred_delay[:, 1]  # 50th percentile
        else:
            pred_median = pred_delay

        metrics[f"{prefix}_delay_mae"] = mean_absolute_error(true_delay, pred_median)

        # Quantile calibration (what fraction of true values fall below predicted quantile?)
        if pred_delay.ndim == 2:
            for qi, q_name in enumerate(["q10", "q50", "q90"]):
                coverage = (true_delay <= pred_delay[:, qi]).mean()
                metrics[f"{prefix}_delay_{q_name}_coverage"] = float(coverage)

        # Travel time ratio metrics
        pred_ratio = all_preds[f"travel_time_ratio_h{hi}"]
        true_ratio = all_labels[f"travel_time_ratio_h{hi}"]
        if pred_ratio.ndim == 2:
            pred_ratio_median = pred_ratio[:, 1]
        else:
            pred_ratio_median = pred_ratio
        metrics[f"{prefix}_ratio_mae"] = mean_absolute_error(true_ratio, pred_ratio_median)

    # Average across horizons
    for metric_name in ["auroc", "avg_precision", "brier", "delay_mae", "ratio_mae"]:
        values = [metrics.get(f"h{hi}_{metric_name}", 0) for hi in range(num_horizons)]
        metrics[f"avg_{metric_name}"] = np.mean(values)

    for key, val in sorted(metrics.items()):
        logger.info(f"  {key}: {val:.4f}")

    return metrics
