"""Evaluate and compare ST-GAT vs LSTM baseline on the validation set.

Usage:
    python -m scripts.evaluate_models [--device cpu]

Outputs:
    - Side-by-side metrics table
    - Calibration analysis
    - Latency benchmarks
    - Results saved to docs/evaluation_results.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import get_settings
from src.graph.builder import TransportGraph
from src.models.stgat import STGAT, LSTMBaseline
from src.training.dataset import DisruptionDataset
from src.training.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(graph, settings):
    """Load validation dataset."""
    incidents = pd.read_parquet(settings.synthetic_dir / "incidents.parquet")
    weather_path = settings.processed_dir / "weather_features.parquet"
    weather = pd.read_parquet(weather_path) if weather_path.exists() else None

    all_dates = pd.to_datetime(incidents["start_time"])
    split_date = all_dates.quantile(0.8).strftime("%Y-%m-%d")
    data_end = all_dates.max().strftime("%Y-%m-%d")

    val_dataset = DisruptionDataset(
        graph=graph, incidents=incidents, weather=weather,
        start_date=split_date, end_date=data_end,
        samples_per_day=2, seed=123,
    )
    return val_dataset


def benchmark_latency(model, dataset, device, num_samples=20):
    """Measure inference latency."""
    model.eval()
    times = []
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        node_feat = sample["node_features"].unsqueeze(0).to(device) if sample["node_features"].dim() == 3 else sample["node_features"].to(device)
        edge_feat = sample["edge_features"].to(device)
        edge_idx = sample["edge_index"].to(device)

        # Use single sample (no batch dim for model forward)
        nf = node_feat[0] if node_feat.dim() == 4 else node_feat

        t0 = time.time()
        with torch.no_grad():
            _ = model(nf, edge_idx, edge_feat)
        times.append(time.time() - t0)

    return {
        "mean_ms": round(np.mean(times) * 1000, 1),
        "p50_ms": round(np.median(times) * 1000, 1),
        "p95_ms": round(np.percentile(times, 95) * 1000, 1),
        "p99_ms": round(np.percentile(times, 99) * 1000, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    settings = get_settings()
    device = args.device

    # Load graph
    train_graph_path = settings.processed_dir / "transport_graph_train.pkl"
    graph_path = settings.processed_dir / "transport_graph.pkl"
    if train_graph_path.exists():
        graph = TransportGraph.load(train_graph_path)
    else:
        graph = TransportGraph.load(graph_path)
    logger.info(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

    # Load validation dataset
    val_dataset = load_dataset(graph, settings)
    logger.info(f"Validation dataset: {len(val_dataset)} samples")

    # Determine feature dims
    sample = val_dataset[0]
    node_feat_dim = sample["node_features"].shape[-1]
    edge_feat_dim = sample["edge_features"].shape[-1]

    results = {}

    # ── Evaluate ST-GAT ──────────────────────────────────────────
    stgat_path = settings.checkpoint_dir / f"{settings.model_name}_best.pt"
    if stgat_path.exists():
        logger.info("=" * 60)
        logger.info("Evaluating ST-GAT model...")
        stgat = STGAT(
            node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim,
            hidden_dim=settings.hidden_dim, num_heads=settings.num_heads,
            num_layers=settings.num_layers, temporal_window=settings.temporal_window,
            forecast_horizons=settings.forecast_horizons,
            num_quantiles=settings.num_quantiles, dropout=settings.dropout,
        )
        checkpoint = torch.load(stgat_path, map_location=device, weights_only=False)
        stgat.load_state_dict(checkpoint["model_state_dict"])
        stgat = stgat.to(device)

        stgat_metrics = evaluate_model(stgat, val_dataset, device=device, batch_size=16)
        stgat_latency = benchmark_latency(stgat, val_dataset, device)
        results["stgat"] = {
            "metrics": {k: round(v, 4) for k, v in stgat_metrics.items()},
            "latency": stgat_latency,
            "params": sum(p.numel() for p in stgat.parameters()),
        }
        logger.info(f"ST-GAT AUROC: {stgat_metrics.get('avg_auroc', 0):.4f}")
        logger.info(f"ST-GAT Latency: {stgat_latency['mean_ms']:.1f}ms")
    else:
        logger.warning(f"No ST-GAT checkpoint at {stgat_path}")

    # ── Evaluate LSTM Baseline ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("Evaluating LSTM baseline (random init)...")
    lstm = LSTMBaseline(
        node_feat_dim=node_feat_dim, hidden_dim=64,
        num_layers=2, num_horizons=len(settings.forecast_horizons),
    ).to(device)

    # Check if trained baseline exists
    lstm_path = settings.checkpoint_dir / "lstm_baseline_best.pt"
    if lstm_path.exists():
        lstm_ckpt = torch.load(lstm_path, map_location=device, weights_only=False)
        lstm.load_state_dict(lstm_ckpt["model_state_dict"])
        logger.info("Loaded trained LSTM baseline")
    else:
        logger.info("Using randomly initialized LSTM baseline (untrained)")

    lstm_metrics = evaluate_model(lstm, val_dataset, device=device, batch_size=16)
    lstm_latency = benchmark_latency(lstm, val_dataset, device)
    results["lstm_baseline"] = {
        "metrics": {k: round(v, 4) for k, v in lstm_metrics.items()},
        "latency": lstm_latency,
        "params": sum(p.numel() for p in lstm.parameters()),
    }
    logger.info(f"LSTM AUROC: {lstm_metrics.get('avg_auroc', 0):.4f}")
    logger.info(f"LSTM Latency: {lstm_latency['mean_ms']:.1f}ms")

    # ── Comparison Summary ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)

    headers = ["Metric", "ST-GAT", "LSTM Baseline", "Improvement"]
    compare_metrics = ["avg_auroc", "avg_brier", "avg_delay_mae", "avg_ratio_mae"]
    for metric in compare_metrics:
        stgat_val = results.get("stgat", {}).get("metrics", {}).get(metric, 0)
        lstm_val = results.get("lstm_baseline", {}).get("metrics", {}).get(metric, 0)
        if "auroc" in metric or "precision" in metric:
            # Higher is better
            improvement = ((stgat_val - lstm_val) / max(lstm_val, 0.001)) * 100
            better = "+" if improvement > 0 else ""
        else:
            # Lower is better
            improvement = ((lstm_val - stgat_val) / max(lstm_val, 0.001)) * 100
            better = "+" if improvement > 0 else ""
        logger.info(f"  {metric:25s}  ST-GAT={stgat_val:.4f}  LSTM={lstm_val:.4f}  {better}{improvement:.1f}%")

    # Latency comparison
    logger.info("\nLATENCY (inference per sample):")
    for model_name, model_results in results.items():
        lat = model_results.get("latency", {})
        logger.info(f"  {model_name:15s}  mean={lat.get('mean_ms', 0):.1f}ms  "
                     f"p95={lat.get('p95_ms', 0):.1f}ms  "
                     f"params={model_results.get('params', 0):,}")

    # Calibration summary
    logger.info("\nQUANTILE CALIBRATION (q90 coverage — ideal = 0.90):")
    for model_name, model_results in results.items():
        m = model_results.get("metrics", {})
        coverages = [m.get(f"h{i}_delay_q90_coverage", 0) for i in range(3)]
        avg_coverage = np.mean(coverages) if coverages else 0
        logger.info(f"  {model_name:15s}  avg_q90_coverage={avg_coverage:.3f}  "
                     f"horizons={[round(c, 3) for c in coverages]}")

    # Save results
    output_dir = Path("docs")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
