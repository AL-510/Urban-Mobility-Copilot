"""Train the ST-GAT disruption forecasting model.

Usage:
    python -m scripts.train_model [--epochs 50] [--batch-size 32] [--device cpu]
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import get_settings
from src.graph.builder import TransportGraph
from src.models.stgat import STGAT, LSTMBaseline
from src.training.dataset import DisruptionDataset
from src.training.evaluate import evaluate_model
from src.training.trainer import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train ST-GAT model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--baseline", action="store_true", help="Train LSTM baseline instead")
    parser.add_argument("--samples-per-day", type=int, default=None,
                        help="Training samples per day (auto-tuned if not set)")
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="Model hidden dimension (default from settings)")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Number of GAT layers (default from settings)")
    args = parser.parse_args()

    settings = get_settings()

    # Load graph
    graph_path = settings.processed_dir / "transport_graph.pkl"
    if graph_path.exists():
        graph = TransportGraph.load(graph_path)
    else:
        logger.info("No saved graph, building demo graph...")
        from src.api.app import _build_demo_graph
        graph = _build_demo_graph(settings)

    num_nodes = graph.num_nodes
    num_edges = graph.num_edges
    logger.info(f"Full graph: {num_nodes} nodes, {num_edges} edges")

    # Auto-subsample large graphs for CPU training
    # ~500 nodes = real Portland downtown intersections, 2x demo size
    MAX_TRAINABLE_NODES = 600
    if num_nodes > MAX_TRAINABLE_NODES:
        logger.info(f"Graph has {num_nodes} nodes. Extracting downtown core for CPU training...")
        target_nodes = 500
        lo, hi = 1.0, settings.city_radius_km
        best_graph = None
        for _ in range(8):
            mid = (lo + hi) / 2
            candidate = graph.extract_subgraph(
                settings.city_center_lat, settings.city_center_lon, radius_km=mid
            )
            if candidate.num_nodes < target_nodes * 0.8:
                lo = mid
            elif candidate.num_nodes > target_nodes * 1.3:
                hi = mid
            else:
                best_graph = candidate
                break
            best_graph = candidate

        graph = best_graph
        # Save the training subgraph for the backend to use
        sub_path = settings.processed_dir / "transport_graph_train.pkl"
        graph.save(sub_path)
        logger.info(f"Training subgraph: {graph.num_nodes} nodes, {graph.num_edges} edges "
                     f"(saved to {sub_path})")

    num_nodes = graph.num_nodes
    num_edges = graph.num_edges
    logger.info(f"Training graph: {num_nodes} nodes, {num_edges} edges")

    # Auto-tune batch size, samples, and epochs based on graph size.
    # Real-world graphs (>200 nodes) need fewer samples to keep training feasible on CPU.
    if num_nodes > 5000:
        default_batch_size = 4
        default_samples_per_day = 4
        default_epochs = 8
        logger.info(f"Large graph ({num_nodes} nodes) - using conservative settings")
    elif num_nodes > 2000:
        default_batch_size = 8
        default_samples_per_day = 4
        default_epochs = 8
    elif num_nodes > 1000:
        default_batch_size = 16
        default_samples_per_day = 6
        default_epochs = 10
    elif num_nodes > 300:
        default_batch_size = 32
        default_samples_per_day = 6
        default_epochs = 10
    else:
        # Small demo graph
        default_batch_size = 32
        default_samples_per_day = 20
        default_epochs = 15

    samples_per_day = args.samples_per_day or default_samples_per_day
    batch_size = args.batch_size or default_batch_size

    # Load incidents
    incidents_path = settings.synthetic_dir / "incidents.parquet"
    if incidents_path.exists():
        incidents = pd.read_parquet(incidents_path)
    else:
        logger.info("No incidents found, generating...")
        from src.data_ingestion.incident_generator import generate_incidents
        incidents = generate_incidents()

    logger.info(f"Loaded {len(incidents)} incidents")

    # Load weather
    weather_path = settings.processed_dir / "weather_features.parquet"
    weather = pd.read_parquet(weather_path) if weather_path.exists() else None
    if weather is not None:
        logger.info(f"Loaded {len(weather)} weather records")

    # Determine date ranges from available data
    if len(incidents) > 0:
        all_dates = pd.to_datetime(incidents["start_time"])
        data_start = all_dates.min().strftime("%Y-%m-%d")
        data_end = all_dates.max().strftime("%Y-%m-%d")
        # Split: 80% train, 20% val
        split_date = all_dates.quantile(0.8).strftime("%Y-%m-%d")
    else:
        data_start = "2024-01-01"
        split_date = "2024-10-01"
        data_end = "2024-12-31"

    logger.info(f"Data range: {data_start} to {data_end}")
    logger.info(f"Train: {data_start} to {split_date}, Val: {split_date} to {data_end}")

    # Create datasets
    logger.info("Creating training dataset...")
    train_dataset = DisruptionDataset(
        graph=graph,
        incidents=incidents,
        weather=weather,
        start_date=data_start,
        end_date=split_date,
        samples_per_day=samples_per_day,
        seed=42,
    )

    val_dataset = DisruptionDataset(
        graph=graph,
        incidents=incidents,
        weather=weather,
        start_date=split_date,
        end_date=data_end,
        samples_per_day=max(samples_per_day // 3, 2),
        seed=123,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Batch size: {batch_size}, Samples/day: {samples_per_day}")

    # Determine actual feature dimensions from data
    sample = train_dataset[0]
    node_feat_dim = sample["node_features"].shape[-1]
    edge_feat_dim = sample["edge_features"].shape[-1]

    logger.info(f"Node feature dim: {node_feat_dim}, Edge feature dim: {edge_feat_dim}")

    # Create model
    if args.baseline:
        logger.info("Training LSTM baseline...")
        model = LSTMBaseline(
            node_feat_dim=node_feat_dim,
            hidden_dim=64,
            num_layers=2,
            num_horizons=len(settings.forecast_horizons),
        )
    else:
        hidden_dim = args.hidden_dim or settings.hidden_dim
        n_layers = args.num_layers or settings.num_layers
        logger.info(f"Training ST-GAT model (hidden={hidden_dim}, layers={n_layers})...")
        model = STGAT(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            num_heads=settings.num_heads,
            num_layers=n_layers,
            temporal_window=settings.temporal_window,
            forecast_horizons=settings.forecast_horizons,
            num_quantiles=settings.num_quantiles,
            dropout=settings.dropout,
        )

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # Train
    epochs = args.epochs or default_epochs
    logger.info(f"Starting training: {epochs} epochs, batch_size={batch_size}, "
                f"samples/day={samples_per_day}")
    name = "lstm_baseline" if args.baseline else settings.model_name
    history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=epochs,
        batch_size=batch_size,
        device=args.device,
        model_name=name,
    )

    # Evaluate
    logger.info("Evaluating on validation set...")
    metrics = evaluate_model(
        model, val_dataset, device=args.device or settings.device
    )

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best val loss: {min(history.get('val_loss', [float('inf')])):.4f}")
    logger.info(f"Avg AUROC: {metrics.get('avg_auroc', 0):.4f}")
    logger.info(f"Avg Delay MAE: {metrics.get('avg_delay_mae', 0):.2f} min")
    logger.info(f"Avg Q90 Coverage: {metrics.get('avg_q90_coverage', 0):.1%}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
