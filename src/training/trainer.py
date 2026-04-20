"""Training loop for the ST-GAT disruption forecasting model."""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.config.settings import get_settings
from src.training.dataset import DisruptionDataset, collate_graph_batch

logger = logging.getLogger(__name__)


class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression."""

    def __init__(self, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, N, Q) predicted quantiles
            target: (B, N) actual values
        """
        target = target.unsqueeze(-1)
        errors = target - pred
        losses = []
        for i, q in enumerate(self.quantiles):
            e = errors[..., i]
            loss = torch.max(q * e, (q - 1) * e)
            losses.append(loss.mean())
        return sum(losses) / len(losses)


class DisruptionLoss(nn.Module):
    """Combined loss for disruption forecasting.

    Components:
    1. Binary cross-entropy for disruption probability
    2. Quantile loss for delay minutes
    3. Quantile loss for travel time ratio
    """

    def __init__(self, num_horizons: int = 3):
        super().__init__()
        self.num_horizons = num_horizons
        self.bce = nn.BCELoss()
        self.quantile_loss = QuantileLoss()
        self.weights = {"disruption": 1.0, "delay": 0.5, "ratio": 0.5}

    def forward(
        self, predictions: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        losses = {}
        total = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        for hi in range(self.num_horizons):
            # Disruption probability
            pred_disrupt = predictions[f"disruption_prob_h{hi}"]
            true_disrupt = labels[f"disruption_prob_h{hi}"]
            # Clamp predictions for numerical stability
            pred_disrupt = pred_disrupt.clamp(1e-6, 1 - 1e-6)
            loss_d = self.bce(pred_disrupt, true_disrupt)
            losses[f"bce_h{hi}"] = loss_d
            total = total + self.weights["disruption"] * loss_d

            # Delay quantiles
            pred_delay = predictions[f"delay_quantiles_h{hi}"]
            true_delay = labels[f"delay_minutes_h{hi}"]
            if pred_delay.dim() == 2 and true_delay.dim() == 2:
                loss_delay = self.quantile_loss(pred_delay.unsqueeze(0), true_delay.unsqueeze(0))
            else:
                loss_delay = self.quantile_loss(pred_delay, true_delay)
            losses[f"delay_ql_h{hi}"] = loss_delay
            total = total + self.weights["delay"] * loss_delay

            # Travel time ratio quantiles
            pred_ratio = predictions[f"travel_time_ratio_h{hi}"]
            true_ratio = labels[f"travel_time_ratio_h{hi}"]
            if pred_ratio.dim() == 2 and true_ratio.dim() == 2:
                loss_ratio = self.quantile_loss(pred_ratio.unsqueeze(0), true_ratio.unsqueeze(0))
            else:
                loss_ratio = self.quantile_loss(pred_ratio, true_ratio)
            losses[f"ratio_ql_h{hi}"] = loss_ratio
            total = total + self.weights["ratio"] * loss_ratio

        losses["total"] = total
        return losses


def train_model(
    model: nn.Module,
    train_dataset: DisruptionDataset,
    val_dataset: DisruptionDataset | None = None,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    patience: int | None = None,
    device: str | None = None,
    model_name: str | None = None,
) -> dict:
    """Full training loop with per-batch progress logging.

    Returns:
        Training history dict with loss curves and best metrics.
    """
    settings = get_settings()
    model_name = model_name or model_name
    num_epochs = num_epochs or settings.num_epochs
    batch_size = batch_size or settings.batch_size
    learning_rate = learning_rate or settings.learning_rate
    patience = patience or settings.patience
    device = device or settings.device

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = DisruptionLoss(num_horizons=len(settings.forecast_horizons))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graph_batch,
        num_workers=0,
    )
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_graph_batch,
            num_workers=0,
        )

    total_batches = len(train_loader)
    total_samples = len(train_dataset)
    logger.info(f"Training: {num_epochs} epochs, {total_samples} samples, "
                f"{total_batches} batches/epoch, batch_size={batch_size}")

    history = {"train_loss": [], "val_loss": [], "epoch_time": []}
    best_val_loss = float("inf")
    no_improve = 0

    checkpoint_dir = settings.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        t0 = time.time()

        # Train
        model.train()
        epoch_losses = []
        for batch_idx, batch in enumerate(train_loader):
            batch_t0 = time.time()
            node_feat = batch["node_features"].to(device)
            edge_feat = batch["edge_features"].to(device)
            edge_idx = batch["edge_index"].to(device)

            # Gradient accumulation: forward+backward per sample, accumulate grads.
            # Much more memory-efficient than building one giant computation graph
            # across all B samples (frees activations after each backward).
            B = node_feat.shape[0]
            optimizer.zero_grad()
            batch_loss_val = 0.0

            for b in range(B):
                predictions = model(node_feat[b], edge_idx, edge_feat)
                labels = {
                    k: v[b].to(device) for k, v in batch.items()
                    if k.startswith(("disruption_prob", "delay_minutes", "travel_time_ratio"))
                }
                losses = criterion(predictions, labels)
                sample_loss = losses["total"] / B
                sample_loss.backward()  # accumulates gradients, frees graph
                batch_loss_val += losses["total"].item()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(batch_loss_val / B)

            # Progress logging every 10 batches or at end
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                batch_elapsed = time.time() - batch_t0
                elapsed = time.time() - t0
                avg_batch_time = elapsed / (batch_idx + 1)
                eta = avg_batch_time * (total_batches - batch_idx - 1)
                avg_loss = np.mean(epoch_losses[-10:])
                logger.info(
                    f"  Epoch {epoch+1} [{batch_idx+1}/{total_batches}] "
                    f"loss={avg_loss:.4f} batch={batch_elapsed:.1f}s "
                    f"elapsed={elapsed:.0f}s ETA={eta:.0f}s"
                )

        train_loss = np.mean(epoch_losses)
        history["train_loss"].append(train_loss)

        # Validate
        val_loss = 0
        if val_loader:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    node_feat = batch["node_features"].to(device)
                    edge_feat = batch["edge_features"].to(device)
                    edge_idx = batch["edge_index"].to(device)

                    B = node_feat.shape[0]
                    for b in range(B):
                        predictions = model(node_feat[b], edge_idx, edge_feat)
                        labels = {
                            k: v[b].to(device) for k, v in batch.items()
                            if k.startswith(("disruption_prob", "delay_minutes", "travel_time_ratio"))
                        }
                        losses = criterion(predictions, labels)
                        val_losses.append(losses["total"].item())

            val_loss = np.mean(val_losses)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    checkpoint_dir / f"{model_name}_best.pt",
                )
                logger.info(f"  ✓ New best val_loss={val_loss:.4f}, checkpoint saved")
            else:
                no_improve += 1
        else:
            # Save periodically without validation
            if (epoch + 1) % 5 == 0:
                torch.save(
                    {"epoch": epoch, "model_state_dict": model.state_dict()},
                    checkpoint_dir / f"{model_name}_epoch{epoch}.pt",
                )

        scheduler.step()
        elapsed = time.time() - t0
        history["epoch_time"].append(elapsed)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} complete | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s | "
            f"No-improve: {no_improve}/{patience}"
        )

        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Save final model
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict()},
        checkpoint_dir / f"{model_name}_final.pt",
    )

    # Save training history
    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history
