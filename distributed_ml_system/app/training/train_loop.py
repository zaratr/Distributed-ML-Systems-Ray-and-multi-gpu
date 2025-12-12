"""Baseline single-node training loop for MNIST."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ..data.dataset import get_dataloaders
from ..models.classifier import SimpleCNN


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total if total else 0.0


def train_single_node(config: Dict, checkpoint_dir: Optional[str] = None) -> Dict[str, float]:
    """Train a model on a single node and return metrics.

    Args:
        config: Hyperparameters such as lr, batch_size, epochs, and hidden_dim.
        checkpoint_dir: Optional directory containing a checkpoint to resume from.

    Returns:
        Metrics dictionary with train loss, validation accuracy, and epochs trained.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(hidden_dim=config.get("hidden_dim", 128)).to(device)
    train_loader, val_loader = get_dataloaders(batch_size=config.get("batch_size", 64))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))

    start_epoch = 0
    best_val_acc = 0.0
    epochs = config.get("epochs", 2)

    checkpoint_dir_path = Path(checkpoint_dir or config.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir) / "model.pt"
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            start_epoch = state.get("epoch", 0)
            best_val_acc = state.get("best_val_acc", 0.0)

    for epoch in range(start_epoch, epochs):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = _evaluate(model, val_loader, device)
        best_val_acc = max(best_val_acc, val_acc)

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_val_acc": best_val_acc,
        }
        torch.save(checkpoint, checkpoint_dir_path / "model.pt")

    return {
        "train_loss": train_loss,
        "val_accuracy": best_val_acc,
        "epochs": epochs,
    }
