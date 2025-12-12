"""Data loading utilities for training and evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_dataloaders(batch_size: int, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Return train and validation dataloaders for MNIST.

    Args:
        batch_size: Per-iteration batch size.
        num_workers: Number of workers for the PyTorch DataLoader.

    Returns:
        Tuple of (train_loader, val_loader).
    """

    data_dir = Path("./data")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset = datasets.MNIST(str(data_dir), download=True, train=True, transform=transform)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
