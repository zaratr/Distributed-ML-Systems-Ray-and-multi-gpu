"""CLI to run distributed training."""
from __future__ import annotations

from ..config import DEFAULT_TRAIN_CONFIG
from ..training.distributed_train import train_distributed


if __name__ == "__main__":
    metrics = train_distributed(DEFAULT_TRAIN_CONFIG.to_dict())
    print(metrics)
