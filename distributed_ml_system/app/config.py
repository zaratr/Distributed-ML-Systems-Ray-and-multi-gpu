"""Centralized configuration for the distributed ML system."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TrainConfig:
    """Training defaults that can be overridden via environment variables."""

    dataset_name: str = os.getenv("DATASET_NAME", "mnist")
    model_type: str = os.getenv("MODEL_TYPE", "cnn")
    num_workers: int = int(os.getenv("NUM_WORKERS", "2"))
    use_gpu: bool = os.getenv("USE_GPU", "false").lower() == "true"
    batch_size: int = int(os.getenv("BATCH_SIZE", "64"))
    lr: float = float(os.getenv("LR", "0.001"))
    epochs: int = int(os.getenv("EPOCHS", "2"))
    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "checkpoints")
    hidden_dim: int = int(os.getenv("HIDDEN_DIM", "128"))

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


DEFAULT_TRAIN_CONFIG = TrainConfig()
