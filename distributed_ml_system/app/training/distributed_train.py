"""Distributed training entrypoints using Ray Train."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import ray
from ray import train
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from ..config import DEFAULT_TRAIN_CONFIG
from .train_loop import train_single_node


def _train_loop_ray(config: Dict):
    train.report(train_single_node(config))


def train_distributed(config: Dict | None = None) -> Dict:
    """Launch distributed training with Ray.

    Args:
        config: Optional training configuration dictionary.

    Returns:
        Final metrics from the Ray training run.
    """

    merged_config = {**DEFAULT_TRAIN_CONFIG.to_dict(), **(config or {})}
    scaling = ScalingConfig(
        num_workers=merged_config.get("num_workers", 2),
        use_gpu=merged_config.get("use_gpu", False),
        resources_per_worker={"CPU": 1, "GPU": 1 if merged_config.get("use_gpu", False) else 0},
    )

    trainer = TorchTrainer(
        train_loop_per_worker=_train_loop_ray,
        train_loop_config=merged_config,
        scaling_config=scaling,
        run_config=RunConfig(
            name="mnist-distributed",
            storage_path=str(Path(merged_config.get("checkpoint_dir", "checkpoints")).resolve()),
        ),
    )

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    result = trainer.fit()
    return result.metrics
