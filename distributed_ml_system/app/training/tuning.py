"""Hyperparameter tuning using Ray Tune."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import ray
from ray import tune
from ray.tune import Tuner

from ..config import DEFAULT_TRAIN_CONFIG
from .train_loop import train_single_node


def tuning_entrypoint(config: Dict):
    metrics = train_single_node(config)
    tune.report(**metrics)


def run_tuning(num_samples: int = 5) -> Dict:
    """Run Ray Tune hyperparameter search.

    Args:
        num_samples: Number of samples to evaluate.

    Returns:
        Dictionary containing best config and metric.
    """

    base_config = DEFAULT_TRAIN_CONFIG.to_dict()
    param_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
        "hidden_dim": tune.choice([64, 128, 256]),
        "epochs": tune.choice([1, 2]),
        "checkpoint_dir": base_config.get("checkpoint_dir", "checkpoints"),
    }

    tuner = Tuner(
        tuning_entrypoint,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric="val_accuracy",
            mode="max",
        ),
        run_config=ray.air.RunConfig(
            name="mnist-tuning",
            storage_path=str(Path(base_config.get("checkpoint_dir", "checkpoints")).resolve()),
        ),
    )

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    results = tuner.fit()
    best_result = results.get_best_result(metric="val_accuracy", mode="max")
    return {
        "best_config": best_result.config,
        "best_metrics": best_result.metrics,
    }
