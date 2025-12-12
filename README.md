# Distributed ML System with Ray and Multi-GPU Support

This project implements a distributed ML training and inference system using **Ray**, **PyTorch**, and **Ray Serve**. It provides single-node and distributed training, hyperparameter tuning, and scalable inference deployments.

## Features
- MNIST data pipeline with PyTorch `DataLoader`s
- Baseline training loop with checkpointing and validation metrics
- Ray Train integration for multi-worker or multi-GPU execution
- Ray Tune hyperparameter search
- Ray Serve deployment for scalable inference
- Simple observability helper to time operations

## Getting Started
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Local Training
```bash
python -m distributed_ml_system.app.scripts.run_distributed_train
```

### Hyperparameter Tuning
```bash
python -m distributed_ml_system.app.scripts.run_tuning
```

### Serving
```bash
python -m distributed_ml_system.app.scripts.run_serve
```

The Ray Dashboard is available at `http://localhost:8265` when Ray is running.
