from distributed_ml_system.app.training.train_loop import train_single_node


def test_train_single_node_runs():
    metrics = train_single_node({"epochs": 1, "batch_size": 16})
    assert "val_accuracy" in metrics
    assert metrics["epochs"] == 1
