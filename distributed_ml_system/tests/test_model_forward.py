from distributed_ml_system.app.models.classifier import SimpleCNN
import torch


def test_model_forward_shape():
    model = SimpleCNN()
    dummy = torch.randn(2, 1, 28, 28)
    output = model(dummy)
    assert output.shape == (2, 10)
