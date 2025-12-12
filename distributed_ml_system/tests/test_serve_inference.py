from distributed_ml_system.app.serving.serve_app import InferenceDeployment
import torch


def test_inference_returns_predictions():
    deployment = InferenceDeployment(checkpoint_dir="checkpoints")
    dummy = torch.zeros((1, 1, 28, 28)).view(1, -1).tolist()
    response = deployment.predict.__wrapped__(deployment, type("obj", (), {"json": lambda self=None: {"inputs": dummy}})())
    # predict is async, so we call result() if coroutine
    if hasattr(response, "__await__"):
        response = response.send(None)
    assert "predictions" in response
