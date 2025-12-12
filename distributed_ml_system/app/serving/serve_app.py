"""Ray Serve application for distributed inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import ray
from fastapi import FastAPI, Request
from ray import serve
import torch

from ..models.classifier import SimpleCNN

app = FastAPI()


def load_model(checkpoint_dir: str) -> torch.nn.Module:
    model = SimpleCNN()
    checkpoint_path = Path(checkpoint_dir) / "model.pt"
    state = torch.load(checkpoint_path, map_location="cpu") if checkpoint_path.exists() else None
    if state:
        model.load_state_dict(state["model_state"])
    model.eval()
    return model


@serve.deployment(route_prefix="/model", num_replicas=1, ray_actor_options={"num_gpus": 0})
@serve.ingress(app)
class InferenceDeployment:
    """Serve deployment wrapping the classifier model."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.model = load_model(checkpoint_dir)

    @app.post("/predict")
    async def predict(self, request: Request) -> Dict[str, Any]:
        payload = await request.json()
        inputs: List[List[float]] = payload["inputs"]
        tensor = torch.tensor(inputs, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).tolist()
        return {"predictions": preds}


def deploy_serve_app(checkpoint_dir: str = "checkpoints") -> str:
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    serve.start(detached=True)
    InferenceDeployment.options(init_args=(checkpoint_dir,)).deploy()
    handle = serve.get_deployment_handle("InferenceDeployment")
    return str(handle)
