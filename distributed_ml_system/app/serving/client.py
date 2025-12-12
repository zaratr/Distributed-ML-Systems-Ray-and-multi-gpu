"""Simple client for querying the Ray Serve endpoint."""
from __future__ import annotations

import json
from typing import List

import requests


def query_model(endpoint: str, inputs: List[List[float]]) -> dict:
    payload = {"inputs": inputs}
    response = requests.post(f"{endpoint}/predict", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    example = [[0.0] * 28 * 28]
    print(query_model("http://localhost:8000/model", example))
