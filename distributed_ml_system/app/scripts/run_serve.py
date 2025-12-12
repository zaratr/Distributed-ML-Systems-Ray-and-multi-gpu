"""CLI to start Ray Serve deployment."""
from __future__ import annotations

from ..serving.serve_app import deploy_serve_app


if __name__ == "__main__":
    handle = deploy_serve_app()
    print(f"Serve deployment started. Handle: {handle}")
