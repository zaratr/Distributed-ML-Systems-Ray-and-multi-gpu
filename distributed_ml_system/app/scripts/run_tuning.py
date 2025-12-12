"""CLI to run Ray Tune search."""
from __future__ import annotations

from ..training.tuning import run_tuning


if __name__ == "__main__":
    results = run_tuning()
    print(results)
