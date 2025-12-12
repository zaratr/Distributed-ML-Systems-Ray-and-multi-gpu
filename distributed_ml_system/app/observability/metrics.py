"""Lightweight observability helpers."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator


@contextmanager
def track_time(metrics: Dict[str, float], key: str) -> Iterator[None]:
    start = time.time()
    try:
        yield
    finally:
        metrics[key] = time.time() - start
