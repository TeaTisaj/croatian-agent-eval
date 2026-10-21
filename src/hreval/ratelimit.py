"""Token-bucket limiter + backoff.

The free tier's requests-per-second, not the monthly token cap, is what gates
this eval (~3,900 model calls). This exists so a long matrix run survives
contact with the rate limiter instead of dying halfway.
"""

from __future__ import annotations

import asyncio
import random
import time


class RateLimiter:
    def __init__(self, rate_per_sec: float, burst: int = 1):
        self._rate = rate_per_sec
        self._capacity = max(1, burst)
        self._tokens = float(self._capacity)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self._capacity, self._tokens + (now - self._updated) * self._rate)
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                await asyncio.sleep((1.0 - self._tokens) / self._rate)


def backoff_delay(attempt: int, *, base: float = 1.0, cap: float = 60.0) -> float:
    """Exponential backoff with full jitter."""
    return random.uniform(0, min(cap, base * (2 ** attempt)))
