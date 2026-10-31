from __future__ import annotations
from typing import List

from latent.sim.request import Request


class Policy:
    """A policy picks which queued request runs next."""
    name = "base"

    def select(self, queue: List[Request], now: float) -> Request:
        raise NotImplementedError


class FIFO(Policy):
    name = "fifo"

    def select(self, queue: List[Request], now: float) -> Request:
        return min(queue, key=lambda r: (r.arrival_time, r.id))