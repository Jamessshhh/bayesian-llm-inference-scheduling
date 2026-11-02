from __future__ import annotations
from typing import List

from latent.sim.request import Request


class Policy:
    name = "base"

    def select(self, queue: List[Request], now: float) -> Request:
        raise NotImplementedError

    def on_complete(self, req: Request, now: float) -> None:
        """Called after a request finishes. Its true size is now observable,
        so adaptive policies may learn from it."""
        return None


class FIFO(Policy):
    name = "fifo"

    def select(self, queue, now):
        return min(queue, key=lambda r: (r.arrival_time, r.id))