from __future__ import annotations
from typing import List

from latent.policies.base import Policy
from latent.sim.request import Request


class SJFOracle(Policy):
    """Shortest job first with PERFECT knowledge. Upper bound on headroom."""
    name = "sjf_oracle"

    def select(self, queue: List[Request], now: float) -> Request:
        return min(queue, key=lambda r: (r.service_time, r.id))


class SJFPoint(Policy):
    """Shortest job first using the (noisy) point prediction."""
    name = "sjf_point"

    def select(self, queue: List[Request], now: float) -> Request:
        return min(queue, key=lambda r: (r.predicted_service, r.id))