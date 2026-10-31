from __future__ import annotations
from typing import List

from latent.policies.base import Policy
from latent.sim.request import Request


class AgingSJF(Policy):
    """score = predicted_cost / (1 + gamma * wait) ; lowest score runs first.
    gamma = 0 is plain SJF; larger gamma lets long-waiting jobs jump the queue."""
    name = "aging_sjf"

    def __init__(self, gamma: float = 0.05):
        self.gamma = gamma

    def select(self, queue: List[Request], now: float) -> Request:
        g = self.gamma
        return min(queue, key=lambda r: (r.predicted_service / (1.0 + g * (now - r.arrival_time)), r.id))