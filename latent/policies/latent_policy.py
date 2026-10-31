from __future__ import annotations
import math
from statistics import NormalDist
from typing import Dict, List, Optional

from latent.policies.base import Policy
from latent.predictor.posterior import LogNormalPrior, posterior_log_params
from latent.sim.request import Request


class Latent(Policy):
    """Uncertainty-aware scheduling with aging.

    1. Posterior: combine population prior + this request's noisy prediction.
    2. Cost c:   alpha=None -> posterior MEAN  exp(mu + var/2)  (best for average latency)
                 alpha=q    -> posterior q-quantile exp(mu + z_q * std)  (risk-averse)
    3. Score:    c / (1 + gamma * wait) ; lowest score runs first.
    """
    name = "latent"

    def __init__(self, prior: LogNormalPrior, alpha: Optional[float] = None, gamma: float = 0.0):
        self.prior = prior
        self.alpha = alpha
        self.gamma = gamma
        self.z = None if alpha is None else NormalDist().inv_cdf(alpha)
        self._cost: Dict[int, float] = {}

    def _c(self, r: Request) -> float:
        c = self._cost.get(r.id)
        if c is None:
            mu, sd = posterior_log_params(r, self.prior)
            c = math.exp(mu + 0.5 * sd * sd) if self.z is None else math.exp(mu + self.z * sd)
            self._cost[r.id] = c
        return c

    def select(self, queue: List[Request], now: float) -> Request:
        g = self.gamma
        return min(queue, key=lambda r: (self._c(r) / (1.0 + g * (now - r.arrival_time)), r.id))