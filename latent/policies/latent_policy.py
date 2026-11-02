from __future__ import annotations
import math
from statistics import NormalDist
from typing import Dict, List, Optional, Tuple

from latent.policies.base import Policy
from latent.predictor.posterior import LogNormalPrior, posterior_log_params
from latent.predictor.recalibrate import SigmaRecalibrator
from latent.sim.request import Request


class Latent(Policy):
    """Uncertainty-aware scheduling with aging and optional online recalibration.

    1. Posterior: combine population prior + this request's noisy prediction.
       If a recalibrator is given, the REPORTED sigma is replaced by a sigma
       learned from finished requests.
    2. Cost c:   alpha=None -> posterior MEAN  exp(mu + var/2)
                 alpha=q    -> posterior q-quantile exp(mu + z_q * std)
    3. Score:    c / (1 + gamma * wait) ; lowest score runs first.
    """
    name = "latent"

    def __init__(self, prior: LogNormalPrior, alpha: Optional[float] = None,
                 gamma: float = 0.0, recal: Optional[SigmaRecalibrator] = None,
                 refresh: int = 200):
        self.prior = prior
        self.alpha = alpha
        self.gamma = gamma
        self.recal = recal
        self.refresh = refresh
        self.z = None if alpha is None else NormalDist().inv_cdf(alpha)
        self._cost: Dict[int, Tuple[int, float]] = {}
        self._version = 0
        self._completed = 0

    def _c(self, r: Request) -> float:
        hit = self._cost.get(r.id)
        if hit is not None and hit[0] == self._version:
            return hit[1]
        sigma = self.recal.sigma(r) if self.recal is not None else None
        mu, sd = posterior_log_params(r, self.prior, sigma)
        c = math.exp(mu + 0.5 * sd * sd) if self.z is None else math.exp(mu + self.z * sd)
        self._cost[r.id] = (self._version, c)
        return c

    def on_complete(self, req: Request, now: float) -> None:
        self._cost.pop(req.id, None)
        if self.recal is not None:
            self.recal.observe(req)
            self._completed += 1
            if self._completed % self.refresh == 0:
                self._version += 1      # invalidate cached costs

    def select(self, queue: List[Request], now: float) -> Request:
        g = self.gamma
        return min(queue, key=lambda r: (self._c(r) / (1.0 + g * (now - r.arrival_time)), r.id))