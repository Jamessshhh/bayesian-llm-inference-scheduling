from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import t as student_t

from latent.policies.base import Policy
from latent.sim.request import Request


class TIE(Policy):
    """Re-implementation of the Tail Inflated Expectation score (Zheng et al.,
    arXiv 2604.00499) for our single-server simulator.

    log(length) ~ location-scale Student-t (nu = 3.5), censored at x_max.
    score = E[X] + beta * CVaR_alpha[X]       (lowest score runs first)
    beta  = fixed, or adaptive: min(0.5, max(0.1, 0.1 * queue_len / batch_size)).

    ASSUMPTIONS made here (not from the paper):
      * our predictor's (mu, sigma) are mapped to the t location/scale by matching
        the log-scale standard deviation: scale = sigma / sqrt(nu / (nu - 2)).
      * aging uses OUR form  score / (1 + gamma * wait), not the paper's
        exponential decay, so that comparisons with AgingSJF are at equal gamma.
      * batch_size = 1 because the simulator has a single worker.
    """
    name = "tie"

    def __init__(self, x_max: float, beta: Optional[float] = None, alpha: float = 0.9,
                 nu: float = 3.5, gamma: float = 0.0, batch_size: int = 1, grid: int = 2000):
        self.x_max = x_max
        self.beta = beta                      # None -> adaptive
        self.gamma = gamma
        self.batch_size = batch_size
        q = (np.arange(grid) + 0.5) / grid    # deterministic quantile grid
        self._y = student_t.ppf(q, nu)
        self._tail = q >= alpha
        self._scale = 1.0 / math.sqrt(nu / (nu - 2.0))
        self._cache: Dict[int, Tuple[float, float]] = {}

    def _stats(self, r: Request) -> Tuple[float, float]:
        hit = self._cache.get(r.id)
        if hit is None:
            mu = math.log(r.predicted_service)
            s = (r.pred_sigma or 0.0) * self._scale
            x = np.minimum(np.exp(mu + s * self._y), self.x_max)
            hit = (float(x.mean()), float(x[self._tail].mean()))
            self._cache[r.id] = hit
        return hit

    def _beta(self, queue_len: int) -> float:
        if self.beta is not None:
            return self.beta
        return min(0.5, max(0.1, 0.1 * queue_len / self.batch_size))

    def on_complete(self, req: Request, now: float) -> None:
        self._cache.pop(req.id, None)

    def select(self, queue: List[Request], now: float) -> Request:
        b = self._beta(len(queue))
        g = self.gamma

        def key(r: Request):
            e, cvar = self._stats(r)
            return ((e + b * cvar) / (1.0 + g * (now - r.arrival_time)), r.id)

        return min(queue, key=key)