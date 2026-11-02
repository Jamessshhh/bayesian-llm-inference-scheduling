from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List

import numpy as np

from latent.sim.request import Request


@dataclass
class LogNormalPrior:
    mu: float       # mean of log(service_time) across the population
    s: float        # std of log(service_time)


def fit_log_prior(history: List[Request]) -> LogNormalPrior:
    """Fit from past (already finished) requests, never from the live queue."""
    logs = np.log([r.service_time for r in history])
    return LogNormalPrior(float(logs.mean()), float(logs.std()))


def posterior_log_params(r: Request, prior: LogNormalPrior, sigma=None):
    """Noisy prediction model: log(pred) = log(true) + N(0, sigma^2).
    Large sigma -> posterior pulled toward the prior.
    `sigma` overrides the request's reported sigma (used by recalibration).
    Returns (posterior mean of log cost, posterior std of log cost)."""
    sg = r.pred_sigma if sigma is None else sigma
    sig2 = (sg or 0.0) ** 2
    s2 = prior.s ** 2
    lp = math.log(r.predicted_service)
    w = s2 / (s2 + sig2)                      # trust placed in the prediction
    mu_post = w * lp + (1.0 - w) * prior.mu
    var_post = (s2 * sig2) / (s2 + sig2)
    return mu_post, math.sqrt(var_post)