from __future__ import annotations
from typing import List

import numpy as np

from latent.sim.request import Request


def add_noisy_predictions(requests: List[Request], sigma: float, seed: int = 123) -> None:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, len(requests))
    for r, e in zip(requests, noise):
        r.predicted_service = r.service_time * float(np.exp(e))


def add_hetero_predictions(requests: List[Request],
                           sigmas=(0.2, 0.6, 1.2),
                           probs=(0.4, 0.4, 0.2),
                           report_scale: float = 1.0,
                           informative: bool = True,
                           seed: int = 123) -> None:
    """Each request has a hidden difficulty class with its own TRUE error level.
    The predictor REPORTS a sigma that may be wrong:
      report_scale < 1  -> overconfident   (claims less uncertainty than it has)
      report_scale > 1  -> underconfident
      informative=False -> reported sigmas are shuffled across requests, so they
                           carry no information about which predictions are shaky.
    """
    rng = np.random.default_rng(seed)
    n = len(requests)
    cls = rng.choice(len(sigmas), size=n, p=probs)
    noise = rng.normal(0.0, 1.0, n)
    true_sig = np.array(sigmas)[cls]
    reported = true_sig * report_scale
    if not informative:
        reported = rng.permutation(reported)
    for r, ts, rs, z in zip(requests, true_sig, reported, noise):
        r.predicted_service = r.service_time * float(np.exp(ts * z))
        r.pred_sigma = float(rs)