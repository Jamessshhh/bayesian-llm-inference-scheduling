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
                           seed: int = 123) -> None:
    """Each request belongs to a hidden 'difficulty class' with its own error
    level. The predictor reports a median and the (calibrated) sigma of its class,
    so some predictions are trustworthy and others are not."""
    rng = np.random.default_rng(seed)
    cls = rng.choice(len(sigmas), size=len(requests), p=probs)
    noise = rng.normal(0.0, 1.0, len(requests))
    for r, c, z in zip(requests, cls, noise):
        sg = sigmas[c]
        r.predicted_service = r.service_time * float(np.exp(sg * z))
        r.pred_sigma = float(sg)