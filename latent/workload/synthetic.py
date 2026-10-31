from __future__ import annotations
from typing import List

import numpy as np

from latent.sim.request import Request


def mm1_workload(n, arrival_rate, service_rate, seed=0):
    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(1.0 / arrival_rate, n))
    services = rng.exponential(1.0 / service_rate, n)
    return [Request(i, float(a), float(s))
            for i, (a, s) in enumerate(zip(arrivals, services))]


def _sample_sizes(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "exponential":
        return rng.exponential(1.0, n)
    if kind == "lognormal":                      # heavy-ish tail
        return rng.lognormal(mean=0.0, sigma=1.5, size=n)
    if kind == "bimodal":                        # 85% short chat, 15% long generation
        is_long = rng.random(n) < 0.15
        short = rng.lognormal(mean=0.0, sigma=0.4, size=n)
        long_ = rng.lognormal(mean=np.log(20.0), sigma=0.4, size=n)
        return np.where(is_long, long_, short)
    raise ValueError(f"unknown workload kind: {kind}")


def make_workload(kind: str, n: int, load: float, seed: int = 0) -> List[Request]:
    """Poisson arrivals; service times drawn from `kind`, rescaled so the mean
    service time is 1.0. Arrival rate = load, so utilization = load."""
    rng = np.random.default_rng(seed)
    sizes = _sample_sizes(kind, n, rng)
    sizes = sizes / sizes.mean()
    arrivals = np.cumsum(rng.exponential(1.0 / load, n))
    return [Request(i, float(a), float(s))
            for i, (a, s) in enumerate(zip(arrivals, sizes))]