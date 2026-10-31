from __future__ import annotations
from typing import Dict, List
import numpy as np
from latent.sim.request import Request


def summarize(done: List[Request], warmup_frac: float = 0.1) -> Dict[str, float]:
    d = done[int(len(done) * warmup_frac):]
    lat = np.array([r.latency for r in d])
    wait = np.array([r.wait for r in d])
    slow = np.array([r.latency / r.service_time for r in d])
    span = done[-1].finish_time - done[0].arrival_time
    return {
        "n": len(d),
        "mean_latency": float(lat.mean()),
        "p50_latency": float(np.percentile(lat, 50)),
        "p95_latency": float(np.percentile(lat, 95)),
        "p99_latency": float(np.percentile(lat, 99)),
        "mean_wait": float(wait.mean()),
        "max_wait": float(wait.max()),
        "mean_slowdown": float(slow.mean()),
        "p99_slowdown": float(np.percentile(slow, 99)),
        "throughput": len(done) / span,
    }