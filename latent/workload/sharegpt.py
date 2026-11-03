from __future__ import annotations
from typing import List

import numpy as np
import pandas as pd

from latent.sim.request import Request


def load_predictions(path: str = "data/test_predictions.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def make_sharegpt_workload(df: pd.DataFrame, n: int, load: float, seed: int = 0) -> List[Request]:
    """Resample n real prompts (with their real output lengths and the learned
    predictor's median + sigma). Poisson arrivals at utilization `load`.
    Service time = output tokens (decode-dominated; prefill is not modelled).
    Sizes are rescaled to mean 1, so arrival rate = load."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(df), n)
    out = df["out_tokens"].to_numpy(float)[idx]
    inp = df["in_tokens"].to_numpy(float)[idx]
    mu = df["pred_mu"].to_numpy(float)[idx]
    sg = df["pred_sigma"].to_numpy(float)[idx]
    scale = out.mean()
    arrivals = np.cumsum(rng.exponential(1.0 / load, n))
    return [
        Request(i, float(arrivals[i]), float(out[i] / scale),
                input_tokens=int(inp[i]), output_tokens=int(out[i]),
                predicted_service=float(np.exp(mu[i]) / scale),
                pred_sigma=float(sg[i]))
        for i in range(n)
    ]