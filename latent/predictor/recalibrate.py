from __future__ import annotations
import math
from collections import defaultdict, deque

from latent.sim.request import Request


class SigmaRecalibrator:
    """Learns how wrong the predictor really is, from finished requests.

    Requests are grouped by their REPORTED sigma (log-spaced buckets). For each
    bucket we keep a rolling window of log-errors e = log(true) - log(pred) and
    estimate the real error scale as sqrt(mean(e^2)). With few observations we
    blend toward the reported sigma (pseudo-count prior)."""

    def __init__(self, window: int = 500, pseudo: int = 30, bin_width: float = 0.35):
        self.window = window
        self.pseudo = pseudo
        self.bin_width = bin_width
        self._errs = defaultdict(lambda: deque(maxlen=window))
        self._sumsq = defaultdict(float)

    def _key(self, sigma: float) -> int:
        return int(round(math.log(max(sigma, 1e-6)) / self.bin_width))

    def observe(self, r: Request) -> None:
        k = self._key(r.pred_sigma)
        e = math.log(r.service_time) - math.log(r.predicted_service)
        buf = self._errs[k]
        if len(buf) == buf.maxlen:                 # about to evict the oldest
            self._sumsq[k] -= buf[0] ** 2
        buf.append(e)
        self._sumsq[k] += e * e

    def sigma(self, r: Request) -> float:
        rep = r.pred_sigma
        buf = self._errs.get(self._key(rep))
        n = len(buf) if buf else 0
        if n == 0:
            return rep
        mean_sq = max(self._sumsq[self._key(rep)], 0.0) / n
        return math.sqrt((n * mean_sq + self.pseudo * rep * rep) / (n + self.pseudo))