import pytest

from latent.policies.base import FIFO
from latent.sim.metrics import summarize
from latent.sim.simulator import Simulator
from latent.workload.synthetic import mm1_workload


@pytest.mark.parametrize("lam", [0.5, 0.7])
def test_fifo_matches_mm1_theory(lam):
    mu = 1.0
    reqs = mm1_workload(200_000, lam, mu, seed=42)
    stats = summarize(Simulator(FIFO()).run(reqs))

    expected_latency = 1.0 / (mu - lam)          # W  = 1/(mu - lambda)
    expected_wait = lam / (mu * (mu - lam))      # Wq = rho/(mu - lambda)

    assert stats["mean_latency"] == pytest.approx(expected_latency, rel=0.05)
    assert stats["mean_wait"] == pytest.approx(expected_wait, rel=0.05)