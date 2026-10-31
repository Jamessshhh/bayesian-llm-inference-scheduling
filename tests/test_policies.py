import math

import pytest

from latent.policies.aging import AgingSJF
from latent.policies.base import FIFO
from latent.policies.sjf import SJFOracle, SJFPoint
from latent.predictor.posterior import LogNormalPrior, posterior_log_params
from latent.sim.request import Request


def make(i, arrival, true, pred=None, sigma=0.0):
    return Request(i, arrival, true,
                   predicted_service=true if pred is None else pred,
                   pred_sigma=sigma)


def test_fifo_picks_earliest_arrival():
    q = [make(1, 5.0, 1.0), make(2, 1.0, 9.0)]
    assert FIFO().select(q, now=10.0).id == 2


def test_oracle_uses_true_size():
    q = [make(1, 0.0, 9.0, pred=1.0), make(2, 0.0, 2.0, pred=5.0)]
    assert SJFOracle().select(q, now=0.0).id == 2


def test_point_uses_prediction_not_truth():
    q = [make(1, 0.0, 9.0, pred=1.0), make(2, 0.0, 2.0, pred=5.0)]
    assert SJFPoint().select(q, now=0.0).id == 1


def test_aging_lets_old_long_job_win():
    old_long = make(1, 0.0, 50.0)
    new_short = make(2, 99.0, 1.0)
    assert AgingSJF(gamma=0.0).select([old_long, new_short], now=100.0).id == 2
    assert AgingSJF(gamma=1.0).select([old_long, new_short], now=100.0).id == 1


def test_posterior_trusts_confident_prediction():
    prior = LogNormalPrior(mu=0.0, s=1.0)
    r = make(1, 0.0, 5.0, pred=5.0, sigma=1e-6)
    mu, sd = posterior_log_params(r, prior)
    assert mu == pytest.approx(math.log(5.0), rel=1e-3)
    assert sd < 1e-3


def test_posterior_shrinks_shaky_prediction_toward_prior():
    prior = LogNormalPrior(mu=0.0, s=1.0)
    r = make(1, 0.0, 5.0, pred=5.0, sigma=100.0)
    mu, sd = posterior_log_params(r, prior)
    assert mu == pytest.approx(prior.mu, abs=0.01)
    assert sd == pytest.approx(prior.s, rel=0.01)