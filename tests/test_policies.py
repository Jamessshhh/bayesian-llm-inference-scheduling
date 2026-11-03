import math

import numpy as np
import pytest

from latent.policies.aging import AgingSJF
from latent.policies.base import FIFO
from latent.policies.sjf import SJFOracle, SJFPoint
from latent.predictor.posterior import LogNormalPrior, posterior_log_params
from latent.predictor.recalibrate import SigmaRecalibrator
from latent.sim.request import Request


def make(i, arrival, true, pred=None, sigma=0.0):
    return Request(i, arrival, true,
                   predicted_service=true if pred is None else pred,
                   pred_sigma=sigma)


# ---- policies ----
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


# ---- posterior ----
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


# ---- recalibration ----
def _obs(rec, n, true_sigma, reported, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        true = float(np.exp(rng.normal(0, 1)))
        pred = true * float(np.exp(true_sigma * rng.normal()))
        rec.observe(Request(i, 0.0, true, predicted_service=pred, pred_sigma=reported))


def test_recalibrator_returns_reported_sigma_with_no_data():
    rec = SigmaRecalibrator()
    r = Request(0, 0.0, 1.0, predicted_service=1.0, pred_sigma=0.3)
    assert rec.sigma(r) == pytest.approx(0.3)


def test_recalibrator_corrects_overconfident_predictor():
    rec = SigmaRecalibrator(window=2000)
    _obs(rec, 2000, true_sigma=0.8, reported=0.2)
    r = Request(0, 0.0, 1.0, predicted_service=1.0, pred_sigma=0.2)
    assert rec.sigma(r) == pytest.approx(0.8, rel=0.1)


def test_recalibrator_corrects_underconfident_predictor():
    rec = SigmaRecalibrator(window=2000)
    _obs(rec, 2000, true_sigma=0.2, reported=0.8)
    r = Request(0, 0.0, 1.0, predicted_service=1.0, pred_sigma=0.8)
    assert rec.sigma(r) == pytest.approx(0.2, rel=0.15)




def test_posterior_without_prior_uses_prediction_directly():
    r = make(1, 0.0, 5.0, pred=5.0, sigma=0.7)
    mu, sd = posterior_log_params(r, None)
    assert mu == pytest.approx(math.log(5.0))
    assert sd == pytest.approx(0.7)



# ---- TIE re-implementation ----
from latent.policies.tie import TIE


def test_tie_penalizes_uncertain_request_with_same_median():
    sure = make(1, 0.0, 5.0, pred=5.0, sigma=0.2)
    shaky = make(2, 0.0, 5.0, pred=5.0, sigma=1.2)
    assert TIE(x_max=1e9, beta=0.0).select([shaky, sure], now=0.0).id == 1


def test_tie_with_equal_sigma_ranks_like_point_sjf():
    a = make(1, 0.0, 9.0, pred=2.0, sigma=0.6)
    b = make(2, 0.0, 1.0, pred=4.0, sigma=0.6)
    assert TIE(x_max=1e9, beta=0.3).select([b, a], now=0.0).id == 1


def test_tie_censoring_caps_the_score():
    r = make(1, 0.0, 5.0, pred=5.0, sigma=3.0)
    e, cvar = TIE(x_max=10.0)._stats(r)
    assert e <= 10.0 and cvar <= 10.0