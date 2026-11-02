# LATENT: Uncertainty-Aware Scheduling for LLM Inference

**Question:** When an LLM server can only *predict* how long each request will take, and the predictions are noisy, how should it schedule? Does modeling prediction uncertainty beat trusting a single point estimate?

This is a controlled simulation study, not a production system. See [Limitations](#limitations).

## Why this matters

Shortest-job-first (SJF) scheduling cuts latency a lot, but it needs job sizes. For LLM requests the output length is unknown until generation finishes, so schedulers rely on predictions. A wrong "this is short" prediction puts a long job ahead of many short ones.

## Approach

A discrete-event simulator with a single, non-preemptive worker, which gives a controlled environment before adding batching and preemption.

```
Workload -> Length predictor (median + uncertainty) -> Scheduling policy -> Simulator -> Metrics
```

- **Workloads:** exponential, lognormal (heavy tail), and bimodal (85% short, 15% long), all at utilization 0.9.
- **Predictor:** synthetic. Each request has a hidden difficulty class with its own error level, and the predictor reports a median and a sigma.
- **LATENT policy:** combines a population prior with each request's noisy prediction (Bayesian shrinkage in log space) and ranks jobs by posterior expected cost, with optional aging.
- **Baselines:** FIFO, oracle SJF (perfect knowledge, an upper bound), point-estimate SJF, and point-estimate SJF with aging.
- **Validation:** the simulator reproduces M/M/1 theory within 5% (`tests/test_queueing.py`).

## Results so far

**1. Headroom is large, and prediction error eats it.** On a lognormal workload at load 0.9, perfect SJF cuts p95 latency about 5x versus FIFO. Noisy predictions lose much of that, and SJF starves: max wait grows several-fold versus FIFO.

**2. Posterior-based LATENT beats point-estimate SJF, modestly.** With a calibrated predictor, 10 seeds, mean change vs. SJF-point (negative = better):

| workload | mean latency | p99 latency |
|---|---|---|
| lognormal | -2.6% ± 1.0 | -3.6% ± 1.9 |
| bimodal | -4.4% ± 0.7 | -7.4% ± 2.4 |

**3. It is only as good as its uncertainty.** If the predictor's uncertainty is overconfident, LATENT is never worse than point-SJF. If it is badly underconfident (x4) or uninformative, LATENT can be worse (up to +13.5% p99 on lognormal).

**4. Aging trades tail for fairness.** Aging cuts max wait sharply (bimodal: 1727 to 444) but raises p95 roughly 2x.

**Negative result:** ranking by a risk-averse quantile (instead of the posterior mean) did not improve on the posterior mean.

## Limitations

- Simulation only, with a single non-preemptive worker. No batching, preemption, KV-cache or prefill/decode modeling yet.
- The predictor is synthetic, and its uncertainty is calibrated by construction in the main experiments.
- Gains are modest and workload-dependent. Do not read these results as production GPU speedups.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib pyyaml pytest
pip install -e .
pytest -q
python experiments/run.py     # headroom: oracle vs noisy SJF
python experiments/run2.py    # LATENT v1 vs baselines
python experiments/run3.py    # miscalibration sweep with CIs
```

## Roadmap

- [ ] Online recalibration of predictor uncertainty
- [ ] Real length predictor trained on ShareGPT / LMSYS-Chat
- [ ] Token-level preemption and batching
- [ ] Validation on a real inference server (e.g. vLLM)