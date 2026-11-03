# LATENT: Uncertainty-Aware Scheduling for LLM Inference

**Question:** When an LLM server can only *predict* how long each request will take, and the predictions are noisy, how should it schedule? Does modeling prediction uncertainty beat trusting a single point estimate?

> **Status:** a controlled simulation study, not a production system, and **not a new method**: uncertainty-aware SJF scheduling already exists (see [Related work](#related-work), especially TIE). This repo re-examines it in a controlled single-server simulator with real ShareGPT lengths.

## TL;DR

- Length-aware scheduling (SJF-style) cuts mean and p95 latency a lot on real ShareGPT output lengths, but **hurts p99 and starvation versus FIFO**, even with a perfect oracle.
- A scheduler that uses the predictor's *uncertainty* (LATENT) beats point-estimate SJF **modestly but consistently**: about 1-3% on mean and 4-6% on p99.
- With aging, LATENT lowers p99 by about 4-6% versus plain aging-SJF at equal aging strength, at about equal mean latency (about 1% worse mean when aging is strong).
- The gains depend on the predictor's uncertainty being roughly honest. A bad uncertainty estimate can make LATENT *worse* than point-SJF, and online recalibration repairs this on synthetic workloads.

## Approach

A discrete-event simulator with one non-preemptive worker.

```
Workload -> Length predictor (median + sigma) -> Scheduling policy -> Simulator -> Metrics
```

- **Policies:** FIFO, oracle SJF (upper bound), point-estimate SJF, aging-SJF, and **LATENT**.
- **LATENT:** ranks requests by the posterior *expected* cost, `exp(mu + sigma^2/2)`, with optional aging `score = cost / (1 + gamma * wait)`. For synthetic predictors it first shrinks the prediction toward a population prior. For the learned predictor the prediction is already a conditional distribution, so no extra shrinkage is applied.
- **Workloads:** synthetic (exponential, lognormal, bimodal) and **real ShareGPT** prompts with a learned length predictor.
- **Learned predictor:** TF-IDF + ridge regression on log output length, with a second model for per-prompt sigma, calibrated on held-out data. On held-out prompts: R² = 0.44, Spearman = 0.56, and the reported sigma is informative (actual error std rises 0.57 -> 0.70 -> 1.03 across sigma terciles).
- **Validation:** the simulator reproduces M/M/1 theory within 5% (`tests/test_queueing.py`).
- All comparisons are paired across seeds, with 95% confidence intervals.

## Results

### 1. Headroom exists, but SJF hurts the extreme tail on real lengths
ShareGPT, load 0.9 (service time = output tokens):

| policy | mean | p95 | p99 | max wait |
|---|---|---|---|---|
| FIFO | 7.8 | 24.2 | **35.4** | **45** |
| SJF oracle | 4.4 | 12.6 | 53.7 | 569 |
| SJF point | 5.7 | 18.0 | 76.8 | 583 |

Real output lengths are far less heavy-tailed than the synthetic lognormal workload. Here SJF improves the mean and p95 but starves long requests, so even the oracle has a worse p99 than FIFO. On the heavy-tailed synthetic lognormal workload SJF improves p99 as well, so conclusions depend on the length distribution.

### 2. LATENT vs point-estimate SJF (real data, no aging)
Paired % change, 10 seeds, 95% CI (negative = better):

| load | mean | p99 |
|---|---|---|
| 0.80 | -1.0 ± 0.3 | -3.8 ± 1.8 |
| 0.90 | -2.2 ± 0.6 | -5.3 ± 2.9 |
| 0.95 | -3.0 ± 1.2 | -5.9 ± 3.4 |

### 3. LATENT vs aging-SJF across the trade-off (load 0.9)
Paired % change at equal aging strength gamma:

| gamma | mean | p95 | p99 | max wait |
|---|---|---|---|---|
| 0 | -2.2 ± 0.6 | -1.5 ± 1.3 | -5.3 ± 2.9 | -5.9 ± 8.2 |
| 0.01 | -0.9 ± 0.2 | -1.2 ± 1.5 | -5.2 ± 2.0 | -14.2 ± 13.1 |
| 0.05 | +0.6 ± 0.3 | -2.5 ± 0.8 | -5.7 ± 1.7 | -9.4 ± 9.6 |
| 0.3 | +1.4 ± 0.1 | -1.3 ± 0.8 | -4.5 ± 0.9 | -11.7 ± 13.0 |
| 1.0 | +1.4 ± 0.1 | -0.8 ± 0.7 | -3.6 ± 1.0 | -9.7 ± 12.2 |

p99 improves at every gamma tested. Mean latency improves with weak aging and is about 1% worse with strong aging. Max-wait improvements are consistent in sign but statistically weak.

![Mean vs tail trade-off](docs/pareto_tradeoff.png)

### 4. Sensitivity to wrong uncertainty (synthetic, heteroscedastic predictor)
With a calibrated predictor, LATENT improves on point-SJF by 3-4% on mean and 4-7% on p99. If the predictor is badly underconfident (x4) or its sigma is uninformative, LATENT can be worse (up to +13.5% p99 on the lognormal workload). **Online recalibration** (learning the real error scale from finished requests) turns the worst case into a small gain and costs nothing when the predictor is already calibrated. On the real-data predictor, which was calibrated offline, recalibration brought no benefit.

### 5. Negative results
- Ranking by a risk-averse quantile did not beat ranking by the posterior mean.
- Online recalibration did not help when the predictor was already calibrated.
- Aging trades tail for mean. No policy tested beats FIFO on p99 on the real workload.

## Related work

The closest prior work is **TIE** (Zheng et al., arXiv 2604.00499), which predicts a per-prompt output-length distribution and schedules by expected length plus a CVaR tail term, with starvation prevention, in vLLM. Related: Chen et al. 2025 (interval predictions, arXiv 2508.14544), MAPS (conformal length bounds), and the Mitzenmacher & Shahout survey. Details and verification levels are in [`docs/related_work.md`](docs/related_work.md). This repo does not claim a new scheduling principle; its contribution is a reproducible controlled study.

## Limitations

- **Simulation only**, one non-preemptive worker. No batching, preemption, KV-cache or memory effects.
- **Service time = output tokens.** Prefill is not modeled. Token counts are approximate (characters / 4), and only the first exchange of each conversation is used.
- **Poisson arrivals**; real traffic is burstier.
- The learned predictor is deliberately simple (TF-IDF + ridge). Better predictors would shift results.
- Gains are modest. Do not read them as production GPU speedups.
- Related work (e.g. learned-ranking and preemptive LLM schedulers, scheduling-with-predictions theory) has not yet been systematically compared against.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q

# synthetic experiments
python experiments/run.py     # headroom: oracle vs noisy SJF
python experiments/run2.py    # LATENT v1 vs baselines
python experiments/run3.py    # miscalibration sweep
python experiments/run4.py    # recalibration

# real data (ShareGPT)
mkdir -p data
curl -L -o data/ShareGPT_V3_unfiltered_cleaned_split.json \
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
python scripts/prepare_sharegpt.py
python scripts/train_predictor.py
python experiments/run5.py    # real-data comparison
python experiments/run6.py    # aging trade-off sweep + figure
```

## Roadmap

- [ ] Related-work review and comparison against a published learned-ranking baseline
- [ ] Bursty arrivals and real arrival traces
- [ ] Separate prefill and decode costs
- [ ] Token-level preemption, batching, and re-estimating length mid-generation