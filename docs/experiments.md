# Experiments

Honest note on method: hypotheses were informal and were refined as results came in; they were **not pre-registered**. Many policy variants were compared and confidence intervals are not corrected for multiple comparisons, so treat small effects with caution. All comparisons are paired across seeds with 95% t-intervals.

| script | question | setting | finding |
|---|---|---|---|
| `tests/test_queueing.py` | Does the simulator match theory? | M/M/1, load 0.5 and 0.7 | Mean latency and wait within 5% of theory |
| `run.py` | How much headroom does length knowledge give, and how fast does noise erode it? | synthetic workloads, multiplicative lognormal noise | Large headroom; oracle SJF also starves long jobs |
| `run2.py` | Does posterior-based LATENT beat point-SJF and aging? | synthetic, heteroscedastic predictor | Modest gain over point-SJF; aging trades p95 for max wait |
| `run3.py` | What if the reported uncertainty is wrong? | synthetic, over/underconfident and uninformative sigma | Safe when overconfident; can be worse than point-SJF when underconfident or uninformative |
| `run4.py` | Can online recalibration repair that? | same as run3 | Repairs harmful cases; neutral when calibrated |
| `scripts/train_predictor.py` | Is the learned predictor informative and honest? | ShareGPT, 60/20/20 split | R² 0.44; sigma informative; intervals slightly conservative |
| `run5.py` | Does it work on real prompts? | ShareGPT predictions, loads 0.8-0.95 | SJF hurts p99 vs FIFO; LATENT beats point-SJF by a few percent |
| `run6.py` | LATENT vs aging-SJF across the trade-off | gamma sweep, load 0.9 | p99 better at every gamma; mean slightly worse with strong aging |
| `run7.py`, `run8.py` | Does a TIE-style CVaR term help? | TIE score re-implemented, gamma sweep | No clear benefit beyond expected-cost ranking |

Not yet tested: bursty arrivals, prefill modelling, batching, preemption, mid-generation re-estimation, other loads for the TIE comparison, TIE's adaptive beta with batch size > 1.