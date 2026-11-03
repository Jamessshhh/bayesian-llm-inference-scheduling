# Design

## Pipeline
Workload -> predictor -> policy -> simulator -> metrics.

## Modules
- `latent/sim/request.py`: `Request` with true `service_time` (hidden from policies except the oracle), `predicted_service` (median), `pred_sigma`.
- `latent/sim/simulator.py`: single worker, non-preemptive. When the server frees, the policy picks from everything that has arrived; `on_complete` is called after each job so adaptive policies can learn from the now-known true size.
- `latent/policies/`: `FIFO`, `SJFOracle`, `SJFPoint`, `AgingSJF`, `Latent`, `TIE`.
- `latent/predictor/`: synthetic noisy predictors, Bayesian posterior, online sigma recalibration, learned TF-IDF + ridge length predictor.
- `latent/workload/`: synthetic and ShareGPT workloads.

## Key decisions
- Non-preemptive single server first: a controlled setting that isolates the scheduling decision under uncertainty (the classical M/G/1 setting, which also allows validation against theory).
- Policies may only read `predicted_service` and `pred_sigma`; only `SJFOracle` reads the true size.
- Service time = output tokens, so prefill is ignored.
- Aging uses `score / (1 + gamma * wait)` for all policies, so comparisons are at equal gamma.
- Learned predictor outputs a conditional distribution, so LATENT applies no extra prior shrinkage to it.
- Held-out predictions are saved to `data/` and resampled with replacement to build workloads.