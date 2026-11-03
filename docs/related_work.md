# Related Work

Verification level is marked for each entry. "Read" means I read the full text; "abstract only" means only the abstract or a summary was seen; "cited" means it is cited by a paper I read but I have not opened it.

## Closest prior work

**TIE: Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions** (Zheng et al., arXiv 2604.00499; code: github.com/Hyzheng-code/TIE). *Read.*
Predicts a log-t distribution over output length per prompt, schedules by Tail Inflated Expectation (expectation + adaptive-weight CVaR), adds waiting-time decay against starvation, implemented in vLLM. Evaluated against FCFS, SSJF and LTR on LMSYS-Chat-1M, ShareGPT and Alpaca. This is the closest work to LATENT: the idea of ranking by an uncertainty-aware score derived from a predicted distribution is **not novel to this repo**.

**Adaptively Robust LLM Inference Optimization under Prediction Uncertainty** (Chen et al., arXiv 2508.14544). *Abstract only.*
Uses prediction intervals; a conservative algorithm uses the upper bound, an adaptive one starts from the lower bound and refines during inference, with a competitive-ratio guarantee. Focus is memory/total latency.

**MAPS: Memory-Aware Predictive Scheduling Framework for LLM Serving** (arXiv 2609.15359). *Search snippet only.*
Conformal calibration of output-length upper bounds from recent runtime prediction errors. Related to the online recalibration in this repo.

**Queueing, Predictions, and LLMs: Challenges and Open Problems** (Mitzenmacher & Shahout, arXiv 2503.07545). *Abstract only.*
Survey of scheduling with predictions in queues and LLM systems; lists predictions-as-distributions as an open question.

## Learned length prediction and scheduling (cited by TIE; not opened)
- SSJF: proxy-model sequence-length prediction (Qiu et al., 2024).
- LTR: learning-to-rank scheduling (Fu et al., 2024).
- Response Length Perception and Sequence Scheduling (Zheng et al., 2023).
- TRAIL: iterative, embedding-based length prediction with preemption (Shahout et al., 2025).
- ELIS (Choi et al., 2025), S3 (Jin et al., 2023).

## Systems background (cited by TIE; not opened)
- Orca / continuous batching (Yu et al., 2022); vLLM / PagedAttention (Kwon et al., 2023).

## Classical foundations
- Shortest Expected Processing Time (SEPT) for stochastic scheduling (Weber, 1983, as cited by TIE).

## How LATENT relates
LATENT does **not** propose a new scheduling principle. It is a controlled single-server simulation study of uncertainty-aware SJF with aging, using real ShareGPT output lengths and a learned predictor. Possible contributions, each still to be checked against the papers above: (1) oracle-based headroom analysis, (2) p99 / max-wait behaviour, where SJF-style policies are worse than FIFO even with an oracle on this length distribution, (3) paired comparison against aging-SJF across the aging trade-off. Sensitivity to miscalibrated uncertainty and online recalibration overlap with existing work (see TIE's robustness appendix and MAPS) and should not be claimed as novel without checking.