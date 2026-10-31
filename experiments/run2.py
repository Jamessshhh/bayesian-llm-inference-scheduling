import pandas as pd

from latent.policies.aging import AgingSJF
from latent.policies.base import FIFO
from latent.policies.latent_policy import Latent
from latent.policies.sjf import SJFOracle, SJFPoint
from latent.predictor.noisy import add_hetero_predictions
from latent.predictor.posterior import fit_log_prior
from latent.sim.metrics import summarize
from latent.sim.simulator import Simulator
from latent.workload.synthetic import make_workload

N = 50_000
SEEDS = [7, 8, 9]
KINDS = ["lognormal", "bimodal"]
LOAD = 0.9


def build_policies(prior):
    return [
        ("fifo", FIFO()),
        ("sjf_oracle", SJFOracle()),
        ("sjf_point", SJFPoint()),
        ("aging_sjf(g=0.1)", AgingSJF(gamma=0.1)),
        ("latent_mean(g=0)", Latent(prior, alpha=None, gamma=0.0)),
        ("latent_mean(g=0.02)", Latent(prior, alpha=None, gamma=0.02)),
        ("latent_mean(g=0.1)", Latent(prior, alpha=None, gamma=0.1)),
        ("latent_q0.9(g=0)", Latent(prior, alpha=0.9, gamma=0.0)),
    ]


rows = []
for kind in KINDS:
    # fit the prior on a separate "historical" workload, not the one being scheduled
    prior = fit_log_prior(make_workload(kind, 20_000, LOAD, seed=999))
    for seed in SEEDS:
        for label, pol in build_policies(prior):
            reqs = make_workload(kind, N, LOAD, seed=seed)
            add_hetero_predictions(reqs, seed=seed + 100)
            st = summarize(Simulator(pol).run(reqs))
            rows.append({"workload": kind, "policy": label, **st})

df = pd.DataFrame(rows)
cols = ["mean_latency", "p95_latency", "p99_latency", "max_wait", "p99_slowdown"]
out = df.groupby(["workload", "policy"], sort=False)[cols].mean().round(2).reset_index()
pd.set_option("display.width", 200)
print(out.to_string(index=False))
df.to_csv("results/latent_v1.csv", index=False)