import pandas as pd

from latent.policies.base import FIFO
from latent.policies.sjf import SJFOracle, SJFPoint
from latent.predictor.noisy import add_noisy_predictions
from latent.sim.metrics import summarize
from latent.sim.simulator import Simulator
from latent.workload.synthetic import make_workload

N = 50_000
SEED = 7
KINDS = ["exponential", "lognormal", "bimodal"]
LOADS = [0.8, 0.9]
SIGMAS = [0.25, 0.5, 1.0, 2.0]


def run_one(policy, kind, load, sigma):
    reqs = make_workload(kind, N, load, seed=SEED)   # fresh requests each run
    add_noisy_predictions(reqs, sigma)
    return summarize(Simulator(policy).run(reqs))


rows = []
for kind in KINDS:
    for load in LOADS:
        base = {"workload": kind, "load": load}
        rows.append({**base, "policy": "fifo", "sigma": None,
                     **run_one(FIFO(), kind, load, 0.0)})
        rows.append({**base, "policy": "sjf_oracle", "sigma": 0.0,
                     **run_one(SJFOracle(), kind, load, 0.0)})
        for s in SIGMAS:
            rows.append({**base, "policy": "sjf_point", "sigma": s,
                         **run_one(SJFPoint(), kind, load, s)})

df = pd.DataFrame(rows)
cols = ["workload", "load", "policy", "sigma",
        "mean_latency", "p95_latency", "p99_latency", "max_wait"]
pd.set_option("display.width", 200)
print(df[cols].round(2).to_string(index=False))
df.to_csv("results/baselines.csv", index=False)