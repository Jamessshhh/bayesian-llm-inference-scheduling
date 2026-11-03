"""Head-to-head: LATENT vs a re-implementation of TIE's score, on real ShareGPT
predictions. All comparisons are paired across seeds."""
import numpy as np
import pandas as pd
from scipy import stats

from latent.policies.aging import AgingSJF
from latent.policies.base import FIFO
from latent.policies.latent_policy import Latent
from latent.policies.sjf import SJFOracle, SJFPoint
from latent.policies.tie import TIE
from latent.sim.metrics import summarize
from latent.sim.simulator import Simulator
from latent.workload.sharegpt import load_predictions, make_sharegpt_workload

N = 15_000
SEEDS = list(range(10))
LOAD = 0.9
MAX_TOKENS = 4096                       # the data prep filtered replies to <= 4096
df = load_predictions()
X_MAX = MAX_TOKENS / df["out_tokens"].mean()    # censor point in normalized units


def no_aging():
    return {
        "fifo": FIFO(),
        "sjf_oracle": SJFOracle(),
        "sjf_point": SJFPoint(),
        "latent": Latent(prior=None),
        "tie(beta=0, SEPT)": TIE(X_MAX, beta=0.0),
        "tie(beta=0.1)": TIE(X_MAX, beta=0.1),
        "tie(beta=0.5)": TIE(X_MAX, beta=0.5),
        "tie(adaptive)": TIE(X_MAX, beta=None),
    }


def with_aging(g):
    return {
        f"aging_sjf(g={g})": AgingSJF(gamma=g),
        f"latent(g={g})": Latent(prior=None, gamma=g),
        f"tie_adaptive(g={g})": TIE(X_MAX, beta=None, gamma=g),
        f"tie_sept(g={g})": TIE(X_MAX, beta=0.0, gamma=g),
    }


GAMMAS = [0.02, 0.1]
rows = []
for seed in SEEDS:
    sets = [no_aging()] + [with_aging(g) for g in GAMMAS]
    for pols in sets:
        for name, pol in pols.items():
            reqs = make_sharegpt_workload(df, N, LOAD, seed=seed)
            rows.append({"seed": seed, "policy": name,
                         **summarize(Simulator(pol).run(reqs))})
res = pd.DataFrame(rows)
res.to_csv("results/tie_comparison_raw.csv", index=False)


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():+.1f} ± {h:.1f}"


def paired(base, names, title):
    piv = res.pivot(index="seed", columns="policy")
    out = []
    for n in names:
        row = {"policy": n}
        for m, lab in (("mean_latency", "mean"), ("p95_latency", "p95"),
                       ("p99_latency", "p99"), ("max_wait", "max_wait")):
            row[lab] = ci(100 * (piv[m][n] / piv[m][base] - 1))
        out.append(row)
    pd.set_option("display.width", 200)
    print(f"\n=== {title} (load {LOAD}; paired % change, negative = better) ===")
    print(pd.DataFrame(out).to_string(index=False))


paired("sjf_point", ["latent", "tie(beta=0, SEPT)", "tie(beta=0.1)", "tie(beta=0.5)", "tie(adaptive)"],
       "vs SJF-point, no aging")
paired("latent", ["tie(beta=0, SEPT)", "tie(beta=0.1)", "tie(beta=0.5)", "tie(adaptive)"],
       "TIE variants vs LATENT, no aging")
for g in GAMMAS:
    paired(f"aging_sjf(g={g})", [f"latent(g={g})", f"tie_sept(g={g})", f"tie_adaptive(g={g})"],
           f"vs AgingSJF at gamma={g}")