"""Does LATENT beat plain aging-SJF across the whole mean-vs-tail trade-off?
Sweeps the aging strength gamma for both policies, with paired comparisons."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from latent.policies.aging import AgingSJF
from latent.policies.base import FIFO
from latent.policies.latent_policy import Latent
from latent.policies.sjf import SJFOracle, SJFPoint
from latent.sim.metrics import summarize
from latent.sim.simulator import Simulator
from latent.workload.sharegpt import load_predictions, make_sharegpt_workload

N = 15_000
SEEDS = list(range(10))
LOAD = 0.9
GAMMAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0]
df = load_predictions()


def run(pol, seed):
    return summarize(Simulator(pol).run(make_sharegpt_workload(df, N, LOAD, seed=seed)))


rows = []
for seed in SEEDS:
    for name, mk in (("fifo", FIFO), ("sjf_oracle", SJFOracle), ("sjf_point", SJFPoint)):
        rows.append({"policy": name, "gamma": np.nan, "seed": seed, **run(mk(), seed)})
    for g in GAMMAS:
        rows.append({"policy": "aging_sjf", "gamma": g, "seed": seed, **run(AgingSJF(gamma=g), seed)})
        rows.append({"policy": "latent", "gamma": g, "seed": seed, **run(Latent(prior=None, gamma=g), seed)})

res = pd.DataFrame(rows)
res.to_csv("results/pareto_raw.csv", index=False)


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():+.1f} ± {h:.1f}"


# ---- paired comparison: latent vs aging_sjf at the same gamma ----
print(f"load {LOAD}: LATENT vs AgingSJF at equal gamma, paired % change (negative = LATENT better)")
out = []
for g in GAMMAS:
    a = res[(res.policy == "aging_sjf") & (res.gamma == g)].set_index("seed")
    l = res[(res.policy == "latent") & (res.gamma == g)].set_index("seed")
    out.append({
        "gamma": g,
        "mean": ci(100 * (l.mean_latency / a.mean_latency - 1)),
        "p95": ci(100 * (l.p95_latency / a.p95_latency - 1)),
        "p99": ci(100 * (l.p99_latency / a.p99_latency - 1)),
        "max_wait": ci(100 * (l.max_wait / a.max_wait - 1)),
    })
pd.set_option("display.width", 200)
print(pd.DataFrame(out).to_string(index=False))

# ---- trade-off plots ----
m = res.groupby(["policy", "gamma"], dropna=False)[["mean_latency", "p99_latency", "max_wait"]].mean().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, ycol, ylab, logy in ((axes[0], "p99_latency", "p99 latency", False),
                             (axes[1], "max_wait", "max wait (starvation)", True)):
    for pol, style in (("aging_sjf", "s--"), ("latent", "o-")):
        d = m[m.policy == pol].sort_values("gamma")
        ax.plot(d.mean_latency, d[ycol], style, label=pol)
    for pol, mk in (("fifo", "^"), ("sjf_point", "x"), ("sjf_oracle", "*")):
        d = m[m.policy == pol]
        ax.scatter(d.mean_latency, d[ycol], marker=mk, s=110, label=pol, zorder=5)
    ax.set_xlabel("mean latency"); ax.set_ylabel(ylab)
    if logy: ax.set_yscale("log")
    ax.grid(alpha=.3)
axes[0].legend()
fig.suptitle(f"Mean vs tail trade-off on ShareGPT (load {LOAD}); lower-left is better. Points along curves = gamma sweep")
fig.tight_layout()
fig.savefig("docs/pareto_tradeoff.png", dpi=150)
print("\nsaved docs/pareto_tradeoff.png")