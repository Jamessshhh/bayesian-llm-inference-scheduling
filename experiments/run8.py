"""Does TIE's CVaR term buy anything beyond moving along the aging trade-off?
Gamma sweep for aging-SJF, LATENT and TIE variants, with a trade-off figure."""
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
from latent.policies.tie import TIE
from latent.sim.metrics import summarize
from latent.sim.simulator import Simulator
from latent.workload.sharegpt import load_predictions, make_sharegpt_workload

N = 15_000
SEEDS = list(range(10))
LOAD = 0.9
GAMMAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0]
df = load_predictions()
X_MAX = 4096 / df["out_tokens"].mean()


def sweep_policies(g):
    return {
        "aging_sjf": AgingSJF(gamma=g),
        "latent": Latent(prior=None, gamma=g),
        "tie_sept": TIE(X_MAX, beta=0.0, gamma=g),
        "tie_beta0.1": TIE(X_MAX, beta=0.1, gamma=g),
        "tie_beta0.5": TIE(X_MAX, beta=0.5, gamma=g),
    }


def run(pol, seed):
    return summarize(Simulator(pol).run(make_sharegpt_workload(df, N, LOAD, seed=seed)))


rows = []
for seed in SEEDS:
    for name, mk in (("fifo", FIFO), ("sjf_oracle", SJFOracle), ("sjf_point", SJFPoint)):
        rows.append({"policy": name, "gamma": np.nan, "seed": seed, **run(mk(), seed)})
    for g in GAMMAS:
        for name, pol in sweep_policies(g).items():
            rows.append({"policy": name, "gamma": g, "seed": seed, **run(pol, seed)})

res = pd.DataFrame(rows)
res.to_csv("results/pareto_tie_raw.csv", index=False)


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():+.1f} ± {h:.1f}"


print(f"load {LOAD}: TIE variants vs LATENT at equal gamma, paired % change (negative = TIE better)")
out = []
for g in GAMMAS:
    base = res[(res.policy == "latent") & (res.gamma == g)].set_index("seed")
    for name in ("tie_sept", "tie_beta0.1", "tie_beta0.5"):
        v = res[(res.policy == name) & (res.gamma == g)].set_index("seed")
        out.append({"gamma": g, "policy": name,
                    "mean": ci(100 * (v.mean_latency / base.mean_latency - 1)),
                    "p99": ci(100 * (v.p99_latency / base.p99_latency - 1)),
                    "max_wait": ci(100 * (v.max_wait / base.max_wait - 1))})
pd.set_option("display.width", 200)
print(pd.DataFrame(out).to_string(index=False))

m = res.groupby(["policy", "gamma"], dropna=False)[["mean_latency", "p99_latency", "max_wait"]].mean().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
styles = {"aging_sjf": "s--", "latent": "o-", "tie_sept": "^-", "tie_beta0.1": "v-", "tie_beta0.5": "d-"}
for ax, ycol, ylab, logy in ((axes[0], "p99_latency", "p99 latency", False),
                             (axes[1], "max_wait", "max wait (starvation)", True)):
    for pol, st in styles.items():
        d = m[m.policy == pol].sort_values("gamma")
        ax.plot(d.mean_latency, d[ycol], st, label=pol, markersize=5)
    for pol, mk in (("fifo", "P"), ("sjf_point", "x"), ("sjf_oracle", "*")):
        d = m[m.policy == pol]
        ax.scatter(d.mean_latency, d[ycol], marker=mk, s=110, label=pol, zorder=5, color="k")
    ax.set_xlabel("mean latency"); ax.set_ylabel(ylab)
    if logy: ax.set_yscale("log")
    ax.grid(alpha=.3)
axes[0].legend(fontsize=8)
fig.suptitle(f"Mean vs tail trade-off, ShareGPT, load {LOAD}; lower-left is better. Points along curves = gamma sweep")
fig.tight_layout()
fig.savefig("docs/pareto_with_tie.png", dpi=150)
print("\nsaved docs/pareto_with_tie.png")