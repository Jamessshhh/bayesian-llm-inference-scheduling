import numpy as np
import pandas as pd
from scipy import stats

from latent.policies.latent_policy import Latent
from latent.policies.sjf import SJFPoint
from latent.predictor.noisy import add_hetero_predictions
from latent.predictor.posterior import fit_log_prior
from latent.sim.metrics import summarize
from latent.sim.simulator import Simulator
from latent.workload.synthetic import make_workload

N = 30_000
SEEDS = list(range(10))
KINDS = ["lognormal", "bimodal"]
LOAD = 0.9
SETTINGS = [
    ("overconfident x0.25", dict(report_scale=0.25)),
    ("overconfident x0.5", dict(report_scale=0.5)),
    ("calibrated x1", dict(report_scale=1.0)),
    ("underconfident x2", dict(report_scale=2.0)),
    ("underconfident x4", dict(report_scale=4.0)),
    ("uninformative sigma", dict(report_scale=1.0, informative=False)),
]


def one(kind, seed, pol, **pred_kw):
    reqs = make_workload(kind, N, LOAD, seed=seed)
    add_hetero_predictions(reqs, seed=seed + 100, **pred_kw)
    return summarize(Simulator(pol).run(reqs))


rows = []
for kind in KINDS:
    prior = fit_log_prior(make_workload(kind, 20_000, LOAD, seed=999))
    for name, kw in SETTINGS:
        for seed in SEEDS:
            base = one(kind, seed, SJFPoint(), **kw)
            lat = one(kind, seed, Latent(prior, alpha=None, gamma=0.0), **kw)
            rows.append({
                "workload": kind, "setting": name, "seed": seed,
                "mean_chg_%": 100 * (lat["mean_latency"] / base["mean_latency"] - 1),
                "p95_chg_%": 100 * (lat["p95_latency"] / base["p95_latency"] - 1),
                "p99_chg_%": 100 * (lat["p99_latency"] / base["p99_latency"] - 1),
            })

df = pd.DataFrame(rows)


def ci(x):
    x = np.asarray(x)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():+.1f} ± {h:.1f}"


out = (df.groupby(["workload", "setting"], sort=False)[["mean_chg_%", "p95_chg_%", "p99_chg_%"]]
         .agg(ci).reset_index())
print("LATENT vs SJF-point: % change (negative = LATENT better), mean ± 95% CI over", len(SEEDS), "seeds")
pd.set_option("display.width", 200)
print(out.to_string(index=False))
df.to_csv("results/miscalibration.csv", index=False)