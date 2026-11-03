import numpy as np
import pandas as pd
from scipy import stats

from latent.policies.aging import AgingSJF
from latent.policies.base import FIFO
from latent.policies.latent_policy import Latent
from latent.policies.sjf import SJFOracle, SJFPoint
from latent.predictor.recalibrate import SigmaRecalibrator
from latent.sim.metrics import summarize
from latent.sim.simulator import Simulator
from latent.workload.sharegpt import load_predictions, make_sharegpt_workload

N = 15_000
SEEDS = list(range(10))
LOADS = [0.8, 0.9, 0.95]
df = load_predictions()


def policies():
    return {
        "fifo": FIFO(),
        "sjf_oracle": SJFOracle(),
        "sjf_point": SJFPoint(),
        "latent": Latent(prior=None),
        "latent+recal": Latent(prior=None, recal=SigmaRecalibrator()),
        "aging_sjf(g=0.02)": AgingSJF(gamma=0.02),
        "latent(g=0.02)": Latent(prior=None, gamma=0.02),
    }


rows = []
for load in LOADS:
    for seed in SEEDS:
        for name, pol in policies().items():
            reqs = make_sharegpt_workload(df, N, load, seed=seed)
            rows.append({"load": load, "seed": seed, "policy": name,
                         **summarize(Simulator(pol).run(reqs))})

res = pd.DataFrame(rows)
res.to_csv("results/sharegpt_raw.csv", index=False)


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():.1f} ± {h:.1f}"


pd.set_option("display.width", 220)
print("=== absolute metrics (mean ± 95% CI over seeds) ===")
tab = (res.groupby(["load", "policy"], sort=False)[["mean_latency", "p95_latency", "p99_latency", "max_wait"]]
          .agg(ci).reset_index())
print(tab.to_string(index=False))

print("\n=== LATENT vs SJF-point: paired % change (negative = better) ===")
out = []
for load in LOADS:
    piv = res[res.load == load].pivot(index="seed", columns="policy")
    for pol in ("latent", "latent+recal", "latent(g=0.02)"):
        row = {"load": load, "policy": pol}
        for m in ("mean_latency", "p99_latency"):
            d = 100 * (piv[m][pol] / piv[m]["sjf_point"] - 1)
            row[m] = ci(d)
        # share of the oracle's headroom (over point-SJF) that is recovered, on p99
        gap = piv["p99_latency"]["sjf_point"] - piv["p99_latency"]["sjf_oracle"]
        got = piv["p99_latency"]["sjf_point"] - piv["p99_latency"][pol]
        row["oracle_gap_closed_p99_%"] = f"{100 * got.mean() / gap.mean():.0f}"
        out.append(row)
print(pd.DataFrame(out).to_string(index=False))