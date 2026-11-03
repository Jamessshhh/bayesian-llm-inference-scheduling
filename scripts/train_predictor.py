"""Train the length predictor on the prepared ShareGPT pairs and report how good
(and how honest) its uncertainty is. Saves held-out predictions for scheduling."""
import json
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from latent.predictor.learned import LengthPredictor

SEED = 0
df = pd.read_csv("data/sharegpt_pairs.csv")
rng = np.random.default_rng(SEED)
idx = rng.permutation(len(df))
n = len(df)
tr, ca, te = idx[: int(.6 * n)], idx[int(.6 * n): int(.8 * n)], idx[int(.8 * n):]
train, calib, test = df.iloc[tr], df.iloc[ca], df.iloc[te]
print(f"train {len(train)}  calib {len(calib)}  test {len(test)}")

pred = LengthPredictor(seed=SEED).fit(train.prompt, train.in_tokens, train.out_tokens)
pred.calibrate(calib.prompt, calib.in_tokens, calib.out_tokens)

mu, sig = pred.predict(test.prompt, test.in_tokens)
y = np.log(test.out_tokens.to_numpy(float))
resid = y - mu

print("\n--- point prediction quality (log scale) ---")
print(f"std of log(out), no model : {y.std():.3f}")
print(f"RMSE of model             : {np.sqrt(np.mean(resid**2)):.3f}")
print(f"R^2                       : {1 - np.mean(resid**2) / y.var():.3f}")
print(f"Spearman(pred, true)      : {spearmanr(mu, y).statistic:.3f}")

print("\n--- is the uncertainty honest? ---")
z = resid / sig
nd = NormalDist()
for p in (0.5, 0.8, 0.9):
    zz = nd.inv_cdf(0.5 + p / 2)
    print(f"nominal {int(p*100)}% interval covers: {np.mean(np.abs(z) < zz)*100:.1f}%")
print(f"corr(sigma, |resid|)      : {np.corrcoef(sig, np.abs(resid))[0,1]:.3f}")
q = np.quantile(sig, [1/3, 2/3])
for name, m in (("low sigma ", sig <= q[0]), ("mid sigma ", (sig > q[0]) & (sig <= q[1])),
                ("high sigma", sig > q[1])):
    print(f"{name}: mean sigma {sig[m].mean():.2f}   actual resid std {resid[m].std():.2f}")

out = test[["in_tokens", "out_tokens"]].copy()
out["pred_mu"], out["pred_sigma"] = mu, sig
out.to_csv("data/test_predictions.csv", index=False)

cmu, csig = pred.predict(calib.prompt, calib.in_tokens)
c = calib[["in_tokens", "out_tokens"]].copy()
c["pred_mu"], c["pred_sigma"] = cmu, csig
c.to_csv("data/calib_predictions.csv", index=False)

logs = np.log(train.out_tokens.to_numpy(float))
json.dump({"mu": float(logs.mean()), "s": float(logs.std())}, open("data/train_prior.json", "w"))
print("\nsaved data/test_predictions.csv, data/calib_predictions.csv, data/train_prior.json")