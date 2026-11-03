from __future__ import annotations
import math

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


class LengthPredictor:
    """Predicts log(output_tokens) from the prompt, plus a per-request sigma.

    mean model  : TF-IDF + log(input length) -> Ridge on log(out_tokens)
    sigma model : Ridge on |out-of-fold residual|, so sigma varies by prompt
    calibrate() : rescales sigma on held-out data so z = resid/sigma has std 1
    """

    def __init__(self, max_features=50_000, alpha=3.0, sigma_floor=0.15,
                 n_folds=4, seed=0):
        self.alpha = alpha
        self.sigma_floor = sigma_floor
        self.n_folds = n_folds
        self.seed = seed
        self.vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                                   min_df=3, sublinear_tf=True)
        self.scale = 1.0

    def _X(self, prompts, in_tokens, fit=False):
        T = self.vec.fit_transform(prompts) if fit else self.vec.transform(prompts)
        x = np.log1p(np.asarray(in_tokens, dtype=float))
        if fit:
            self._mu, self._sd = x.mean(), x.std() + 1e-9
        return sparse.hstack([T, sparse.csr_matrix(((x - self._mu) / self._sd)[:, None])]).tocsr()

    def fit(self, prompts, in_tokens, out_tokens):
        y = np.log(np.asarray(out_tokens, dtype=float))
        X = self._X(prompts, in_tokens, fit=True)
        self.mean_model = Ridge(alpha=self.alpha).fit(X, y)
        oof = np.zeros(len(y))
        for tr, va in KFold(self.n_folds, shuffle=True, random_state=self.seed).split(X):
            oof[va] = Ridge(alpha=self.alpha).fit(X[tr], y[tr]).predict(X[va])
        self.sigma_model = Ridge(alpha=self.alpha).fit(X, np.abs(y - oof))
        return self

    def predict(self, prompts, in_tokens):
        """Returns (mu, sigma) on the log scale: log(out) ~ N(mu, sigma^2)."""
        X = self._X(prompts, in_tokens)
        mu = self.mean_model.predict(X)
        raw = self.sigma_model.predict(X) * math.sqrt(math.pi / 2.0)   # E|e| -> std
        return mu, np.maximum(raw * self.scale, self.sigma_floor)

    def calibrate(self, prompts, in_tokens, out_tokens, rounds=3):
        y = np.log(np.asarray(out_tokens, dtype=float))
        for _ in range(rounds):
            mu, s = self.predict(prompts, in_tokens)
            self.scale *= float(np.sqrt(np.mean(((y - mu) / s) ** 2)))
        return self