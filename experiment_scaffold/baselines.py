"""A5 floor: the trivial age->VWM regression (no graph).

Separates "age leaking in" from genuine brain signal -- the mandatory floor the
thesis uses to bound any age-concatenation gain. numpy-only (1-feature OLS with
k-fold CV R^2), so the substrate stays dependency-light.
"""
from __future__ import annotations

import numpy as np


def age_vwm_baseline(age, vwm, cv: int = 5, seed: int = 0):
    """Fit age->VWM (1-feature OLS) and return ``((slope, intercept), mean_cv_r2)``."""
    x = np.asarray(age, dtype=float).ravel()
    y = np.asarray(vwm, dtype=float).ravel()
    n = len(y)
    folds = np.array_split(np.random.default_rng(seed).permutation(n), cv)
    r2s = []
    for k in range(cv):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(cv) if j != k])
        coef = np.linalg.lstsq(np.c_[x[tr], np.ones(len(tr))], y[tr], rcond=None)[0]
        pred = coef[0] * x[te] + coef[1]
        ss_res = float(((y[te] - pred) ** 2).sum())
        ss_tot = float(((y[te] - y[te].mean()) ** 2).sum())
        r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    coef = np.linalg.lstsq(np.c_[x, np.ones(n)], y, rcond=None)[0]
    return (float(coef[0]), float(coef[1])), float(np.mean(r2s))
