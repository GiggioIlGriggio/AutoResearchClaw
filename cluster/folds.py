"""Subject-aligned nested folds for the within-subject age→VWM transfer.

Folds are defined ONCE over the subjects common to both bundles (743). For each
(rep, outer-fold) the same subject partition is mapped to per-bundle POSITIONS
(orderings differ), so the source pretrains on a fold's train subjects' age and the
target fine-tunes on the SAME subjects' VWM, with the held-out test subjects excluded
from both. Reps reshuffle the 5-fold split (random_state=rep).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold

from cluster._common import AGE_BUNDLE, OUTER, VWM_BUNDLE
from cluster._scaffold import import_scaffold

AGE, VWM = "age", "vwm"
_dl = None
_sid = {}          # bundle -> np.ndarray[str] of subject_ids (bundle order)
_pos = {}          # bundle -> {subject_id: position}
_common = None     # sorted list[str] of common subject_ids


def _ensure_loaded():
    global _dl, _common
    if _common is not None:          # guard on the final assignment, not the first
        return
    dl, *_ = import_scaffold()
    for tag, name in ((AGE, AGE_BUNDLE), (VWM, VWM_BUNDLE)):
        _, _, meta = dl.load_fc_bundle(name, matrix_key="sc")
        sids = np.asarray(meta["subject_id"]).astype(str)
        _sid[tag] = sids
        _pos[tag] = {s: i for i, s in enumerate(sids)}
    _common = sorted(set(_sid[AGE]) & set(_sid[VWM]))
    _dl = dl                          # only set after all state is valid


def n_common() -> int:
    _ensure_loaded()
    return len(_common)


def subjects_at(bundle, idx):
    _ensure_loaded()
    return [_sid[bundle][i] for i in idx]


def _positions(bundle, subjects):
    return np.array([_pos[bundle][s] for s in subjects], dtype=int)


@dataclass
class FoldIdx:
    rep: int
    outer: int
    train_subjects: set
    test_subjects: set
    train_age: np.ndarray
    test_age: np.ndarray
    train_vwm: np.ndarray
    test_vwm: np.ndarray


def outer_split(rep: int, k: int) -> FoldIdx:
    _ensure_loaded()
    kf = KFold(n_splits=OUTER, shuffle=True, random_state=rep)
    common = np.asarray(_common)
    for ki, (tr, te) in enumerate(kf.split(common)):
        if ki != k:
            continue
        train_s, test_s = sorted(common[tr]), sorted(common[te])
        return FoldIdx(
            rep=rep, outer=k,
            train_subjects=set(train_s), test_subjects=set(test_s),
            train_age=_positions(AGE, train_s), test_age=_positions(AGE, test_s),
            train_vwm=_positions(VWM, train_s), test_vwm=_positions(VWM, test_s),
        )
    raise ValueError(f"outer fold {k} out of range (OUTER={OUTER})")


def inner_split(train_idx, seed: int, frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Single inner train/val holdout WITHIN a fold's train positions (for Optuna)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_idx))
    cut = int(round(len(train_idx) * frac))
    arr = np.asarray(train_idx)
    return arr[perm[:cut]], arr[perm[cut:]]
