"""M3 cluster-harness unit tests. Run under the ARC venv (torch+pyg+optuna):
    RC_DATASET_DIR=~/rc_brain_data .venv/bin/python -m pytest tests/test_cluster_harness.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
HAS_DATA = bool(os.environ.get("RC_DATASET_DIR")) and (
    Path(os.path.expanduser(os.environ.get("RC_DATASET_DIR", "/nonexistent")))
    / "pnc_sc400_vwm_reg.npz"
).is_file()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="RC_DATASET_DIR bundles absent")


@needs_data
def test_folds_are_subject_aligned_and_disjoint():
    from cluster import folds as F
    from cluster._common import OUTER

    # Every outer fold: train/test disjoint, and the SAME subjects map across bundles.
    for rep in (0, 3):
        seen_test = []
        for k in range(OUTER):
            fold = F.outer_split(rep, k)
            # disjoint within a bundle
            assert set(fold.train_vwm).isdisjoint(fold.test_vwm)
            assert set(fold.train_age).isdisjoint(fold.test_age)
            # subject sets identical across bundles (within-subject transfer)
            assert fold.train_subjects == set(F.subjects_at(F.VWM, fold.train_vwm))
            assert fold.train_subjects == set(F.subjects_at(F.AGE, fold.train_age))
            assert fold.test_subjects == set(F.subjects_at(F.AGE, fold.test_age))
            # cross-bundle alignment is direct: AGE positions and VWM positions
            # resolve to the SAME subjects (not just transitively via train_subjects)
            assert set(F.subjects_at(F.AGE, fold.train_age)) == set(F.subjects_at(F.VWM, fold.train_vwm))
            assert set(F.subjects_at(F.AGE, fold.test_age)) == set(F.subjects_at(F.VWM, fold.test_vwm))
            seen_test.append(fold.test_subjects)
        # the 5 outer test sets partition the common subjects
        union = set().union(*seen_test)
        assert len(union) == sum(len(t) for t in seen_test)  # no overlap across folds
        assert len(union) == F.n_common()


@needs_data
def test_inner_split_holds_out_within_train():
    from cluster import folds as F

    fold = F.outer_split(1, 2)
    itr, iva = F.inner_split(fold.train_vwm, seed=1)
    assert set(itr).isdisjoint(iva)
    assert set(itr) | set(iva) == set(fold.train_vwm)
