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


def test_cells_map_is_consistent():
    from cluster._common import CELLS, SOURCE_CELLS, HPO_CELLS, carrier_kwargs

    assert set(CELLS) == {"A1","A2","A3","A4","A5","B1","B2","B3","B4","C1","C2"}
    assert set(SOURCE_CELLS) == {"A1", "A4"}
    assert set(HPO_CELLS) == {"A2","A3","B1","B2","B3","B4","C1","C2"}
    # every transfer cell points at a real source whose carrier matches (strict=True load)
    for c, spec in CELLS.items():
        if spec.get("mode") in {"finetune", "frozen"}:
            assert CELLS[spec["source"]]["carrier"] == spec["carrier"], f"{c} carrier mismatch"
    # carrier_kwargs never leaks a 'carrier' kwarg into the loader
    for carrier in ("identity", "glm_diagonal"):
        assert "carrier" not in carrier_kwargs(carrier)
    assert CELLS["C1"]["global_dim"] == 1 and CELLS["C2"]["global_dim"] == 400


@needs_data
def test_load_fold_graphs_shapes():
    from cluster import data as D
    from cluster import folds as F

    fold = F.outer_split(0, 0)
    g_tr, g_te = D.load_cell_graphs("A3", fold, limit=24)  # glm_diagonal carrier
    assert g_tr[0].x.shape[1] == 400 and len(g_te) > 0
    assert len(g_tr) == 24 and len(g_te) == 24  # limit slices both train and test
    g_tr, g_te = D.load_cell_graphs("C1", fold, limit=24)  # + age @head -> data.u U=1
    assert g_tr[0].u.shape[-1] == 1
    g_tr, g_te = D.load_cell_graphs("C2", fold, limit=24)  # GLM @head -> data.u U=400
    assert g_tr[0].u.shape[-1] == 400
