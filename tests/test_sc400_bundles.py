"""Validate the baked PNC SC-400 thesis bundles via the scaffold loader.

Skips if RC_DATASET_DIR (or ~/rc_brain_data) lacks the bundles, so the suite
still passes on a box where Task 7 hasn't run. Run under the ARC venv:
    .venv/bin/python -m pytest tests/test_sc400_bundles.py -v
"""
from __future__ import annotations

import os
import shutil
import importlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("RC_DATASET_DIR", os.path.expanduser("~/rc_brain_data")))


@pytest.fixture(scope="module")
def loader(tmp_path_factory):
    root = tmp_path_factory.mktemp("scaffold_root"); pkg = root / "scaffold"; pkg.mkdir()
    for py in (REPO / "experiment_scaffold").glob("*.py"):
        shutil.copy(py, pkg / py.name)
    (pkg / "__init__.py").write_text("")
    import sys; sys.path.insert(0, str(root))
    os.environ["RC_DATASET_DIR"] = str(DATA)
    return importlib.import_module("scaffold.data_loader")


@pytest.mark.skipif(not (DATA / "pnc_sc400_vwm_reg.npz").is_file(),
                    reason="pnc_sc400_vwm_reg not baked (run scripts/bake_sc400_thesis.py)")
def test_vwm_bundle_identity_carrier_loads(loader):
    g = loader.load_fc_graphs("pnc_sc400_vwm_reg", matrix_key="sc",
                              edge_weight_norm="abs_max", node_features="identity",
                              limit=8)
    assert len(g) == 8
    assert tuple(g[0].x.shape) == (400, 400)          # identity carrier on 400 nodes
    assert g[0].y.dtype.is_floating_point


@pytest.mark.skipif(not (DATA / "pnc_sc400_vwm_reg.npz").is_file(),
                    reason="pnc_sc400_vwm_reg not baked")
def test_vwm_bundle_glm_diagonal_and_age(loader):
    g = loader.load_fc_graphs("pnc_sc400_vwm_reg", matrix_key="sc",
                              edge_weight_norm="abs_max",
                              node_feature_key="glm_2back_vs_0back", glm_diagonal=True,
                              graph_feature_key="age", limit=8)
    assert tuple(g[0].x.shape) == (400, 400)          # glm_diagonal carrier
    assert tuple(g[0].u.shape) == (1, 1)              # co-stored age side channel
    _, _, meta = loader.load_fc_bundle("pnc_sc400_vwm_reg", matrix_key="sc")
    assert "age" in meta and "glm_2back_vs_0back" in meta


@pytest.mark.skipif(not (DATA / "pnc_sc400_age_reg.npz").is_file(),
                    reason="pnc_sc400_age_reg not baked")
def test_age_bundle_loads_for_pretraining(loader):
    mat, y, meta = loader.load_fc_bundle("pnc_sc400_age_reg", matrix_key="sc")
    assert mat.shape[1:] == (400, 400)
    assert 8.0 <= float(np.nanmin(y)) and float(np.nanmax(y)) <= 22.0
    assert meta["task"] == "regression"
