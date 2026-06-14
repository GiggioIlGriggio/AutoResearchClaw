"""Load a fold's train/test graphs for a cell, using the EXACT scaffold loader API.

Graphs are built eagerly, so a (cell, rep, outer) task loads its train+test once and
reuses them across all Optuna trials.
"""
from __future__ import annotations

from cluster._common import AGE_BUNDLE, CELLS, VWM_BUNDLE, carrier_kwargs
from cluster._scaffold import import_scaffold

_dl = None


def _loader():
    global _dl
    if _dl is None:
        _dl, *_ = import_scaffold()
    return _dl


def _bundle_name(tag: str) -> str:
    if tag == "age":
        return AGE_BUNDLE
    if tag == "vwm":
        return VWM_BUNDLE
    raise ValueError(f"unknown bundle tag {tag!r}")


def load_cell_graphs(cell: str, fold, *, limit=None):
    """Return (train_graphs, test_graphs) for ``cell`` on its fold partition.

    Not for trivial cells (A5): those carry no carrier/bundle and are handled by the
    caller's trivial branch, never the graph loader.
    """
    spec = CELLS[cell]
    dl = _loader()
    name = _bundle_name(spec["bundle"])
    tr_idx = fold.train_age if spec["bundle"] == "age" else fold.train_vwm
    te_idx = fold.test_age if spec["bundle"] == "age" else fold.test_vwm
    if limit is not None:
        tr_idx, te_idx = tr_idx[:limit], te_idx[:limit]
    kw = dict(matrix_key="sc", edge_weight_norm="abs_max", **carrier_kwargs(spec["carrier"]))
    if spec.get("graph_feature_key"):
        kw["graph_feature_key"] = spec["graph_feature_key"]
    g_tr = dl.load_fc_graphs(name, indices=list(tr_idx), **kw)
    g_te = dl.load_fc_graphs(name, indices=list(te_idx), **kw)
    return g_tr, g_te
