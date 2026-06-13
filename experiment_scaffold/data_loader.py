"""Load baked FC-graph datasets (``.npz`` bundles under ``RC_DATASET_DIR``).

Replaces a framework-specific loader: depends only on numpy, torch_geometric,
and the sibling ``fc_graph`` module. Each bundle holds at least:

    fc : (N, R, R) float  -- functional-connectivity matrices (one graph/subject)
    y  : (N,)             -- graph-level label (int class) or target (float)

and optionally ``task`` (b'classification' / b'regression'), ``subject_id``,
and extra modalities (``sc``, ``glm_*``). Discover what was baked with
``list_datasets()``; the dataset root is exposed by the pipeline at the
``RC_DATASET_DIR`` environment variable -- never hardcode a path.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from torch_geometric.data import InMemoryDataset

from .fc_graph import fc_arrays_to_data_list, glm_diagonal_features


def dataset_root() -> Path:
    """Absolute dataset root, read from the ``RC_DATASET_DIR`` environment variable."""
    root = os.environ.get("RC_DATASET_DIR")
    if not root:
        raise RuntimeError(
            "RC_DATASET_DIR is not set. The pipeline exposes the scaffold dataset "
            "there; point it at the folder holding the baked .npz bundles."
        )
    return Path(root)


def list_datasets() -> list[str]:
    """Stems of the ``.npz`` bundles available under ``RC_DATASET_DIR``."""
    return sorted(p.stem for p in dataset_root().glob("*.npz"))


def _resolve(name) -> Path:
    p = Path(name)
    if not p.is_absolute():
        p = dataset_root() / p
    if p.suffix != ".npz":
        p = p.with_suffix(".npz")
    if not p.is_file():
        avail = ", ".join(list_datasets()) or "(none)"
        raise FileNotFoundError(f"dataset {p} not found; available: {avail}")
    return p


def load_fc_bundle(name, matrix_key: str = "fc"):
    """Load a baked bundle -> ``(matrix [N, R, R], y [N], meta dict)``.

    ``matrix_key`` selects which stored matrix is the graph adjacency: ``"fc"``
    (default) or ``"sc"`` for a structural-connectivity bundle (whose nodes are
    a *different* set than FC -- e.g. 452 vs 400). The non-selected matrix and
    every other stored array (subject_id, glm_*, the other modality) are
    returned in ``meta``. ``meta`` also carries the baked-in ``task`` (default
    'classification').
    """
    z = np.load(_resolve(name), allow_pickle=True)
    if matrix_key not in z.files:
        avail = ", ".join(k for k in z.files if k not in ("y", "task")) or "(none)"
        raise KeyError(f"matrix_key '{matrix_key}' not in {name}; available: {avail}")
    mat, y = z[matrix_key], z["y"]
    task = "classification"
    if "task" in z.files:
        t = z["task"]
        t = t.item() if hasattr(t, "item") else t
        task = t.decode() if isinstance(t, bytes) else str(t)
    meta = {k: z[k] for k in z.files if k not in (matrix_key, "y")}
    meta["task"] = task
    return mat, y, meta


def load_fc_graphs(name, *, k: int = 10, node_features: str = "degree",
                   task: str | None = None, edge_construction: str = "topk",
                   density: float = 0.10, edge_weight_norm: str | None = None,
                   matrix_key: str = "fc", node_feature_key: str | None = None,
                   glm_diagonal: bool = False, glm_normalize: bool = True,
                   graph_feature_key: str | None = None,
                   limit: int | None = None, indices=None):
    """Load a baked bundle as a ``list[torch_geometric.data.Data]`` (one graph/subject).

    ``task`` defaults to the value baked into the bundle (classification ->
    long labels, regression -> float targets). The edge-construction,
    node-feature, and edge_weight_norm options are forwarded to
    ``fc_graph.fc_to_data`` -- see that module for the full menu.

    ``matrix_key`` selects which stored matrix builds the graph (``"fc"`` or
    ``"sc"``; SC weights are large/positive -- use ``edge_weight_norm="abs_max"``
    for GCN). ``node_feature_key`` names a baked per-node array (e.g.
    ``"glm_2back_vs_0back"``, shape ``(N, R)``) to attach as node features,
    overriding ``node_features``.

    ``indices`` (explicit list) or ``limit`` (first N) subset the bundle before
    building graphs -- ``load_fc_graphs`` builds *all* graphs eagerly, so use
    these for fast smokes on the large bundles. ``indices`` takes precedence.
    """
    fc, y, meta = load_fc_bundle(name, matrix_key=matrix_key)
    if task is None:
        task = meta.get("task", "classification")
    node_attr = None
    if node_feature_key is not None:
        if node_feature_key not in meta:
            avail = ", ".join(k for k in meta if k != "task") or "(none)"
            raise KeyError(f"node_feature_key '{node_feature_key}' not in {name}; "
                           f"available arrays: {avail}")
        arr = meta[node_feature_key]
        if glm_diagonal:
            node_attr = np.stack([glm_diagonal_features(arr[i], normalize=glm_normalize)
                                  for i in range(len(arr))])      # (N, R, R)
        else:
            node_attr = arr                                       # (N, R) -> (R, 1)
    graph_attr = None
    if graph_feature_key is not None:
        if graph_feature_key not in meta:
            avail = ", ".join(k for k in meta if k != "task") or "(none)"
            raise KeyError(f"graph_feature_key '{graph_feature_key}' not in {name}; "
                           f"available arrays: {avail}")
        graph_attr = meta[graph_feature_key]
    if indices is not None:
        sel = np.asarray(indices)
        fc, y = fc[sel], y[sel]
        if node_attr is not None: node_attr = node_attr[sel]
        if graph_attr is not None: graph_attr = graph_attr[sel]
    elif limit is not None:
        fc, y = fc[:limit], y[:limit]
        if node_attr is not None: node_attr = node_attr[:limit]
        if graph_attr is not None: graph_attr = graph_attr[:limit]
    return fc_arrays_to_data_list(
        fc, y, k=k, node_features=node_features, task=task,
        edge_construction=edge_construction, density=density,
        edge_weight_norm=edge_weight_norm, node_attr=node_attr, graph_attr=graph_attr,
    )


class FCInMemoryDataset(InMemoryDataset):
    """Wrap a list of PyG Data graphs as an in-memory PyG dataset."""

    def __init__(self, data_list):
        super().__init__(root=None)
        self.data, self.slices = self.collate(data_list)


def load_fc_dataset(name, **kwargs) -> FCInMemoryDataset:
    """Convenience: ``load_fc_graphs`` wrapped as an ``InMemoryDataset``."""
    return FCInMemoryDataset(load_fc_graphs(name, **kwargs))
