"""Build PyG graphs from dense functional-connectivity (FC) matrices."""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data


def density_mst_edge_index(fc: np.ndarray, density: float = 0.10):
    """Undirected density-thresholded graph with an MST union for connectivity.

    Keep the strongest ``density`` fraction of off-diagonal edges by ``|weight|``,
    then union with a maximum-spanning-tree-style backbone so every node is
    reachable (no isolated nodes / disconnected components). Edges are undirected
    and carry the SIGNED original FC value as weight.

    The MST is computed on a distance ``max|w| - |w|`` so that strong (high |w|)
    edges are *preferred* by the minimum-spanning-tree, i.e. it behaves like a
    maximum-spanning-tree on |w|. Self-loops are excluded.

    Parameters
    ----------
    fc : np.ndarray
        Dense (R, R) connectivity matrix (FC or SC). Need not be symmetric; the
        upper triangle (``|fc[i,j]|`` symmetrised by max) defines edge strength.
    density : float
        Fraction of the R*(R-1)/2 possible undirected edges to keep by |weight|.

    Returns
    -------
    (edge_index [2, E], edge_weight [E])
        Undirected COO edges (both (i,j) and (j,i)) with signed FC weights.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    fc = np.asarray(fc, dtype=np.float64)
    n = fc.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    # symmetric signed value and magnitude per upper-triangle pair
    signed = np.where(np.abs(fc[iu, ju]) >= np.abs(fc[ju, iu]), fc[iu, ju], fc[ju, iu])
    mag = np.abs(signed)

    n_pairs = iu.size
    n_keep = max(0, int(round(density * n_pairs)))
    # indices of the strongest n_keep pairs by magnitude
    if n_keep > 0:
        keep = np.argpartition(-mag, kth=min(n_keep, n_pairs) - 1)[:n_keep]
    else:
        keep = np.array([], dtype=int)
    kept = set(zip(iu[keep].tolist(), ju[keep].tolist()))

    # MST backbone: distance = max|w| - |w| (strong edges -> small distance)
    max_mag = mag.max() if mag.size else 0.0
    dist = (max_mag - mag) + 1e-9  # strictly positive so MST keeps all n-1 edges
    dmat = np.zeros((n, n), dtype=np.float64)
    dmat[iu, ju] = dist
    dmat[ju, iu] = dist
    mst = minimum_spanning_tree(csr_matrix(dmat)).tocoo()
    for i, j in zip(mst.row.tolist(), mst.col.tolist()):
        kept.add((min(i, j), max(i, j)))

    # signed weight lookup for any (i,j) pair via symmetrised value
    src, dst, w = [], [], []
    for i, j in kept:
        wij = float(fc[i, j]) if abs(fc[i, j]) >= abs(fc[j, i]) else float(fc[j, i])
        src += [i, j]
        dst += [j, i]
        w += [wij, wij]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float)
    return edge_index, edge_weight


def topk_edge_index(fc: np.ndarray, k: int = 10):
    """Undirected top-k graph from a dense FC matrix.

    For each node keep its k strongest neighbours by |FC| (diagonal excluded),
    then union (i->j) and (j->i). Returns (edge_index [2, E], edge_weight [E])
    with signed FC values as weights.
    """
    fc = np.asarray(fc, dtype=np.float64)
    n = fc.shape[0]
    a = np.abs(fc).copy()
    np.fill_diagonal(a, -np.inf)
    kk = min(k, n - 1)
    idx = np.argpartition(-a, kth=kk - 1, axis=1)[:, :kk]
    pairs = set()
    for r in range(n):
        for c in idx[r]:
            c = int(c)
            if r != c:
                pairs.add((min(r, c), max(r, c)))
    src, dst, w = [], [], []
    for i, j in pairs:
        wij = float(fc[i, j])
        src += [i, j]
        dst += [j, i]
        w += [wij, wij]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float)
    return edge_index, edge_weight


def fc_to_data(fc, y, k: int = 10, node_features: str = "degree",
               task: str = "classification", edge_construction: str = "topk",
               density: float = 0.10, edge_weight_norm: str | None = None,
               node_attr=None) -> Data:
    """Convert one FC matrix + label into a graph-level PyG Data sample.

    ``task`` controls the label dtype: ``classification`` stores ``y`` as a
    long class index (cross-entropy expects this); ``regression`` stores it as
    a float target.

    ``node_attr`` optionally supplies an *external* per-node feature array
    (shape ``(R,)`` or ``(R, F)``, e.g. a parcellated GLM activation vector
    aligned to the FC nodes). When given it becomes ``x`` directly (reshaped to
    ``(R, F)``), overriding ``node_features``; its length must equal ``R``.

    ``edge_construction`` selects the graph builder: ``"topk"`` (per-node k
    strongest neighbours, the legacy default) or ``"density_mst"`` (keep the
    strongest ``density`` fraction of edges by |weight|, union an MST so the
    graph is connected).

    ``edge_weight_norm`` optionally rescales edge weights (huge SC streamline
    counts can destabilise training): ``None`` (raw), ``"max"`` (per-graph
    divide by max |weight|), ``"log1p"`` (signed log1p), ``"abs"`` (drop the
    sign -> |weight|), or ``"abs_max"`` (|weight| then per-graph /max, in
    [0,1]). NOTE: symmetric-normalised convs (GCN, ARMAConv) divide by node
    degree and produce NaN on negative edge weights, so signed FC weights must
    be passed through ``"abs"``/``"abs_max"`` for those backbones.

    The signed edge weight is stored under BOTH ``edge_attr`` (shape [E, 1],
    legacy) and ``edge_weight`` (shape [E], consumed by GNN wrappers).
    """
    if edge_construction == "topk":
        edge_index, edge_weight = topk_edge_index(fc, k=k)
    elif edge_construction == "density_mst":
        edge_index, edge_weight = density_mst_edge_index(fc, density=density)
    else:
        raise ValueError(f"unknown edge_construction: {edge_construction}")
    if edge_weight_norm == "max":
        m = edge_weight.abs().max()
        if m > 0:
            edge_weight = edge_weight / m
    elif edge_weight_norm == "log1p":
        edge_weight = torch.sign(edge_weight) * torch.log1p(edge_weight.abs())
    elif edge_weight_norm == "abs":
        edge_weight = edge_weight.abs()
    elif edge_weight_norm == "abs_max":
        edge_weight = edge_weight.abs()
        m = edge_weight.max()
        if m > 0:
            edge_weight = edge_weight / m
    elif edge_weight_norm not in (None, "none"):
        raise ValueError(f"unknown edge_weight_norm: {edge_weight_norm}")
    n = fc.shape[0]
    if node_attr is not None:
        arr = np.asarray(node_attr)
        if arr.shape[0] != n:
            raise ValueError(
                f"node_attr has {arr.shape[0]} nodes but fc has {n}")
        x = torch.as_tensor(arr, dtype=torch.float).reshape(n, -1)
    elif node_features == "degree":
        deg = np.zeros(n)
        ei = edge_index.numpy()
        for s, d in zip(ei[0], ei[1]):
            deg[int(s)] += abs(float(fc[int(s), int(d)]))
        mu, sd = deg.mean(), deg.std()
        x = torch.tensor(((deg - mu) / (sd + 1e-8))[:, None], dtype=torch.float)
    elif node_features == "identity":
        x = torch.eye(n, dtype=torch.float)
    else:
        raise ValueError(f"unknown node_features: {node_features}")
    if task == "classification":
        y_tensor = torch.tensor([int(y)], dtype=torch.long)
    elif task == "regression":
        y_tensor = torch.tensor([float(y)], dtype=torch.float)
    else:
        raise ValueError(f"unknown task: {task}")
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight.view(-1, 1),
        edge_weight=edge_weight,
        y=y_tensor,
    )


def glm_diagonal_features(glm_vec, normalize: bool = True) -> np.ndarray:
    """Per-node GLM scalar ``(R,)`` -> diagonal node-feature matrix ``(R, R)``.

    Each node i's feature is ``e_i * z(glm)_i`` (zero except at its own index),
    i.e. ``diag(glm)`` -- the thesis's ``glm_diagonal`` carrier. ``normalize=True``
    z-scores the GLM across nodes per subject (``glm_normalize=true``).
    """
    g = np.asarray(glm_vec, dtype=np.float64).ravel()
    if normalize:
        g = (g - g.mean()) / (g.std() + 1e-8)
    return (np.eye(g.shape[0]) * g).astype(np.float32)


def fc_arrays_to_data_list(fc, y, k: int = 10, node_features: str = "degree",
                           task: str = "classification",
                           edge_construction: str = "topk",
                           density: float = 0.10,
                           edge_weight_norm: str | None = None,
                           node_attr=None):
    """(N, R, R) FC stack + (N,) labels -> list[Data].

    ``node_attr`` optionally supplies external per-node features as an
    ``(N, R)`` or ``(N, R, F)`` stack, sliced per sample into each graph's
    ``x`` (overrides ``node_features``) -- e.g. a baked ``glm_<contrast>``
    activation array aligned to the FC nodes.
    """
    return [
        fc_to_data(fc[i], y[i], k=k, node_features=node_features, task=task,
                   edge_construction=edge_construction, density=density,
                   edge_weight_norm=edge_weight_norm,
                   node_attr=None if node_attr is None else node_attr[i])
        for i in range(len(y))
    ]
