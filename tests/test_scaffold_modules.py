"""Behavioral smoke tests for the experiment-scaffold *modules* themselves
(``fc_graph`` / ``data_loader`` / ``models``), as opposed to the pipeline's
scaffold machinery (covered by ``test_scaffold.py``).

These materialize ``experiment_scaffold/*.py`` into an importable ``scaffold``
package — exactly the way the pipeline does at experiment time — then exercise
the public API on small synthetic arrays. They are skipped if torch_geometric is
unavailable, so the rest of the suite still runs on a torch-less box.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

REPO = Path(__file__).resolve().parents[1]
SCAFFOLD_SRC = REPO / "experiment_scaffold"


@pytest.fixture(scope="session")
def scaffold_pkg(tmp_path_factory):
    """Copy the flat scaffold dir into a ``scaffold/`` package and import it."""
    import importlib
    import sys

    root = tmp_path_factory.mktemp("scaffold_root")
    pkg = root / "scaffold"
    pkg.mkdir()
    for py in SCAFFOLD_SRC.glob("*.py"):
        shutil.copy(py, pkg / py.name)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    sys.path.insert(0, str(root))
    try:
        fc_graph = importlib.import_module("scaffold.fc_graph")
        data_loader = importlib.import_module("scaffold.data_loader")
        models = importlib.import_module("scaffold.models")
    finally:
        # keep root on sys.path for the session so submodules stay importable
        pass
    return {"root": root, "fc_graph": fc_graph,
            "data_loader": data_loader, "models": models}


def _signed_fc(n=12, seed=0):
    """A symmetric, signed FC-like matrix with unit diagonal."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    a = (a + a.T) / 2
    np.fill_diagonal(a, 1.0)
    return a.astype(np.float32)


def test_fc_to_data_uses_external_node_attr(scaffold_pkg):
    """A supplied per-node vector becomes ``x`` (shape (R,1)), overriding degree."""
    fc_graph = scaffold_pkg["fc_graph"]
    fc = _signed_fc(10)
    attr = np.arange(10, dtype=np.float32)        # one scalar feature per node
    data = fc_graph.fc_to_data(fc, 0, node_attr=attr, task="classification")
    assert tuple(data.x.shape) == (10, 1)
    assert torch.allclose(data.x.view(-1), torch.arange(10, dtype=torch.float))


def test_fc_to_data_rejects_wrong_length_node_attr(scaffold_pkg):
    fc_graph = scaffold_pkg["fc_graph"]
    with pytest.raises(ValueError, match="node_attr"):
        fc_graph.fc_to_data(_signed_fc(10), 0, node_attr=np.zeros(7))


def test_fc_arrays_to_data_list_threads_node_attr(scaffold_pkg):
    """An (N, R) node_attr stack is sliced per-sample into each graph's x."""
    fc_graph = scaffold_pkg["fc_graph"]
    fc = np.stack([_signed_fc(8, seed=i) for i in range(3)])
    y = np.array([0, 1, 0])
    attr = np.stack([np.full(8, i, dtype=np.float32) for i in range(3)])  # (3, 8)
    dl = fc_graph.fc_arrays_to_data_list(fc, y, node_attr=attr)
    assert len(dl) == 3
    assert tuple(dl[2].x.shape) == (8, 1)
    assert torch.allclose(dl[2].x.view(-1), torch.full((8,), 2.0))


# --- data_loader -------------------------------------------------------------


@pytest.fixture()
def ds_dir(tmp_path, monkeypatch):
    """A throwaway RC_DATASET_DIR; returns a writer that bakes synthetic npz."""
    monkeypatch.setenv("RC_DATASET_DIR", str(tmp_path))

    def write(name, **arrays):
        np.savez(tmp_path / f"{name}.npz", **arrays)

    return tmp_path, write


def test_load_fc_bundle_matrix_key_selects_sc(scaffold_pkg, ds_dir):
    """matrix_key='sc' returns the SC matrix as the primary graph; fc stays in meta."""
    _, write = ds_dir
    data_loader = scaffold_pkg["data_loader"]
    fc = np.stack([_signed_fc(6, seed=i) for i in range(4)]).astype(np.float32)
    sc = np.stack([_signed_fc(9, seed=i + 100) for i in range(4)]).astype(np.float32)
    write("orbit_sc_age_reg", fc=fc, sc=sc, y=np.arange(4.0),
          task=np.array("regression"))

    mat, y, meta = data_loader.load_fc_bundle("orbit_sc_age_reg", matrix_key="sc")
    assert mat.shape == (4, 9, 9)          # SC node set, not FC's 6
    assert meta["task"] == "regression"
    assert meta["fc"].shape == (4, 6, 6)   # FC preserved in meta


def test_load_fc_graphs_node_feature_key_attaches_glm(scaffold_pkg, ds_dir):
    """node_feature_key pulls a baked glm_<c> array and uses it as node features."""
    _, write = ds_dir
    data_loader = scaffold_pkg["data_loader"]
    n_nodes = 7
    fc = np.stack([_signed_fc(n_nodes, seed=i) for i in range(3)]).astype(np.float32)
    glm = np.stack([np.full(n_nodes, i + 1, dtype=np.float32) for i in range(3)])
    write("orbit_fcglm_age_reg", fc=fc, glm_2back_vs_0back=glm,
          y=np.arange(3.0), task=np.array("regression"))

    graphs = data_loader.load_fc_graphs(
        "orbit_fcglm_age_reg", node_feature_key="glm_2back_vs_0back",
        edge_weight_norm="abs")
    assert len(graphs) == 3
    assert tuple(graphs[0].x.shape) == (n_nodes, 1)
    assert torch.allclose(graphs[2].x.view(-1), torch.full((n_nodes,), 3.0))


def test_load_fc_graphs_node_feature_key_missing_is_clear(scaffold_pkg, ds_dir):
    _, write = ds_dir
    data_loader = scaffold_pkg["data_loader"]
    fc = np.stack([_signed_fc(5, seed=i) for i in range(2)]).astype(np.float32)
    write("nofeat", fc=fc, y=np.arange(2.0))
    with pytest.raises(KeyError, match="glm_nope"):
        data_loader.load_fc_graphs("nofeat", node_feature_key="glm_nope")


def test_load_fc_graphs_limit_and_indices_subset(scaffold_pkg, ds_dir):
    """limit takes the first N graphs; indices selects an explicit subset (fast smokes)."""
    _, write = ds_dir
    data_loader = scaffold_pkg["data_loader"]
    fc = np.stack([_signed_fc(5, seed=i) for i in range(20)]).astype(np.float32)
    write("big", fc=fc, y=np.arange(20.0), task=np.array("regression"))

    assert len(data_loader.load_fc_graphs("big", limit=5)) == 5

    g = data_loader.load_fc_graphs("big", indices=[0, 7, 19])
    assert len(g) == 3
    assert [float(d.y) for d in g] == [0.0, 7.0, 19.0]


# --- end-to-end forward (the GCN NaN gate) -----------------------------------


def _gcn_forward(models, graphs, in_channels):
    """Batch graphs, run a GCN backbone + mean-pool, return the pooled output."""
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import global_mean_pool

    backbone = models.build_gcn(in_channels=in_channels, hidden_channels=16,
                                out_channels=8, num_layers=3)
    backbone.eval()
    batch = next(iter(DataLoader(graphs, batch_size=len(graphs))))
    with torch.no_grad():
        emb = backbone(batch.x, batch.edge_index, edge_weight=batch.edge_weight)
        return global_mean_pool(emb, batch.batch)


def test_gcn_forward_finite_on_signed_fc(scaffold_pkg, ds_dir):
    """Signed FC through abs-normalized edges -> GCN forward is NaN/Inf-free."""
    _, write = ds_dir
    dl, models = scaffold_pkg["data_loader"], scaffold_pkg["models"]
    fc = np.stack([_signed_fc(20, seed=i) for i in range(5)]).astype(np.float32)
    write("fc_b", fc=fc, y=np.array([0, 1, 0, 1, 0]), task=np.array("classification"))
    graphs = dl.load_fc_graphs("fc_b", edge_weight_norm="abs")
    out = _gcn_forward(models, graphs, in_channels=1)
    assert out.shape == (5, 8)
    assert torch.isfinite(out).all()


def test_gcn_forward_finite_on_sc_matrix(scaffold_pkg, ds_dir):
    """SC bundle (large positive weights) via matrix_key='sc' + abs_max -> finite."""
    _, write = ds_dir
    dl, models = scaffold_pkg["data_loader"], scaffold_pkg["models"]
    rng = np.random.default_rng(0)
    sc = np.stack([rng.gamma(2.0, 500.0, size=(15, 15)).astype(np.float32)
                   for _ in range(4)])   # big streamline-count-like weights
    sc = (sc + sc.transpose(0, 2, 1)) / 2
    fc = np.stack([_signed_fc(12, seed=i) for i in range(4)]).astype(np.float32)
    write("sc_b", fc=fc, sc=sc, y=np.arange(4.0), task=np.array("regression"))
    graphs = dl.load_fc_graphs("sc_b", matrix_key="sc", edge_weight_norm="abs_max")
    out = _gcn_forward(models, graphs, in_channels=1)
    assert out.shape == (4, 8)
    assert torch.isfinite(out).all()


def test_gcn_forward_with_glm_node_features(scaffold_pkg, ds_dir):
    """GLM node-feature bundle feeds an (R,1) x into the GCN; forward is finite."""
    _, write = ds_dir
    dl, models = scaffold_pkg["data_loader"], scaffold_pkg["models"]
    n = 18
    fc = np.stack([_signed_fc(n, seed=i) for i in range(5)]).astype(np.float32)
    glm = np.random.default_rng(1).standard_normal((5, n)).astype(np.float32)
    write("fcglm_b", fc=fc, glm_2back_vs_0back=glm,
          y=np.arange(5.0), task=np.array("regression"))
    graphs = dl.load_fc_graphs("fcglm_b", node_feature_key="glm_2back_vs_0back",
                               edge_weight_norm="abs")
    assert tuple(graphs[0].x.shape) == (n, 1)
    out = _gcn_forward(models, graphs, in_channels=1)
    assert torch.isfinite(out).all()


def test_glm_diagonal_features_builds_normalized_diagonal(scaffold_pkg):
    fc_graph = scaffold_pkg["fc_graph"]
    glm = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)   # mean 4, std ~2.236
    D = fc_graph.glm_diagonal_features(glm, normalize=True)
    assert D.shape == (4, 4)
    # off-diagonal is zero; diagonal is the z-scored glm
    assert np.allclose(D[~np.eye(4, dtype=bool)], 0.0)
    z = (glm - glm.mean()) / (glm.std() + 1e-8)
    assert np.allclose(np.diag(D), z, atol=1e-5)

def test_glm_diagonal_features_no_normalize(scaffold_pkg):
    fc_graph = scaffold_pkg["fc_graph"]
    glm = np.array([2.0, -4.0, 0.0], dtype=np.float32)
    D = fc_graph.glm_diagonal_features(glm, normalize=False)
    assert np.allclose(np.diag(D), glm)
    assert D.shape == (3, 3)

def test_fc_to_data_attaches_graph_attr_u(scaffold_pkg):
    fc_graph = scaffold_pkg["fc_graph"]
    fc = _signed_fc(8)
    d = fc_graph.fc_to_data(fc, 0.0, task="regression",
                            graph_attr=np.array([2.5, -1.0]))   # U=2
    assert tuple(d.u.shape) == (1, 2)
    assert np.allclose(d.u.view(-1).numpy(), [2.5, -1.0])

def test_fc_arrays_to_data_list_threads_graph_attr(scaffold_pkg):
    fc_graph = scaffold_pkg["fc_graph"]
    fc = np.stack([_signed_fc(6, seed=i) for i in range(3)])
    y = np.arange(3.0)
    gattr = np.array([[1.0], [2.0], [3.0]])    # (N, U=1)  e.g. age scalar
    dl = fc_graph.fc_arrays_to_data_list(fc, y, task="regression", graph_attr=gattr)
    assert tuple(dl[2].u.shape) == (1, 1)
    assert float(dl[2].u) == 3.0
