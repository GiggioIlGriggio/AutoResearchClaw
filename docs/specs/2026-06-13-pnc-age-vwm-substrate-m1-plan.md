# PNC Age→VWM Substrate (Milestone 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and unit-test the data + model substrate for the 11-cell PNC age→VWM matrix — SC-400 cortical bundles (VWM target + co-stored age) plus scaffold primitives (`glm_diagonal` carrier, `data.u` side-channel, `GraphRegressor`, checkpoint transfer/freeze, A5 baseline).

**Architecture:** Extend the flat `experiment_scaffold/*.py` modules (the scaffold contract only sees `glob("*.py")`) with thesis primitives, all TDD'd in `tests/test_scaffold_modules.py` via the existing "materialize as a `scaffold/` package" fixture. A standalone bake script produces two new `~/rc_brain_data` bundles on a verified-aligned SC-400 graph (`sc[:400,:400]`; the GLM zmap already matches those 400 cortical nodes in order).

**Tech Stack:** Python, numpy, PyTorch, PyTorch-Geometric (GCN backbone), pytest. Bake reuses the LLM_Playground brain loader (`core.data.brain`).

**Spec:** `docs/specs/2026-06-13-pnc-age-vwm-substrate-design.md`.

**Verified facts (alignment gate R1 retired):**
- SC `schaefer400` matrix is 452×452; the **first 400** indices are the cortical Schaefer parcels (52 subcortical/cerebellar at the tail). SC-400 = `sc[:400, :400]`.
- GLM zmap `glm["2back_vs_0back"]` is 400-d and aligns to those 400 cortical nodes **position-by-position** (400/400 on hemisphere+network).
- `SubjectRecord`: `.sc` (452×452 native), `.glm` = `{contrast: (R,)}`, `.subject_id` (`"sub-…"`), `.label`, `.has_sc`, `.has_glm`.

**Conventions for every task:**
- Run the scaffold-module tests with the ARC venv that has torch+pyg:
  `.venv/bin/python -m pytest tests/test_scaffold_modules.py -v`
- Each new test takes the existing `scaffold_pkg` fixture (it copies `experiment_scaffold/*.py` into an importable `scaffold/` package and keeps it on `sys.path`); new modules are then imported with `importlib.import_module("scaffold.<mod>")`.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `experiment_scaffold/fc_graph.py` | FC/SC matrix → PyG `Data` | **Modify**: add `glm_diagonal_features`; add `graph_attr`→`data.u` to `fc_to_data`/`fc_arrays_to_data_list` |
| `experiment_scaffold/data_loader.py` | baked-bundle → `list[Data]` | **Modify**: `glm_diagonal`/`glm_normalize`/`graph_feature_key` options on `load_fc_graphs` |
| `experiment_scaffold/heads.py` | model assembly (backbone+pool+side-channel+head) | **Create** |
| `experiment_scaffold/transfer.py` | checkpoint save/load + freeze | **Create** |
| `experiment_scaffold/baselines.py` | A5 trivial age→VWM regressor | **Create** |
| `scripts/bake_sc400_thesis.py` | bake `pnc_sc400_{age,vwm}_reg` bundles | **Create** |
| `tests/test_scaffold_modules.py` | behavioral tests for all of the above | **Modify**: append tests |
| `tests/test_sc400_bundles.py` | schema validation of the baked bundles (skips if absent) | **Create** |
| `experiment_scaffold/SCAFFOLD.md` | scaffold manifest (LLM-facing) | **Modify**: document new bundles + primitives |

---

## Task 1: `glm_diagonal_features` (the GLM carrier)

**Files:**
- Modify: `experiment_scaffold/fc_graph.py`
- Test: `tests/test_scaffold_modules.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_scaffold_modules.py -k glm_diagonal_features -v`
Expected: FAIL with `AttributeError: module 'scaffold.fc_graph' has no attribute 'glm_diagonal_features'`

- [ ] **Step 3: Write minimal implementation** (add to `experiment_scaffold/fc_graph.py`)

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_scaffold_modules.py -k glm_diagonal_features -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add experiment_scaffold/fc_graph.py tests/test_scaffold_modules.py
git commit -m "feat(scaffold): add glm_diagonal node-feature carrier"
```

---

## Task 2: `data.u` graph-level side-channel

**Files:**
- Modify: `experiment_scaffold/fc_graph.py` (`fc_to_data`, `fc_arrays_to_data_list`)
- Test: `tests/test_scaffold_modules.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_scaffold_modules.py -k graph_attr -v`
Expected: FAIL — `fc_to_data() got an unexpected keyword argument 'graph_attr'`

- [ ] **Step 3: Write minimal implementation**

In `fc_to_data` add the parameter `graph_attr=None` to the signature, and just before the `return Data(...)`, build the optional global feature and pass it:

```python
    u = None
    if graph_attr is not None:
        u = torch.as_tensor(np.asarray(graph_attr), dtype=torch.float).reshape(1, -1)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight.view(-1, 1),
        edge_weight=edge_weight,
        y=y_tensor,
        u=u,
    )
```

In `fc_arrays_to_data_list` add `graph_attr=None` to the signature and pass the per-sample slice:

```python
        fc_to_data(fc[i], y[i], k=k, node_features=node_features, task=task,
                   edge_construction=edge_construction, density=density,
                   edge_weight_norm=edge_weight_norm,
                   node_attr=None if node_attr is None else node_attr[i],
                   graph_attr=None if graph_attr is None else graph_attr[i])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_scaffold_modules.py -k graph_attr -v`
Expected: PASS (2 tests). Also run the full file to confirm no regression:
`.venv/bin/python -m pytest tests/test_scaffold_modules.py -v` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add experiment_scaffold/fc_graph.py tests/test_scaffold_modules.py
git commit -m "feat(scaffold): attach optional graph-level feature as data.u"
```

---

## Task 3: `data_loader` glm_diagonal + side-channel wiring

**Files:**
- Modify: `experiment_scaffold/data_loader.py` (`load_fc_graphs`)
- Test: `tests/test_scaffold_modules.py`

- [ ] **Step 1: Write the failing test**

```python
def test_load_fc_graphs_glm_diagonal_node_features(scaffold_pkg, ds_dir):
    _, write = ds_dir
    data_loader = scaffold_pkg["data_loader"]
    n = 6
    sc = np.stack([_signed_fc(n, seed=i) for i in range(3)]).astype(np.float32)
    glm = np.stack([np.arange(1.0, n + 1.0, dtype=np.float32) for _ in range(3)])  # (3, n)
    write("pnc_sc400_vwm_reg", sc=sc, glm_2back_vs_0back=glm,
          y=np.arange(3.0), age=np.array([10.0, 12.0, 14.0]),
          task=np.array("regression"))
    graphs = data_loader.load_fc_graphs(
        "pnc_sc400_vwm_reg", matrix_key="sc", edge_weight_norm="abs_max",
        node_feature_key="glm_2back_vs_0back", glm_diagonal=True)
    assert tuple(graphs[0].x.shape) == (n, n)        # diagonal carrier (R, R)
    z = (glm[0] - glm[0].mean()) / (glm[0].std() + 1e-8)
    assert np.allclose(np.diag(graphs[0].x.numpy()), z, atol=1e-5)

def test_load_fc_graphs_graph_feature_key_age(scaffold_pkg, ds_dir):
    _, write = ds_dir
    data_loader = scaffold_pkg["data_loader"]
    sc = np.stack([_signed_fc(5, seed=i) for i in range(3)]).astype(np.float32)
    write("pnc_sc400_vwm_reg2", sc=sc, y=np.arange(3.0),
          age=np.array([8.0, 9.0, 10.0]), task=np.array("regression"))
    graphs = data_loader.load_fc_graphs(
        "pnc_sc400_vwm_reg2", matrix_key="sc", edge_weight_norm="abs_max",
        graph_feature_key="age")
    assert tuple(graphs[1].u.shape) == (1, 1)
    assert float(graphs[1].u) == 9.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_scaffold_modules.py -k "glm_diagonal_node_features or graph_feature_key_age" -v`
Expected: FAIL — `load_fc_graphs() got an unexpected keyword argument 'glm_diagonal'`

- [ ] **Step 3: Write minimal implementation**

Update `load_fc_graphs` signature to add `glm_diagonal: bool = False`, `glm_normalize: bool = True`, `graph_feature_key: str | None = None`. Import the helper at module top: `from .fc_graph import fc_arrays_to_data_list, glm_diagonal_features`. Replace the node-attr block and the final return:

```python
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
```

Apply the existing `indices`/`limit` subsetting to `graph_attr` too (mirror the `node_attr` lines), then:

```python
    return fc_arrays_to_data_list(
        fc, y, k=k, node_features=node_features, task=task,
        edge_construction=edge_construction, density=density,
        edge_weight_norm=edge_weight_norm, node_attr=node_attr, graph_attr=graph_attr,
    )
```

For `indices`/`limit`, add alongside the existing `node_attr` subsetting:

```python
    if indices is not None:
        sel = np.asarray(indices)
        fc, y = fc[sel], y[sel]
        if node_attr is not None: node_attr = node_attr[sel]
        if graph_attr is not None: graph_attr = graph_attr[sel]
    elif limit is not None:
        fc, y = fc[:limit], y[:limit]
        if node_attr is not None: node_attr = node_attr[:limit]
        if graph_attr is not None: graph_attr = graph_attr[:limit]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_scaffold_modules.py -v`
Expected: all PASS (new + existing).

- [ ] **Step 5: Commit**

```bash
git add experiment_scaffold/data_loader.py tests/test_scaffold_modules.py
git commit -m "feat(scaffold): glm_diagonal + graph_feature_key loader options"
```

---

## Task 4: `heads.py` — `GraphRegressor`

**Files:**
- Create: `experiment_scaffold/heads.py`
- Test: `tests/test_scaffold_modules.py`

- [ ] **Step 1: Write the failing test**

```python
def _toy_batch(scaffold_pkg, n=10, r=6, with_u=0):
    from torch_geometric.loader import DataLoader
    fc_graph = scaffold_pkg["fc_graph"]
    fcs = np.stack([_signed_fc(r, seed=i) for i in range(n)])
    y = np.arange(float(n))
    gattr = None if not with_u else np.zeros((n, with_u), dtype=np.float32) + 0.5
    dl = fc_graph.fc_arrays_to_data_list(fcs, y, task="regression",
                                         node_features="identity",
                                         edge_weight_norm="abs", graph_attr=gattr)
    return next(iter(DataLoader(dl, batch_size=n))), r

def test_graph_regressor_plain(scaffold_pkg):
    import importlib
    heads = importlib.import_module("scaffold.heads")
    models = scaffold_pkg["models"]
    batch, r = _toy_batch(scaffold_pkg, with_u=0)
    backbone = models.build_gcn(in_channels=r, hidden_channels=16, out_channels=8)
    model = heads.GraphRegressor(backbone, pooled_dim=8, global_dim=0)
    out = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
    assert tuple(out.shape) == (10,)
    assert torch.isfinite(out).all()

def test_graph_regressor_with_side_channel(scaffold_pkg):
    import importlib
    heads = importlib.import_module("scaffold.heads")
    models = scaffold_pkg["models"]
    batch, r = _toy_batch(scaffold_pkg, with_u=1)        # age @head dim 1
    backbone = models.build_gcn(in_channels=r, hidden_channels=16, out_channels=8)
    model = heads.GraphRegressor(backbone, pooled_dim=8, global_dim=1)
    out = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch, u=batch.u)
    assert tuple(out.shape) == (10,)
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_scaffold_modules.py -k graph_regressor -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scaffold.heads'`

- [ ] **Step 3: Write minimal implementation** (`experiment_scaffold/heads.py`)

```python
"""Graph-level regression model: GCN backbone + pooling + optional @head side-channel.

``GraphRegressor`` wraps a ``models.build_gcn`` backbone, mean-pools node
embeddings to a graph embedding, optionally concatenates an MLP-encoded
graph-level side channel (``data.u`` -- age scalar for C1, GLM zmap for C2),
and maps the result to a scalar with a small MLP head.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool


class GraphRegressor(nn.Module):
    """Backbone -> mean-pool -> [optional u-encoder ++] -> MLP head -> scalar."""

    def __init__(self, backbone, pooled_dim: int, global_dim: int = 0,
                 global_hidden: int = 32, head_hidden: int = 64):
        super().__init__()
        self.backbone = backbone
        self.global_dim = global_dim
        if global_dim > 0:
            self.global_enc = nn.Sequential(
                nn.Linear(global_dim, global_hidden), nn.ReLU())
            head_in = pooled_dim + global_hidden
        else:
            self.global_enc = None
            head_in = pooled_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))

    def forward(self, x, edge_index, edge_weight, batch, u=None):
        h = self.backbone(x, edge_index, edge_weight=edge_weight)   # [n_nodes, pooled_dim]
        h = global_mean_pool(h, batch)                              # [B, pooled_dim]
        if self.global_enc is not None:
            if u is None:
                raise ValueError("global_dim>0 requires u (the @head side channel)")
            h = torch.cat([h, self.global_enc(u)], dim=-1)
        return self.head(h).squeeze(-1)                             # [B]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_scaffold_modules.py -k graph_regressor -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add experiment_scaffold/heads.py tests/test_scaffold_modules.py
git commit -m "feat(scaffold): GraphRegressor with optional @head side-channel"
```

---

## Task 5: `transfer.py` — checkpoint save/load + freeze

**Files:**
- Create: `experiment_scaffold/transfer.py`
- Test: `tests/test_scaffold_modules.py`

- [ ] **Step 1: Write the failing test**

```python
def test_backbone_checkpoint_roundtrip_and_freeze(scaffold_pkg, tmp_path):
    import importlib
    transfer = importlib.import_module("scaffold.transfer")
    models = scaffold_pkg["models"]
    src = models.build_gcn(in_channels=6, hidden_channels=16, out_channels=8)
    dst = models.build_gcn(in_channels=6, hidden_channels=16, out_channels=8)
    # weights differ before transfer
    p_src = next(src.parameters()); p_dst = next(dst.parameters())
    assert not torch.allclose(p_src, p_dst)
    ckpt = tmp_path / "age_backbone.pt"
    transfer.save_backbone(src, ckpt)
    transfer.load_backbone(dst, ckpt, strict=True)
    for a, b in zip(src.parameters(), dst.parameters()):
        assert torch.allclose(a, b)
    # freeze: no grads on the backbone
    transfer.freeze(dst)
    assert all(not p.requires_grad for p in dst.parameters())

def test_set_trainable_toggles_grad(scaffold_pkg):
    import importlib
    transfer = importlib.import_module("scaffold.transfer")
    models = scaffold_pkg["models"]
    m = models.build_gcn(in_channels=6, hidden_channels=8, out_channels=4)
    transfer.set_trainable(m, False)
    assert all(not p.requires_grad for p in m.parameters())
    transfer.set_trainable(m, True)
    assert all(p.requires_grad for p in m.parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_scaffold_modules.py -k "checkpoint_roundtrip or set_trainable" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scaffold.transfer'`

- [ ] **Step 3: Write minimal implementation** (`experiment_scaffold/transfer.py`)

```python
"""Backbone checkpoint transfer + freezing for the age->VWM sequential cells.

Save an age-pretrained GCN backbone, reload it into a fresh target model
(``strict=True`` is safe: identity and glm_diagonal carriers are both 400-dim,
so the backbone input dim is identical across all 11 cells), and freeze it for
the frozen-feature modes (B2, B4).
"""
from __future__ import annotations

import torch


def save_backbone(backbone, path) -> None:
    """Persist just the backbone ``state_dict`` to ``path``."""
    torch.save(backbone.state_dict(), str(path))


def load_backbone(backbone, path, strict: bool = True):
    """Load a saved backbone ``state_dict`` into ``backbone`` (returns it)."""
    backbone.load_state_dict(torch.load(str(path), map_location="cpu"), strict=strict)
    return backbone


def freeze(module):
    """Set ``requires_grad=False`` on every parameter (frozen-feature mode)."""
    return set_trainable(module, False)


def set_trainable(module, flag: bool):
    """Set ``requires_grad=flag`` on every parameter of ``module`` (returns it)."""
    for p in module.parameters():
        p.requires_grad_(flag)
    return module
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_scaffold_modules.py -k "checkpoint_roundtrip or set_trainable" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add experiment_scaffold/transfer.py tests/test_scaffold_modules.py
git commit -m "feat(scaffold): backbone checkpoint transfer + freeze helpers"
```

---

## Task 6: `baselines.py` — A5 trivial age→VWM

**Files:**
- Create: `experiment_scaffold/baselines.py`
- Test: `tests/test_scaffold_modules.py`

- [ ] **Step 1: Write the failing test**

```python
def test_age_vwm_baseline_recovers_linear_signal(scaffold_pkg):
    import importlib
    baselines = importlib.import_module("scaffold.baselines")
    rng = np.random.default_rng(0)
    age = rng.uniform(8, 21, size=300)
    vwm = 0.1 * age + rng.normal(0, 0.05, size=300)     # strong linear age->vwm
    (slope, intercept), r2 = baselines.age_vwm_baseline(age, vwm, cv=5)
    assert 0.5 < r2 <= 1.0
    assert abs(slope - 0.1) < 0.03

def test_age_vwm_baseline_floor_on_noise(scaffold_pkg):
    import importlib
    baselines = importlib.import_module("scaffold.baselines")
    rng = np.random.default_rng(1)
    age = rng.uniform(8, 21, size=200)
    vwm = rng.normal(0, 1, size=200)                    # no relationship
    _, r2 = baselines.age_vwm_baseline(age, vwm, cv=5)
    assert r2 < 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_scaffold_modules.py -k age_vwm_baseline -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scaffold.baselines'`

- [ ] **Step 3: Write minimal implementation** (`experiment_scaffold/baselines.py`)

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_scaffold_modules.py -k age_vwm_baseline -v`
Expected: PASS (2 tests). Then full file: `.venv/bin/python -m pytest tests/test_scaffold_modules.py -v` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add experiment_scaffold/baselines.py tests/test_scaffold_modules.py
git commit -m "feat(scaffold): A5 trivial age->VWM baseline"
```

---

## Task 7: Bake the SC-400 bundles

**Files:**
- Create: `scripts/bake_sc400_thesis.py`
- Reference: `scripts/bake_scaffold_dataset.py` (existing bake — patterns for `LoadConfig`/staging)

> This task produces data, not a pure-unit-test target. It is validated by running
> the bake and then Task 8's schema test. Run with the full-stack venv (torch+pyg+
> pandas+scipy), per the `gnn-scaffold-dataset` memory:
> `/home/compa/Documents/working_dir/EF_neural_substrate/.venv/bin/python3`.

- [ ] **Step 1: Write the bake script** (`scripts/bake_sc400_thesis.py`)

```python
"""Bake the PNC age->VWM thesis bundles on a 400-node cortical SC graph.

Two bundles in ~/rc_brain_data, both with the SC-400 graph (sc[:400,:400] of the
452-node schaefer400 SC -- cortical Schaefer-400, verified to align in order with
the 400-d GLM zmap) and the 400-d glm_2back_vs_0back node-feature array:

  pnc_sc400_age_reg : y = age            (A1 identity->age, A4 glm_diagonal->age; source ckpts)
  pnc_sc400_vwm_reg : y = VWM_overall_dprime, + age (N,) co-stored (A5 floor, C1 age@head)

Both filter to has_sc AND has_glm so every cell's carrier is available. Run with a
venv that has torch+torch_geometric+pandas+scipy.
"""
import os, sys, json, shutil, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

CORE = "/home/compa/Documents/working_dir/LLM_Playground/core"
sys.path.insert(0, CORE)
from core.data.brain.config import LoadConfig          # noqa: E402
from core.data.brain.records import load_cohort         # noqa: E402

REAL_ROOT = Path("/media/compa/DATA1/Compa/DATA_DERIVATIVES")
OUT = Path(os.path.expanduser("~/rc_brain_data")); OUT.mkdir(parents=True, exist_ok=True)
GLM_CONTRAST, GLM_AGG, GLM_MAPTYPE = "2back_vs_0back", "mean", "zmap"
N_CORTICAL = 400


def stage_pnc():
    """Symlinked PNC root with cleaned labels (Sex->F/M, age coerced + <5y dropped).

    Adapted from scripts/bake_scaffold_dataset.py:stage_pnc (kept local so this
    one-off thesis bake does not import that script's top-level bake).
    """
    stage = Path(tempfile.mkdtemp(prefix="pnc_sc400_"))
    t0 = stage / "PNC" / "T0"; t0.mkdir(parents=True)
    src = REAL_ROOT / "PNC" / "T0"
    (t0 / "Functional_Mats").symlink_to(src / "Functional_Mats")
    (t0 / "Structural_maps").symlink_to(src / "Structural_maps")
    (t0 / "GLM_Maps").symlink_to(src / "GLM_Maps")
    tab = t0 / "Tabular_data"; tab.mkdir()
    df = pd.read_csv(src / "Tabular_data" / "PNC_ALL_SCORES.csv", low_memory=False)
    df["Sex"] = df["Sex"].where(df["Sex"].isin(["F", "M"]))
    age = pd.to_numeric(df["age_at_cnb"], errors="coerce")
    df["age_at_cnb"] = age.where(age >= 5)
    df.to_csv(tab / "PNC_ALL_SCORES.csv", index=False)
    shutil.copy(src / "Tabular_data" / "subject_mapping.tsv", tab / "subject_mapping.tsv")
    return stage


def load(label, root):
    cfg = LoadConfig(cohort="pnc", label_column=label, task="regression", load_sc=True,
                     glm_contrast=GLM_CONTRAST, glm_agg=GLM_AGG, glm_maptype=GLM_MAPTYPE,
                     root=root)
    recs = [r for r in load_cohort(cfg) if r.has_sc and r.has_glm]
    return recs


def arrays(recs):
    sc = np.stack([np.asarray(r.sc[:N_CORTICAL, :N_CORTICAL], np.float32) for r in recs])
    glm = np.stack([np.asarray(r.glm[GLM_CONTRAST], np.float32) for r in recs])
    assert sc.shape[1:] == (N_CORTICAL, N_CORTICAL), sc.shape
    assert glm.shape[1] == N_CORTICAL, glm.shape          # alignment gate
    sid = np.array([r.subject_id for r in recs])
    y = np.array([float(r.label) for r in recs], np.float32)
    return sc, glm, sid, y


def write(name, sc, glm, sid, y, age=None):
    z = dict(sc=sc, glm_2back_vs_0back=glm, subject_id=sid, y=y,
             task=np.array("regression"),
             glm_contrast=np.array(GLM_CONTRAST), glm_agg=np.array(GLM_AGG),
             glm_maptype=np.array(GLM_MAPTYPE))
    if age is not None:
        z["age"] = age
    np.savez(OUT / f"{name}.npz", **z)
    info = {"name": name, "n": int(len(y)), "R_sc": N_CORTICAL,
            "graph_matrix_key": "sc", "node_feature_key": "glm_2back_vs_0back",
            "y_range": [round(float(y.min()), 3), round(float(y.max()), 3)],
            "size_mb": round((OUT / f"{name}.npz").stat().st_size / 1e6, 1)}
    if age is not None:
        info["age_range"] = [round(float(np.nanmin(age)), 2), round(float(np.nanmax(age)), 2)]
        info["n_age_nan"] = int(np.isnan(age).sum())
    print(f"[OK] {name}: N={len(y)} {info['y_range']} ({info['size_mb']} MB)", flush=True)
    return info


def main():
    stage = stage_pnc()
    manifest = []
    try:
        # age->subject_id map (for co-storing age on the VWM bundle)
        age_recs = load("age_at_cnb", stage)
        age_map = {r.subject_id: float(r.label) for r in age_recs}

        # pnc_sc400_age_reg (source checkpoints A1/A4)
        sc, glm, sid, y = arrays(age_recs)
        manifest.append(write("pnc_sc400_age_reg", sc, glm, sid, y))

        # pnc_sc400_vwm_reg (targets; age co-stored)
        vwm_recs = load("VWM_overall_dprime", stage)
        sc, glm, sid, y = arrays(vwm_recs)
        age = np.array([age_map.get(s, np.nan) for s in sid], np.float32)
        manifest.append(write("pnc_sc400_vwm_reg", sc, glm, sid, y, age=age))
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    mpath = OUT / "MANIFEST_sc400.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print("\n=== DONE ===\n" + json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the bake**

Run: `/home/compa/Documents/working_dir/EF_neural_substrate/.venv/bin/python3 scripts/bake_sc400_thesis.py`
Expected: two `[OK]` lines and a DONE manifest. `pnc_sc400_age_reg` N≈700 (sc∩glm), `pnc_sc400_vwm_reg` N smaller (∩ VWM-labelled). Both `R_sc=400`. If either `assert` in `arrays()` trips, STOP — the 400-cortical/GLM alignment assumption broke; re-examine before proceeding.

- [ ] **Step 3: Eyeball the outputs**

Run: `ls -la ~/rc_brain_data/pnc_sc400_*.npz && cat ~/rc_brain_data/MANIFEST_sc400.json`
Expected: two files present; manifest Ns and ranges sane (age 8–21; VWM dprime roughly −1…3).

- [ ] **Step 4: Commit**

```bash
git add scripts/bake_sc400_thesis.py
git commit -m "feat(data): bake PNC SC-400 cortical age/VWM thesis bundles"
```

(The `.npz` bundles live in `~/rc_brain_data`, outside the repo — not committed, same as the existing dataset.)

---

## Task 8: Schema test for the baked bundles

**Files:**
- Create: `tests/test_sc400_bundles.py`

- [ ] **Step 1: Write the test** (skips cleanly if the bundles aren't baked / wrong env)

```python
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
```

- [ ] **Step 2: Run the test**

Run: `.venv/bin/python -m pytest tests/test_sc400_bundles.py -v`
Expected: PASS if Task 7 ran (bundles present); otherwise SKIPPED with clear reasons.

- [ ] **Step 3: Commit**

```bash
git add tests/test_sc400_bundles.py
git commit -m "test(data): schema validation for SC-400 thesis bundles"
```

---

## Task 9: Document the new substrate in `SCAFFOLD.md`

**Files:**
- Modify: `experiment_scaffold/SCAFFOLD.md`

- [ ] **Step 1: Add a "Thesis primitives (PNC age→VWM)" section** documenting, with one line each: `glm_diagonal_features` / the `glm_diagonal` + `glm_normalize` loader options; `graph_feature_key` → `data.u`; `heads.GraphRegressor(backbone, pooled_dim, global_dim)`; `transfer.{save_backbone,load_backbone,freeze,set_trainable}`; `baselines.age_vwm_baseline`. Add the two bundles to the dataset table:

```markdown
| `pnc_sc400_age_reg` | PNC | age (years) | regression | ~700 | SC 400 (+GLM) | 8.0 – 21.0 |
| `pnc_sc400_vwm_reg` | PNC | VWM dprime  | regression | (∩ VWM) | SC 400 (+GLM, +age) | dprime range |
```

with a note: graph is the **400 cortical Schaefer** SC (`sc[:400,:400]`); carriers are `node_features="identity"` or `node_feature_key="glm_2back_vs_0back", glm_diagonal=True`; `graph_feature_key="age"` / `"glm_2back_vs_0back"` supply the @head side channel; load with `matrix_key="sc", edge_weight_norm="abs_max"`.

- [ ] **Step 2: Commit**

```bash
git add experiment_scaffold/SCAFFOLD.md
git commit -m "docs(scaffold): document SC-400 bundles + thesis primitives"
```

---

## Self-Review

**Spec coverage** (each §3/§4/§5 spec item → task):
- §3 `pnc_sc400_age_reg` / `pnc_sc400_vwm_reg` with age co-stored, SC-400 subset, alignment gate → **Task 7** (gate = `arrays()` asserts; retired in plan preamble).
- §4.1 `glm_diagonal` carrier → **Task 1**; wired in loader → **Task 3**.
- §4.2 `graph_feature_key`→`data.u` → **Tasks 2 + 3**.
- §4.3 `GraphRegressor` (global_dim 0/1/400) → **Task 4**.
- §4.4 `transfer` save/load/freeze → **Task 5**.
- §4.5 A5 `age_vwm_baseline` → **Task 6**.
- §5 tests (glm_diagonal, data.u, GraphRegressor finite, checkpoint round-trip+freeze, A5) → **Tasks 1–6**; bundle schema → **Task 8**.
- §8 DoD: SCAFFOLD.md update → **Task 9**.

**Placeholder scan:** no TBD/TODO; every code step has complete code; commands have expected output. (`global_dim=400` for C2 GLM @head isn't separately unit-tested beyond the `global_dim=1` path — same code path, dimension-parametrized; the loader's `graph_feature_key="glm_2back_vs_0back"` at U=400 is covered structurally by Task 3's age test + Task 8.)

**Type/name consistency:** `GraphRegressor(backbone, pooled_dim, global_dim=…)` used identically in Tasks 4 & 8 reasoning; `glm_diagonal_features(glm_vec, normalize=)` consistent Tasks 1/3/7; `load_fc_graphs(..., node_feature_key, glm_diagonal, glm_normalize, graph_feature_key)` consistent Tasks 3/8; bundle keys `sc`, `glm_2back_vs_0back`, `age`, `y`, `task` consistent Tasks 3/7/8.

**Deferred (not this plan):** D1 @node surgery; ARC config/prompts (M2); cluster/SLURM (M3); analysis/paper (M4).
