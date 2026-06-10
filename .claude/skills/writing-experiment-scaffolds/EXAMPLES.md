# Worked Example — Brain-Graph GNN Scaffold

A complete scaffold for brain-graph (fMRI connectivity) classification: a GCN baseline, a
loader that reads the dataset via `RC_DATASET_DIR`, shared utils, and the `SCAFFOLD.md`
manifest. Note the **package-qualified import** in `data_loader.py`.

## `models.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Dense GCN layer: H' = act(A_norm @ H @ W)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj_norm):
        return self.lin(adj_norm @ x)


class BaseGNN(nn.Module):
    """3-layer dense GCN baseline for brain-graph graph classification.

    Args:
        in_dim:    node-feature dimension
        hidden:    hidden width
        n_classes: number of target classes
        n_layers:  number of GCN layers (default 3)

    forward(x, adj_norm):
        x        (N, in_dim) node features
        adj_norm (N, N)      symmetric-normalised adjacency
        returns  (n_classes,) graph logits via global mean pooling
    """

    def __init__(self, in_dim, hidden, n_classes, n_layers=3):
        super().__init__()
        dims = [in_dim] + [hidden] * n_layers
        self.layers = nn.ModuleList(GCNLayer(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x, adj_norm):
        for layer in self.layers:
            x = F.relu(layer(x, adj_norm))
        return self.head(x.mean(dim=0))
```

## `utils.py`

```python
import numpy as np
import torch


def set_seed(seed):
    """Seed numpy + torch for reproducible runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_adj(adj):
    """Symmetric normalisation D^-1/2 (A + I) D^-1/2. adj: (N, N) tensor."""
    a = adj + torch.eye(adj.size(0), dtype=adj.dtype)
    deg = a.sum(dim=1).clamp(min=1.0)
    dinv = deg.pow(-0.5)
    return dinv.unsqueeze(1) * a * dinv.unsqueeze(0)
```

## `data_loader.py`

```python
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scaffold.utils import normalize_adj  # package-qualified intra-scaffold import (rule 1)


def dataset_root():
    """Dataset root provided by the pipeline (RC_DATASET_DIR). Never hardcode a path (rule 2)."""
    root = os.environ.get("RC_DATASET_DIR")
    if not root:
        raise RuntimeError(
            "RC_DATASET_DIR is not set. The pipeline sets it at run time; "
            "set it yourself only for local testing."
        )
    return Path(root)


def load_brain_graphs(split="train"):
    """Load ABIDE brain-connectivity graphs for one split.

    Files under RC_DATASET_DIR:
        adjacency.npy  (S, N, N) float32 connectivity matrices
        features.npy   (S, N, F) float32 node features
        labels.csv     columns: 'label' (int), 'split' (train/val/test)

    Returns a list of dicts: {'adj_norm': (N,N) tensor, 'x': (N,F) tensor, 'y': int}.
    """
    root = dataset_root()
    adj = np.load(root / "adjacency.npy")
    feats = np.load(root / "features.npy")
    meta = pd.read_csv(root / "labels.csv")
    out = []
    for i in meta.index[meta["split"] == split]:
        a = torch.from_numpy(adj[i]).float()
        out.append({
            "adj_norm": normalize_adj(a),
            "x": torch.from_numpy(feats[i]).float(),
            "y": int(meta.loc[i, "label"]),
        })
    return out
```

## `SCAFFOLD.md`

````markdown
# Brain-Graph GNN Scaffold

Base models, loaders, and utilities for brain-graph (fMRI connectivity) classification.
Import these directly; do not modify them. Write your own top-level module only when a
research idea needs a variant.

## Dataset

Brain-connectivity graphs (e.g. ABIDE), located at `RC_DATASET_DIR`:
- `adjacency.npy` — `(S, N, N)` float32 connectivity matrices
- `features.npy`  — `(S, N, F)` float32 node features
- `labels.csv`    — columns `label` (int class), `split` (`train`/`val`/`test`)

Access via `os.environ['RC_DATASET_DIR']` or `scaffold.data_loader.dataset_root()`.

## Models — `scaffold/models.py`

- `BaseGNN(in_dim, hidden, n_classes, n_layers=3)` — 3-layer dense GCN baseline for graph
  classification. `forward(x, adj_norm) -> (n_classes,)` logits via global mean pooling.

## Loaders — `scaffold/data_loader.py`

- `load_brain_graphs(split="train") -> list[dict]` — graphs as
  `{'adj_norm': (N,N) tensor, 'x': (N,F) tensor, 'y': int}`; adjacency is already
  symmetric-normalised.
- `dataset_root() -> Path` — the `RC_DATASET_DIR` root.

## Utils — `scaffold/utils.py`

- `set_seed(seed)` — seed numpy + torch.
- `normalize_adj(adj) -> tensor` — `D^-1/2 (A+I) D^-1/2` symmetric normalisation.

## When to vary

Use `BaseGNN` as-is for the baseline arm. For a new architecture (attention pooling, GAT
layers, …), write your own top-level `models.py` that may import building blocks from
`scaffold`, and leave `scaffold/models.py` untouched.
````

## How generated experiment code uses it

```python
# main.py — LLM-generated, "use-as-is" path
import os
from scaffold.models import BaseGNN
from scaffold.data_loader import load_brain_graphs
from scaffold.utils import set_seed

set_seed(0)
graphs = load_brain_graphs("train")          # dataset found via RC_DATASET_DIR
model = BaseGNN(in_dim=graphs[0]["x"].shape[1], hidden=64, n_classes=2)
# ... training loop ...
```

```python
# models.py — LLM-generated "variant" path (scaffold/models.py stays untouched)
import torch.nn as nn
from scaffold.models import GCNLayer        # reuse the building block

class AttentionPoolGNN(nn.Module):
    """Variant: same GCN stack, attention pooling instead of mean pooling."""
    ...
```
