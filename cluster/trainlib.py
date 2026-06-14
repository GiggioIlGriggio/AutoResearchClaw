"""Shared train/eval for source + target cells (z-scored targets, early stopping, R²)."""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn

from cluster._common import ARCH, BATCH_SIZE, MAX_EPOCHS, PATIENCE
from cluster._scaffold import import_scaffold

_models = _heads = _transfer = None


def _prim():
    global _models, _heads, _transfer
    if _models is None:
        _, _models, _heads, _transfer, _ = import_scaffold()
    return _models, _heads, _transfer


def _zscore(graphs, ref_idx):
    ys = np.array([float(graphs[i].y) for i in range(len(graphs))], dtype=np.float64)
    mu, sd = float(ys[ref_idx].mean()), float(ys[ref_idx].std() + 1e-8)
    for g in graphs:
        g.y = torch.tensor([(float(g.y) - mu) / sd], dtype=torch.float)
    return mu, sd


def _r2(model, dl, device, global_dim):
    model.eval()
    p, t = [], []
    with torch.no_grad():
        for b in dl:
            b = b.to(device)
            u = torch.nan_to_num(b.u) if global_dim > 0 else None
            p.append(model(b.x, b.edge_index, b.edge_weight, b.batch, u=u).cpu())
            t.append(b.y.view(-1).cpu())
    p, t = torch.cat(p).numpy(), torch.cat(t).numpy()
    ss_res = float(((t - p) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def build_model(in_ch, global_dim, hps, device, *, ckpt_in=None, freeze=False):
    models, heads, transfer = _prim()
    backbone = models.build_gcn(in_channels=in_ch, hidden_channels=ARCH["hidden_channels"],
                                out_channels=ARCH["out_channels"], num_layers=ARCH["num_layers"],
                                norm=ARCH["norm"], dropout=hps.get("dropout", 0.0))
    if ckpt_in is not None:
        transfer.load_backbone(backbone, ckpt_in, strict=True)
    if freeze:
        transfer.freeze(backbone)
    model = heads.GraphRegressor(backbone, pooled_dim=ARCH["out_channels"], global_dim=global_dim,
                                 global_hidden=hps.get("global_hidden", 32),
                                 head_hidden=hps.get("head_hidden", 64)).to(device)
    return model


def train_eval(train_graphs, val_graphs, *, global_dim, hps, device,
               ckpt_in=None, freeze=False, ckpt_out=None, test_graphs=None,
               max_epochs=MAX_EPOCHS):
    """Train on train_graphs, EARLY-STOP on val_graphs, score test_graphs ONCE at best-val.

    Targets are z-scored with TRAIN stats only. The returned test_r2 is never used for
    epoch selection (no test peeking) — it is measured a single time after restoring the
    best-val checkpoint. Returns (best_val_r2, test_r2_or_None).

    Input graph lists are NOT mutated: graphs are shallow-copied before z-scoring, so the
    same graph objects may be passed across many calls (e.g. an Optuna inner loop).
    """
    from torch_geometric.loader import DataLoader

    torch.manual_seed(0)
    train_graphs, val_graphs = list(train_graphs), list(val_graphs)
    test_graphs = list(test_graphs) if test_graphs is not None else []
    n_tr, n_va = len(train_graphs), len(val_graphs)
    all_g = [copy.copy(g) for g in train_graphs + val_graphs + test_graphs]
    _zscore(all_g, np.arange(n_tr))                 # z-score using TRAIN stats only
    tr_dl = DataLoader(all_g[:n_tr], batch_size=BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(all_g[n_tr:n_tr + n_va], batch_size=BATCH_SIZE)
    te_dl = DataLoader(all_g[n_tr + n_va:], batch_size=BATCH_SIZE) if test_graphs else None

    _, _, transfer = _prim()
    in_ch = all_g[0].x.shape[1]
    model = build_model(in_ch, global_dim, hps, device, ckpt_in=ckpt_in, freeze=freeze)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                           lr=hps.get("lr", 1e-3), weight_decay=hps.get("weight_decay", 0.0))
    lossf = nn.MSELoss()

    best, best_state, bad = -1e9, None, 0
    for _ in range(max_epochs):
        model.train()
        for b in tr_dl:
            b = b.to(device)
            u = torch.nan_to_num(b.u) if global_dim > 0 else None
            opt.zero_grad()
            out = model(b.x, b.edge_index, b.edge_weight, b.batch, u=u)
            lossf(out, b.y.view(-1)).backward()
            opt.step()
        score = _r2(model, va_dl, device, global_dim)        # EARLY-STOP on VAL, not test
        if score > best:
            best, best_state, bad = score, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if ckpt_out is not None:
        transfer.save_backbone(model.backbone, ckpt_out)
    test_r2 = _r2(model, te_dl, device, global_dim) if te_dl is not None else None
    return best, test_r2
