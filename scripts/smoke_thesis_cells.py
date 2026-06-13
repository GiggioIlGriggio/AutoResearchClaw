"""Reference 4-cell smoke for the PNC age->VWM thesis (A1, A3, B1, C1).

Deterministic validation that the M1 scaffold primitives + the pinned per-cell
wiring recipe actually TRAIN FORWARD on the real ``~/rc_brain_data`` SC-400 bundles,
transfer the A1->B1 backbone, and emit ``val_r2`` per cell. This is the *correct*
implementation of exactly what ``config.thesis.smoke.yaml`` pins ARC to generate
(using the real scaffold API: ``node_features=`` / ``node_feature_key=``+``glm_diagonal=``,
NOT a made-up ``carrier=`` kwarg). It validates the recipe/substrate independently of
ARC's autonomous (ACP-flaky) codegen — it is NOT the experiment of record.

Run (ARC venv with torch+pyg):
    RC_DATASET_DIR=~/rc_brain_data .venv/bin/python scripts/smoke_thesis_cells.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
LIMIT = 64          # subjects per bundle (fast graph build)
EPOCHS = 3
BATCH = 8
SEED = 0


def _import_scaffold():
    """Materialize experiment_scaffold/*.py into an importable ``scaffold/`` pkg."""
    root = Path(tempfile.mkdtemp(prefix="smoke_scaffold_"))
    pkg = root / "scaffold"
    pkg.mkdir()
    for py in (REPO / "experiment_scaffold").glob("*.py"):
        shutil.copy(py, pkg / py.name)
    (pkg / "__init__.py").write_text("")
    sys.path.insert(0, str(root))
    import importlib
    return (
        importlib.import_module("scaffold.data_loader"),
        importlib.import_module("scaffold.models"),
        importlib.import_module("scaffold.heads"),
        importlib.import_module("scaffold.transfer"),
    )


def _split(n, frac=0.8, seed=SEED):
    idx = np.random.default_rng(seed).permutation(n)
    cut = int(round(n * frac))
    return idx[:cut], idx[cut:]


def _standardize_targets(graphs, tr):
    ys = np.array([float(graphs[i].y) for i in range(len(graphs))], dtype=np.float64)
    mu, sd = float(ys[tr].mean()), float(ys[tr].std() + 1e-8)
    for g in graphs:
        g.y = torch.tensor([(float(g.y) - mu) / sd], dtype=torch.float)
    return mu, sd


def _train_eval(graphs, models, heads, transfer, *, global_dim, device,
                ckpt_in=None, freeze=False, ckpt_out=None):
    from torch_geometric.loader import DataLoader

    torch.manual_seed(SEED)
    tr, va = _split(len(graphs))
    _standardize_targets(graphs, tr)
    tr_dl = DataLoader([graphs[i] for i in tr], batch_size=BATCH, shuffle=True)
    va_dl = DataLoader([graphs[i] for i in va], batch_size=BATCH)

    in_ch = graphs[0].x.shape[1]                       # 400 for both carriers
    backbone = models.build_gcn(in_channels=in_ch, hidden_channels=32,
                                out_channels=16, num_layers=2, norm="batch_norm")
    if ckpt_in is not None:                            # transfer (B1)
        transfer.load_backbone(backbone, ckpt_in, strict=True)
    if freeze:
        transfer.freeze(backbone)
    model = heads.GraphRegressor(backbone, pooled_dim=16, global_dim=global_dim).to(device)

    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    lossf = nn.MSELoss()
    model.train()
    for _ in range(EPOCHS):
        for b in tr_dl:
            b = b.to(device)
            u = torch.nan_to_num(b.u) if global_dim > 0 else None
            opt.zero_grad()
            out = model(b.x, b.edge_index, b.edge_weight, b.batch, u=u)
            loss = lossf(out, b.y.view(-1))
            loss.backward()
            opt.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for b in va_dl:
            b = b.to(device)
            u = torch.nan_to_num(b.u) if global_dim > 0 else None
            preds.append(model(b.x, b.edge_index, b.edge_weight, b.batch, u=u).cpu())
            trues.append(b.y.view(-1).cpu())
    p, t = torch.cat(preds).numpy(), torch.cat(trues).numpy()
    ss_res = float(((t - p) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if ckpt_out is not None:                           # A1 saves for transfer
        transfer.save_backbone(model.backbone, ckpt_out)
    return float(r2)


def main():
    os.environ.setdefault("RC_DATASET_DIR", os.path.expanduser("~/rc_brain_data"))
    dl, models, heads, transfer = _import_scaffold()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  RC_DATASET_DIR={os.environ['RC_DATASET_DIR']}", flush=True)

    common = dict(matrix_key="sc", edge_weight_norm="abs_max", limit=LIMIT)
    ckpt = Path(tempfile.mkdtemp(prefix="smoke_ckpt_")) / "A1_identity_age.pt"

    # A1: identity -> age (source backbone; saves checkpoint)
    g = dl.load_fc_graphs("pnc_sc400_age_reg", node_features="identity", **common)
    r2_a1 = _train_eval(g, models, heads, transfer, global_dim=0, device=device,
                        ckpt_out=ckpt)
    print(f"cell_A1_val_r2: {r2_a1:.4f}", flush=True)

    # A3: glm_diagonal -> VWM (from-scratch baseline)
    g = dl.load_fc_graphs("pnc_sc400_vwm_reg", node_feature_key="glm_2back_vs_0back",
                          glm_diagonal=True, glm_normalize=True, **common)
    r2_a3 = _train_eval(g, models, heads, transfer, global_dim=0, device=device)
    print(f"cell_A3_val_r2: {r2_a3:.4f}", flush=True)

    # B1: identity -> VWM, fine-tune from the A1 backbone (transfer, strict=True)
    g = dl.load_fc_graphs("pnc_sc400_vwm_reg", node_features="identity", **common)
    r2_b1 = _train_eval(g, models, heads, transfer, global_dim=0, device=device,
                        ckpt_in=ckpt)
    print(f"cell_B1_val_r2: {r2_b1:.4f}", flush=True)

    # C1: glm_diagonal -> VWM, age injected @head (U=1)
    g = dl.load_fc_graphs("pnc_sc400_vwm_reg", node_feature_key="glm_2back_vs_0back",
                          glm_diagonal=True, glm_normalize=True,
                          graph_feature_key="age", **common)
    r2_c1 = _train_eval(g, models, heads, transfer, global_dim=1, device=device)
    print(f"cell_C1_val_r2: {r2_c1:.4f}", flush=True)

    best = max(r2_a1, r2_a3, r2_b1, r2_c1)
    print(f"val_r2: {best:.4f}", flush=True)
    print("SMOKE_OK: 4/4 cells trained forward, A1->B1 transfer ran, metrics emitted",
          flush=True)


if __name__ == "__main__":
    main()
