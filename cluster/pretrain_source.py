"""Pretrain ONE source cell (A1/A4) for ONE (rep, outer) on its fold-train subjects.

Saves the backbone checkpoint keyed (cell, rep, outer) for the dependent B/C cells,
and writes the source's held-out age-R² (for A1/A4 reporting).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cluster import data as D
from cluster import folds as F
from cluster._common import CELLS, DEFAULT_HPS


def ckpt_path(out_dir: Path, cell, rep, outer) -> Path:
    return Path(out_dir) / "ckpts" / f"{cell}_rep{rep}_outer{outer}.pt"


def run(cell, rep, outer, out_dir, *, limit=None, max_epochs=None):
    spec = CELLS[cell]
    assert spec["mode"] == "source", f"{cell} is not a source cell"
    fold = F.outer_split(rep, outer)
    g_tr, g_te = D.load_cell_graphs(cell, fold, limit=limit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cp = ckpt_path(Path(out_dir), cell, rep, outer)
    cp.parent.mkdir(parents=True, exist_ok=True)
    hps = dict(DEFAULT_HPS, lr=1e-3, weight_decay=1e-5)
    age_r2 = train_eval_source(g_tr, g_te, hps, device, cp, rep=rep, outer=outer,
                               max_epochs=max_epochs)
    res = dict(cell=cell, rep=rep, outer=outer, test_r2=age_r2, ckpt=str(cp),
               n_train=len(g_tr), n_test=len(g_te), kind="source_age")
    rp = Path(out_dir) / "results" / f"{cell}_rep{rep}_outer{outer}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(res, indent=2))
    print(f"cell_{cell}_val_r2: {age_r2:.4f}  ckpt={cp}", flush=True)
    return res


def train_eval_source(g_tr, g_te, hps, device, ckpt_out, *, rep=0, outer=0, max_epochs=None):
    """Train the source backbone on an inner-train split of fold-train (early-stop on the
    held-out inner-val), save it, and report held-out age-R² on fold-test (scored once)."""
    import numpy as np
    from cluster import folds as F
    from cluster.trainlib import train_eval
    itr, iva = F.inner_split(np.arange(len(g_tr)), seed=1000 + rep * 10 + outer)
    tr = [g_tr[i] for i in itr]
    va = [g_tr[i] for i in iva]
    kw = {} if max_epochs is None else dict(max_epochs=max_epochs)
    _, test_r2 = train_eval(tr, va, global_dim=0, hps=hps, device=device,
                            ckpt_out=ckpt_out, test_graphs=g_te, **kw)
    return test_r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=["A1", "A4"])
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--outer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.cell, a.rep, a.outer, a.out,
        limit=32 if a.smoke else None, max_epochs=2 if a.smoke else None)


if __name__ == "__main__":
    main()
