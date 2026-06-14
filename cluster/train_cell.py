"""Run ONE (cell, rep, outer) target unit of the nested CV → result JSON.

Inner Optuna HPO on an inner holdout, refit on full fold-train, eval held-out fold-test.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import optuna
import torch

from cluster import data as D
from cluster import folds as F
from cluster._common import (CELLS, INNER_TRIALS, suggest_hps)
from cluster._scaffold import import_scaffold
from cluster.pretrain_source import ckpt_path
from cluster.trainlib import train_eval


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _write_json(path: Path, obj: dict):
    """Atomically write JSON (tmp + os.replace) so a preempted task can't leave a
    truncated result file that breaks the reducer."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _a5(fold, rep, outer):
    """Trivial age→VWM OLS floor on this fold (no graph)."""
    dl, *_ = import_scaffold()
    _, y, meta = dl.load_fc_bundle("pnc_sc400_vwm_reg", matrix_key="sc")
    age = np.asarray(meta["age"], dtype=np.float64)
    vwm = np.asarray(y, dtype=np.float64)
    tr, te = fold.train_vwm, fold.test_vwm
    m = ~np.isnan(age[tr]) & ~np.isnan(vwm[tr])
    slope, intercept = np.polyfit(age[tr][m], vwm[tr][m], 1)
    pred = slope * age[te] + intercept
    keep = ~np.isnan(age[te]) & ~np.isnan(vwm[te])
    t, p = vwm[te][keep], pred[keep]
    ss_res = float(((t - p) ** 2).sum()); ss_tot = float(((t - t.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return dict(cell="A5", rep=rep, outer=outer, test_r2=r2, best_hps={"slope": float(slope)},
                n_train=int(m.sum()), n_test=int(keep.sum()), kind="trivial")


def run(cell, rep, outer, out_dir, *, source_dir=None, limit=None,
        n_trials=INNER_TRIALS, max_epochs=None):
    spec = CELLS[cell]
    fold = F.outer_split(rep, outer)
    out_dir = Path(out_dir)
    rp = out_dir / "results" / f"{cell}_rep{rep}_outer{outer}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)

    if spec["mode"] == "trivial":
        res = _a5(fold, rep, outer)
        _write_json(rp, res)
        print(f"cell_A5_val_r2: {res['test_r2']:.4f}", flush=True)
        return res

    device = _device()
    g_tr, g_te = D.load_cell_graphs(cell, fold, limit=limit)
    gd = spec["global_dim"]
    is_transfer = spec["mode"] in {"finetune", "frozen"}
    freeze = spec["mode"] == "frozen"
    ck = None
    if is_transfer:
        sdir = Path(source_dir or out_dir)
        ck = str(ckpt_path(sdir, spec["source"], rep, outer))
        assert Path(ck).is_file(), f"missing source ckpt {ck} (run pretrain_source first)"

    itr, iva = F.inner_split(np.arange(len(g_tr)), seed=rep * 100 + outer)
    inner_tr = [g_tr[i] for i in itr]
    inner_va = [g_tr[i] for i in iva]

    def objective(trial):
        hps = suggest_hps(trial, global_dim=gd)
        score, _ = train_eval([g.clone() for g in inner_tr], [g.clone() for g in inner_va],
                              global_dim=gd, hps=hps, device=device, ckpt_in=ck, freeze=freeze,
                              **({} if max_epochs is None else dict(max_epochs=max_epochs)))
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=rep * 100 + outer))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   catch=(RuntimeError,))   # tolerate a transient trial failure on the cluster
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"all {n_trials} Optuna trials failed for cell={cell} rep={rep} outer={outer}")
    best_hps = study.best_params

    # refit best HPs: early-stop on a val carved from fold-train, score fold-test ONCE (no peek)
    ritr, riva = F.inner_split(np.arange(len(g_tr)), seed=rep * 100 + outer + 7)
    refit_tr = [g_tr[i] for i in ritr]
    refit_va = [g_tr[i] for i in riva]
    _, test_r2 = train_eval([g.clone() for g in refit_tr], [g.clone() for g in refit_va],
                            global_dim=gd, hps=best_hps, device=device, ckpt_in=ck, freeze=freeze,
                            test_graphs=[g.clone() for g in g_te],
                            **({} if max_epochs is None else dict(max_epochs=max_epochs)))
    res = dict(cell=cell, rep=rep, outer=outer, test_r2=float(test_r2),
               inner_val_r2=float(study.best_value), best_hps=best_hps,
               n_train=len(g_tr), n_test=len(g_te), mode=spec["mode"], kind="target")
    _write_json(rp, res)
    print(f"cell_{cell}_val_r2: {test_r2:.4f}  (inner_val {study.best_value:.4f})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=list(CELLS))
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--outer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-dir", default=None, help="where source ckpts live (default: --out)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    run(a.cell, a.rep, a.outer, a.out, source_dir=a.source_dir,
        limit=32 if a.smoke else None,
        n_trials=2 if a.smoke else INNER_TRIALS,
        max_epochs=2 if a.smoke else None)


if __name__ == "__main__":
    main()
