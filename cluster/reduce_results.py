"""Aggregate per-(cell,rep,outer) result JSONs into the per-cell val_r2 table,
side-by-side with the registered predictions + falsifier checks.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from cluster._common import OUTER, REPS, SOURCE_CELLS

# Registered predictions (point/range midpoints) from docs/thesis/pnc-age-vwm-matrix.md
REGISTERED = {
    "A2": "≈ -0.03 (identity floor)", "A3": "≈ 0.18-0.21 (baseline to beat)",
    "A5": "small positive", "B1": "≈ 0-0.05 (<< A3)", "B2": "<= B1",
    "B3": "≈ A3", "B4": "<= B3", "C1": "A3 < C1 <= A3⊕A5", "C2": "≈ A3",
}
CELL_ORDER = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "C1", "C2"]


def aggregate(results_dir):
    by = {}
    for f in Path(results_dir).glob("*.json"):
        r = json.loads(f.read_text())
        by.setdefault(r["cell"], []).append(float(r["test_r2"]))
    table = []
    for cell in CELL_ORDER:
        vals = by.get(cell)
        if not vals:
            continue
        table.append(dict(cell=cell, task=("age" if cell in SOURCE_CELLS else "vwm"),
                          mean_r2=st.fmean(vals),
                          std_r2=(st.pstdev(vals) if len(vals) > 1 else 0.0),
                          n_folds=len(vals), registered=REGISTERED.get(cell, "—")))
    return table


def falsifiers(table):
    m = {r["cell"]: r["mean_r2"] for r in table}
    out = []
    if "A3" in m and "B1" in m and "B2" in m:
        out.append(("signal-location (B1/B2 ≈ A3 falsifies)",
                    f"A3={m['A3']:.3f} B1={m.get('B1'):.3f} B2={m.get('B2'):.3f}"))
    if all(c in m for c in ("A3", "A5", "C1")):
        out.append(("trivial-trend (C1 > A3⊕A5 super-additively falsifies)",
                    f"A3={m['A3']:.3f} A5={m['A5']:.3f} C1={m['C1']:.3f}"))
    if "A3" in m and "B3" in m:
        out.append(("saturation (B3 ≫ A3 falsifies)", f"A3={m['A3']:.3f} B3={m['B3']:.3f}"))
    return out


def write_table(table, path_stem):
    stem = Path(path_stem)
    hdr = "| cell | task | mean R² | std | folds | registered |\n|---|---|---|---|---|---|\n"
    rows = "".join(f"| {r['cell']} | {r['task']} | {r['mean_r2']:.4f} | {r['std_r2']:.4f} | "
                   f"{r['n_folds']} | {r['registered']} |\n" for r in table)
    md = stem.with_suffix(".md")
    md.write_text("# M3 per-cell val_r2\n\n"
                  "> `task` = the cell's prediction target: **age** (A1/A4 source backbones, "
                  "held-out age-R²) vs **vwm** (all baseline/transfer cells, the thesis metric). "
                  "Compare R² only within the same task.\n\n" + hdr + rows)
    csv = stem.with_suffix(".csv")
    csv.write_text("cell,task,mean_r2,std_r2,n_folds\n" +
                   "".join(f"{r['cell']},{r['task']},{r['mean_r2']},{r['std_r2']},{r['n_folds']}\n"
                           for r in table))
    return md, csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="m3_results")
    a = ap.parse_args()
    table = aggregate(a.results)
    md, csv = write_table(table, a.out)
    print(f"wrote {md} and {csv}")
    expected = REPS * OUTER
    incomplete = [r for r in table if r["n_folds"] < expected]
    if incomplete:
        print(f"[WARNING] incomplete cells (expected {expected} folds): "
              + ", ".join(f"{r['cell']}={r['n_folds']}" for r in incomplete))
    for name, line in falsifiers(table):
        print(f"[falsifier] {name}: {line}")


if __name__ == "__main__":
    main()
