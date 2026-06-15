"""Map a flat SLURM array index to a (cell, rep, outer) unit, given a cell kind.

Pure python: imports only cluster._common (no torch/sklearn), so it runs with any
python that can see the repo root — locally under .venv for the wire-check, and in
the container on the cluster. Layout for kind="cell" follows HPO_CELLS order:
idx // (REPS*OUTER) -> cell, then (rep, outer) within that cell's 50-fold block.
"""
import sys

from cluster._common import HPO_CELLS, OUTER, REPS, SOURCE_CELLS


def _cells(kind):
    if kind == "source":
        return SOURCE_CELLS
    if kind == "cell":
        return HPO_CELLS
    raise ValueError(f"unknown kind {kind!r} (expected 'source' or 'cell')")


def unit(kind, idx):
    cells = _cells(kind)
    per = REPS * OUTER
    if not 0 <= idx < len(cells) * per:
        raise IndexError(f"idx {idx} out of range for kind {kind!r} (0..{len(cells) * per - 1})")
    cell = cells[idx // per]
    rem = idx % per
    return cell, rem // OUTER, rem % OUTER  # cell, rep, outer


def count(kind):
    return len(_cells(kind)) * REPS * OUTER


if __name__ == "__main__":
    kind, idx = sys.argv[1], int(sys.argv[2])
    c, r, o = unit(kind, idx)
    print(f"{c} {r} {o}")
