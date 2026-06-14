"""Materialize experiment_scaffold/*.py into an importable ``scaffold`` package.

Mirrors scripts/smoke_thesis_cells.py so the cluster harness imports the SAME
trusted M1 primitives without duplicating or drifting from them. Ships fine to the
cluster: experiment_scaffold/ travels via git; this copies it into a temp pkg at
import time. Call ``import_scaffold()`` once near program start.
"""
from __future__ import annotations

import importlib
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_SCAFFOLD = None


def import_scaffold():
    """Materialize + import the scaffold once per process; cached on repeat calls."""
    global _SCAFFOLD
    if _SCAFFOLD is not None:
        return _SCAFFOLD
    root = Path(tempfile.mkdtemp(prefix="cluster_scaffold_"))
    pkg = root / "scaffold"
    pkg.mkdir()
    for py in (REPO / "experiment_scaffold").glob("*.py"):
        shutil.copy(py, pkg / py.name)
    (pkg / "__init__.py").write_text("")
    sys.path.insert(0, str(root))
    _SCAFFOLD = (
        importlib.import_module("scaffold.data_loader"),
        importlib.import_module("scaffold.models"),
        importlib.import_module("scaffold.heads"),
        importlib.import_module("scaffold.transfer"),
        importlib.import_module("scaffold.baselines"),
    )
    return _SCAFFOLD
