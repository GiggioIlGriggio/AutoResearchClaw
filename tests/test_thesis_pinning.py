"""M2: the thesis pinning artifacts (matrix docs + configs) are correct and wired.

Pure-text + config-parse checks; no ARC run. Run under the ARC venv:
    .venv/bin/python -m pytest tests/test_thesis_pinning.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MATRIX = REPO / "docs" / "thesis" / "pnc-age-vwm-matrix.md"
MATRIX_SMOKE = REPO / "docs" / "thesis" / "pnc-age-vwm-matrix-smoke.md"
CFG_FULL = REPO / "config.thesis.yaml"
CFG_SMOKE = REPO / "config.thesis.smoke.yaml"

FULL_CELLS = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "C1", "C2"]
SMOKE_CELLS = ["A1", "A3", "B1", "C1"]
SCAFFOLD_SYMBOLS = [
    "load_fc_graphs", "GraphRegressor", "save_backbone", "load_backbone",
    "freeze", "age_vwm_baseline", "glm_diagonal", "graph_feature_key",
    "pnc_sc400_age_reg", "pnc_sc400_vwm_reg",
]


def test_matrix_doc_lists_all_eleven_cells_and_excludes_d1():
    assert MATRIX.is_file(), f"missing {MATRIX}"
    text = MATRIX.read_text(encoding="utf-8")
    for cell in FULL_CELLS:
        assert cell in text, f"matrix doc missing cell {cell}"
    # D1 is deferred and must NOT appear as a run cell. The doc may mention
    # it only in a clearly-marked 'deferred / out of scope' line.
    for line in text.splitlines():
        if "D1" in line:
            assert any(w in line.lower() for w in ("defer", "out of scope", "not implemented")), \
                f"D1 referenced outside a deferral note: {line!r}"


def test_matrix_doc_names_the_scaffold_primitives_and_bundles():
    text = MATRIX.read_text(encoding="utf-8")
    for sym in SCAFFOLD_SYMBOLS:
        assert sym in text, f"matrix doc missing scaffold symbol/bundle {sym}"


def test_matrix_doc_has_per_stage_instruction_sections():
    text = MATRIX.read_text(encoding="utf-8").lower()
    assert "experiment-design stage" in text
    assert "code-generation stage" in text
    # The canonical metric the thesis sweeps on.
    assert "val_r2" in text
