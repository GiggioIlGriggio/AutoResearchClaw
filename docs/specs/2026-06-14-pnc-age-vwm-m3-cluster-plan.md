# PNC Age→VWM Milestone 3 (Cluster Execution) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Operational (non-TDD) tasks are marked **[OPS]**; for those, the "test" is the stated acceptance check, not pytest.

**Goal:** Run the pre-registered 11-cell PNC age→VWM matrix end-to-end on the Slurm cluster under the full nested CV (10 reps × 5 outer × 20 inner Optuna), producing a per-cell `val_r2` table to compare against the registered predictions — by (1) generating + inspecting the 11-cell ARC code as a wiring cross-check, (2) authoring a hand-written nested-CV harness that scales the validated `scripts/smoke_thesis_cells.py` recipe, and (3) driving `cluster-helper` to bootstrap the project, upload the bundles, build the container, and run an `A1/A4 → B/C` Slurm matrix DAG.

**Architecture:** ARC's pinned Stage-10 codegen deliberately emits *reduced-protocol* code ("the full nested CV is the cluster's job, not this code's" — `docs/thesis/pnc-age-vwm-matrix.md`), so the heavy protocol is **hand-authored** in a new `cluster/` package that imports the trusted M1 `experiment_scaffold/` primitives (never re-implements them) exactly as `scripts/smoke_thesis_cells.py` does, scaled to nested CV + Optuna. The transfer is **within-subject by design** (locked 2026-06-14): per `(rep, outer-fold)` the source cell (A1/A4) is pretrained on that fold's *train* subjects' age, then the dependent B/C cell fine-tunes on the *same* train subjects' VWM and is evaluated on the held-out test subjects — source and target share one fold partition, the test subjects excluded from both stages. **Option A (locked):** the GCN architecture is **fixed** (`hidden_channels/out_channels/num_layers`) so one per-fold source checkpoint is shared by all its dependents; the inner Optuna sweeps only optimization/regularization HPs. This yields the clean `A1/A4`-source-array → `B/C`-dependent-array `afterok` DAG. The ARC-generated `experiment/` is a wiring cross-check + the artifact M4's analysis stage attaches to — it is **not** the cluster payload.

**Tech Stack:** the M1 `experiment_scaffold/` primitives (GCN factory, `GraphRegressor`, `transfer`, `glm_diagonal`/`graph_feature_key` loaders, `age_vwm_baseline`), the baked `~/rc_brain_data` SC-400 bundles, **Optuna** (new dep) for inner HPO, `scikit-learn` `KFold` for folds, the ACP/Claude Code backend for the ARC codegen run (`researchclaw-claude-code-backend` memory), and the **`cluster-helper`** CLI (`cluster-init`/`-push-container`/`-upload-dataset`/`-gpus`/`-submit`/`-tail`/`-status`/`-fetch`) targeting `gpunode02` (`rad2`/`16cpu`). pytest for the harness unit tests under the ARC `.venv` (torch+pyg).

**Spec / sources (read; do not duplicate):**
- `docs/thesis/pnc-age-vwm-matrix.md` — the 11 cells, exact scaffold API, per-cell recipe, registered predictions/falsifiers, metric (`val_r2`, maximize).
- `docs/specs/2026-06-13-pnc-age-vwm-substrate-design.md` §0 (locked decisions), §6 (M3 scope: project + `.sif` + SLURM DAG + full nested CV).
- `docs/thesis/SMOKE_RESULTS.md` + `scripts/smoke_thesis_cells.py` — the deterministic ground-truth recipe the harness must match (real loader kwargs; **no invented `carrier=`**).
- `experiment_scaffold/SCAFFOLD.md` — the primitive signatures + the two `pnc_sc400_*` bundles.
- Memories: `pnc-age-vwm-thesis-effort` (now carries the M3 transfer×CV decision), `gnn-scaffold-dataset`, `researchclaw-claude-code-backend`, `feedback-one-question-at-a-time`.

---

## Verified facts (investigated 2026-06-14; do not re-derive)

1. **Cluster is greenfield.** No `.cluster-helper.yaml`, `slurm/`, `Dockerfile`, or `requirements.txt` exist in the repo → M3 includes the `cluster-init` bootstrap. The `cluster-*` CLI is on PATH (`/home/compa/.claude/skills/cluster-helper/bin/`).
2. **Repo is already on GitHub:** `origin → https://github.com/aiming-lab/AutoResearchClaw.git`. `cluster-submit` clones the deployed SHA and auto-rewrites HTTPS→SSH for the cluster clone — **no `gh repo create`**. The `thesis/pnc-age-vwm` branch must be **pushed** so its SHA is reachable on origin (Task 8 Step 5).
3. **Local stack (mirror in the container):** python 3.12.3, `torch==2.11.0+cu128` (CUDA 12.8), `torch_geometric==2.8.0`, `numpy==2.4.6`, `scipy==1.17.1`, `pandas==3.0.3`, `scikit-learn==1.9.0`. **`optuna` is NOT installed** — install into `.venv` (Task 0) and pin it in the container (Task 8). Local smoke GPU = RTX A4000 16 GB.
4. **Subject alignment:** `subject_id` is `<U14` (string) in both bundles. `pnc_sc400_age_reg` N=747, `pnc_sc400_vwm_reg` N=744, **743 subjects common**. Folds are defined on the 743 common subjects so every transfer is within-subject (no subject in a transfer chain is missing its age or VWM). The 1 age-NaN subject (M1 note) and the 1 vwm-only subject fall outside the intersection and are dropped — record the realized N=743 in the run notes.
5. **Loader subsetting is positional.** `load_fc_graphs(..., indices=[...])` subsets a bundle by **position**, not `subject_id`, and returns graphs in that order with the bundle's baked `y`. So the fold builder maps fold subject-ids → positions **per bundle** (age vs vwm orderings differ). Graphs are built eagerly → load each fold's `train∪test` once per array task and reuse across all 20 Optuna trials.
6. **Exact scaffold API (call verbatim — `scripts/smoke_thesis_cells.py` is the reference):**
   - identity carrier: `load_fc_graphs(name, matrix_key="sc", edge_weight_norm="abs_max", node_features="identity", indices=...)`.
   - glm_diagonal carrier: `load_fc_graphs(name, matrix_key="sc", edge_weight_norm="abs_max", node_feature_key="glm_2back_vs_0back", glm_diagonal=True, glm_normalize=True, indices=...)`.
   - age @head (C1): add `graph_feature_key="age"` (→ `data.u`, U=1), `GraphRegressor(..., global_dim=1)`, `forward(..., u=batch.u)`.
   - GLM @head (C2): add `graph_feature_key="glm_2back_vs_0back"` (→ `data.u`, U=400), `global_dim=400`.
   - `build_gcn(in_channels=400, hidden_channels=H, out_channels=P, num_layers=L, norm="batch_norm", dropout=...)`; `GraphRegressor(backbone, pooled_dim=P, global_dim=0|1|400, global_hidden=.., head_hidden=..)`; `transfer.save_backbone/load_backbone(strict=True)/freeze`; `load_fc_bundle(name, matrix_key="sc")` → `(matrix, y, meta)` with `meta["age"]`. **No `carrier=`/`mode=` kwargs exist.**
7. **Locked protocol parameters (Option A; confirm only if compute forces a change):**
   - `REPS=10`, `OUTER=5`, `INNER_TRIALS=20`; primary metric = outer-test R² (reported as `cell_<X>_val_r2`), maximize.
   - **Fixed architecture:** `hidden_channels=64, out_channels=32, num_layers=3, norm="batch_norm"`. (These can be re-pinned from a quick pre-sweep before the official run; keep them in one constant block — `ARCH` in `cluster/_common.py`.)
   - **Inner Optuna search space (optimization/reg only):** `lr ∈ loguniform[1e-4, 5e-3]`, `weight_decay ∈ loguniform[1e-6, 1e-3]`, `dropout ∈ uniform[0.0, 0.5]`, `head_hidden ∈ {32,64,128}`, `global_hidden ∈ {16,32,64}` (C1/C2 only). Training: `max_epochs=200`, early-stop `patience=20` on inner-val, `batch_size=32`, Adam, MSE on z-scored targets.
   - **Source (A1/A4) training:** fixed HPs `lr=1e-3, weight_decay=1e-5, max_epochs=200, patience=20`, the same `ARCH` (so `strict=True` load always fits).
8. **Cell → (carrier, bundle, mode, source, global_dim, graph_feature_key) map** (drives both source + cell scripts) — see `cluster/_common.py` (Task 3). Source cells A1 (identity→age), A4 (glm_diagonal→age). Dependents: A1→{B1 ft, B2 frozen, C2 ft+GLM@head}; A4→{B3 ft, B4 frozen}. From-scratch on VWM: A2 (identity), A3 (glm_diagonal — the baseline to beat), C1 (glm_diagonal + age@head). A5 = trivial age→VWM OLS (no graph).

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `cluster/_scaffold.py` | Materialize `experiment_scaffold/*.py` into an importable `scaffold` package + import the modules (the proven `scripts/smoke_thesis_cells.py` pattern — no scaffold duplication, no drift). | Create |
| `cluster/_common.py` | Single source of truth for `REPS/OUTER/INNER_TRIALS`, `ARCH`, the Optuna search space, the `CELLS` config map, and `carrier_kwargs()`. Imported by every other module + the tests. | Create |
| `cluster/folds.py` | Subject-aligned nested folds over the 743 common subjects: `outer_split(rep, k) -> FoldIdx` returning per-bundle positional index arrays for `train`/`test`, identical subjects across bundles; `inner_split(train_idx, seed)`. | Create |
| `cluster/pretrain_source.py` | CLI: pretrain one source cell (A1/A4) for one `(rep, outer)` on its fold-train subjects → save backbone ckpt keyed `(cell, rep, outer)` + emit the source's held-out age-R² JSON. | Create |
| `cluster/train_cell.py` | CLI: run one `(cell, rep, outer)` target unit — inner Optuna HPO over the search space, refit best HPs on full fold-train (loading the source ckpt for transfer cells / freezing for B2,B4 / from-scratch for A2,A3,C1 / trivial OLS for A5), eval held-out fold-test → write result JSON. `--smoke` shrinks everything for the local gate. | Create |
| `cluster/reduce_results.py` | Aggregate all per-fold result JSONs → per-cell mean±std outer-test R² table (md+csv) + a side-by-side vs the registered predictions + the falsifier checks. | Create |
| `tests/test_cluster_harness.py` | Unit tests: fold subject-alignment + no train/test subject overlap across stages; `carrier_kwargs` correctness; a tiny end-to-end `train_cell --smoke` per cell-mode; the reducer aggregation. Materialize-`scaffold/` pattern; `importorskip("torch")`. | Create |
| `slurm/source.sh` | Slurm array (A1,A4 × 10 reps × 5 outer = 100 tasks) → `cluster/pretrain_source.py`. | Create |
| `slurm/cell.sh` | Slurm array (8 HPO cells × 10 × 5 = 400 tasks) → `cluster/train_cell.py`; submitted `--dependency=afterok:<source_job>`. (A5 folded into a tiny tail task or the reducer.) | Create |
| `Dockerfile` + `requirements.txt` | Container image pinning the §-3 stack + optuna for `cluster-push-container`. | Create (edit cluster-init template) |
| `.cluster-helper.yaml` | Project manifest (name, container, dataset remote path). | Create (cluster-init) |
| `docs/thesis/M3_CLUSTER_RESULTS.md` | Evidence artifact: local-smoke proof, the launched job ids, the fetched per-cell table vs predictions. | Create (Tasks 6 + 12) |

> **Why a hand-authored harness, not the ARC code.** ARC's Stage-10 output is single-split/few-epoch by pin design and ARC would overwrite it on any re-run (it's a deploy-from-git world on the cluster). The nested-CV + Optuna + checkpoint-DAG logic must be deterministic, locally TDD-able, and version-controlled — so it lives in `cluster/` and imports the same trusted `scaffold/` primitives the ARC code does. The ARC `experiment/` (Task 1) is kept as a wiring cross-check and the M4 analysis anchor.

---

## Task 0: Preflight — env, optuna, bundles, folds sanity **[OPS]**

**Files:** none (verification only).

- [ ] **Step 1: Install optuna into the ARC venv** (needed for the harness + its tests)

```bash
cd /home/compa/Documents/working_dir/ResearchClaude/AutoResearchClaw
.venv/bin/python -m pip install 'optuna>=3.6,<5'
.venv/bin/python -c "import optuna; print('optuna', optuna.__version__)"
```
Expected: a version prints (record it for `requirements.txt` in Task 8).

- [ ] **Step 2: Confirm substrate + bundles still resolve**

```bash
ls ~/rc_brain_data/pnc_sc400_age_reg.npz ~/rc_brain_data/pnc_sc400_vwm_reg.npz
RC_DATASET_DIR=~/rc_brain_data .venv/bin/python scripts/smoke_thesis_cells.py 2>&1 | tail -3
```
Expected: both `.npz` present; the reference smoke prints `SMOKE_OK: 4/4 cells ...` (the recipe the harness mirrors still trains forward on this box).

- [ ] **Step 3: Note the realized common-subject N** (drives fold sizes; already verified = **743**). No action — just confirm in the run notes that folds will run on 743 subjects.

---

## Task 1: Generate + inspect the 11-cell ARC code (wiring cross-check) **[OPS]**

**Files:** ARC run dir under `artifacts/rc-*` (not committed); note its path.

> Runs ARC via the ACP/Claude Code backend (`researchclaw-claude-code-backend` memory: acpx 0.10.0, isolated `CLAUDE_CONFIG_DIR=~/.researchclaw-acp-home`, model `claude-opus-4-8`, **run from repo root** so `extra_prompts` relative paths resolve). `config.thesis.yaml` HITL-gates execution (stage 12), so ARC pauses after codegen. Suggest the user launch the interactive command with `! <command>` so its output lands in the session.

- [ ] **Step 1: Run ARC to the execution gate with the full config**

```bash
cd /home/compa/Documents/working_dir/ResearchClaude/AutoResearchClaw
CLAUDE_CONFIG_DIR=~/.researchclaw-acp-home .venv/bin/researchclaw run \
  --config config.thesis.yaml --mode full-auto --to-stage CODE_GENERATION
```
Expected: Stage 9 emits exactly the 11 conditions (A1–A5, B1–B4, C1, C2; datasets = the two `pnc_sc400_*` bundles; `val_r2`/maximize); Stage 10 generates `experiment/`. **If acpx flakes (exit 1 mid-codegen — the known SMOKE_RESULTS.md flake), just re-run Step 1.** If it flakes repeatedly, set `experiment.code_agent.enabled: false` in a scratch copy of the config to force the single-shot legacy codegen path, and re-run.

- [ ] **Step 2: Inspect the generated code for the loader-API + wiring contract**

```bash
RUN=$(ls -dt artifacts/rc-* | head -1)        # newest run dir
EXP=$(ls -d "$RUN"/stage-10*/experiment 2>/dev/null | head -1)
echo "experiment dir: $EXP"
# (a) imports scaffold, does NOT re-implement, does NOT use the invented carrier= kwarg
grep -RnE "from scaffold|import scaffold" "$EXP"/*.py
! grep -RnE "carrier\s*=" "$EXP"/*.py && echo "OK: no carrier= kwarg"
grep -RnE "load_fc_graphs|GraphRegressor|save_backbone|load_backbone|age_vwm_baseline|RC_DATASET_DIR" "$EXP"/*.py
# (b) all 11 cells emit a metric line; transfer wiring present
grep -oE "cell_(A[1-5]|B[1-4]|C[12])_val_r2" "$EXP"/*.py | sort -u   # expect 11 distinct
grep -RnE "load_backbone\(.*strict=True|transfer\.freeze|global_dim\s*=\s*(1|400)" "$EXP"/*.py
ls "$EXP"/scaffold/   # materialized scaffold package present
```
Expected: scaffold imports only; **no `carrier=`**; 11 distinct `cell_*_val_r2`; A1/A4 `save_backbone`, B*/C2 `load_backbone(strict=True)`, B2/B4 `freeze`, C1 `global_dim=1`, C2 `global_dim=400`, A5 `age_vwm_baseline`. **If the `carrier=` bug reappears**, that confirms the hardened pin still leaks — note it; the cluster harness does NOT depend on the ARC code, so this does not block M3, but flag it for an M2 pin follow-up.

- [ ] **Step 3 (no commit — artifacts not committed):** Record the `experiment/` path; it is the M4 analysis anchor + a wiring cross-check for the harness in Tasks 3–5.

---

## Task 2: Subject-aligned nested folds (`cluster/folds.py`)

**Files:**
- Create: `cluster/__init__.py` (empty), `cluster/_scaffold.py`, `cluster/_common.py` (skeleton; fleshed in Task 3), `cluster/folds.py`
- Test: `tests/test_cluster_harness.py`

- [ ] **Step 1: Write `cluster/_scaffold.py`** (materialize + import the scaffold — the proven pattern)

```python
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


def import_scaffold():
    root = Path(tempfile.mkdtemp(prefix="cluster_scaffold_"))
    pkg = root / "scaffold"
    pkg.mkdir()
    for py in (REPO / "experiment_scaffold").glob("*.py"):
        shutil.copy(py, pkg / py.name)
    (pkg / "__init__.py").write_text("")
    sys.path.insert(0, str(root))
    return (
        importlib.import_module("scaffold.data_loader"),
        importlib.import_module("scaffold.models"),
        importlib.import_module("scaffold.heads"),
        importlib.import_module("scaffold.transfer"),
        importlib.import_module("scaffold.baselines"),
    )
```

- [ ] **Step 2: Write a minimal `cluster/_common.py`** (constants the fold test imports; the full `CELLS` map lands in Task 3)

```python
"""Single source of truth for the M3 protocol constants + cell map."""
from __future__ import annotations

REPS = 10
OUTER = 5
INNER_TRIALS = 20

AGE_BUNDLE = "pnc_sc400_age_reg"
VWM_BUNDLE = "pnc_sc400_vwm_reg"

# Fixed GCN architecture (Option A). Re-pin from a pre-sweep before the official run.
ARCH = dict(hidden_channels=64, out_channels=32, num_layers=3, norm="batch_norm")
```

- [ ] **Step 3: Write the failing test** (`tests/test_cluster_harness.py`)

```python
"""M3 cluster-harness unit tests. Run under the ARC venv (torch+pyg+optuna):
    RC_DATASET_DIR=~/rc_brain_data .venv/bin/python -m pytest tests/test_cluster_harness.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
HAS_DATA = bool(os.environ.get("RC_DATASET_DIR")) and (
    Path(os.path.expanduser(os.environ.get("RC_DATASET_DIR", "/nonexistent")))
    / "pnc_sc400_vwm_reg.npz"
).is_file()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="RC_DATASET_DIR bundles absent")


@needs_data
def test_folds_are_subject_aligned_and_disjoint():
    from cluster import folds as F
    from cluster._common import OUTER

    # Every outer fold: train/test disjoint, and the SAME subjects map across bundles.
    for rep in (0, 3):
        seen_test = []
        for k in range(OUTER):
            fold = F.outer_split(rep, k)
            # disjoint within a bundle
            assert set(fold.train_vwm).isdisjoint(fold.test_vwm)
            assert set(fold.train_age).isdisjoint(fold.test_age)
            # subject sets identical across bundles (within-subject transfer)
            assert fold.train_subjects == set(F.subjects_at(F.VWM, fold.train_vwm))
            assert fold.train_subjects == set(F.subjects_at(F.AGE, fold.train_age))
            assert fold.test_subjects == set(F.subjects_at(F.AGE, fold.test_age))
            seen_test.append(fold.test_subjects)
        # the 5 outer test sets partition the common subjects
        union = set().union(*seen_test)
        assert len(union) == sum(len(t) for t in seen_test)  # no overlap across folds
        assert len(union) == F.n_common()


@needs_data
def test_inner_split_holds_out_within_train():
    from cluster import folds as F

    fold = F.outer_split(1, 2)
    itr, iva = F.inner_split(fold.train_vwm, seed=1)
    assert set(itr).isdisjoint(iva)
    assert set(itr) | set(iva) == set(fold.train_vwm)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `RC_DATASET_DIR=~/rc_brain_data .venv/bin/python -m pytest tests/test_cluster_harness.py -k folds -v`
Expected: FAIL — `ModuleNotFoundError: cluster.folds` / missing attrs.

- [ ] **Step 5: Write `cluster/folds.py`**

```python
"""Subject-aligned nested folds for the within-subject age→VWM transfer.

Folds are defined ONCE over the subjects common to both bundles (743). For each
(rep, outer-fold) the same subject partition is mapped to per-bundle POSITIONS
(orderings differ), so the source pretrains on a fold's train subjects' age and the
target fine-tunes on the SAME subjects' VWM, with the held-out test subjects excluded
from both. Reps reshuffle the 5-fold split (random_state=rep).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold

from cluster._common import AGE_BUNDLE, OUTER, VWM_BUNDLE
from cluster._scaffold import import_scaffold

AGE, VWM = "age", "vwm"
_dl = None
_sid = {}          # bundle -> np.ndarray[str] of subject_ids (bundle order)
_pos = {}          # bundle -> {subject_id: position}
_common = None     # sorted list[str] of common subject_ids


def _ensure_loaded():
    global _dl, _common
    if _dl is not None:
        return
    dl, *_ = import_scaffold()
    _dl = dl
    for tag, name in ((AGE, AGE_BUNDLE), (VWM, VWM_BUNDLE)):
        _, _, meta = dl.load_fc_bundle(name, matrix_key="sc")
        sids = np.asarray(meta["subject_id"]).astype(str)
        _sid[tag] = sids
        _pos[tag] = {s: i for i, s in enumerate(sids)}
    _common = sorted(set(_sid[AGE]) & set(_sid[VWM]))


def n_common() -> int:
    _ensure_loaded()
    return len(_common)


def subjects_at(bundle, idx):
    _ensure_loaded()
    return [_sid[bundle][i] for i in idx]


def _positions(bundle, subjects):
    return np.array([_pos[bundle][s] for s in subjects], dtype=int)


@dataclass
class FoldIdx:
    rep: int
    outer: int
    train_subjects: set
    test_subjects: set
    train_age: np.ndarray
    test_age: np.ndarray
    train_vwm: np.ndarray
    test_vwm: np.ndarray


def outer_split(rep: int, k: int) -> FoldIdx:
    _ensure_loaded()
    kf = KFold(n_splits=OUTER, shuffle=True, random_state=rep)
    common = np.asarray(_common)
    for ki, (tr, te) in enumerate(kf.split(common)):
        if ki != k:
            continue
        train_s, test_s = sorted(common[tr]), sorted(common[te])
        return FoldIdx(
            rep=rep, outer=k,
            train_subjects=set(train_s), test_subjects=set(test_s),
            train_age=_positions(AGE, train_s), test_age=_positions(AGE, test_s),
            train_vwm=_positions(VWM, train_s), test_vwm=_positions(VWM, test_s),
        )
    raise ValueError(f"outer fold {k} out of range (OUTER={OUTER})")


def inner_split(train_idx, seed: int, frac: float = 0.8):
    """Single inner train/val holdout WITHIN a fold's train positions (for Optuna)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_idx))
    cut = int(round(len(train_idx) * frac))
    arr = np.asarray(train_idx)
    return arr[perm[:cut]], arr[perm[cut:]]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `RC_DATASET_DIR=~/rc_brain_data .venv/bin/python -m pytest tests/test_cluster_harness.py -k "folds or inner_split" -v`
Expected: PASS (2 tests). (If `meta["subject_id"]` key differs, fix `_ensure_loaded` to the real key — verified present as `subject_id`.)

- [ ] **Step 7: Commit**

```bash
git add cluster/__init__.py cluster/_scaffold.py cluster/_common.py cluster/folds.py tests/test_cluster_harness.py
git commit -m "feat(thesis): M3 subject-aligned nested folds + scaffold import shim"
```

---

## Task 3: Cell config + carrier/graph helpers (`cluster/_common.py` complete)

**Files:**
- Modify: `cluster/_common.py`
- Create: `cluster/data.py` (graph-loading helper that builds a fold's train/test graphs once)
- Test: `tests/test_cluster_harness.py`

- [ ] **Step 1: Flesh out `cluster/_common.py`** — append the search space + `CELLS` map + `carrier_kwargs`

```python
# --- inner Optuna search space (Option A: optimization/reg only; ARCH is fixed) ---
def suggest_hps(trial, *, global_dim: int):
    hps = dict(
        lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        head_hidden=trial.suggest_categorical("head_hidden", [32, 64, 128]),
    )
    if global_dim > 0:
        hps["global_hidden"] = trial.suggest_categorical("global_hidden", [16, 32, 64])
    return hps


DEFAULT_HPS = dict(lr=1e-3, weight_decay=1e-5, dropout=0.0, head_hidden=64, global_hidden=32)

MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 32

# carrier -> load_fc_graphs kwargs (NEVER a `carrier=` kwarg — see SMOKE_RESULTS.md)
def carrier_kwargs(carrier: str) -> dict:
    if carrier == "identity":
        return dict(node_features="identity")
    if carrier == "glm_diagonal":
        return dict(node_feature_key="glm_2back_vs_0back", glm_diagonal=True, glm_normalize=True)
    raise ValueError(f"unknown carrier {carrier!r}")


# cell -> spec. mode: source|scratch|finetune|frozen|trivial. bundle: age|vwm.
CELLS = {
    "A1": dict(carrier="identity",     bundle="age", mode="source",   global_dim=0),
    "A2": dict(carrier="identity",     bundle="vwm", mode="scratch",  global_dim=0),
    "A3": dict(carrier="glm_diagonal", bundle="vwm", mode="scratch",  global_dim=0),
    "A4": dict(carrier="glm_diagonal", bundle="age", mode="source",   global_dim=0),
    "A5": dict(mode="trivial"),
    "B1": dict(carrier="identity",     bundle="vwm", mode="finetune", source="A1", global_dim=0),
    "B2": dict(carrier="identity",     bundle="vwm", mode="frozen",   source="A1", global_dim=0),
    "B3": dict(carrier="glm_diagonal", bundle="vwm", mode="finetune", source="A4", global_dim=0),
    "B4": dict(carrier="glm_diagonal", bundle="vwm", mode="frozen",   source="A4", global_dim=0),
    "C1": dict(carrier="glm_diagonal", bundle="vwm", mode="scratch",  global_dim=1,
               graph_feature_key="age"),
    "C2": dict(carrier="identity",     bundle="vwm", mode="finetune", source="A1", global_dim=400,
               graph_feature_key="glm_2back_vs_0back"),
}
SOURCE_CELLS = [c for c, s in CELLS.items() if s.get("mode") == "source"]          # A1, A4
HPO_CELLS = [c for c, s in CELLS.items() if s.get("mode") in {"scratch", "finetune", "frozen"}]
```

- [ ] **Step 2: Write the failing test** (append)

```python
def test_cells_map_is_consistent():
    from cluster._common import CELLS, SOURCE_CELLS, HPO_CELLS, carrier_kwargs

    assert set(CELLS) == {"A1","A2","A3","A4","A5","B1","B2","B3","B4","C1","C2"}
    assert set(SOURCE_CELLS) == {"A1", "A4"}
    assert set(HPO_CELLS) == {"A2","A3","B1","B2","B3","B4","C1","C2"}
    # every transfer cell points at a real source whose carrier matches (strict=True load)
    for c, spec in CELLS.items():
        if spec.get("mode") in {"finetune", "frozen"}:
            assert CELLS[spec["source"]]["carrier"] == spec["carrier"], f"{c} carrier mismatch"
    # carrier_kwargs never leaks a 'carrier' kwarg into the loader
    for carrier in ("identity", "glm_diagonal"):
        assert "carrier" not in carrier_kwargs(carrier)
    assert CELLS["C1"]["global_dim"] == 1 and CELLS["C2"]["global_dim"] == 400


@needs_data
def test_load_fold_graphs_shapes():
    from cluster import data as D
    from cluster import folds as F

    fold = F.outer_split(0, 0)
    g_tr, g_te = D.load_cell_graphs("A3", fold, limit=24)  # glm_diagonal carrier
    assert g_tr[0].x.shape[1] == 400 and len(g_te) > 0
    g_tr, g_te = D.load_cell_graphs("C1", fold, limit=24)  # + age @head -> data.u U=1
    assert g_tr[0].u.shape[-1] == 1
    g_tr, g_te = D.load_cell_graphs("C2", fold, limit=24)  # GLM @head -> data.u U=400
    assert g_tr[0].u.shape[-1] == 400
```

- [ ] **Step 3: Run test to verify it fails**

Run: `RC_DATASET_DIR=~/rc_brain_data .venv/bin/python -m pytest tests/test_cluster_harness.py -k "cells_map or load_fold" -v`
Expected: FAIL — `ModuleNotFoundError: cluster.data` / missing helpers.

- [ ] **Step 4: Write `cluster/data.py`**

```python
"""Load a fold's train/test graphs for a cell, using the EXACT scaffold loader API.

Graphs are built eagerly, so a (cell, rep, outer) task loads its train+test once and
reuses them across all Optuna trials.
"""
from __future__ import annotations

from cluster._common import AGE_BUNDLE, CELLS, VWM_BUNDLE, carrier_kwargs
from cluster._scaffold import import_scaffold

_dl = None


def _loader():
    global _dl
    if _dl is None:
        _dl, *_ = import_scaffold()
    return _dl


def _bundle_name(tag: str) -> str:
    return AGE_BUNDLE if tag == "age" else VWM_BUNDLE


def load_cell_graphs(cell: str, fold, *, limit=None):
    """Return (train_graphs, test_graphs) for ``cell`` on its fold partition."""
    spec = CELLS[cell]
    dl = _loader()
    name = _bundle_name(spec["bundle"])
    tr_idx = fold.train_age if spec["bundle"] == "age" else fold.train_vwm
    te_idx = fold.test_age if spec["bundle"] == "age" else fold.test_vwm
    if limit is not None:
        tr_idx, te_idx = tr_idx[:limit], te_idx[:limit]
    kw = dict(matrix_key="sc", edge_weight_norm="abs_max", **carrier_kwargs(spec["carrier"]))
    if spec.get("graph_feature_key"):
        kw["graph_feature_key"] = spec["graph_feature_key"]
    g_tr = dl.load_fc_graphs(name, indices=list(tr_idx), **kw)
    g_te = dl.load_fc_graphs(name, indices=list(te_idx), **kw)
    return g_tr, g_te
```

- [ ] **Step 5: Run test to verify it passes**

Run: `RC_DATASET_DIR=~/rc_brain_data .venv/bin/python -m pytest tests/test_cluster_harness.py -k "cells_map or load_fold" -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add cluster/_common.py cluster/data.py tests/test_cluster_harness.py
git commit -m "feat(thesis): M3 cell config map + fold graph loader (exact scaffold API)"
```

---

## Task 4: Source pretraining (`cluster/pretrain_source.py`)

**Files:**
- Create: `cluster/trainlib.py` (shared train/eval loop — the single forward/eval used by source + target so they cannot drift)
- Create: `cluster/pretrain_source.py`
- Test: `tests/test_cluster_harness.py`

- [ ] **Step 1: Write `cluster/trainlib.py`** (one train/eval loop, mirrors `smoke_thesis_cells.py`)

```python
"""Shared train/eval for source + target cells (z-scored targets, early stopping, R²)."""
from __future__ import annotations

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
    """
    from torch_geometric.loader import DataLoader

    torch.manual_seed(0)
    train_graphs, val_graphs = list(train_graphs), list(val_graphs)
    test_graphs = list(test_graphs) if test_graphs is not None else []
    n_tr, n_va = len(train_graphs), len(val_graphs)
    all_g = train_graphs + val_graphs + test_graphs
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
```

- [ ] **Step 2: Write `cluster/pretrain_source.py`**

```python
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
```

- [ ] **Step 3: Write the failing test** (append) — a tiny smoke that the source trains + saves a loadable backbone

```python
@needs_data
def test_pretrain_source_smoke(tmp_path):
    from cluster import pretrain_source as P
    res = P.run("A1", rep=0, outer=0, out_dir=tmp_path, limit=24, max_epochs=2)
    assert Path(res["ckpt"]).is_file()
    assert isinstance(res["test_r2"], float)
    # the saved backbone reloads strict=True into a fresh identity model (transfer contract)
    import torch
    from cluster.trainlib import build_model
    from cluster._common import DEFAULT_HPS
    m = build_model(400, 0, DEFAULT_HPS, torch.device("cpu"))
    from cluster._scaffold import import_scaffold
    _, _, _, transfer, _ = import_scaffold()
    transfer.load_backbone(m.backbone, res["ckpt"], strict=True)  # must not raise
```

- [ ] **Step 4: Run test to verify it fails, then passes**

Run: `RC_DATASET_DIR=~/rc_brain_data .venv/bin/python -m pytest tests/test_cluster_harness.py -k pretrain_source -v`
Expected: PASS once `trainlib.py` + `pretrain_source.py` exist (it exercises the real GPU/CPU train + the `save_backbone`→`load_backbone(strict=True)` round trip).

- [ ] **Step 5: Commit**

```bash
git add cluster/trainlib.py cluster/pretrain_source.py tests/test_cluster_harness.py
git commit -m "feat(thesis): M3 source pretraining (A1/A4 per-fold backbone checkpoints)"
```

---

## Task 5: The nested-CV target harness (`cluster/train_cell.py`)

**Files:**
- Create: `cluster/train_cell.py`
- Test: `tests/test_cluster_harness.py`

> One `(cell, rep, outer)` unit: build fold graphs once → inner Optuna (INNER_TRIALS) over the search space on an inner holdout (for transfer cells, each trial loads the source ckpt; for B2/B4 it freezes) → refit best HPs on full fold-train → eval held-out fold-test → write result JSON. A5 is the trivial age→VWM OLS per fold.

- [ ] **Step 1: Write `cluster/train_cell.py`**

```python
"""Run ONE (cell, rep, outer) target unit of the nested CV → result JSON.

Inner Optuna HPO on an inner holdout, refit on full fold-train, eval held-out fold-test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def _a5(fold, out_dir, rep, outer):
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
        res = _a5(fold, out_dir, rep, outer)
        rp.write_text(json.dumps(res, indent=2))
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
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
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
    rp.write_text(json.dumps(res, indent=2))
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
```

- [ ] **Step 2: Write the failing test** (append) — one smoke per cell-mode, end-to-end

```python
@needs_data
def test_train_cell_smoke_all_modes(tmp_path):
    import torch
    from cluster import pretrain_source as P
    from cluster import train_cell as T

    # source ckpts A1/A4 for (rep0,outer0) so the transfer cells can load them
    for s in ("A1", "A4"):
        P.run(s, 0, 0, tmp_path, limit=24, max_epochs=2)

    smoke = dict(limit=24, n_trials=2, max_epochs=2)
    for cell in ("A2", "A3", "B1", "B2", "B3", "B4", "C1", "C2", "A5"):
        res = T.run(cell, 0, 0, tmp_path, source_dir=tmp_path, **(smoke if cell != "A5" else {}))
        assert np.isfinite(res["test_r2"]), f"{cell} produced non-finite R²"
        assert (tmp_path / "results" / f"{cell}_rep0_outer0.json").is_file()
```

- [ ] **Step 3: Run test to verify it passes**

Run: `RC_DATASET_DIR=~/rc_brain_data .venv/bin/python -m pytest tests/test_cluster_harness.py -k train_cell_smoke -v`
Expected: PASS — every cell-mode (scratch, finetune, frozen, age@head, GLM@head, trivial) runs the inner Optuna + refit + eval and writes a finite-R² JSON. (This is the single most important test: it proves the whole per-unit harness before any cluster cost.)

- [ ] **Step 4: Commit**

```bash
git add cluster/train_cell.py tests/test_cluster_harness.py
git commit -m "feat(thesis): M3 nested-CV target harness (inner Optuna + refit + transfer)"
```

---

## Task 6: Local end-to-end harness smoke on the A4000 (acceptance gate) **[OPS]**

**Files:** Create `docs/thesis/M3_CLUSTER_RESULTS.md` (start it here).

> Proves the full per-fold pipeline (source pretrain → all dependents loading the ckpt → scratch cells → A5) runs end-to-end on real GPU data at reduced scale **before** any cluster spend. This is the M3 local gate, mirroring M2's reference smoke.

- [ ] **Step 1: Run a one-fold reduced matrix locally**

```bash
cd /home/compa/Documents/working_dir/ResearchClaude/AutoResearchClaw
OUT=$(mktemp -d)
export RC_DATASET_DIR=~/rc_brain_data
for s in A1 A4; do .venv/bin/python -m cluster.pretrain_source --cell $s --rep 0 --outer 0 --out "$OUT" --smoke; done
for c in A2 A3 B1 B2 B3 B4 C1 C2 A5; do .venv/bin/python -m cluster.train_cell --cell $c --rep 0 --outer 0 --out "$OUT" --source-dir "$OUT" --smoke; done
echo "results:"; ls "$OUT"/results/
```
Expected: 11 `cell_*_val_r2:` lines (A1/A4 from the source step, the rest from train_cell), 11 result JSONs in `$OUT/results/`, no `load_backbone` strict/shape errors, no `carrier=` TypeError. Numbers are meaningless at `--smoke` scale — **the gate is mechanics**, exactly as M2.

- [ ] **Step 2: Record the outcome** in `docs/thesis/M3_CLUSTER_RESULTS.md` — date, the 11 emitted lines, confirmation that the A1→{B1,B2,C2} and A4→{B3,B4} transfers loaded clean, and that A5/C1/C2 (trivial / age@head / GLM@head) all ran. Commit:

```bash
git add docs/thesis/M3_CLUSTER_RESULTS.md
git commit -m "docs(thesis): M3 local harness smoke — full per-fold matrix runs forward"
```

---

## Task 7: Results reducer (`cluster/reduce_results.py`)

**Files:**
- Create: `cluster/reduce_results.py`
- Test: `tests/test_cluster_harness.py`

- [ ] **Step 1: Write the failing test** (append) — reducer aggregates per-fold JSONs to per-cell mean±std

```python
def test_reducer_aggregates_per_cell(tmp_path):
    import json
    from cluster import reduce_results as R

    rdir = tmp_path / "results"; rdir.mkdir()
    # two folds for A3, one for B1
    for k, v in [("A3", 0.20), ("A3", 0.18), ("B1", 0.02)]:
        i = sum(1 for _ in rdir.glob(f"{k}_*"))
        (rdir / f"{k}_rep0_outer{i}.json").write_text(json.dumps(
            dict(cell=k, rep=0, outer=i, test_r2=v, kind="target")))
    table = R.aggregate(rdir)
    a3 = next(r for r in table if r["cell"] == "A3")
    assert abs(a3["mean_r2"] - 0.19) < 1e-9 and a3["n_folds"] == 2
    md, csv = R.write_table(table, tmp_path / "m3_results")
    assert md.is_file() and csv.is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cluster_harness.py -k reducer -v`
Expected: FAIL — `ModuleNotFoundError: cluster.reduce_results`.

- [ ] **Step 3: Write `cluster/reduce_results.py`**

```python
"""Aggregate per-(cell,rep,outer) result JSONs into the per-cell val_r2 table,
side-by-side with the registered predictions + falsifier checks.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

# Registered predictions (point/▒range midpoints) from docs/thesis/pnc-age-vwm-matrix.md
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
        table.append(dict(cell=cell, mean_r2=st.fmean(vals),
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
    hdr = "| cell | mean R² | std | folds | registered |\n|---|---|---|---|---|\n"
    rows = "".join(f"| {r['cell']} | {r['mean_r2']:.4f} | {r['std_r2']:.4f} | "
                   f"{r['n_folds']} | {r['registered']} |\n" for r in table)
    md = stem.with_suffix(".md"); md.write_text("# M3 per-cell val_r2\n\n" + hdr + rows)
    csv = stem.with_suffix(".csv")
    csv.write_text("cell,mean_r2,std_r2,n_folds\n" +
                   "".join(f"{r['cell']},{r['mean_r2']},{r['std_r2']},{r['n_folds']}\n" for r in table))
    return md, csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="m3_results")
    a = ap.parse_args()
    table = aggregate(a.results)
    md, csv = write_table(table, a.out)
    print(f"wrote {md} and {csv}")
    for name, line in falsifiers(table):
        print(f"[falsifier] {name}: {line}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cluster_harness.py -k reducer -v`
Expected: PASS.

- [ ] **Step 5: Run the FULL harness suite + commit**

```bash
RC_DATASET_DIR=~/rc_brain_data .venv/bin/python -m pytest tests/test_cluster_harness.py -v
git add cluster/reduce_results.py tests/test_cluster_harness.py
git commit -m "feat(thesis): M3 results reducer (per-cell table + falsifier checks)"
```
Expected: all harness tests green.

---

## Task 8: Cluster bootstrap — project, container, branch push **[OPS]**

**Files:** Create `.cluster-helper.yaml`, `Dockerfile`, `requirements.txt`, `slurm/` (cluster-init scaffolds these; edit them).

- [ ] **Step 1: Bootstrap the project**

```bash
cd /home/compa/Documents/working_dir/ResearchClaude/AutoResearchClaw
cluster-init
```
Expected: writes `.cluster-helper.yaml` + `slurm/` templates + a `Dockerfile`. (Read `cluster-init --help` / the skill's `README.md` if a field is unclear — do not invent verbs.)

- [ ] **Step 2: Edit `.cluster-helper.yaml`** — set the project name, container image, and the dataset remote subdir. Keep dataset OUT of the git deploy (rule 2). Example fields:
```yaml
project_name: pnc-age-vwm
container_image: pnc-age-vwm.sif
dataset_remote_subdir: pnc_sc400        # -> /data/bdip_ssd/al5165/pnc-age-vwm/data/pnc_sc400
```
(Use the actual keys cluster-init emits; match them, don't rename.)

- [ ] **Step 3: Write `requirements.txt`** (mirror the §-3 stack; pin optuna from Task 0)

```
torch==2.11.0
torch_geometric==2.8.0
optuna>=3.6,<5
numpy==2.4.6
scipy==1.17.1
pandas==3.0.3
scikit-learn==1.9.0
```

- [ ] **Step 4: Write/adjust the `Dockerfile`** to install the CUDA-12.8 torch wheel + the rest. Skeleton (adapt to the cluster-init base + the cluster's CUDA/driver):

```dockerfile
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3.12 python3-pip git && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m pip install --upgrade pip
RUN python3.12 -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt /tmp/requirements.txt
RUN python3.12 -m pip install -r /tmp/requirements.txt
```
> **Container/GPU compatibility is the top M3 infra risk** (torch cu128 needs a recent cluster driver). Do NOT assume it works — Step 7 runs a one-task GPU sanity job before the matrix.

- [ ] **Step 5: Commit the cluster scaffolding + push the branch** (the deployed SHA must be on origin)

```bash
git add .cluster-helper.yaml Dockerfile requirements.txt slurm/
git commit -m "feat(thesis): M3 cluster bootstrap (cluster-helper manifest + container + slurm)"
git push -u origin thesis/pnc-age-vwm
```

- [ ] **Step 6: Build + push the container**

```bash
cluster-push-container
```
Expected: `docker build` → `docker save` → push → `.sif` on the cluster. (Re-run only when `Dockerfile`/`requirements.txt` change.)

- [ ] **Step 7: One-task GPU sanity job** (cheapest possible — proves torch sees the GPU in the `.sif` before the 500-task matrix). Add `slurm/sanity.sh` running `python -c "import torch,torch_geometric,optuna; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`, then:

```bash
cluster-gpus                                   # see gpunode02 free GPUs
cluster-submit --node gpunode02 slurm/sanity.sh -J smoke-pnc-sanity
# read JOB_ID=, then:
cluster-tail <JOB_ID>
```
Expected: `True <gpu name>`. **If torch reports CUDA unavailable / driver mismatch, fix the Dockerfile CUDA tag to match the cluster driver and re-push** before proceeding. This is a hard gate.

---

## Task 9: Upload the datasets **[OPS]**

**Files:** none (data transfer).

- [ ] **Step 1: Upload the two bundles** (datasets travel via `cluster-upload-dataset`, not git — rule 2)

```bash
cluster-upload-dataset ~/rc_brain_data/pnc_sc400_age_reg.npz pnc_sc400
cluster-upload-dataset ~/rc_brain_data/pnc_sc400_vwm_reg.npz pnc_sc400
# include the manifest if the loader reads it:
cluster-upload-dataset ~/rc_brain_data/MANIFEST_sc400.json pnc_sc400
```
Expected: each prints the remote path under `/data/bdip_ssd/al5165/pnc-age-vwm/data/pnc_sc400/`. The Slurm scripts (Task 10) export `RC_DATASET_DIR` to that dir. (If the scaffold loader needs the bundles at a bare `RC_DATASET_DIR`, upload them flat — match what `data_loader.dataset_root()` expects.)

---

## Task 10: SLURM matrix DAG scripts (`slurm/source.sh`, `slurm/cell.sh`)

**Files:**
- Create: `slurm/source.sh`, `slurm/cell.sh`, `slurm/_index.py` (maps `$SLURM_ARRAY_TASK_ID` → `(cell, rep, outer)`)
- Test: a local dry-run of `_index.py` (the only unit-testable piece; the `.sh` are ops).

- [ ] **Step 1: Write `slurm/_index.py`** (deterministic array-index → work-unit; shared by both scripts)

```python
"""Map a flat SLURM array index to a (cell, rep, outer) unit, given a cell list."""
import sys
from cluster._common import OUTER, REPS, SOURCE_CELLS, HPO_CELLS

def unit(kind, idx):
    cells = SOURCE_CELLS if kind == "source" else HPO_CELLS
    per = REPS * OUTER
    cell = cells[idx // per]
    rem = idx % per
    return cell, rem // OUTER, rem % OUTER          # cell, rep, outer

def count(kind):
    return len(SOURCE_CELLS if kind == "source" else HPO_CELLS) * REPS * OUTER

if __name__ == "__main__":
    kind, idx = sys.argv[1], int(sys.argv[2])
    c, r, o = unit(kind, idx)
    print(f"{c} {r} {o}")
```
Source array size = `2*10*5 = 100`; cell array size = `8*10*5 = 400`. Print them: `python -m slurm._index countcheck` (add a tiny branch) or just compute by hand and hard-code `--array=0-99` / `--array=0-399`.

- [ ] **Step 2: Write `slurm/source.sh`** (adapt the `#SBATCH` lines to the cluster-init template; `--node` sets partition/qos/account — do NOT hand-set them)

```bash
#!/bin/bash
#SBATCH --job-name=pnc-source
#SBATCH --array=0-99
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%A_%a.out
set -euo pipefail
export RC_DATASET_DIR=/data/bdip_ssd/al5165/pnc-age-vwm/data/pnc_sc400
OUT=/data/bdip_ssd/al5165/pnc-age-vwm/runs/matrix
read CELL REP OUTER < <(python -m slurm._index source ${SLURM_ARRAY_TASK_ID})
echo "source $CELL rep=$REP outer=$OUTER"
python -m cluster.pretrain_source --cell "$CELL" --rep "$REP" --outer "$OUTER" --out "$OUT"
```

- [ ] **Step 3: Write `slurm/cell.sh`** (same template; reads source ckpts from the same `$OUT`)

```bash
#!/bin/bash
#SBATCH --job-name=pnc-cell
#SBATCH --array=0-399
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%A_%a.out
set -euo pipefail
export RC_DATASET_DIR=/data/bdip_ssd/al5165/pnc-age-vwm/data/pnc_sc400
OUT=/data/bdip_ssd/al5165/pnc-age-vwm/runs/matrix
read CELL REP OUTER < <(python -m slurm._index cell ${SLURM_ARRAY_TASK_ID})
echo "cell $CELL rep=$REP outer=$OUTER"
python -m cluster.train_cell --cell "$CELL" --rep "$REP" --outer "$OUTER" --out "$OUT" --source-dir "$OUT"
```
> A5 (trivial) isn't in `HPO_CELLS`, so it's not in the 400-task array — run it as a tiny tail (Step 5) or fold it into the reducer call. Keep it explicit so A5 is never silently skipped.

- [ ] **Step 4: Wire-check the index mapping locally** (no cluster)

```bash
.venv/bin/python -c "from slurm._index import count,unit; print('source',count('source'),'cell',count('cell')); print(unit('cell',0),unit('cell',399))"
```
Expected: `source 100 cell 400`, first `('A2',0,0)`, last `('C2',9,4)` (order follows `HPO_CELLS`). Commit:

```bash
git add slurm/_index.py slurm/source.sh slurm/cell.sh
git commit -m "feat(thesis): M3 SLURM matrix DAG (source array + dependent cell array)"
git push origin thesis/pnc-age-vwm
```

---

## Task 11: Launch the matrix, monitor, fetch **[OPS]**

**Files:** none (cluster run); results fetched to `runs/matrix-fetched/`.

> **Official run — name it + pick the node (cluster-helper rules 4 & 5).** Use `AskUserQuestion` with a suggested name (e.g. `pnc-age-vwm-matrix`) and node (`gpunode02`, default) after showing `cluster-gpus`. Confirm before submitting (the cell array is 400 GPU tasks).

- [ ] **Step 1: Submit the source array, capture its job id**

```bash
cluster-gpus
cluster-submit --node gpunode02 slurm/source.sh -J pnc-age-vwm-source
# capture: SRC=$(cluster-submit ... | grep ^JOB_ID= | cut -d= -f2)
```

- [ ] **Step 2: Submit the cell array depending on the source array** (`afterok` — all source ckpts done first)

```bash
cluster-submit --node gpunode02 slurm/cell.sh -J pnc-age-vwm-matrix --dependency=afterok:$SRC
```
> Optimization (optional): `--dependency=aftercorr:$SRC` starts cell task *i* when source task *i* finishes, but the index orderings differ (2 source cells vs 8 target cells), so `afterok` (whole-array barrier) is the correct, simple choice here. Source pretraining is fast relative to the inner HPO, so the barrier costs little.

- [ ] **Step 3: Monitor**

```bash
cluster-status
cluster-tail <cell_job_id>          # spot-check a running task
```
Expected: tasks progress; logs show `cell_<X>_val_r2:` lines. Spot-check that transfer cells found their source ckpt (no "missing source ckpt" assert).

- [ ] **Step 4: Run A5 (trivial) tail + fetch results** once the matrix completes

```bash
# A5 is cheap CPU work — one task over (rep,outer) or just compute in the reducer.
cluster-fetch /data/bdip_ssd/al5165/pnc-age-vwm/runs/matrix/results runs/matrix-fetched
ls runs/matrix-fetched/ | wc -l       # expect ~450 JSONs (8 cells×50 + A1/A4×50 + A5×50)
```

---

## Task 12: Aggregate, compare to predictions, hand off to M4 **[OPS]**

**Files:** `docs/thesis/M3_CLUSTER_RESULTS.md` (final), memory update.

- [ ] **Step 1: Reduce the fetched results**

```bash
.venv/bin/python -m cluster.reduce_results --results runs/matrix-fetched --out docs/thesis/m3_results
```
Expected: `docs/thesis/m3_results.md` + `.csv` with per-cell mean±std `val_r2`, and `[falsifier]` lines for the three pre-registered checks. Sanity-anchor against the registered predictions (A3 ≈ 0.18–0.21 is the load-bearing baseline; if A3 is wildly off, suspect a protocol bug before interpreting transfer).

- [ ] **Step 2: Write the M3 evidence section** in `docs/thesis/M3_CLUSTER_RESULTS.md` — the launched job ids, realized N=743, the per-cell table, and a one-paragraph read of the falsifiers (do NOT over-claim; report what the numbers say vs the registered predictions).

- [ ] **Step 3: Verification before "done"** (superpowers:verification-before-completion)

```bash
RC_DATASET_DIR=~/rc_brain_data .venv/bin/python -m pytest tests/test_cluster_harness.py -v
git status   # only intended files; leave the pre-existing untracked files alone
git log --oneline -12
```
Expected: harness suite green; the M3 commits present; the per-cell table committed.

- [ ] **Step 4: Commit + update memory + offer the M4 handoff**

```bash
git add docs/thesis/M3_CLUSTER_RESULTS.md docs/thesis/m3_results.md docs/thesis/m3_results.csv
git commit -m "docs(thesis): M3 cluster matrix results vs registered predictions"
git push origin thesis/pnc-age-vwm
```
Update the `pnc-age-vwm-thesis-effort` memory with an "M3 DONE" paragraph (the harness package, the DAG, realized N, the result-table path, any protocol deviation from the pre-reg). Then offer to write the **M4 handoff** (ARC analysis stage on the fetched results + the ARC `experiment/` anchor from Task 1) via the `handoff` skill.

---

## Self-Review

**Spec coverage (substrate-design §6 M3 scope + the M3 handoff):**
- "cluster project + `.cluster-helper.yaml`" → **Task 8**.
- "dataset upload" → **Task 9**.
- "`.sif` container" → **Task 8** (+ the GPU sanity gate, Step 7).
- "SLURM matrix DAG (pretrain→checkpoint→dependent cells)" → **Tasks 4, 5, 10, 11** (A1/A4 source array → B/C dependent array via `afterok`; within-subject shared folds).
- "full nested CV (10×5×20 Optuna, HPO on val_r2)" → **Tasks 2–5** (`folds.py` + `train_cell.py` inner Optuna; locked params in `_common.py`).
- "generate the 11-cell code, inspect for the loader-API fix before launching" (handoff task 1–2) → **Task 1**.
- "feed results back into analysis (M4)" → **Task 12**.
- The within-subject transfer decision (locked 2026-06-14) → **Task 2** folds (`outer_split` shares the partition across age/vwm; `test_*` excluded from both stages) + the `test_folds_are_subject_aligned_and_disjoint` proof.

**Placeholder scan:** every module + Slurm script + Dockerfile is shown in full; each TDD step has runnable code + an expected result. Two values are confirmed-at-runtime, not placeholders, and flagged as hard gates: (a) the container CUDA tag vs the cluster driver (**Task 8 Step 7** sanity job); (b) the exact `.cluster-helper.yaml`/Slurm `#SBATCH` keys the cluster-init template emits (**Task 8 Step 2 / Task 10** — "match what cluster-init emits, don't rename"). The fixed `ARCH` + search space are deliberate Option-A choices (Verified fact 7), re-pinnable in one constant block.

**Type/name consistency:** the `CELLS` map (Task 3) is the single source for carrier/bundle/mode/source/global_dim, consumed identically by `data.py`, `pretrain_source.py`, `train_cell.py`, `_index.py`, and the tests; cell ids A1–A5/B1–B4/C1/C2, bundle names `pnc_sc400_age_reg`/`pnc_sc400_vwm_reg`, scaffold symbols (`load_fc_graphs` with `node_features=`/`node_feature_key=`+`glm_diagonal=`/`graph_feature_key=` — **never `carrier=`**, `GraphRegressor(pooled_dim/global_dim)`, `transfer.save_backbone`/`load_backbone(strict=True)`/`freeze`, `age_vwm_baseline`) match `SCAFFOLD.md` + `scripts/smoke_thesis_cells.py`; metric `val_r2`/maximize and `ARCH`/search-space live only in `_common.py`. The `ckpt_path()` keying `(cell,rep,outer)` is defined once in `pretrain_source.py` and imported by `train_cell.py` so source/target agree on the checkpoint filename.

**Compute envelope (sanity, not a gate):** ~8 HPO cells × 10 reps × 5 outer × (20 inner + 1 refit) ≈ 8.4k target trainings + 100 source pretrains; each loads its fold's ~595-graph train once and reuses across trials. Heavy but array-parallel across the source/cell arrays. If `cluster-gpus` shows gpunode02 too contended or per-task `--time` is exceeded, the cheapest rigorous reductions (confirm with the user first) are `REPS 10→5` or `INNER_TRIALS 20→10` in `_common.py` — both one-line, both honest to report. Do NOT silently truncate.

**Deferred (not M3):** D1 @node surgery; the ARC analysis + paper write-up vs predictions/falsifiers (M4); any FC/ORBIT replication (out of scope per §0).
```
