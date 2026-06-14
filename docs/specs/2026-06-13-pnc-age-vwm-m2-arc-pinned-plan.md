# PNC Age→VWM Milestone 2 (ARC-Pinned) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make AutoResearchClaw's *autonomous* design + code-generation stages reproduce the pre-registered 11-cell PNC age→VWM matrix on top of the M1 scaffold primitives — by authoring a thesis matrix/seed artifact, two pinned configs (full + laptop smoke), and a tiny local A4000 smoke that proves the pinned pipeline emits runnable, correct-shaped code (imports `scaffold.*`, loads a `pnc_sc400_*` bundle, trains forward, transfers a checkpoint, emits a metric).

**Architecture:** No ARC source changes — M2 is *configure + prompt + smoke*, not re-architect. The scaffold→ARC plumbing already exists (`researchclaw/experiment/scaffold.py`, `ScaffoldConfig`, and the Stage-10 scaffold-guidance injection). We pin the two autonomous stages with **one matrix/seed document** injected verbatim into **both** `experiment_design` (Stage 9) and `code_generation` (Stage 10) via `prompts.extra_prompts` — the same document is simultaneously the "thesis seed artifact" and the "per-stage pinning prompt." A canonical `config.thesis.yaml` (full 11-cell, execution-gated for the M3 cluster handoff) and a `config.thesis.smoke.yaml` (4-cell reduced protocol, runs locally) drive it. Lit/synthesis/hypotheses/analysis/paper stages run normally.

**Tech Stack:** YAML config (`RCConfig.load`), `researchclaw.prompts.PromptManager` (`extra_prompts` → `_load_extras` → `for_stage`), the M1 `experiment_scaffold/` primitives, the baked `~/rc_brain_data` SC-400 bundles, the Claude Code/ACP backend for running ARC (`researchclaw-claude-code-backend` memory), pytest.

**Spec / sources:**
- `docs/specs/2026-06-13-pnc-age-vwm-substrate-design.md` §0 (locked decisions), §6 (M2 = this scope).
- `docs/specs/2026-06-10-experiment-scaffold-design.md` (scaffold↔ARC contract — already implemented; do not rebuild).
- The authoritative matrix: `~/Documents/LLM Wiki/Transfer Learning/wiki/theses/pnc-age-vwm-transfer-vs-concatenation.md` (the 11 cells, predictions, falsifiers — transcribed in Task 1; **D1 is deferred and NOT one of the 11**).
- M1 plan + `experiment_scaffold/SCAFFOLD.md` (the primitives + the two bundles M2 wires).

---

## Verified facts (investigated 2026-06-13; do not re-derive)

1. **The 11 cells = A1–A5, B1–B4, C1, C2** (D1 deferred). Full table + per-cell scaffold recipe in Task 1.
2. **`extra_prompts` injection is live on both codegen paths.** `prompts.extra_prompts.<stage>` parses (`config.py:986`) into `PromptsConfig.extra_prompts` (tuple of `(stage, path_or_text)`); the executor rebuilds the dict and constructs a `PromptManager(..., extra_prompts=...)` per stage (`executor.py:657`). `PromptManager._load_extras` (`manager.py:181`) **reads the value as a file when the path exists, else inline**, and drops unknown stage names with a warning. `for_stage` (`manager.py:231`) appends the resolved text under `## Additional Stage Guidance`. The default **CodeAgent** path (`code_agent.enabled=True`) calls `self._pm.for_stage("code_generation", ...)` (`code_agent.py:1011`), and the legacy single-shot path calls `for_stage("code_generation", ...)` (`_code_generation.py:734`) — so the injection works **regardless** of which codegen path runs. Stage 9 calls `for_stage("experiment_design", ...)` (`_experiment_design.py:201`). **The exact stage keys are `experiment_design` and `code_generation`** — any typo is silently dropped, so Task 5 asserts they resolved.
2a. **Silent-failure trap in `extra_prompts` path resolution.** `_load_extras` resolves the value with `Path(text).expanduser()` against the **current working directory**, and — critically — if the path does **not** exist it does *not* drop the stage; it injects the **literal string verbatim** (`manager.py:196–209`). So a relative path that misresolves silently turns the "pin" into the one-line string `./docs/thesis/pnc-age-vwm-matrix.md` instead of the matrix — and ARC then invents its own model. ARC runs with CWD = repo root (as does pytest), so the relative paths below resolve; but the **`len(body) > 500` assertion in Task 5 is the guard** that catches a misresolved path, and the implementer may switch to absolute paths (like `scaffold.dir` already is) if ARC is ever launched from elsewhere.
3. **Two corruption risks the configs must defuse:**
   - **Condition-count trim.** Stage 9 caps `len(baselines)+len(proposed_methods)+len(ablations)` at **8** for `time_budget_sec ≤ 3600`, **12** for `> 3600`, **20** for `> 7200` (`_experiment_design.py:495–540`). The full 11-cell config **must set `time_budget_sec > 3600`** (we use `7201` → cap 20) or cells get silently trimmed. The smoke pins only 4 cells, so its small budget is fine.
   - **BenchmarkAgent dataset hijack.** For ML-domain topics Stage 9 runs the BenchmarkAgent, which **overwrites `plan["datasets"]` and prepends its own baselines** (`_experiment_design.py:458–471`) — it would inject CIFAR/HF datasets over our two PNC bundles. **Both configs must set `experiment.benchmark_agent.enabled: false`.**
4. **Scaffold guidance is already injected into codegen** (`build_scaffold_guidance`, `_code_generation.py:392`) — the LLM already sees `SCAFFOLD.md` verbatim, so the pin prompt only has to say *which* primitives to wire per cell, not re-describe them.
5. **Config loads with `RCConfig.load(path, check_paths=False)`** (`config.py:1691` → `load_config`). The scaffold-config parse helpers (`_parse_scaffold_config`, `_parse_experiment_config`) and `PromptsConfig` are tested directly in `tests/test_scaffold_config.py` — mirror that style. `pm.extra_prompts()` (`manager.py:301`) returns the resolved per-stage texts — the clean public hook for Task 5.
6. **Decoupled handoff (§0).** Full config gates *execution* via HITL so ARC pauses and M3 drives the cluster; the smoke config does **not** gate execution so it runs end-to-end locally. Confirm the execution stage number with `researchclaw info` in Task 2 (design = 9, code-gen = 10, execution = 12 per the scaffold design doc).

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `docs/thesis/pnc-age-vwm-matrix.md` | **The seed + per-stage pin.** Full 11-cell table, per-cell scaffold wiring recipe, checkpoint dependency graph, registered predictions/falsifiers, and explicit Stage-9 + Stage-10 instructions. Injected verbatim into both stages. | Create |
| `config.thesis.yaml` | Canonical full config: scaffold→`experiment_scaffold` + `~/rc_brain_data`; topic=thesis; `time_budget_sec: 7201`; `benchmark_agent.enabled:false`; `extra_prompts.{experiment_design,code_generation}` → the matrix doc; HITL gates execution (cluster handoff). | Create |
| `docs/thesis/pnc-age-vwm-matrix-smoke.md` | Smoke seed + pin: subset **{A1, A3, B1, C1}** + reduced protocol (limit subjects, ≤ epochs, single split). Same structure as the full matrix doc. | Create |
| `config.thesis.smoke.yaml` | Laptop A4000 smoke config: same pinning, small budget, `extra_prompts` → the smoke matrix doc, HITL does **not** gate execution (runs locally). | Create |
| `tests/test_thesis_pinning.py` | Validates both configs parse with the right fields, both matrix docs contain exactly their cells (full has 11 incl. no-D1; smoke has 4), and a `PromptManager` built from each config injects the matrix into **both** `experiment_design` and `code_generation`. | Create |
| `experiment_scaffold/SCAFFOLD.md` | (Touch only if the smoke surfaces a manifest gap.) | Maybe |

> **Why one doc, not three.** The handoff lists a "seed artifact" (#2) and "per-stage pinning prompts" (#3) separately. `extra_prompts` injects exactly one resolved text per stage, so the DRY, drift-free design is a **single authoritative matrix document injected into both stages**. It carries clearly-labelled `## Instructions — experiment-design stage` and `## Instructions — code-generation stage` sections; each stage reads its own section and the rest is harmless consistent context. The seed *is* the pin.

---

## Task 1: Author the thesis matrix/seed document (full)

**Files:**
- Create: `docs/thesis/pnc-age-vwm-matrix.md`
- Test: `tests/test_thesis_pinning.py` (Step 1 below; the test file is fleshed out across Tasks 1–5)

- [ ] **Step 1: Write the failing test** (create `tests/test_thesis_pinning.py`)

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_thesis_pinning.py -k matrix_doc -v`
Expected: FAIL — `AssertionError: missing .../docs/thesis/pnc-age-vwm-matrix.md`

- [ ] **Step 3: Write the matrix document** (`docs/thesis/pnc-age-vwm-matrix.md`)

Create `docs/thesis/` and write this file verbatim. It is the single source of truth injected into Stages 9 and 10.

````markdown
# PNC Age→VWM Transfer-vs-Concatenation — Pinned Experiment Matrix

> This document is injected verbatim into the AutoResearchClaw **experiment-design**
> (Stage 9) and **code-generation** (Stage 10) stages. It is the authoritative,
> pre-registered specification. The design and code-generation stages must
> **reproduce exactly the cells and wiring below** on top of the provided
> `scaffold/` package and the baked PNC SC-400 bundles. **Do not invent cells,
> models, datasets, or metrics. Do not re-implement the model or the data loader.**

## Thesis (context)

On PNC, with a fixed-topology identity-encoded GCN on the 400-node cortical
Schaefer structural-connectivity graph, test whether sequential **age→VWM transfer**
or **age/GLM concatenation** beats a **from-scratch GLM-feature model** at predicting
visual working memory (`VWM_overall_dprime`). Source label = chronological **age**;
target = **VWM d′**. Carriers: **ID** = identity one-hot (`node_features="identity"`);
**GLM** = `glm_diagonal` (per-node 2back-vs-0back z-map, `glm_normalize=true`).
**@head** = a graph-level side channel concatenated into the pooled embedding.

## Data (provided; read via `RC_DATASET_DIR`, never hardcode paths)

| Bundle | y (task) | Carrier arrays present | Used by |
|---|---|---|---|
| `pnc_sc400_age_reg` | age (regression), N=747 | `sc (N,400,400)`, `glm_2back_vs_0back (N,400)` | A1, A4 — **source checkpoints** |
| `pnc_sc400_vwm_reg` | `VWM_overall_dprime` (regression), N=744 | `sc`, `glm_2back_vs_0back`, **`age (N,)`** co-stored | A2, A3, A5, B1–B4, C1, C2 |

Load every graph with `matrix_key="sc", edge_weight_norm="abs_max"` (SC edges are
large positive streamline counts). Backbone input dim is **400 for both carriers**,
so transferred backbones load `strict=True` across all cells.

## The 11 cells (D1 is DEFERRED — out of scope, do not implement)

| Cell | Node feat | Pretrain | Mode | Inject (@head) | Bundle / Target | Role |
|------|-----------|----------|------|----------------|-----------------|------|
| A1 | ID  | —   | scratch   | — | `pnc_sc400_age_reg` / age | source ckpt → B1,B2,C2 |
| A2 | ID  | —   | scratch   | — | `pnc_sc400_vwm_reg` / VWM | identity floor |
| A3 | GLM | —   | scratch   | — | `pnc_sc400_vwm_reg` / VWM | **GLM-from-scratch — baseline to beat** |
| A4 | GLM | —   | scratch   | — | `pnc_sc400_age_reg` / age | source ckpt → B3,B4 |
| A5 | none (age scalar) | — | — | — | `pnc_sc400_vwm_reg` / VWM | trivial age→VWM floor |
| B1 | ID  | age | fine-tune | — | `pnc_sc400_vwm_reg` / VWM | transfer (load A1) |
| B2 | ID  | age | **frozen** (head only) | — | `pnc_sc400_vwm_reg` / VWM | transfer (load A1) |
| B3 | GLM | age | fine-tune | — | `pnc_sc400_vwm_reg` / VWM | transfer (load A4) |
| B4 | GLM | age | **frozen** | — | `pnc_sc400_vwm_reg` / VWM | transfer (load A4) |
| C1 | GLM | —   | scratch   | **age** (U=1) | `pnc_sc400_vwm_reg` / VWM | age @head concat |
| C2 | ID  | age | fine-tune | **GLM zmap** (U=400) | `pnc_sc400_vwm_reg` / VWM | GLM @head (load A1) |

**Checkpoint dependency graph (execution order):** train **A1** and **A4** first and
save their backbones; **A1 → {B1, B2, C2}**, **A4 → {B3, B4}** load them.

## Per-cell scaffold wiring recipe (import from `scaffold/`; do not re-implement)

Common backbone (same hyperparameters across cells; HPO-swept in the full run):
`backbone = scaffold.models.build_gcn(in_channels=400, hidden_channels=H,
out_channels=P, num_layers=L, norm="batch_norm")`.

- **Carrier — identity:** `scaffold.data_loader.load_fc_graphs(name, matrix_key="sc",
  edge_weight_norm="abs_max", node_features="identity")` → `x` is `(400,400)`.
- **Carrier — glm_diagonal:** `...load_fc_graphs(name, matrix_key="sc",
  edge_weight_norm="abs_max", node_feature_key="glm_2back_vs_0back",
  glm_diagonal=True, glm_normalize=True)` → `x` is `(400,400)` diagonal.
- **Head (plain, global_dim=0):** `scaffold.heads.GraphRegressor(backbone, pooled_dim=P,
  global_dim=0)`; `forward(x, edge_index, edge_weight, batch)`.
- **Head (age @head, C1):** add `graph_feature_key="age"` to the loader (→ `data.u`,
  `U=1`); `GraphRegressor(backbone, pooled_dim=P, global_dim=1)`;
  `forward(..., u=batch.u)`.
- **Head (GLM @head, C2):** add `graph_feature_key="glm_2back_vs_0back"` to the loader
  (→ `data.u`, `U=400`); `GraphRegressor(backbone, pooled_dim=P, global_dim=400)`;
  `forward(..., u=batch.u)`.
- **Transfer (B1,B3,C2 — fine-tune):** after training the source cell,
  `scaffold.transfer.save_backbone(model.backbone, ckpt_path)`; in the target cell
  build a fresh model with the matching carrier and
  `scaffold.transfer.load_backbone(model.backbone, ckpt_path, strict=True)`; leave all
  params trainable.
- **Transfer (B2,B4 — frozen):** same load, then
  `scaffold.transfer.freeze(model.backbone)` (only the head trains).
- **A5 (no graph):** `(slope, intercept), cv_r2 =
  scaffold.baselines.age_vwm_baseline(age, vwm, cv=5)` using the co-stored `age` array
  and `y` from `pnc_sc400_vwm_reg` (load arrays via
  `scaffold.data_loader.load_fc_bundle("pnc_sc400_vwm_reg", matrix_key="sc")` → `meta["age"]`).

Source-cell carriers: **A1 = identity**, **A4 = glm_diagonal**. The target cells reuse
the matching carrier (B1/B2/C2 = identity to match A1; B3/B4 = glm_diagonal to match A4;
C1 = glm_diagonal scratch with an age side channel).

## Registered predictions (do not optimize toward these; report what you get)

| Cell | Predicted R² | Rationale |
|---|---|---|
| A2 (ID scratch) | ≈ −0.03 (identity floor) | identity carries no VWM signal |
| A3 (GLM scratch) | ≈ 0.18–0.21 | the VWM-bearing baseline to beat |
| A5 (trivial age→VWM) | small positive | developmental age→VWM trend floor |
| B1 (ID age→FT) | ≈ 0–0.05 (≪ A3) | transfer can't manufacture VWM signal absent from SC |
| B2 (ID age→frozen) | ≤ B1 | frozen is the weakest variant |
| B3 (GLM age→FT) | ≈ A3 | GLM-from-scratch already saturates the signal |
| B4 (GLM age→frozen) | ≤ B3 | frozen head can't adapt the GLM read-out |
| C1 (GLM + age@head) | A3 < C1 ≤ A3 ⊕ A5 | age boost is the developmental trend, additive at best |
| C2 (ID age-FT + GLM@head) | ≈ A3 | GLM @head recovers the signal; the age backbone adds little |

**Falsifiers:** signal-location claim falsifies if B1/B2 ≈ A3; trivial-trend claim
falsifies if C1 > A3 ⊕ A5 super-additively; saturation claim falsifies if B3 ≫ A3.

## Metric & protocol

- Primary metric key = **`val_r2`** (regression R² on a held-out split; **higher is
  better**). Report per-cell `val_r2`.
- Full protocol (cluster, M3): nested CV (10 reps × 5 outer × 20 inner Optuna), HPO on
  `val_r2`. **For any local run, use the reduced protocol the stage instructions below
  specify** — do not attempt the full nested CV locally.

---

## Instructions — experiment-design stage (Stage 9)

Emit an experiment plan whose conditions are **exactly the 11 cells above and nothing
else**:
- `baselines:` → A1, A2, A3, A4, A5 (one entry each, named e.g. `A1_id_age_source`,
  `A2_id_vwm_floor`, `A3_glm_vwm_scratch`, `A4_glm_age_source`, `A5_trivial_age_vwm`).
- `proposed_methods:` → B1, B2, B3, B4, C1, C2 (named e.g. `B1_id_age_finetune`,
  `B2_id_age_frozen`, `B3_glm_age_finetune`, `B4_glm_age_frozen`, `C1_glm_age_head`,
  `C2_id_glm_head`).
- `ablations:` → leave empty or list only carrier/mode contrasts already covered above;
  **do not add new conditions**.
- `datasets:` → exactly `pnc_sc400_age_reg` and `pnc_sc400_vwm_reg` (the provided
  bundles). **Do not propose CIFAR, MNIST, HuggingFace, or any standard ML benchmark.**
- `metrics:` → `val_r2` (primary). `objectives:` → the transfer-vs-concatenation
  comparison. Keep the total condition count at 11.

## Instructions — code-generation stage (Stage 10)

Generate a single self-contained experiment (`main.py` + optional helpers) that:
- **Imports the provided primitives** — `from scaffold.data_loader import load_fc_graphs,
  load_fc_bundle`, `from scaffold.models import build_gcn`,
  `from scaffold.heads import GraphRegressor`, `from scaffold import transfer`,
  `from scaffold.baselines import age_vwm_baseline`. **Never re-implement the model,
  loader, head, or transfer logic. Never modify anything under `scaffold/`.**
- Reads the dataset root from `os.environ['RC_DATASET_DIR']`; loads the two bundles by
  name; uses `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.
- Implements **one function per cell** following the per-cell recipe above, runs the
  **source cells (A1, A4) first**, saves their backbones with `transfer.save_backbone`,
  and loads them in the dependent cells with `transfer.load_backbone(..., strict=True)`
  (and `transfer.freeze` for B2/B4). A5 uses `age_vwm_baseline` (no graph).
- Emits one metric line per cell in `name: value` format, e.g.
  `cell_A3_val_r2: 0.19`, plus a primary `val_r2: <best or A3>` line.
- Uses the reduced protocol passed via the stage budget for any local run (few epochs,
  a single train/val split); the full nested CV is the cluster's job, not this code's.
````

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_thesis_pinning.py -k matrix_doc -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add docs/thesis/pnc-age-vwm-matrix.md tests/test_thesis_pinning.py
git commit -m "feat(thesis): pinned 11-cell age->VWM matrix seed/design artifact"
```

---

## Task 2: Author `config.thesis.yaml` (canonical full config)

**Files:**
- Create: `config.thesis.yaml`
- Reference: `config.gnn.yaml` (the closest existing GNN config — copy its `llm`,
  `experiment.scaffold`, `experiment.sandbox`, `figure_agent` blocks)
- Test: `tests/test_thesis_pinning.py`

- [ ] **Step 1: Confirm the execution stage number**

Run: `.venv/bin/python -m researchclaw info` (or `researchclaw info` if on PATH)
Expected: a stage list. Note the **execution** stage number (expected `12`) and that
`experiment_design` (9) and `code_generation` (10) appear. Use the execution number in
`security.hitl_required_stages` below so ARC pauses *before* running (cluster handoff).
If `info` is unavailable, grep the stage enum: `grep -n "EXPERIMENT_DESIGN\|CODE_GENERATION\|EXECUTION" researchclaw/pipeline/stages.py`.

- [ ] **Step 2: Write the failing test** (append to `tests/test_thesis_pinning.py`)

```python
from researchclaw.config import load_config


def _load(path):
    # check_paths=False: the scaffold dirs/bundles live outside the repo and may
    # be absent on a CI box; we assert the *parsed* values, not on-disk presence.
    return load_config(path, check_paths=False)


def test_full_config_pins_scaffold_and_defuses_corruption():
    cfg = _load(CFG_FULL)
    s = cfg.experiment.scaffold
    assert s.enabled is True
    assert s.dir.endswith("experiment_scaffold")
    assert "rc_brain_data" in s.dataset_dir
    assert s.require is True
    # condition-count trim guard: 11 cells need a budget > 3600 (cap becomes 12/20)
    assert cfg.experiment.time_budget_sec > 3600
    # BenchmarkAgent must not hijack plan["datasets"]/baselines
    assert cfg.experiment.benchmark_agent.enabled is False
    # regression: higher R2 is better
    assert cfg.experiment.metric_key == "val_r2"
    assert cfg.experiment.metric_direction == "maximize"


def test_full_config_injects_matrix_into_both_stages():
    cfg = _load(CFG_FULL)
    extras = dict(cfg.prompts.extra_prompts)
    assert "experiment_design" in extras
    assert "code_generation" in extras
    # both point at the (existing) full matrix doc
    for v in (extras["experiment_design"], extras["code_generation"]):
        assert "pnc-age-vwm-matrix.md" in v
        assert v.endswith("pnc-age-vwm-matrix.md")
    # and the execution stage is HITL-gated (cluster handoff in M3)
    assert any(int(s) >= 11 for s in cfg.security.hitl_required_stages), \
        "full config must gate execution so ARC pauses for the cluster handoff"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_thesis_pinning.py -k full_config -v`
Expected: FAIL — `FileNotFoundError`/load error for `config.thesis.yaml`.

- [ ] **Step 4: Write `config.thesis.yaml`**

Adjust `12` below to the execution stage number confirmed in Step 1.

```yaml
project:
  name: "pnc-age-vwm-transfer-vs-concatenation"
  mode: "full-auto"

research:
  topic: >-
    Within-PNC age->visual-working-memory transfer vs. age/GLM concatenation in a
    fixed-topology identity-encoded GCN on the 400-node cortical Schaefer
    structural-connectivity graph. Does reusing an age-pretrained backbone
    (fine-tuned or frozen) or concatenating age/GLM as a side channel beat a
    from-scratch GLM-feature model at predicting VWM d-prime on the same subjects?
  domains:
    - "machine-learning"
    - "graph-neural-networks"
    - "transfer-learning"
    - "neuroimaging"
  daily_paper_count: 5
  quality_threshold: 3.0

runtime:
  timezone: "America/New_York"
  max_parallel_tasks: 3
  approval_timeout_hours: 12

notifications:
  channel: "console"
  target: ""
  on_stage_start: true
  on_stage_fail: true
  on_gate_required: true

knowledge_base:
  backend: "markdown"
  root: "docs/kb"

llm:
  provider: "acp"
  base_url: ""
  wire_api: "chat_completions"
  api_key_env: ""
  api_key: ""
  primary_model: "gpt-4o"
  fallback_models:
    - "gpt-4.1"
    - "gpt-4o-mini"

security:
  # Gate the execution stage so ARC PAUSES before running the full nested CV —
  # M3 drives the cluster (decoupled handoff). 9 = experiment design review.
  hitl_required_stages: [9, 12, 20]
  allow_publish_without_approval: false
  redact_sensitive_logs: true

experiment:
  mode: "sandbox"
  # >3600 so Stage 9 does NOT trim the 11 conditions (cap: 12 for >3600, 20 for >7200).
  time_budget_sec: 7201
  max_iterations: 4
  metric_key: "val_r2"
  metric_direction: "maximize"
  scaffold:
    enabled: true
    dir: "/home/compa/Documents/working_dir/ResearchClaude/AutoResearchClaw/experiment_scaffold"
    manifest: "SCAFFOLD.md"
    dataset_dir: "~/rc_brain_data"
    require: true
  sandbox:
    python_path: ".venv/bin/python3"
    gpu_required: true
    max_memory_mb: 8192
  # The pinned matrix is the from-scratch + transfer + concat design itself — the
  # BenchmarkAgent would overwrite plan["datasets"]/baselines with ML benchmarks.
  benchmark_agent:
    enabled: false
  opencode:
    enabled: false

prompts:
  custom_file: ""
  # The matrix doc is the seed AND the per-stage pin: injected verbatim into both
  # the experiment-design and code-generation stages (resolved as a file path).
  extra_prompts:
    experiment_design: ./docs/thesis/pnc-age-vwm-matrix.md
    code_generation: ./docs/thesis/pnc-age-vwm-matrix.md
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_thesis_pinning.py -k full_config -v`
Expected: PASS (2 tests). If `metric_direction`/`hitl_required_stages` attribute paths
differ, fix the test to the real attribute names (verify with
`.venv/bin/python -c "from researchclaw.config import load_config; c=load_config('config.thesis.yaml', check_paths=False); print(c.security.hitl_required_stages, c.experiment.metric_direction)"`).

- [ ] **Step 6: Commit**

```bash
git add config.thesis.yaml tests/test_thesis_pinning.py
git commit -m "feat(thesis): config.thesis.yaml pins scaffold + 11-cell matrix (full)"
```

---

## Task 3: Author the smoke matrix/seed document (4-cell reduced)

**Files:**
- Create: `docs/thesis/pnc-age-vwm-matrix-smoke.md`
- Test: `tests/test_thesis_pinning.py`

> **Smoke subset = {A1, A3, B1, C1}.** This covers everything the smoke must prove with
> the fewest cells: **A1** (identity carrier, source checkpoint → exercises
> `save_backbone`), **A3** (glm_diagonal carrier, the from-scratch baseline), **B1**
> (loads A1, fine-tune → exercises `load_backbone` strict=True transfer), **C1**
> (glm_diagonal + age `data.u` @head → exercises `graph_feature_key` + `global_dim=1`).
> 4 conditions ≤ the cap of 8, so the smoke's small budget will not trim.

- [ ] **Step 1: Write the failing test** (append to `tests/test_thesis_pinning.py`)

```python
def test_smoke_matrix_doc_is_the_four_cell_subset():
    assert MATRIX_SMOKE.is_file(), f"missing {MATRIX_SMOKE}"
    text = MATRIX_SMOKE.read_text(encoding="utf-8")
    for cell in SMOKE_CELLS:
        assert cell in text, f"smoke matrix missing cell {cell}"
    # The transfer/concat cells the smoke is built around must be wired here.
    for sym in ("load_fc_graphs", "save_backbone", "load_backbone",
                "graph_feature_key", "pnc_sc400_age_reg", "pnc_sc400_vwm_reg"):
        assert sym in text, f"smoke matrix missing {sym}"
    # reduced protocol must be explicit (so the smoke is tiny on the A4000)
    low = text.lower()
    assert "limit" in low and "epoch" in low
    # cells NOT in the smoke subset must not be pinned as run cells
    for cell in ("A2", "A4", "B2", "B3", "B4", "C2"):
        assert f"| {cell} " not in text, f"{cell} should not be a smoke run cell"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_thesis_pinning.py -k smoke_matrix -v`
Expected: FAIL — `AssertionError: missing .../pnc-age-vwm-matrix-smoke.md`

- [ ] **Step 3: Write the smoke matrix document** (`docs/thesis/pnc-age-vwm-matrix-smoke.md`)

````markdown
# PNC Age→VWM — Pinned SMOKE Matrix (laptop A4000, reduced protocol)

> Injected verbatim into the AutoResearchClaw **experiment-design** and
> **code-generation** stages for a **local smoke validation only**. Goal: prove the
> pinned pipeline emits runnable, correct-shaped code — not real numbers. Implement
> **exactly the 4 cells below and nothing else.** Import the provided `scaffold/`
> primitives; never re-implement the model/loader/head/transfer; never touch
> `scaffold/`.

## Data (read via `RC_DATASET_DIR`)

- `pnc_sc400_age_reg` — y=age — source checkpoint for B1.
- `pnc_sc400_vwm_reg` — y=`VWM_overall_dprime`, with `age (N,)` co-stored — targets.

Load with `matrix_key="sc", edge_weight_norm="abs_max"`. Backbone `in_channels=400`.

## Smoke cells (subset of the full 11)

| Cell | Node feat | Pretrain | Mode | Inject (@head) | Bundle / Target | Role |
|------|-----------|----------|------|----------------|-----------------|------|
| A1 | ID  | —   | scratch   | — | `pnc_sc400_age_reg` / age | source ckpt (save backbone) |
| A3 | GLM | —   | scratch   | — | `pnc_sc400_vwm_reg` / VWM | from-scratch GLM baseline |
| B1 | ID  | age | fine-tune | — | `pnc_sc400_vwm_reg` / VWM | transfer: load A1, fine-tune |
| C1 | GLM | —   | scratch   | **age** (U=1) | `pnc_sc400_vwm_reg` / VWM | age @head concat |

**Order:** train A1 first, `transfer.save_backbone(model.backbone, ckpt)`; B1 builds a
fresh identity-carrier model and `transfer.load_backbone(model.backbone, ckpt,
strict=True)`.

## Wiring recipe (identical to the full matrix; import from `scaffold/`)

- identity carrier: `load_fc_graphs(name, matrix_key="sc",
  edge_weight_norm="abs_max", node_features="identity")`.
- glm_diagonal carrier: `load_fc_graphs(name, matrix_key="sc",
  edge_weight_norm="abs_max", node_feature_key="glm_2back_vs_0back",
  glm_diagonal=True, glm_normalize=True)`.
- A1/A3/B1 head: `GraphRegressor(backbone, pooled_dim=P, global_dim=0)`.
- C1 head: loader `graph_feature_key="age"` → `data.u (B,1)`;
  `GraphRegressor(backbone, pooled_dim=P, global_dim=1)`; `forward(..., u=batch.u)`.
- transfer: `from scaffold import transfer` → `save_backbone` / `load_backbone(...,
  strict=True)`.

## REDUCED PROTOCOL (mandatory for the smoke — keep it tiny)

- `limit=64` subjects per bundle (loader `limit=` argument) — fast graph build.
- `epochs <= 5`, batch size 8, a single 80/20 train/val split (NO nested CV, NO Optuna).
- Small backbone: `hidden_channels=32, out_channels=16, num_layers=2`.
- Emit `cell_A1_val_r2`, `cell_A3_val_r2`, `cell_B1_val_r2`, `cell_C1_val_r2`, and a
  primary `val_r2:` line. Numbers will be noisy/meaningless — that is expected.

## Instructions — experiment-design stage (Stage 9)

`baselines:` → A1, A3. `proposed_methods:` → B1, C1. `ablations:` → empty.
`datasets:` → `pnc_sc400_age_reg`, `pnc_sc400_vwm_reg` only. `metrics:` → `val_r2`.
4 conditions total. Do not add cells or standard ML benchmark datasets.

## Instructions — code-generation stage (Stage 10)

Single self-contained `main.py` importing `scaffold.*`, reading `RC_DATASET_DIR`, using
`device=cuda if available`, implementing the 4 cells with the reduced protocol above,
training A1 → save → load into B1, and emitting the `cell_*_val_r2` + `val_r2` lines.
````

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_thesis_pinning.py -k smoke_matrix -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add docs/thesis/pnc-age-vwm-matrix-smoke.md tests/test_thesis_pinning.py
git commit -m "feat(thesis): pinned 4-cell smoke matrix (reduced protocol)"
```

---

## Task 4: Author `config.thesis.smoke.yaml` (laptop A4000 smoke)

**Files:**
- Create: `config.thesis.smoke.yaml`
- Test: `tests/test_thesis_pinning.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_thesis_pinning.py`)

```python
def test_smoke_config_runs_locally_and_pins_smoke_matrix():
    cfg = _load(CFG_SMOKE)
    s = cfg.experiment.scaffold
    assert s.enabled is True and "rc_brain_data" in s.dataset_dir
    assert cfg.experiment.benchmark_agent.enabled is False
    assert cfg.experiment.mode == "sandbox"
    extras = dict(cfg.prompts.extra_prompts)
    for stage in ("experiment_design", "code_generation"):
        assert stage in extras
        assert extras[stage].endswith("pnc-age-vwm-matrix-smoke.md")
    # the smoke must EXECUTE locally: execution stage is NOT HITL-gated
    assert not any(int(s) >= 11 for s in cfg.security.hitl_required_stages), \
        "smoke must run end-to-end locally (do not gate execution)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_thesis_pinning.py -k smoke_config -v`
Expected: FAIL — load error for `config.thesis.smoke.yaml`.

- [ ] **Step 3: Write `config.thesis.smoke.yaml`**

```yaml
project:
  name: "pnc-age-vwm-smoke"
  mode: "full-auto"

research:
  topic: >-
    SMOKE: within-PNC age->VWM transfer vs. age concatenation on a 400-node cortical
    Schaefer structural-connectivity GCN. Local validation of the pinned pipeline on a
    4-cell subset (A1, A3, B1, C1) with a reduced protocol — code-shape check, not real
    numbers.
  domains:
    - "machine-learning"
    - "graph-neural-networks"
    - "transfer-learning"
  daily_paper_count: 3
  quality_threshold: 3.0

runtime:
  timezone: "America/New_York"
  max_parallel_tasks: 2
  approval_timeout_hours: 12

notifications:
  channel: "console"
  target: ""
  on_stage_start: true
  on_stage_fail: true
  on_gate_required: true

knowledge_base:
  backend: "markdown"
  root: "docs/kb"

llm:
  provider: "acp"
  base_url: ""
  wire_api: "chat_completions"
  api_key_env: ""
  api_key: ""
  primary_model: "gpt-4o"
  fallback_models:
    - "gpt-4.1"
    - "gpt-4o-mini"

security:
  # No execution gate: the smoke MUST run end-to-end locally. (9 = design review only.)
  hitl_required_stages: [9]
  allow_publish_without_approval: true
  redact_sensitive_logs: true

experiment:
  mode: "sandbox"
  # Small budget is fine: only 4 conditions (≤ the cap of 8), reduced protocol.
  time_budget_sec: 900
  max_iterations: 2
  metric_key: "val_r2"
  metric_direction: "maximize"
  scaffold:
    enabled: true
    dir: "/home/compa/Documents/working_dir/ResearchClaude/AutoResearchClaw/experiment_scaffold"
    manifest: "SCAFFOLD.md"
    dataset_dir: "~/rc_brain_data"
    require: true
  sandbox:
    python_path: ".venv/bin/python3"
    gpu_required: true
    max_memory_mb: 8192
  benchmark_agent:
    enabled: false
  opencode:
    enabled: false

prompts:
  custom_file: ""
  extra_prompts:
    experiment_design: ./docs/thesis/pnc-age-vwm-matrix-smoke.md
    code_generation: ./docs/thesis/pnc-age-vwm-matrix-smoke.md
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_thesis_pinning.py -k smoke_config -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add config.thesis.smoke.yaml tests/test_thesis_pinning.py
git commit -m "feat(thesis): config.thesis.smoke.yaml — local 4-cell smoke config"
```

---

## Task 5: Prove the pin actually reaches both stages (PromptManager injection test)

**Files:**
- Test: `tests/test_thesis_pinning.py`

> This is the linchpin test: it builds a `PromptManager` exactly as the executor does
> (`executor.py:657`) from each config's `extra_prompts`, and asserts the matrix text is
> registered for **both** `experiment_design` and `code_generation` (i.e. the stage keys
> are spelled correctly — a typo would be silently dropped, `manager.py:190`).

- [ ] **Step 1: Write the failing test** (append to `tests/test_thesis_pinning.py`)

```python
from researchclaw.prompts import PromptManager


def _pm_from_config(cfg):
    """Mirror executor.py: build a PromptManager from config.prompts.extra_prompts."""
    return PromptManager(
        cfg.prompts.custom_file or None,
        domain="ml",
        extra_prompts={k: v for k, v in cfg.prompts.extra_prompts} or None,
    )


@pytest.mark.parametrize("cfg_path, cells", [
    (CFG_FULL, FULL_CELLS),
    (CFG_SMOKE, SMOKE_CELLS),
])
def test_pin_resolves_into_both_stages(cfg_path, cells):
    cfg = _load(cfg_path)
    pm = _pm_from_config(cfg)
    extras = pm.extra_prompts()          # resolved file texts, keyed by stage
    # Both stage keys resolved (not dropped as unknown, not left as bare paths)
    assert set(extras) >= {"experiment_design", "code_generation"}
    for stage in ("experiment_design", "code_generation"):
        body = extras[stage]
        assert len(body) > 500, f"{stage} extra looks unresolved (got {body!r})"
        for cell in cells:
            assert cell in body, f"{stage} pin missing cell {cell}"
        assert "scaffold" in body and "RC_DATASET_DIR" in body


def test_for_stage_appends_pin_under_additional_guidance():
    """The resolved pin is actually appended to the rendered code_generation prompt."""
    cfg = _load(CFG_SMOKE)
    pm = _pm_from_config(cfg)
    rendered = pm.for_stage(
        "code_generation",
        topic="t", metric="val_r2", pkg_hint="", exp_plan="",
    )
    assert "## Additional Stage Guidance" in rendered.user
    assert "load_fc_graphs" in rendered.user
    assert "A1" in rendered.user and "B1" in rendered.user
```

- [ ] **Step 2: Run test to verify it fails / passes**

Run: `.venv/bin/python -m pytest tests/test_thesis_pinning.py -k "pin_resolves or for_stage_appends" -v`
Expected: PASS if Tasks 1–4 landed (the configs + docs exist). If `for_stage` raises a
`KeyError` on a missing template variable, reduce the assertion to the
`pm.extra_prompts()` path only (the `for_stage` render needs whatever variables the ml
`code_generation` template declares; pass empty strings for any it complains about, or
drop `test_for_stage_appends...` and rely on `test_pin_resolves_into_both_stages`, which
is sufficient proof).

- [ ] **Step 3: Run the whole M2 test file + confirm no regressions elsewhere**

Run:
```bash
.venv/bin/python -m pytest tests/test_thesis_pinning.py -v
.venv/bin/python -m pytest tests/test_scaffold_config.py tests/test_rc_config.py tests/test_rc_prompts.py -q
```
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_thesis_pinning.py
git commit -m "test(thesis): prove matrix pin reaches design + codegen stages"
```

---

## Task 6: Local smoke run on the A4000 (the M2 acceptance gate)

**Files:**
- Run artifact (not committed): the ARC run dir under the project's runs directory.
- Create (notes, committed): `docs/thesis/SMOKE_RESULTS.md`

> **This task RUNS ARC** via the Claude Code/ACP backend — it is not a pytest. Follow the
> `researchclaw-claude-code-backend` memory (acpx 0.10.0, isolated `CLAUDE_CONFIG_DIR`,
> the `defaultMode:auto` gotcha) and `acp-compaction-doc`. Interactive login/launch steps
> should be run by the user via `! <command>` so their output lands in the session.

- [ ] **Step 1: Pre-flight — confirm substrate + config resolve on this box**

```bash
.venv/bin/python -c "import os; from researchclaw.experiment.scaffold import validate_scaffold; from researchclaw.config import load_config; c=load_config('config.thesis.smoke.yaml', check_paths=False); print('scaffold errors:', validate_scaffold(c.experiment))"
ls ~/rc_brain_data/pnc_sc400_age_reg.npz ~/rc_brain_data/pnc_sc400_vwm_reg.npz
.venv/bin/python -m pytest tests/test_sc400_bundles.py -q   # bundles still load via the scaffold loader
```
Expected: `scaffold errors: []`, both `.npz` present, bundle tests PASS.

- [ ] **Step 2: Launch the pinned smoke run via the ACP backend**

Use the working ACP recipe from the `researchclaw-claude-code-backend` memory (isolated
`CLAUDE_CONFIG_DIR`, `defaultMode` set appropriately). Drive the pipeline with
`config.thesis.smoke.yaml`. The run will: lit/synthesis/hypotheses (normal) → **Stage 9**
emits a 4-condition plan (A1, A3, B1, C1; datasets = the two bundles) → **Stage 10**
generates `main.py` importing `scaffold.*` → **Stage 12** executes it in the sandbox.
Suggest the user run the interactive launch with `! <command>` if a TTY login is needed.

- [ ] **Step 3: Inspect the generated experiment (the real acceptance check)**

In the run's `stage-10*/experiment/` dir, confirm the generated code:
```bash
RUN=<path-to-run-dir>
grep -RnE "from scaffold|import scaffold" "$RUN"/stage-10*/experiment/*.py
grep -RnE "load_fc_graphs|GraphRegressor|save_backbone|load_backbone|RC_DATASET_DIR" "$RUN"/stage-10*/experiment/*.py
ls "$RUN"/stage-10*/experiment/scaffold/      # materialized scaffold/ package present
```
Expected: imports come from `scaffold.*` (NOT re-implemented), `RC_DATASET_DIR` is read,
the cells use the pinned primitives, and `scaffold/` was materialized alongside `main.py`.
**If the generated code re-implements the GCN/loader instead of importing `scaffold.*`,
the pin is too weak** — strengthen the `## Instructions — code-generation stage` section
in the smoke matrix doc (make "import only from `scaffold/`; do not re-implement" more
emphatic) and, if needed, set `experiment.code_agent.enabled: false` in the smoke config
to force the legacy single-shot path (which renders `for_stage("code_generation")`
directly), then re-run.

- [ ] **Step 4: Confirm execution emitted a metric**

```bash
grep -RnE "cell_(A1|A3|B1|C1)_val_r2|^val_r2:" "$RUN"/stage-12*/   # or wherever exec logs land
```
Expected: per-cell `cell_*_val_r2` lines + a `val_r2:` line. Values are noisy/meaningless
(reduced protocol) — **the gate is "runs forward, transfers a checkpoint, emits a
correct-shaped metric," not the numbers.** Confirm the A1→B1 transfer path ran (no
`load_backbone` shape/strict error in the exec log).

- [ ] **Step 5: Record the outcome** (`docs/thesis/SMOKE_RESULTS.md`)

Write a short note: date, run dir, which cells generated + executed, whether the code
imported `scaffold.*` as-is, the transfer path result, the emitted metric lines, and any
prompt tweak that was needed to keep the pin tight. This is the M2 evidence artifact.

```bash
git add docs/thesis/SMOKE_RESULTS.md docs/thesis/pnc-age-vwm-matrix-smoke.md
git commit -m "docs(thesis): M2 smoke results — pinned pipeline emits runnable code"
```

---

## Task 7: Full-config code-gen dry-run (generate 11-cell code, do NOT execute)

**Files:**
- Run artifact (not committed): the full-config run dir, paused at the execution gate.

> Optional-but-recommended: proves the *full* pin (not just the smoke subset) yields
> correct-shaped 11-cell code that M3 can lift onto the cluster. The full nested CV is NOT
> run here — `config.thesis.yaml` gates the execution stage, so ARC pauses after Stage 10.

- [ ] **Step 1: Run ARC with `config.thesis.yaml` up to the execution gate**

Launch as in Task 6 Step 2 but with `config.thesis.yaml`. ARC runs through Stage 10 and
**pauses at the HITL execution gate** (stage 12). Do not approve execution — the full
matrix is the cluster's job (M3).

- [ ] **Step 2: Verify the 11-cell generated code is correct-shaped**

```bash
RUN=<full-run-dir>
grep -oE "cell_(A[1-5]|B[1-4]|C[12])_val_r2" "$RUN"/stage-10*/experiment/*.py | sort -u   # expect 11
grep -RnE "save_backbone|load_backbone\(.*strict=True" "$RUN"/stage-10*/experiment/*.py    # transfer wiring
```
Expected: all 11 `cell_*_val_r2` emitters present; A1/A4 saved and B1/B2/B3/B4/C2 load via
`strict=True`; A5 uses `age_vwm_baseline`; C1 uses `global_dim=1`, C2 `global_dim=400`.
Stash this generated `experiment/` (it is the M3 cluster payload) — note its path in the
M3 handoff.

- [ ] **Step 3 (no commit — run artifacts are not committed).** Capture the run-dir path
in the Task 8 memory/handoff update.

---

## Task 8: Update memory + write the M3 handoff

**Files:**
- Modify: `~/.claude/projects/.../memory/pnc-age-vwm-thesis-effort.md` + `MEMORY.md` pointer.

- [ ] **Step 1: Update the project memory** — append an "M2 (ARC-pinned) DONE" paragraph:
the two configs (`config.thesis.yaml` full / `config.thesis.smoke.yaml` smoke), the single
matrix doc injected into both stages (`docs/thesis/pnc-age-vwm-matrix.md` + the smoke
variant), the two gotchas defused (`time_budget_sec>3600` for the 11-condition cap;
`benchmark_agent.enabled:false`), the smoke result, and the stashed full 11-cell generated
`experiment/` path for M3. Keep it one tight paragraph; link `[[gnn-scaffold-dataset]]`,
`[[researchclaw-claude-code-backend]]`.

- [ ] **Step 2: Verification before "done"** (superpowers:verification-before-completion)

```bash
.venv/bin/python -m pytest tests/test_thesis_pinning.py tests/test_scaffold_config.py tests/test_sc400_bundles.py -v
git log --oneline -8
git status
```
Expected: all M2 tests green; commits present; only intended files changed (leave the
pre-existing untracked files — `config.gnn.yaml`, `frontend`, etc. — alone).

- [ ] **Step 3:** Offer to write the M3 handoff (cluster execution) via the `handoff`
skill, mirroring this handoff's structure: M2 artifacts as the substrate, the stashed
11-cell `experiment/`, the `cluster-helper` skill, and the decoupled ARC↔cluster handoff.

---

## Self-Review

**Spec coverage (handoff §"M2 goal & likely deliverables"):**
- #1 `config.thesis.yaml` pinning scaffold + bundles + 11-cell matrix, design/codegen
  hard-constrained → **Task 2** (+ the matrix doc it injects, **Task 1**).
- #2 thesis seed / design artifact ARC ingests → **Task 1** (`pnc-age-vwm-matrix.md`,
  injected into Stage 9 + Stage 10; it *is* the seed and the pin — see "Why one doc").
- #3 per-stage pinning prompts for `_experiment_design` / `_code_generation` → **Task 1**
  (the matrix doc's two `## Instructions —` sections) wired via `extra_prompts` in
  **Tasks 2 & 4**, proven reaching both stages in **Task 5**.
- #4 tiny local A4000 smoke (couple of cells, few epochs; proves imports scaffold + loads
  a bundle + trains forward + emits a result; NOT full nested CV) → **Tasks 3, 4, 6**.
- Decoupled cluster handoff (§0) → full config gates execution (**Task 2**); 11-cell code
  generated but not run (**Task 7**); M3 handoff (**Task 8**).

**Placeholder scan:** every config + doc is shown in full; every test step has complete
code and an expected result; the one parameter to confirm at runtime (the execution stage
number, expected 12) has an explicit command in Task 2 Step 1 and a fallback grep.

**Type/name consistency:** stage keys `experiment_design` / `code_generation` used
identically in the configs, the matrix docs, and Task 5; cell IDs A1–A5/B1–B4/C1/C2 and
scaffold symbols (`load_fc_graphs`, `GraphRegressor`, `save_backbone`/`load_backbone`/
`freeze`, `age_vwm_baseline`, `glm_diagonal`, `graph_feature_key`) and bundle names
(`pnc_sc400_age_reg`, `pnc_sc400_vwm_reg`) match the M1 plan + `SCAFFOLD.md`; metric
`val_r2` / `maximize` consistent across configs, docs, and tests.

**Two known runtime checks (flagged, not placeholders):** (a) `for_stage` render in Task 5
needs whatever template variables the ml `code_generation` prompt declares — Task 5 Step 2
gives the fallback (rely on `pm.extra_prompts()`); (b) if the CodeAgent re-implements
instead of importing `scaffold.*`, Task 6 Step 3 gives the remediation (tighten the pin /
force the legacy path). Both are real validation branches, not gaps.

**Deferred (not M2):** D1 @node surgery; the full nested-CV cluster run, `.sif`, SLURM DAG
(M3); analysis + paper vs registered predictions (M4).
```