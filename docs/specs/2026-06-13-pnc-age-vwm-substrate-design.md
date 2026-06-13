# PNC Age→VWM Transfer-vs-Concatenation — Substrate (Milestone 1) Design Spec

> Status: approved (design). Date: 2026-06-13. Author: Alessio Comparini + Claude.
> Scope: a single implementation plan (Milestone 1 of a 4-milestone effort).
> Terminal step: `writing-plans`.

## 0. Context — the larger effort this milestone belongs to

We are implementing, **from scratch using AutoResearchClaw (ARC) as the vehicle**, the
thesis `pnc-age-vwm-transfer-vs-concatenation` — a pre-registered 12-cell experiment
matrix asking whether age→VWM *transfer* or age/GLM *concatenation* beats a from-scratch
GLM-feature model on PNC, with a fixed-topology identity-encoded GCN. The old codebase
(`LLM_codebase`) is **not** being continued; only its cluster infrastructure pattern and
its baseline numbers (as reference bands) carry over.

**Decisions locked for the whole effort** (drive this and later milestones):

| Axis | Decision |
|---|---|
| Reconciliation with ARC | **Pin ARC's codegen to the thesis** — ARC runs lit/synthesis/design/codegen/analysis/paper, but the design & code-generation stages are hard-constrained (config + per-stage prompts) to reproduce the exact matrix on top of trusted scaffold primitives. |
| Compute | **Cluster** (`gpunode02`, account `rad2`/QoS `16cpu`) via the `cluster-helper` skill. Laptop A4000 is used only for a tiny **smoke** validation of generated code. |
| ARC↔cluster seam | **Decoupled handoff**: ARC pauses at execution (HITL); we drive cluster-helper; results flow back into analysis. (No `cluster_sandbox` ARC backend — deferred.) |
| Cohort | **PNC only.** (ORBIT replication out of scope.) |
| Matrix scope | **Full minus D1 = 11 cells**: A1–A5, B1–B4, C1, C2. D1 (GLM @node surgery) deferred. |
| Graph topology | **Structural connectivity (SC), 400 cortical Schaefer ROIs. Functional connectivity is not used at all.** |
| CV protocol | Cluster makes the thesis's full nested CV (10 reps × 5 outer × 20 inner Optuna) feasible; reduced protocol for local smokes. Final protocol confirmed in M3. |

**Milestones:** **M1 — Substrate (this spec).** M2 — ARC pinned (config + prompts + smoke).
M3 — Cluster execution (project + dataset upload + `.sif` + SLURM DAG + full nested CV).
M4 — Analysis + paper vs. registered predictions.

## 1. Problem

The thesis matrix needs three things the current scaffold + baked dataset do not provide:

1. **No VWM target is baked.** All 12 existing bundles target age/sex. The thesis target
   is `VWM_overall_dprime` (confirmed present in the raw PNC scores: 1,454 labelled
   subjects). The A5/C1 cells additionally need **chronological age co-stored** per subject.
2. **The baked SC is the wrong node set.** The `schaefer400` structural matrix in the raw
   `.mat` files is **452×452** (Schaefer-400 cortical + 52 subcortical/cerebellar). The
   existing `pnc_sc_*` bundles bake that 452-node graph and target age/sex. The thesis
   carrier `glm_diagonal` (and the GLM zmap) live on the **400 cortical** Schaefer ROIs, so
   the matrix must run on a **400-node cortical SC** graph that does not exist yet.
3. **No transfer / side-channel / glm_diagonal machinery.** The scaffold has a GCN backbone
   factory and FC/SC/GLM loaders, but no checkpoint transfer, no frozen-vs-finetune control,
   no @head side-channel concatenation, no `glm_diagonal` node-feature builder, and no
   trivial age→VWM baseline.

These are precisely the subtle pieces that should **not** be left to autonomous codegen —
hence "pin ARC's codegen to thesis-validated primitives."

## 2. Goal

Build and unit-test, locally and deterministically, the full data + model substrate for the
11-cell matrix on an **SC-400 fixed topology** with **identity / glm_diagonal** carriers, so
that M2's pinned codegen only has to *wire* primitives, not invent them.

## 3. Datasets to bake (extend `scripts/bake_scaffold_dataset.py`)

Two new bundles, both on the **400-node cortical SC** graph. Named `sc400` to avoid colliding
with the existing 452-node `pnc_sc_*` bundles (which stay untouched).

| New bundle | Graph (`sc`) | `y` (`task`) | Extra stored arrays | Used by |
|---|---|---|---|---|
| `pnc_sc400_age_reg` | SC `(N,400,400)` | age (regression) | `glm_2back_vs_0back (N,400)`, `subject_id` | A1 (id→age), A4 (glm→age) — *source checkpoints* |
| `pnc_sc400_vwm_reg` | SC `(N,400,400)` | `VWM_overall_dprime` (regression) | `glm_2back_vs_0back`, **`age (N,)`**, `subject_id` | A2, A3, B1–B4, C1, C2 |

Bake rules:

- **SC-400 = the 400 cortical Schaefer parcels** subset out of the 452-node
  `schaefer400_sift_invnodevol_radius2_count_connectivity` matrix, dropping the 52
  subcortical/cerebellar regions identified via `schaefer400_region_labels`.
- **HARD GATE (first build step).** Confirm the subset yields **exactly 400** regions and
  that their ordering is **identical** to the GLM zmap's 400-node ordering (same Schaefer-400
  atlas). If the 400↔400 identity does not hold cleanly, **stop and surface it** — do not
  silently scramble `glm_diagonal` (a misaligned diagonal would invalidate every GLM cell).
- The `age_reg` bundle spans all SC-400 subjects with a valid age (~780). The `vwm_reg`
  bundle is the SC-400 ∩ VWM-labelled subset; `age` is joined onto it by `subject_id` from
  the cleaned scores CSV (same cleaning as the existing PNC bake: age coerced numeric, `<5`
  dropped).
- Edge weights are large positive streamline counts → loaders must use
  `edge_weight_norm="abs_max"` for the GCN (already the documented SC default).
- Re-bake recipe & venv unchanged from `gnn-scaffold-dataset` memory (full torch+pyg+pandas+
  scipy stack; bake via the system-python symlink venv).

## 4. Scaffold module additions

All additions are **flat top-level `*.py`** in `experiment_scaffold/` (the scaffold contract
only sees `glob("*.py")`; nested packages are invisible to preflight/guidance).

### 4.1 `fc_graph.py` (extend)
- `glm_diagonal_features(glm_vec, normalize=True)` → `(R, R)`: `diag(z(glm_vec))`, i.e. each
  node's feature is a length-R vector that is zero except at its own index, where it carries
  that node's GLM z-value. `normalize=True` z-scores the GLM scalar across nodes per subject
  (the thesis's `glm_normalize=true`). This is the `glm_diagonal` carrier; `identity`
  (`eye(R)`) already exists.
- Threaded through `fc_to_data` / `fc_arrays_to_data_list` via a `node_feature_mode` option
  (`"identity" | "glm_diagonal"`), built from an attached glm array.

### 4.2 `data_loader.py` (extend `load_fc_graphs`)
- `graph_feature_key=<bundle array>` → attaches a per-graph **side-channel** vector as
  `data.u` (shape `[1, U]`, batched by PyG to `[B, U]`): `age (N,)`→`U=1` (C1);
  `glm_2back_vs_0back (N,400)`→`U=400` (C2).
- `node_feature_mode="glm_diagonal"` wiring: pulls the bundle glm array and builds the
  diagonal carrier (overrides `node_features`).

### 4.3 `heads.py` (new)
- `class GraphRegressor(nn.Module)`: `backbone (GCN) → global_mean_pool → [optional MLP
  encode of data.u] → concat → linear head → scalar [B]`. Constructor:
  `GraphRegressor(backbone, in_dim, pool="mean", global_dim=0, global_hidden=32, head_hidden=64)`
  (the two hidden sizes are sensible defaults, HPO-swept in M3). `global_dim=0` → plain graph
  regressor (A1–A4, A2/A3, B*); `global_dim=1` → age @head (C1); `global_dim=400` → GLM @head
  (C2), where `u` is the 400-d GLM scalar zmap (not the diagonal). `forward(x, edge_index,
  edge_weight, batch, u=None)`.

### 4.4 `transfer.py` (new)
- `save_backbone(backbone, path)` / `load_backbone(backbone, path, strict=True)` — transfer
  just the GCN backbone `state_dict`. Input dim is **400 for both identity and glm_diagonal**,
  so all 11 cells load `strict=True` cleanly (only the deferred D1 changes input dim).
- `freeze(module)` / `set_trainable(module, flag)` — for the frozen modes (B2, B4): backbone
  params frozen, head trains.

### 4.5 `baselines.py` (new)
- `age_vwm_baseline(age, vwm)` — the A5 trivial age→VWM regressor (1-feature linear/ridge, no
  graph). Returns fitted model + CV R² so A5 is a first-class, reproducible floor.

## 5. Tests (`tests/test_scaffold_modules.py`, existing materialize-as-`scaffold/` pattern)

- `glm_diagonal_features`: shape `(R,R)`, diagonal carries z-scored glm, off-diagonal zero;
  `normalize` toggles z-scoring.
- `data.u` attaches at `U=1` and `U=400`; batches to `[B, U]`.
- `GraphRegressor` forward is finite with `global_dim ∈ {0, 1, 400}`; output shape `[B]`.
- **Checkpoint round-trip**: save a backbone from a fake "age" model → `load_backbone` into a
  fresh "VWM" model → assert weights equal; `freeze` → assert backbone params have
  `requires_grad=False` and receive no grad while the head does.
- `age_vwm_baseline` fits and returns a finite R² on synthetic age/vwm.

All tests run under the ARC `.venv` (torch+pyg), `importorskip` so a torch-less box still
collects the suite.

## 6. Out of scope (deferred to later milestones)
- **D1** (GLM @node, first-layer input-dim extension + differential LR + zero-init new columns).
- **M2**: `config.thesis.yaml`, thesis seed artifact, per-stage pinning prompts, local smoke run.
- **M3**: cluster project + `.cluster-helper.yaml`, dataset upload, `.sif` container, SLURM
  matrix DAG (pretrain→checkpoint→dependent cells), full nested CV.
- **M4**: ARC analysis + paper writing, comparison to registered predictions/falsifiers.

## 7. Risks & open items
- **R1 — SC-400 / GLM alignment (gate in §3).** Highest risk; retired before any bundle is
  written. Mitigation: explicit 400↔400 label-ordering assertion; stop-and-flag on mismatch.
- **R2 — VWM ∩ SC-400 sample size.** ~1,454 VWM-labelled ∩ ~780 SC-imaged → the `vwm_reg`
  bundle N may land well below the FC-based ~940. Record the realized N in `MANIFEST.json`;
  it informs M3 power, not M1 correctness.
- **R3 — `glm_diagonal` 400-dim node features → 400-dim backbone input.** Large but tractable
  on gpunode02; confirm the GCN trains (not just forward-finite) during M2 smoke, not M1.

## 8. Definition of done (Milestone 1)
1. `pnc_sc400_age_reg` and `pnc_sc400_vwm_reg` baked to `~/rc_brain_data` with the §3 contract;
   `MANIFEST.json` updated; realized Ns recorded.
2. SC-400/GLM alignment gate passed and documented.
3. `fc_graph` / `data_loader` extensions + `heads.py` / `transfer.py` / `baselines.py` added.
4. New tests green under the ARC `.venv`; existing scaffold tests still green.
5. `SCAFFOLD.md` updated to document the new bundles and primitives.
