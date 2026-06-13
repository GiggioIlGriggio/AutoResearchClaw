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
**GLM** = `glm_diagonal` (per-node 2back-vs-0back z-map, `glm_normalize=True`).
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
