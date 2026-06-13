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

> **EXACT API — `load_fc_graphs` has NO `carrier=`/`mode=` argument.** Select the carrier
> only via `node_features=` (identity) or `node_feature_key=`+`glm_diagonal=` (glm_diagonal),
> exactly as shown below. A made-up kwarg like `carrier="identity"` raises
> `TypeError: load_fc_graphs() got an unexpected keyword argument 'carrier'` and the run
> fails. If you write a `_load(cell, carrier=...)` helper, translate `carrier` to the real
> kwargs INSIDE it; never forward a `carrier` string to `load_fc_graphs`. Other signatures:
> `build_gcn(in_channels=400, hidden_channels, out_channels, num_layers=2, norm="batch_norm")`;
> `GraphRegressor(backbone, pooled_dim=P, global_dim=0|1)`, `forward(x, edge_index, edge_weight, batch, u=None)`;
> `transfer.save_backbone(backbone, path)` / `load_backbone(backbone, path, strict=True)`.

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
