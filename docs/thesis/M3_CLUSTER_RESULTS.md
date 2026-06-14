# M3 Cluster Results — PNC age→VWM matrix

Evidence artifact for Milestone 3 (cluster execution). Sections are filled as M3
progresses: (1) the local harness smoke gate, (2) the launched cluster jobs, (3) the
fetched per-cell `val_r2` table vs the registered predictions.

---

## 1. Local harness smoke (acceptance gate) — 2026-06-14

Proves the full per-fold pipeline runs end-to-end on real GPU data **before** any cluster
spend, mirroring M2's reference smoke. Run on the local RTX A4000 under the ARC `.venv`
(torch 2.11.0+cu128, torch_geometric 2.8.0, optuna 4.9.0), `RC_DATASET_DIR=~/rc_brain_data`,
realized common **N=743** subjects.

Command (one fold, reduced `--smoke` scale = limit 24/32, 2 epochs, 2 inner trials):

```bash
OUT=$(mktemp -d); export RC_DATASET_DIR=~/rc_brain_data
for s in A1 A4; do .venv/bin/python -m cluster.pretrain_source --cell $s --rep 0 --outer 0 --out "$OUT" --smoke; done
for c in A2 A3 B1 B2 B3 B4 C1 C2 A5; do .venv/bin/python -m cluster.train_cell --cell $c --rep 0 --outer 0 --out "$OUT" --source-dir "$OUT" --smoke; done
```

### Emitted metric lines (all 11 cells)

```
cell_A1_val_r2: -0.0667  ckpt=.../ckpts/A1_rep0_outer0.pt      # source (identity → age)
cell_A4_val_r2: -0.0662  ckpt=.../ckpts/A4_rep0_outer0.pt      # source (glm_diagonal → age)
cell_A2_val_r2: -0.0246  (inner_val -0.0788)                   # scratch, identity
cell_A3_val_r2: -0.0240  (inner_val -0.0784)                   # scratch, glm_diagonal (baseline)
cell_B1_val_r2: -0.0222  (inner_val -0.0755)                   # finetune  A1 → VWM
cell_B2_val_r2: -0.0212  (inner_val -0.0743)                   # frozen    A1 → VWM
cell_B3_val_r2: -0.0223  (inner_val -0.0759)                   # finetune  A4 → VWM
cell_B4_val_r2: -0.0216  (inner_val -0.0753)                   # frozen    A4 → VWM
cell_C1_val_r2: -0.5035  (inner_val -0.0844)                   # scratch + age @head  (global_dim=1)
cell_C2_val_r2:  0.0202  (inner_val  0.0354)                   # finetune A1 + GLM @head (global_dim=400)
cell_A5_val_r2: -0.0469                                        # trivial age→VWM OLS floor
```

### Gate result: PASS (mechanics, not numbers)

The R² values are **meaningless at `--smoke` scale** (24 graphs, 2 epochs, 2 trials) — exactly
as the M2 reference smoke. The gate is that every mechanism runs forward without error:

- **11/11 cells** emit a `cell_*_val_r2:` line and write a result JSON to `$OUT/results/`.
- **Transfer chains load clean** (no `load_backbone` strict / shape errors): A1 → {B1 finetune,
  B2 frozen, C2 finetune+GLM@head}; A4 → {B3 finetune, B4 frozen}. No "missing source ckpt"
  assert fired — the per-fold checkpoint keying `(cell, rep, outer)` resolves.
- **Head wiring runs**: C1 (age @head, `data.u` U=1, `global_dim=1`) and C2 (GLM @head,
  `data.u` U=400, `global_dim=400`).
- **From-scratch + trivial run**: A2/A3 from-scratch, A5 trivial OLS floor.
- **No `carrier=` TypeError** — the loader is called with the exact scaffold API
  (`node_features=` / `node_feature_key=`+`glm_diagonal=` / `graph_feature_key=`).

Harness unit suite (`tests/test_cluster_harness.py`) is green under the ARC venv, including
the no-leakage guard (`train_eval` early-stops on a val split and scores fold-test once;
inputs are not mutated) and the subject-aligned within-subject fold partition.

---

## 2. Launched cluster jobs

_(filled at Task 11 — source array + dependent cell array job ids, node, container tag)_

---

## 3. Fetched per-cell results vs registered predictions

_(filled at Task 12 — per-cell mean±std `val_r2` table + falsifier checks)_
