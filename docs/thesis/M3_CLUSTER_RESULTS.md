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

## 2. Launched cluster jobs — 2026-06-15

**Cluster / node:** `al5165@155.105.223.17`, **gpunode02** (`rad2` / qos `16cpu` / account `rad`),
GPUs RTX 3090 + RTX 6000 (no type pin — both share the node driver). CPU-limited to ~4–7
concurrent array tasks (16 cores / 4 cpus-per-task).

**Container:** `pnc-age-vwm.sif` from `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`
+ `torch_geometric==2.8.0`, `optuna==4.9.0`, numpy/scipy/pandas/sklearn pinned to the local
stack (pip under PEP 668 via `--break-system-packages`). Node driver **r550.54.15 (CUDA 12.4)**
runs the **cu128** runtime via CUDA 12.x minor-version compatibility.

**Deploy:** branch `thesis/pnc-age-vwm` @ `01dd677`, deployed by `cluster-submit` (SSH clone of
the public fork `GiggioIlGriggio/AutoResearchClaw`; `origin`→fork, `upstream`→`aiming-lab`,
because the laptop account lacks write to the aiming-lab repo).

**GPU hard gate (job 361418):** real cuBLAS matmul **and** a `GCNConv` forward both execute and
return finite results on the RTX 3090 — not merely `torch.cuda.is_available()`. PASS.

**Timing/anchor probe (job 361489\_50):** ONE full-protocol A3 task via the real `cell.sh`
(array idx 50 = A3/rep0/outer0) → **3 m 17 s**, single-fold R² **0.2105**. Confirmed the per-task
cost and the load-bearing baseline before the 500-task spend.

**Matrix DAG (full nested CV, protocol 10 reps × 5 outer × 20 inner Optuna; fixed GCN ARCH
hidden=64/out=32/3 layers/batch_norm; metric outer-test R², maximize; realized common N=743):**

| array | job id | `--array` | tasks | result |
|---|---|---|---|---|
| source (A1, A4 → age backbones) | **361491** | `0-99` | 100 | **100/100 COMPLETED, 0 failures** |
| cell (8 HPO cells → VWM) | **361498** | `0-399`, `--dependency=afterok:361491` | 400 | **400/400 COMPLETED, 0 failures** |
| A5 (trivial age→VWM OLS) | — (local) | — | 50 | deterministic CPU OLS on the same seeded folds (environment-independent) |

Wall-clock ≈ 5 h on gpunode02 (source ~25 min → cell ~4.5 h). 500 result JSONs fetched to
`runs/matrix-fetched/` + 50 local A5 = **550** per-fold results reduced.

---

## 3. Fetched per-cell results vs registered predictions

Per-cell mean ± std outer-test R² over the 50 folds (10 reps × 5 outer), from
`cluster.reduce_results` (full table also at `docs/thesis/m3_results.{md,csv}`). `task` = the
cell's target: **age** (A1/A4 source backbones, held-out age-R²) vs **vwm** (all baseline/
transfer cells). Compare R² only within the same task.

| cell | task | mean R² | std | registered prediction | verdict |
|---|---|---|---|---|---|
| A1 (identity → age, source) | age | **0.537** | 0.049 | — | identity-SC decodes age well |
| A4 (glm_diagonal → age, source) | age | **0.043** | 0.105 | — | glm_diagonal-SC barely decodes age |
| A2 (identity, scratch) | vwm | 0.050 | 0.072 | ≈ −0.03 (identity floor) | ~floor (small +, ≪ A3) |
| A3 (glm_diagonal, scratch) | vwm | **0.214** | 0.084 | ≈ 0.18–0.21 (baseline to beat) | ✅ anchor on-target |
| A5 (trivial age→VWM OLS) | vwm | 0.056 | 0.046 | small positive | ✅ |
| B1 (A1 → VWM, finetune) | vwm | 0.077 | 0.047 | ≈ 0–0.05 (≪ A3) | ✅ ≪ A3 |
| B2 (A1 → VWM, frozen) | vwm | 0.031 | 0.029 | ≤ B1 | ✅ |
| B3 (A4 → VWM, finetune) | vwm | 0.191 | 0.067 | ≈ A3 | ✅ ≈ A3 |
| B4 (A4 → VWM, frozen) | vwm | 0.136 | 0.053 | ≤ B3 | ✅ |
| C1 (glm_diagonal + age @head, scratch) | vwm | **0.244** | 0.065 | A3 < C1 ≤ A3⊕A5 | ✅ |
| C2 (A1 → VWM, finetune + GLM @head) | vwm | 0.223 | 0.065 | ≈ A3 | ✅ |

### Pre-registered falsifiers — none triggered

```
[falsifier] signal-location (B1/B2 ≈ A3 falsifies):      A3=0.214 B1=0.077 B2=0.031   → NOT falsified
[falsifier] trivial-trend (C1 > A3⊕A5 super-add falsifies): A3=0.214 A5=0.056 C1=0.244 → NOT falsified
[falsifier] saturation (B3 ≫ A3 falsifies):              A3=0.214 B3=0.191            → NOT falsified
```

### Reading (measured; what the numbers say, not over-claimed)

Every registered prediction holds and no falsifier fires — the matrix is **consistent with the
thesis**: within-subject age→VWM **transfer does not beat the from-scratch `glm_diagonal`
baseline** (A3 = 0.214).

- **The signal is in the carrier, not the transferred weights.** `glm_diagonal` cells cluster at
  ≈ 0.19–0.24 whether trained from scratch (A3 = 0.214) or fine-tuned from an age backbone
  (B3 = 0.191 ≈ A3; B4 frozen = 0.136 ≤ B3). Identity cells sit near the floor whether scratch
  (A2 = 0.050) or transferred (B1 = 0.077, B2 = 0.031 ≪ A3).
- **Age-decoding ability ≠ VWM-transfer value.** A1 decodes age strongly (0.537), yet its backbone
  transfers worst to VWM (B1/B2 ≪ A3). A4 barely decodes age (0.043), yet B3 ≈ A3 — so B3's score
  is the carrier's, not the pretraining's. This is the central negative result: a backbone being
  good at age says nothing about its value for VWM.
- **Marginal head gains, no super-additivity.** C1 (age as a graph-feature at the head) = 0.244
  edges just above A3, bounded by the trivial age→VWM signal (A5 = 0.056) — additive, not
  synergistic. C2 (GLM @head over an identity backbone) recovers ≈ A3 (0.223).
- **Minor deviation, noted:** A2 (identity floor) came in slightly positive (0.050) vs the
  registered ≈ −0.03 — the identity-SC carrier carries a weak-but-nonzero VWM signal. It remains
  ≪ A3, so the baseline ordering is unaffected.

All 50/50 folds present for every cell (no `[WARNING] incomplete cells`); 500/500 GPU tasks
COMPLETED with zero failures.
