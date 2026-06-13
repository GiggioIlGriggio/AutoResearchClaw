# M2 Smoke Results — PNC age→VWM ARC pinning

> Date: 2026-06-13/14. Branch `thesis/pnc-age-vwm`. Validates that the pinned ARC
> pipeline (`config.thesis.smoke.yaml` + `docs/thesis/pnc-age-vwm-matrix-smoke.md`)
> reproduces the 4-cell smoke subset {A1, A3, B1, C1} on the M1 scaffold primitives and
> that the recipe trains forward + transfers + emits metrics. Acceptance gate is
> **mechanics** (runs forward, transfers a checkpoint, emits correct-shaped metrics),
> **not** real numbers — the reduced protocol (64 subjects, 3 epochs, tiny backbone,
> no HPO) cannot produce meaningful R².

## Run 1 — ARC full pipeline (`rc-20260613-140125-27541c`)

Command: `CLAUDE_CONFIG_DIR=~/.researchclaw-acp-home .venv/bin/researchclaw run --config
config.thesis.smoke.yaml --auto-approve --mode full-auto --to-stage EXPERIMENT_RUN`.
Result: **12/12 stages done, 0 failed** (~47 min).

**The pin worked — strongly:**
- **Stage 9 (design)** emitted `REGISTERED_CONDITIONS` = exactly the 4 pinned cells with
  the correct carrier/target/role/global_dim each: A1 (identity→age, source_backbone,
  gd0), A3 (glm_diagonal→VWM, from_scratch, gd0), B1 (identity→VWM, transfer_finetune
  from A1, gd0), C1 (glm_diagonal→VWM, age @head, gd1). `METRIC_DEF val_r2 maximize`.
- **Stage 10 (codegen)** imported the scaffold primitives as-is (`load_fc_graphs`,
  `GraphRegressor`, `transfer.save_backbone`/`load_backbone`), read `RC_DATASET_DIR`,
  materialized `scaffold/`, wired the A1→B1 checkpoint transfer — it did **not**
  re-implement the model or pull in standard ML benchmarks.

→ **The core M2 thesis is validated:** pinning makes ARC's autonomous design+codegen
reproduce the matrix on the trusted scaffold instead of inventing its own model.

**But execution failed on a codegen API-fidelity bug.** The generated `main.py` invented a
`load_fc_graphs(carrier="identity"/"glm_diagonal")` kwarg — conflating the prose term
"carrier" with a real parameter — which the loader does not accept:
`CELL_ERROR cell=A1: load_fc_graphs() got an unexpected keyword argument 'carrier'`
→ the per-cell error sentinel became a non-numeric `val_r2` → `TypeError: must be real
number, not str` → Stage 12 failed in ~1.8s (no training).

## Fix — harden the pin (commit `16fe68a`)

Added an explicit **exact-API contract** to both matrix docs (`pnc-age-vwm-matrix.md` and
`-smoke.md`): real signatures for every scaffold symbol, an explicit
"`load_fc_graphs` has NO `carrier=`/`mode=` argument" prohibition, and an instruction
that any per-cell carrier helper must translate to the real
`node_features=` / `node_feature_key=`+`glm_diagonal=` kwargs. Hardens the pin for the
M3 cluster run too (where the same bug would crash all 11 cells).

## Run 2 — ARC re-run with hardened pin (`rc-20260613-223953-27541c`)

Same command. Result: **FAILED at Stage 10 (1224s)** with a **transient ACP backend
flake** — `acp_client.py:482 RuntimeError: ACP prompt failed (exit 1): [acpx] session …
agent connected`. The acpx subprocess exited 1 mid-codegen with only the connection
banner as stderr; **no `experiment/` was generated.** This is an infrastructure flake on
the long codegen ACP interaction (not a regression from the doc fix — no code reached
validation). The hardened-pin codegen output was therefore not produced this run.

## Reference smoke — deterministic substrate validation (`scripts/smoke_thesis_cells.py`)

To verify the recipe + corrected loader API end-to-end without depending on ARC's
ACP-flaky codegen, the *correct* 4-cell experiment (the reference implementation of what
the smoke pins ARC to generate, using the real scaffold API) was run on the real bundles:

```
RC_DATASET_DIR=~/rc_brain_data .venv/bin/python scripts/smoke_thesis_cells.py
device=cuda  RC_DATASET_DIR=/home/compa/rc_brain_data
cell_A1_val_r2: -0.0623
cell_A3_val_r2: -0.0113
cell_B1_val_r2: -0.0138
cell_C1_val_r2: -0.0048
val_r2: -0.0048
SMOKE_OK: 4/4 cells trained forward, A1->B1 transfer ran, metrics emitted
```

**Validated deterministically on the A4000 (cuda):**
- all 4 cells train forward on the real SC-400 bundles (`limit=64`, 3 epochs);
- identity and `glm_diagonal` carriers both build (400-dim node features);
- the A1→B1 transfer path runs clean (`save_backbone` → `load_backbone(strict=True)`);
- the age `data.u` @head side channel works (C1, `global_dim=1`);
- correct-shaped `cell_*_val_r2` + `val_r2` metrics emit. (Near-zero R² is expected for
  the reduced protocol — mechanics gate, not performance.)

## Status

| Claim | Status |
|---|---|
| ARC design+codegen reproduce the pinned matrix on the scaffold (don't re-invent) | ✅ Run 1 |
| Recipe + substrate train forward + transfer + emit metrics on real GPU data | ✅ reference smoke |
| Loader-API fidelity bug found and hardened against in both matrix docs | ✅ commit `16fe68a` |
| ARC's *hardened* autonomous codegen emits the correct loader API end-to-end | ⏳ pending — Run 2 hit a transient acpx flake before codegen completed |

**For M3:** the same ACP codegen path generates the full 11-cell code locally before the
cluster handoff. Inspect that generated code for the `carrier=`/loader-API issue (and
re-run codegen if acpx flakes) **before** launching the 11-cell cluster job. The reference
script here is the ground-truth the generated code should match.
