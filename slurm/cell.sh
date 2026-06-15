#!/bin/bash
# Dependent-cell array: 8 HPO cells (A2,A3,B1,B2,B3,B4,C1,C2) x 10 reps x 5 outer = 400 tasks.
# Each task runs one (cell, rep, outer): inner Optuna HPO -> refit -> eval fold-test.
# Transfer cells (B*/C2) load the source ckpt from the SAME OUT dir. Submit AFTER the
# source array completes, via:
#   cluster-submit --node gpunode02 slurm/cell.sh -J pnc-age-vwm-matrix --dependency=afterok:<SRC_JOB_ID>
# A5 (trivial) is NOT in HPO_CELLS / this array — run it as a separate tail (see plan Task 11).
#SBATCH --job-name=pnc-cell
#SBATCH --array=0-399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%A_%a.out
#SBATCH --error=slurm/logs/%A_%a.err

set -euo pipefail

SIF="$(pwd)/pnc-age-vwm.sif"
RC_DATASET_DIR_IN=/workspace/data/pnc_sc400          # bound from <project_root>/data/pnc_sc400
OUT_IN=/workspace/runs/matrix                         # source ckpts + result JSONs live here

echo "[cell] node=$(hostname) sha=$(git rev-parse --short HEAD) task=${SLURM_ARRAY_TASK_ID}"

# index -> (cell, rep, outer) via the container python (pure cluster._common).
# tail -n1 guards against any singularity stdout banner; pipefail still propagates
# a real failure of the exec so a bad mapping never silently runs the wrong unit.
UNIT=$(singularity exec --bind "$(pwd):/workspace" --pwd /workspace "$SIF" \
    python -m slurm._index cell "${SLURM_ARRAY_TASK_ID}" | tail -n1)
read -r CELL REP OUTER <<<"$UNIT"
echo "[cell] cell=$CELL rep=$REP outer=$OUTER"

singularity exec --nv \
    --bind "$(pwd):/workspace" \
    --pwd /workspace \
    --env RC_DATASET_DIR="$RC_DATASET_DIR_IN" \
    "$SIF" \
    python -m cluster.train_cell --cell "$CELL" --rep "$REP" --outer "$OUTER" \
        --out "$OUT_IN" --source-dir "$OUT_IN"
