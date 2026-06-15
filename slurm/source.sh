#!/bin/bash
# Source-pretrain array: A1,A4 x 10 reps x 5 outer = 100 tasks.
# Each task pretrains one (cell, rep, outer) backbone on its fold-train age subjects
# and saves the checkpoint the dependent B/C cells load. Submit via:
#   cluster-submit --node gpunode02 slurm/source.sh -J pnc-age-vwm-source
# (--node resolves partition/qos/account; this script declares only resources.)
#SBATCH --job-name=pnc-source
#SBATCH --array=0-99
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%A_%a.out
#SBATCH --error=slurm/logs/%A_%a.err

set -euo pipefail

SIF="$(pwd)/pnc-age-vwm.sif"
RC_DATASET_DIR_IN=/workspace/data/pnc_sc400          # bound from <project_root>/data/pnc_sc400
OUT_IN=/workspace/runs/matrix                         # -> <project_root>/runs/matrix on host

echo "[source] node=$(hostname) sha=$(git rev-parse --short HEAD) task=${SLURM_ARRAY_TASK_ID}"

# index -> (cell, rep, outer) via the container python (pure cluster._common).
# tail -n1 guards against any singularity stdout banner; pipefail still propagates
# a real failure of the exec so a bad mapping never silently runs the wrong unit.
UNIT=$(singularity exec --bind "$(pwd):/workspace" --pwd /workspace "$SIF" \
    python -m slurm._index source "${SLURM_ARRAY_TASK_ID}" | tail -n1)
read -r CELL REP OUTER <<<"$UNIT"
echo "[source] cell=$CELL rep=$REP outer=$OUTER"

singularity exec --nv \
    --bind "$(pwd):/workspace" \
    --pwd /workspace \
    --env RC_DATASET_DIR="$RC_DATASET_DIR_IN" \
    "$SIF" \
    python -m cluster.pretrain_source --cell "$CELL" --rep "$REP" --outer "$OUTER" --out "$OUT_IN"
