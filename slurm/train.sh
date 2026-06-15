#!/bin/bash
#SBATCH --job-name=AutoResearchClaw
#SBATCH --partition=rad2
#SBATCH --qos=16cpu
#SBATCH --account=rad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:59:00
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

set -euo pipefail

SIF="$(pwd)/AutoResearchClaw.sif"

echo "[run] git SHA: $(git rev-parse HEAD)"
echo "[run] container: $SIF"
echo "[run] node: $(hostname)"
nvidia-smi || true

singularity exec --nv \
    --bind "$(pwd):/workspace" \
    --pwd /workspace \
    "$SIF" \
    python src/train.py "$@"
