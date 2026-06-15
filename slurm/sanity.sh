#!/bin/bash
# One-task GPU sanity job — HARD GATE before the 500-task matrix.
# Proves the .sif imports the full stack AND torch sees the GPU on the target node.
# Submit via: cluster-submit --node gpunode02 slurm/sanity.sh -J smoke-pnc-sanity
# (--node resolves partition/qos/account; this script only declares the resources.)
#SBATCH --job-name=smoke-pnc-sanity
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err

set -euo pipefail

SIF="$(pwd)/pnc-age-vwm.sif"
echo "[sanity] node: $(hostname)"
echo "[sanity] git SHA: $(git rev-parse HEAD)"
echo "[sanity] sif: $SIF"
nvidia-smi || true

singularity exec --nv \
    --bind "$(pwd):/workspace" \
    --pwd /workspace \
    "$SIF" \
    python -c "import torch, torch_geometric, optuna, numpy, scipy, pandas, sklearn; \
print('CUDA_AVAILABLE', torch.cuda.is_available()); \
print('DEVICE', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); \
print('VERSIONS', 'torch', torch.__version__, 'pyg', torch_geometric.__version__, 'optuna', optuna.__version__, 'numpy', numpy.__version__)"
