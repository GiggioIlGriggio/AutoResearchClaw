#!/bin/bash
# One-task GPU sanity job — HARD GATE before the 500-task matrix.
# Proves the .sif imports the full stack AND the GPU actually computes on the target
# node (real cuBLAS matmul + a torch_geometric GCNConv forward — the matrix's hot
# path), not merely torch.cuda.is_available(). Matters because the gpunode02 driver
# (r550 / reports CUDA 12.4) runs our cu128 runtime via CUDA 12.x minor-version compat.
# Submit via: cluster-submit --node gpunode02 slurm/sanity.sh -J smoke-pnc-sanity
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
    python - <<'PY'
import torch
from torch_geometric.nn import GCNConv
assert torch.cuda.is_available(), "CUDA not available on the node"
dev = torch.device("cuda")
# real cuBLAS kernel
a = torch.randn(512, 512, device=dev)
b = torch.randn(512, 512, device=dev)
c = float((a @ b).sum())
assert c == c, "matmul produced NaN"                       # finite check
# real pyg message-passing kernel on GPU — the matrix's actual hot path
x = torch.randn(10, 400, device=dev)
ei = torch.randint(0, 10, (2, 40), device=dev)
out = GCNConv(400, 32).to(dev)(x, ei)
assert tuple(out.shape) == (10, 32) and bool(torch.isfinite(out).all()), "GCNConv GPU forward failed"
print("GPU_COMPUTE_OK matmul_sum", round(c, 2), "gcn_out", tuple(out.shape))
print("CUDA_AVAILABLE", torch.cuda.is_available(), "DEVICE", torch.cuda.get_device_name(0))
import torch_geometric, optuna, numpy
print("VERSIONS torch", torch.__version__, "pyg", torch_geometric.__version__,
      "optuna", optuna.__version__, "numpy", numpy.__version__)
PY
