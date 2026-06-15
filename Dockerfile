# M3 cluster container: torch 2.11.0 + CUDA 12.8 from the official PyTorch image
# (mirrors the local .venv: torch==2.11.0+cu128, python 3.12, pyg 2.8.0).
# The CUDA-12.8 runtime vs the gpunode02 driver is the top infra risk — the
# one-task GPU sanity job (slurm/sanity.sh) gates this before the 500-task matrix.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python"]
