#!/bin/bash
# Evaluate a checkpoint with the deployed actor on one GPU:
#   sbatch slurm/eval.sbatch runs/<run>/ckpt_final.pkl [--episodes 256 --set env.hmap_noise=0.02]
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
[ -f cluster.env ] && source cluster.env
source "${JPW_VENV:-$HOME/venvs/jpw}/bin/activate"
# Triton GEMM kernels return wrong results on this ROCm stack (see slurm/diag_sweep.sbatch).
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_enable_triton_gemm=false"
export MUJOCO_GL="${MUJOCO_GL:-disable}"
python -u -m jpw.eval --ckpt "$@"
