#!/bin/bash
# Train the JEPA world model + policy on one AMD GPU. Submit with scripts/submit.sh, or on the
# cluster:  sbatch slurm/train.sbatch --preset smoke
# Extra sbatch flags (partition, account, time) go in cluster.env as JPW_SBATCH_ARGS.
#SBATCH --job-name=jpw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
[ -f cluster.env ] && source cluster.env

if [ -n "${JPW_MODULES:-}" ]; then
  # shellcheck disable=SC2086
  module load ${JPW_MODULES}
fi
source "${JPW_VENV:-$HOME/venvs/jpw}/bin/activate"

# JAX on ROCm: avoid XLA command buffers (per AMD's MJX guide). Training never renders, and
# MUJOCO_GL=egl segfaults on import on this cluster, so disable MuJoCo's GL entirely.
# Triton GEMM kernels return wrong results on this ROCm stack (MJX gets NaNs on ~75% of envs; see
# slurm/diag_sweep.sbatch), so turn them off.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_enable_triton_gemm=false"
export MUJOCO_GL="${MUJOCO_GL:-disable}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"

# W&B: log live when the node can reach wandb, otherwise offline (sync later with scripts/wandb_sync.sh).
if [ -z "${WANDB_MODE:-}" ]; then
  if curl -s --max-time 5 -o /dev/null https://api.wandb.ai; then
    export WANDB_MODE=online
  else
    export WANDB_MODE=offline
  fi
fi

echo "job=${SLURM_JOB_ID} node=$(hostname) wandb=${WANDB_MODE} args: $*"
rocm-smi --showproductname 2>/dev/null | head -n 20 || true

srun python -u -m jpw.train "$@"
