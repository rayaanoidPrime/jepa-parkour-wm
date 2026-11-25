#!/bin/bash
# How throughput scales with the number of envs (see scripts/bench.py):
#   sbatch slurm/bench_scale.sbatch [preset] "2048 4096 8192 16384"
#SBATCH --job-name=benchscale
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
[ -f cluster.env ] && source cluster.env
source "${JPW_VENV:-$HOME/venvs/jpw}/bin/activate"
# Triton GEMM kernels return wrong results on this ROCm stack (see slurm/diag_sweep.sbatch).
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_enable_triton_gemm=false"
export MUJOCO_GL="${MUJOCO_GL:-disable}"
PRESET="${1:-parkour}"
SIZES="${2:-2048 4096 8192 16384}"

for N in ${SIZES}; do
  echo "##### num_envs=${N}"
  BENCH_LABEL="n${N}" python -u scripts/bench.py --preset "${PRESET}" \
    --set env.num_envs=${N} train.replay_capacity=200 2>&1 | grep -E "^BENCH|Error|Traceback|out of memory|RESOURCE"
done
echo "##### SCALE-DONE"
