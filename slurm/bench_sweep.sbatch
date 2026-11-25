#!/bin/bash
# A/B throughput of XLA settings on one GPU (see scripts/bench.py):
#   sbatch slurm/bench_sweep.sbatch [preset] [num_envs]
#SBATCH --job-name=bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
[ -f cluster.env ] && source cluster.env
source "${JPW_VENV:-$HOME/venvs/jpw}/bin/activate"
export MUJOCO_GL="${MUJOCO_GL:-disable}"
# W&B host: your runs live on wandb.ai (newer wandb defaults to another host).
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
PRESET="${1:-parkour}"
N="${2:-2048}"
BASE="--xla_gpu_enable_command_buffer="
SET="env.num_envs=${N} train.replay_capacity=200"

run() {  # label, XLA flags, extra env assignments
  local label="$1" flags="$2"; shift 2
  echo "##### ${label}"
  env BENCH_LABEL="${label}" XLA_FLAGS="${flags}" "$@" \
    python -u scripts/bench.py --preset "${PRESET}" --set ${SET} 2>&1 | grep -E "^BENCH|Error|Traceback"
}

run triton_off   "${BASE} --xla_gpu_enable_triton_gemm=false"  A=1
run highest_prec "${BASE}"                                      JAX_DEFAULT_MATMUL_PRECISION=highest
run triton_on    "${BASE}"                                      A=1
echo "##### BENCH-DONE"
