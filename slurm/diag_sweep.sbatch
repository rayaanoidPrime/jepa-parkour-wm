#!/bin/bash
# Run the MJX NaN diagnostic under several XLA / precision settings on one GPU and print the
# NaN lines of each:  sbatch slurm/diag_sweep.sbatch [env_name] [num_envs]
#SBATCH --job-name=diagsweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
[ -f cluster.env ] && source cluster.env
source "${JPW_VENV:-$HOME/venvs/jpw}/bin/activate"
export MUJOCO_GL="${MUJOCO_GL:-disable}"
ENV="${1:-Go1JoystickRoughTerrain}"
N="${2:-256}"
BASE="--xla_gpu_enable_command_buffer="

run() {  # label, XLA flags, extra env assignments
  local label="$1" flags="$2"; shift 2
  echo "##### ${label}  XLA_FLAGS='${flags}' $*"
  env XLA_FLAGS="${flags}" "$@" python -u scripts/diag_mjx.py "${ENV}" "${N}" 2>&1 \
    | grep -E "^\[diag\] (nan|zeros step (0|29)|random step 29)|Error|Traceback"
}

run baseline        "${BASE}"                                                       A=1
run no_triton_gemm  "${BASE} --xla_gpu_enable_triton_gemm=false"                    A=1
run autotune0       "${BASE} --xla_gpu_autotune_level=0"                            A=1
run both            "${BASE} --xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0" A=1
run highest_prec    "${BASE}"                                                       JAX_DEFAULT_MATMUL_PRECISION=highest
echo "##### SWEEP-DONE"
