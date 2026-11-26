#!/bin/bash
# Experiment: is the learner update-starved? Flat-terrain Go1, same model, three update/data ratios.
#   A: 24 env steps per iteration, 750 update iterations  (current ratio, ~18.6M env steps)
#   C:  2 env steps per iteration, 750 update iterations  (same #updates as A, ~11x less data)
#   D:  2 env steps per iteration, 1500 update iterations (2x the updates of C)
# Each iteration does wm_updates=8 world-model + ac_updates=8 actor steps. Warm-up (collect only)
# gives every arm >= 128 replay rows per env before the first update.
#   sbatch slurm/exp_update_ratio.sbatch
#SBATCH --job-name=exp-ratio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
[ -f cluster.env ] && source cluster.env
source "${JPW_VENV:-$HOME/venvs/jpw}/bin/activate"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_enable_triton_gemm=false"
export MUJOCO_GL="${MUJOCO_GL:-disable}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"

COMMON="env.name=Go1JoystickFlatTerrain env.num_envs=1024 train.log_every=25 train.ckpt_every=1000000 \
train.wandb_group=flat-update-ratio train.wm_updates=8 train.ac_updates=8"

run_arm() {  # label collect_steps warmup_iters iters
  local label="$1" cs="$2" wu="$3" iters="$4"
  echo "##### ARM ${label}: collect_steps=${cs} warmup=${wu} iters=${iters}"
  local t0=$SECONDS
  python -u -m jpw.train --run-name "exp-flat-${label}" --set ${COMMON} \
    train.collect_steps="${cs}" train.warmup_iters="${wu}" train.iters="${iters}" \
    2>&1 | grep -E "^\[step|Error|Traceback|data parallel" | awk 'NR%4==1 || /Error|Traceback/' | cut -c1-250
  echo "ARM ${label} train_seconds=$((SECONDS - t0))"
  python -u -m jpw.eval --ckpt "runs/exp-flat-${label}/ckpt_final.pkl" --episodes 256 --num-envs 256 2>&1 \
    | grep -E "^EVAL|Error|Traceback" | sed "s/^/ARM ${label} /"
  echo "##### ARM ${label} done"
}

run_arm A 24 6 756
run_arm C 2 64 814
run_arm D 2 64 1564
echo "##### EXPERIMENT-DONE"
