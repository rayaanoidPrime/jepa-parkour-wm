#!/bin/bash
# Experiment: does removing the reward clip (and an alive bonus) let the learner train on flat Go1?
# Same model and update ratio as arm D of exp_update_ratio (2 env steps / iteration, 1500 update
# iterations), only the reward changes:
#   E0: unclipped, no alive bonus      E1: unclipped + alive 0.05      E2: unclipped + alive 0.10
#   sbatch slurm/exp_reward.sbatch
#SBATCH --job-name=exp-reward
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
train.wandb_group=flat-unclipped-reward train.wm_updates=8 train.ac_updates=8 \
train.collect_steps=2 train.warmup_iters=64 train.iters=1564 env.unclipped_reward=true"

run_arm() {  # label alive_bonus
  local label="$1" ab="$2"
  echo "##### ARM ${label}: unclipped, alive_bonus=${ab}"
  local t0=$SECONDS
  python -u -m jpw.train --run-name "exp-reward-${label}" --set ${COMMON} env.alive_bonus="${ab}" \
    2>&1 | grep -E "^\[step|Error|Traceback" | awk 'NR%4==1 || /Error|Traceback/' | cut -c1-250
  echo "ARM ${label} train_seconds=$((SECONDS - t0))"
  python -u -m jpw.eval --ckpt "runs/exp-reward-${label}/ckpt_final.pkl" --episodes 256 --num-envs 256 \
    --set env.unclipped_reward=true env.alive_bonus="${ab}" 2>&1 \
    | grep -E "^EVAL|Error|Traceback" | sed "s/^/ARM ${label} /"
  echo "##### ARM ${label} done"
}

run_arm E0 0.0
run_arm E1 0.05
run_arm E2 0.10
echo "##### EXPERIMENT-DONE"
