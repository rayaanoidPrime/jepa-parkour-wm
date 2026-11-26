#!/bin/bash
# Experiment: SuperMimic-style fixed action noise instead of a learned std (flat Go1, unclipped
# reward + alive bonus 0.05, same update ratio as exp_long). Compare with exp-long-E1 at equal
# updates (its checkpoints at 12k / 24k updates: lengths 926 / 974, task reward -0.019 / -0.014).
#   F0: fixed std 0.1, ent_coef 0      F1: fixed std 0.2, ent_coef 0
#   sbatch slurm/exp_noise.sbatch
#SBATCH --job-name=exp-noise
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

run_arm() {  # label fixed_std
  local label="$1" std="$2"
  echo "##### ARM ${label}: fixed_std=${std}"
  local t0=$SECONDS
  python -u -m jpw.train --run-name "exp-noise-${label}" --set env.name=Go1JoystickFlatTerrain \
    env.num_envs=1024 env.unclipped_reward=true env.alive_bonus=0.05 ac.fixed_std="${std}" ac.ent_coef=0.0 \
    train.collect_steps=2 train.warmup_iters=64 train.wm_updates=8 train.ac_updates=8 train.iters=3064 \
    train.ckpt_every=1500 train.log_every=25 train.wandb_group=flat-fixed-noise > /dev/null 2>&1
  echo "ARM ${label} train_seconds=$((SECONDS - t0))"
  for c in runs/exp-noise-${label}/ckpt_*.pkl; do
    python -u -m jpw.eval --ckpt "$c" --episodes 256 --num-envs 256 2>&1 \
      | grep -E "^EVAL|^TERM (energy|feet_clearance|ang_vel_xy|tracking_lin_vel|tracking_ang_vel|pose|orientation|TOTAL)|Error|Traceback" \
      | sed "s/^/ARM ${label} $(basename $c .pkl) /"
  done
  echo "##### ARM ${label} done"
}

run_arm F0 0.1
run_arm F1 0.2
echo "##### EXPERIMENT-DONE"
