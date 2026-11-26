#!/bin/bash
# Experiment: decoder heads + analytic reward / termination vs the learned reward / continue heads.
# Flat Go1, unclipped reward + alive bonus 0.05, fixed action noise 0.1 (the best earlier setting), same
# schedule as exp_noise F0 (3064 iterations = 24k updates), privileged observations in both arms.
#   G: wm.decode_state + ac.analytic_reward        B: learned reward / continue heads (baseline)
# Two seeds each; checkpoints at 12k / 24k / 24.5k updates are evaluated.
#   sbatch slurm/exp_decode.sbatch
#SBATCH --job-name=exp-decode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
[ -f cluster.env ] && source cluster.env
source "${JPW_VENV:-$HOME/venvs/jpw}/bin/activate"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_enable_triton_gemm=false"
export MUJOCO_GL="${MUJOCO_GL:-disable}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"

COMMON="env.name=Go1JoystickFlatTerrain env.num_envs=1024 env.use_privileged=true env.unclipped_reward=true \
env.alive_bonus=0.05 ac.fixed_std=0.1 ac.ent_coef=0.0 train.collect_steps=2 train.warmup_iters=64 \
train.wm_updates=8 train.ac_updates=8 train.iters=3064 train.ckpt_every=1500 train.log_every=25 \
train.wandb_group=flat-decode"

run_arm() {  # label seed extra-overrides...
  local label="$1" seed="$2"; shift 2
  echo "##### ARM ${label}: seed=${seed} $*"
  local t0=$SECONDS
  python -u -m jpw.train --run-name "exp-decode-${label}" --set ${COMMON} train.seed="${seed}" "$@" \
    > "logs/arm-decode-${label}.txt" 2>&1
  echo "ARM ${label} train_seconds=$((SECONDS - t0))"
  grep -E "^\[step" "logs/arm-decode-${label}.txt" | tail -1 | cut -c1-330 | sed "s/^/ARM ${label} LAST /"
  grep -E "Error|Traceback" "logs/arm-decode-${label}.txt" | head -3 | sed "s/^/ARM ${label} /"
  for c in runs/exp-decode-${label}/ckpt_*.pkl; do
    python -u -m jpw.eval --ckpt "$c" --episodes 256 --num-envs 256 2>&1 \
      | grep -E "^EVAL|^TERM (TOTAL|tracking_lin_vel|tracking_ang_vel)|Error|Traceback" \
      | sed "s/^/ARM ${label} $(basename $c .pkl) /"
  done
  echo "##### ARM ${label} done"
}

run_arm G-s0 0 wm.decode_state=true ac.analytic_reward=true
run_arm B-s0 0
run_arm G-s1 1 wm.decode_state=true ac.analytic_reward=true
run_arm B-s1 1
echo "##### EXPERIMENT-DONE"
