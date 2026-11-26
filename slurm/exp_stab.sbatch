#!/bin/bash
# Experiment: SuperMimic / TGI stabilisers on the decoder + analytic-reward recipe (arm G of exp_decode).
# Same env, schedule and seeds as exp_decode G-s0 / G-s1 (their evals at 12k / 24k / 24.5k updates:
# lengths 1000/861/665 and 951/798/966); adds
#   ac.ema=0.995          deployed actor = EMA of the trained one
#   train.lr_decay_iters  cosine decay to 0.1x over the run (all learning rates)
#   ac.grad_clip_value=0.1, ac.beta2=0.95   element-wise actor gradient clip, Adam beta2
# Checkpoints are evaluated with the EMA actor; the final one also with the raw actor.
#   sbatch slurm/exp_stab.sbatch
#SBATCH --job-name=exp-stab
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

COMMON="env.name=Go1JoystickFlatTerrain env.num_envs=1024 env.use_privileged=true env.unclipped_reward=true \
env.alive_bonus=0.05 ac.fixed_std=0.1 ac.ent_coef=0.0 train.collect_steps=2 train.warmup_iters=64 \
train.wm_updates=8 train.ac_updates=8 train.iters=3064 train.ckpt_every=1500 train.log_every=25 \
train.wandb_group=flat-stabilised wm.decode_state=true ac.analytic_reward=true \
ac.ema=0.995 train.lr_decay_iters=3064 ac.grad_clip_value=0.1 ac.beta2=0.95"

run_arm() {  # label seed
  local label="$1" seed="$2"
  echo "##### ARM ${label}: seed=${seed}"
  local t0=$SECONDS
  python -u -m jpw.train --run-name "exp-stab-${label}" --set ${COMMON} train.seed="${seed}" \
    > "logs/arm-stab-${label}.txt" 2>&1
  echo "ARM ${label} train_seconds=$((SECONDS - t0))"
  grep -E "^\[step" "logs/arm-stab-${label}.txt" | tail -1 | cut -c1-330 | sed "s/^/ARM ${label} LAST /"
  grep -E "Error|Traceback" "logs/arm-stab-${label}.txt" | head -3 | sed "s/^/ARM ${label} /"
  for c in runs/exp-stab-${label}/ckpt_*.pkl; do
    python -u -m jpw.eval --ckpt "$c" --episodes 256 --num-envs 256 2>&1 \
      | grep -E "^EVAL|^TERM (TOTAL|tracking_lin_vel|tracking_ang_vel)|Error|Traceback" \
      | sed "s/^/ARM ${label} ema $(basename $c .pkl) /"
  done
  python -u -m jpw.eval --ckpt "runs/exp-stab-${label}/ckpt_final.pkl" --episodes 256 --num-envs 256 \
    --set ac.ema=0 2>&1 | grep -E "^EVAL|^TERM TOTAL|Error|Traceback" | sed "s/^/ARM ${label} raw ckpt_final /"
  echo "##### ARM ${label} done"
}

run_arm S-s0 0
run_arm S-s1 1
echo "##### EXPERIMENT-DONE"
