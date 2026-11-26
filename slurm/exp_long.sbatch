#!/bin/bash
# 1) Per-term reward breakdown of the three existing flat policies (E0/E1/E2 from exp_reward).
# 2) A longer E1 run (unclipped reward + alive bonus 0.05; 6000 update iterations = 48k updates)
#    with checkpoints every 1500 update iterations, each evaluated with the per-term breakdown.
#   sbatch slurm/exp_long.sbatch
#SBATCH --job-name=exp-long
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

eval_ckpt() {  # label ckpt
  echo "##### EVAL $1"
  python -u -m jpw.eval --ckpt "$2" --episodes 256 --num-envs 256 2>&1 | grep -E "^EVAL|^TERM|Error|Traceback" | sed "s/^/$1 /"
}

for a in E0 E1 E2; do eval_ckpt "short-$a" "runs/exp-reward-$a/ckpt_final.pkl"; done

echo "##### LONG RUN: unclipped + alive 0.05"
t0=$SECONDS
python -u -m jpw.train --run-name exp-long-E1 --set env.name=Go1JoystickFlatTerrain env.num_envs=1024 \
  env.unclipped_reward=true env.alive_bonus=0.05 train.collect_steps=2 train.warmup_iters=64 \
  train.wm_updates=8 train.ac_updates=8 train.iters=6064 train.ckpt_every=1500 train.log_every=25 \
  train.wandb_group=flat-unclipped-long 2>&1 | grep -E "^\[step|Error|Traceback" \
  | awk 'NR%12==1 || /Error|Traceback/' | cut -c1-250
echo "long run train_seconds=$((SECONDS - t0))"
for c in runs/exp-long-E1/ckpt_*.pkl; do eval_ckpt "long-$(basename $c .pkl)" "$c"; done
echo "##### EXPERIMENT-DONE"
