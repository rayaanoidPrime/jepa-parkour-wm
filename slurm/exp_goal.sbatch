#!/bin/bash
# One terrain-grounded goal-reaching run (single seed) + eval of every checkpoint.
#   sbatch slurm/exp_goal.sbatch LABEL [train overrides, e.g. train.iters=3000 ac.fixed_std=0.1]
# Uses the `goal` preset (Go1Parkour goal task, 1024 envs, decoder heads, grounded actor, EMA, stabilisers).
#SBATCH --job-name=exp-goal
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

LABEL="${1:?label}"; shift
echo "##### ARM ${LABEL}: $*"
t0=$SECONDS
python -u -m jpw.train --preset goal --run-name "goal-${LABEL}" --set train.wandb_group=goal-${LABEL%%-*} "$@" \
  > "logs/arm-goal-${LABEL}.txt" 2>&1
echo "ARM ${LABEL} train_seconds=$((SECONDS - t0))"
grep -E "^env=|^\[step" "logs/arm-goal-${LABEL}.txt" | sed -n "1p" | cut -c1-250
grep -E "^\[step" "logs/arm-goal-${LABEL}.txt" | tail -2 | cut -c1-330 | sed "s/^/ARM ${LABEL} LAST /"
grep -E "Error|Traceback" "logs/arm-goal-${LABEL}.txt" | head -3 | sed "s/^/ARM ${LABEL} /"
for c in runs/goal-${LABEL}/ckpt_*.pkl; do
  python -u -m jpw.eval --ckpt "$c" --episodes 256 --num-envs 256 2>&1 \
    | grep -E "^EVAL|Error|Traceback" | sed "s/^/ARM ${LABEL} $(basename $c .pkl) /"
done
echo "##### ARM ${LABEL} done"
