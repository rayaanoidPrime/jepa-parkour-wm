#!/bin/bash
# Evaluate a goal-task checkpoint under randomised physics (friction, masses, COM, motor strength, ...) and
# heightmap noise, on its training course and two held-out courses.
#   sbatch slurm/eval_robust.sbatch runs/goal-g4/ckpt_final.pkl [heightmap noise std in m, default 0.02]
#SBATCH --job-name=eval-robust
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
[ -f cluster.env ] && source cluster.env
source "${JPW_VENV:-$HOME/venvs/jpw}/bin/activate"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_enable_triton_gemm=false"
export MUJOCO_GL="${MUJOCO_GL:-disable}"
CK="$1"; NOISE="${2:-0.02}"
for s in train 1 2; do
  extra=""; [ "$s" != "train" ] && extra="env.terrain_seed=$s"
  python -u -m jpw.eval --ckpt "$CK" --episodes 256 --num-envs 256 \
    --set env.domain_randomization=true env.hmap_noise=$NOISE $extra 2>&1 | grep -E "^EVAL|Error|Traceback" \
    | sed "s/^/robust-$s /"
done
echo "##### ROBUST-DONE"
