#!/bin/bash
# Evaluate a goal-task checkpoint on its training course and on held-out courses (other terrain seeds).
#   sbatch slurm/eval_heldout.sbatch runs/goal-g3/ckpt_final.pkl [seeds...]
#SBATCH --job-name=eval-held
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
CK="$1"; shift
SEEDS="${*:-1 2 3}"
echo "##### training course (terrain_seed from the checkpoint)"
python -u -m jpw.eval --ckpt "$CK" --episodes 256 --num-envs 256 2>&1 | grep -E "^EVAL|Error|Traceback" | sed "s/^/seed-train /"
for s in $SEEDS; do
  python -u -m jpw.eval --ckpt "$CK" --episodes 256 --num-envs 256 --set env.terrain_seed=$s 2>&1 \
    | grep -E "^EVAL|Error|Traceback" | sed "s/^/seed-$s /"
done
echo "##### HELDOUT-DONE"
