#!/bin/bash
# Probe the easy levels of the weak Go1Tiles terrain types.   sbatch slurm/eval_tiles_probe.sbatch CKPT
#SBATCH --job-name=probe
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
CK="$1"
for t in stones gap climb_up stairs_down; do
  for lv in "0 1" "2 3"; do
    python -u -m jpw.eval --ckpt "$CK" --episodes 128 --num-envs 128 --types $t --levels $lv 2>&1 \
      | grep -E "^EVAL|Error|Traceback" | sed "s/^/PROBE $t levels[$lv] /" | cut -c1-260
  done
done
echo "##### PROBE-DONE"
