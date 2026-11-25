#!/bin/bash
# Staged MJX diagnostic on one GPU:  sbatch slurm/diag.sbatch [env_name] [num_envs]
#SBATCH --job-name=diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
[ -f cluster.env ] && source cluster.env
source "${JPW_VENV:-$HOME/venvs/jpw}/bin/activate"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export MUJOCO_GL="${MUJOCO_GL:-disable}"
python -u scripts/diag_mjx.py "${1:-Go1JoystickRoughTerrain}" "${2:-256}"
