#!/usr/bin/env bash
# One-time, bare-metal (no container) Python env on the cluster. Run on a login node from the
# repo root:  bash scripts/setup_cluster_env.sh      (or remotely: scripts/remote.sh setup)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f cluster.env ] && source cluster.env

VENV="${JPW_VENV:-$HOME/venvs/jpw}"
PY="${JPW_PYTHON:-python3}"
# JAX extra matching the cluster's ROCm install; "rocm7-local" uses the system ROCm (/opt/rocm).
JAX_EXTRA="${JPW_JAX_EXTRA:-rocm7-local}"

if [ -n "${JPW_MODULES:-}" ]; then
  # shellcheck disable=SC2086
  module load ${JPW_MODULES}
fi

echo "Creating venv at ${VENV} with $(${PY} --version)"
"${PY}" -m venv "${VENV}"
source "${VENV}/bin/activate"
pip install -U pip wheel

pip install "jax[${JAX_EXTRA}]"
JAX_VER="$(python -c 'import jax; print(jax.__version__)')"

pip install -e ".[sim,track]"

# Simulator packages may pull a different jax; put the ROCm-matched version back.
if [ "$(python -c 'import jax; print(jax.__version__)')" != "${JAX_VER}" ]; then
  echo "Re-pinning jax==${JAX_VER} for ROCm"
  pip install "jax[${JAX_EXTRA}]==${JAX_VER}"
fi

python - <<'EOF'
import jax
print("jax", jax.__version__, "backend:", jax.default_backend())
print("devices:", jax.devices())
EOF
echo "Login-node check done (login nodes may have no GPU; the Slurm smoke test is the real check)."
echo "If W&B is not logged in yet: source ${VENV}/bin/activate && wandb login"
