#!/usr/bin/env bash
# One-time, bare-metal (no container) Python env on the cluster, managed with uv. Run on a login
# node from the repo root:  bash scripts/setup_cluster_env.sh   (or remotely: scripts/remote.sh setup)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f cluster.env ] && source cluster.env

VENV="${JPW_VENV:-$HOME/venvs/jpw}"
PY="${JPW_PYTHON:-3.12}"
# pyproject extra that provides jax for this machine's ROCm (rocm60 = ROCm 6.x).
JAX_EXTRA="${JPW_JAX_EXTRA:-rocm60}"
export PATH="$HOME/.local/bin:$PATH"

if [ -n "${JPW_MODULES:-}" ]; then
  # shellcheck disable=SC2086
  module load ${JPW_MODULES}
fi

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

rm -rf "${VENV}"
uv venv --python "${PY}" "${VENV}"
uv pip install --python "${VENV}/bin/python" -e ".[${JAX_EXTRA},sim,track]"

"${VENV}/bin/python" - <<'PY'
import jax
print("jax", jax.__version__, "backend:", jax.default_backend())
print("devices:", jax.devices())
PY
echo "Login-node check done (login nodes may have no GPU; the Slurm smoke test is the real check)."
echo "If W&B is not logged in yet: ${VENV}/bin/wandb login"
