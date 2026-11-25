#!/usr/bin/env bash
# Drive the cluster over SSH from your machine.
#
#   scripts/remote.sh sync                         copy the repo to the cluster (no git needed there)
#   scripts/remote.sh setup                        sync + build the venv on the login node
#   scripts/remote.sh submit [train args...]       sync + sbatch, e.g. submit --preset smoke
#   scripts/remote.sh status                       your queued/running jobs
#   scripts/remote.sh logs [JOB_ID]                tail a job log (default: newest)
#   scripts/remote.sh cancel JOB_ID
#   scripts/remote.sh wandb-sync                   upload offline W&B runs from the login node
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f cluster.env ] && source cluster.env

HOST="${JPW_HOST:-iisc_amd}"
RDIR="${JPW_REMOTE_DIR:-jepa-parkour-wm}"
SBATCH_ARGS="${JPW_SBATCH_ARGS:-}"

sync() {
  tar czf - --exclude=.git --exclude=runs --exclude=logs --exclude=wandb \
      --exclude=__pycache__ --exclude='*.egg-info' . \
    | ssh "${HOST}" "mkdir -p ${RDIR} && tar xzf - -C ${RDIR}"
  echo "synced -> ${HOST}:${RDIR}"
}

cmd="${1:-}"; shift || true
case "${cmd}" in
  sync)   sync ;;
  setup)  sync; ssh "${HOST}" "cd ${RDIR} && bash scripts/setup_cluster_env.sh" ;;
  submit)
    sync
    args="$(printf '%q ' "$@")"
    ssh "${HOST}" "cd ${RDIR} && mkdir -p logs && sbatch ${SBATCH_ARGS} slurm/train.sbatch ${args}"
    ;;
  status) ssh "${HOST}" 'squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %.6D %R"' ;;
  logs)
    if [ -n "${1:-}" ]; then
      ssh "${HOST}" "cd ${RDIR} && tail -n ${JPW_TAIL:-60} logs/*-${1}.out"
    else
      ssh "${HOST}" "cd ${RDIR} && f=\$(ls -t logs/*.out | head -n 1) && echo \"== \$f\" && tail -n ${JPW_TAIL:-60} \"\$f\""
    fi
    ;;
  cancel) ssh "${HOST}" "scancel ${1:?job id}" ;;
  wandb-sync)
    ssh "${HOST}" "cd ${RDIR} && source \${JPW_VENV:-\$HOME/venvs/jpw}/bin/activate && \
      find runs -type d -name 'offline-run-*' -exec wandb sync {} \;"
    ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
