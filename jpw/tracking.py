"""Weights & Biases tracking with a stdout fallback."""
from __future__ import annotations

import os
import time

from jpw.config import Config


class Tracker:
    def __init__(self, cfg: Config, run_name: str, run_dir: str):
        self.run = None
        self._t0 = time.time()
        t = cfg.train
        if t.wandb_mode == "disabled":
            return
        try:
            import wandb

            config = cfg.to_dict()
            config["slurm"] = {
                k: os.environ.get(k, "")
                for k in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION")
            }
            self.run = wandb.init(
                project=t.wandb_project,
                entity=t.wandb_entity or None,
                name=run_name,
                group=t.wandb_group or None,
                tags=[x for x in t.wandb_tags.split(",") if x],
                config=config,
                # The Slurm script sets WANDB_MODE=offline on nodes without internet.
                mode=os.environ.get("WANDB_MODE", t.wandb_mode),
                dir=run_dir,
                id=os.environ.get("WANDB_RUN_ID") or None,
                resume="allow",
            )
        except Exception as e:  # never let tracking kill a cluster job
            print(f"[tracking] W&B unavailable ({e!r}); logging to stdout only.", flush=True)
            self.run = None

    def log(self, metrics: dict, step: int) -> None:
        metrics = {k: float(v) for k, v in metrics.items()}
        metrics["perf/wall_min"] = (time.time() - self._t0) / 60.0
        keys = ("env/reward_per_step", "env/ep_return", "wm/jepa", "wm/latent_std", "wm/reward",
                "wm/contact_acc", "wm/contact_event_acc", "wm/student", "wm/phys", "wm/phys_linvel",
                "wm/phys_gravity", "wm/phys_jvel", "ac/imag_return",
                "perf/env_sps")
        brief = "  ".join(f"{k.split('/')[-1]}={metrics[k]:.4g}" for k in keys if k in metrics)
        print(f"[step {step}] {brief}", flush=True)
        if self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
