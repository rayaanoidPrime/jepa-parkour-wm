"""Print imagined vs real reward over training for runs in the W&B project.

  python scripts/wandb_curve.py exp-stab-S-s0 [exp-decode-G-s0 ...]

ac/imag_reward is the mean imagined per-step reward (analytic reward on the decoded state, plus any
alive bonus); env/reward_per_step is the mean real per-step reward of the data collected that
iteration (Playground reward, unclipped, plus the same bonus). Both are on the same scale, so a large
positive gap means the model believes the policy does better than it really does.
"""
import sys

import numpy as np
import wandb

api = wandb.Api()
for name in sys.argv[1:]:
    runs = api.runs("world_models_iisc/jepa-parkour-wm", filters={"display_name": name})
    if not runs:
        print(f"{name}: no run found")
        continue
    h = runs[0].history(keys=["ac/imag_reward", "env/reward_per_step", "wm/phys_linvel", "ac/entropy",
                              "ac/actor_grad_norm", "perf/iter"], samples=4000, pandas=True)
    h = h.dropna(subset=["ac/imag_reward", "env/reward_per_step"])
    it = h["perf/iter"].to_numpy()
    print(f"\n== {name}: {len(h)} logged points")
    print(f"{'iter':>6} {'imag_r/step':>12} {'real_r/step':>12} {'gap':>9} {'phys_linvel':>12} {'actor_gn':>9}")
    for lo in range(0, int(it.max()) + 1, 400):
        m = (it >= lo) & (it < lo + 400)
        if m.sum() == 0:
            continue
        ir, rr = h["ac/imag_reward"][m].mean(), h["env/reward_per_step"][m].mean()
        print(f"{lo:>6} {ir:>12.5f} {rr:>12.5f} {ir - rr:>+9.5f} {h['wm/phys_linvel'][m].mean():>12.3f} "
              f"{h['ac/actor_grad_norm'][m].mean():>9.4f}")
