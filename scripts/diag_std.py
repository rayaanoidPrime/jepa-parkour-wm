"""Action noise of a checkpoint's actor on states it actually visits.

  python scripts/diag_std.py runs/exp-long-E1/ckpt_final.pkl [steps]

Rolls the deterministic policy for a few steps and reports the actor's std (the noise used while
collecting data and in imagination) and the action magnitude.
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

from jpw import checkpoint
from jpw.envs import get_hmap, get_obs, make_env

cfg, spec, agent, ts, meta = checkpoint.load(sys.argv[1])
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 60
cfg.env.num_envs = 64
cfg.env.domain_randomization = False
env = make_env(cfg.env)
st = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 64))
step = jax.jit(env.step)
stds, means = [], []
for _ in range(steps):
    obs = get_obs(st.obs, cfg.env.use_privileged)
    z = agent.policy_latent(ts.wm, obs, get_hmap(st.obs, cfg.env.use_heightmap))
    mean, std = agent.actor.apply(ts.actor, z)
    stds.append(np.asarray(std)), means.append(np.asarray(jnp.tanh(mean)))
    st = step(st, jnp.tanh(mean))
stds, means = np.stack(stds), np.stack(means)
print(f"{sys.argv[1]} (it={meta['it']}, fixed_std={cfg.ac.fixed_std})")
print(f"actor std : mean {stds.mean():.3f}  min {stds.min():.3f}  max {stds.max():.3f}  (config range "
      f"[{cfg.ac.min_std}, {cfg.ac.max_std}])")
print(f"|tanh(mean) action| : mean {np.abs(means).mean():.3f}  max {np.abs(means).max():.3f}")
print(f"noise-to-signal  std / |action| = {stds.mean() / max(np.abs(means).mean(), 1e-6):.2f}")
