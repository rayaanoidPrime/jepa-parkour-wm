"""Why does a standing Go1 get terminated? Print the condition that fires, with the state at that moment."""
import sys

import jax
import jax.numpy as jnp
import numpy as np

from jpw.config import EnvCfg
from jpw.envs import make_env

cfg = EnvCfg(name="Go1Tiles", num_envs=8, episode_length=1000, goal_task=True, use_heightmap=True, use_privileged=True)
env = make_env(cfg)
grid = env.unwrapped._grid
field = env.unwrapped._field
reset, step = jax.jit(env.reset), jax.jit(env.step)
zero = jnp.zeros((8, env.action_size))
names = ("tilt", "belly", "fell", "left_tile", "base_done")
for kind, lvl in (("flat", 4), ("stones", 0), ("gap", 4)):
    cell = lvl * len(grid.types) + grid.types.index(kind)
    keys = jax.random.split(jax.random.PRNGKey(lvl), 8)
    st = reset(jnp.concatenate([keys, jnp.full((8, 1), cell, keys.dtype)], -1))
    qpos0 = np.asarray(st.data.qpos)
    qvel0 = np.asarray(st.data.qvel)
    print(f"\n== {kind} level {lvl}: initial base velocity (lin x y z | ang x y z) per env")
    print(np.round(qvel0[:, :6], 2))
    for t in range(1, 101):
        st = step(st, zero)
        done = np.asarray(st.done) > 0
        if done.any():
            why = np.asarray(st.info["why"])
            q = np.asarray(st.data.qpos)
            for e in np.flatnonzero(done):
                print(f"  step {t:3d} env {e}: fired " + ",".join(n for n, w in zip(names, why[e]) if w > 0)
                      + f"  (note: state shown is after respawn)  cell->{int(np.asarray(st.info['cell'])[e])}")
            break
    # same envs without the respawn mixing: track base height above ground per env for the first done
