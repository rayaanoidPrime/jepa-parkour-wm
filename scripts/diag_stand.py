"""Does a Go1 holding the default pose (zero action) stay up on the start platform of each tile type?

  python scripts/diag_stand.py [steps]

If a type collapses even though the platform is flat, the terrain collision / generation is at fault, not the policy.
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

from jpw import terrain_grid as tg
from jpw.config import EnvCfg
from jpw.envs import make_env

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
cfg = EnvCfg(name="Go1Tiles", num_envs=8, episode_length=1000, goal_task=True, use_heightmap=True,
             use_privileged=True)
env = make_env(cfg)
grid = env.unwrapped._grid
reset, step = jax.jit(env.reset), jax.jit(env.step)
zero = jnp.zeros((8, env.action_size))
print(f"{'type':12s} {'lvl':>3s}  z_above_ground(0.5s/1s/2s)   min hb   tilt_ok   alive@2s")
for kind in ("flat", "stones", "gap", "climb_up", "stairs_up"):
    for lvl in (0, 4):
        cell = lvl * len(grid.types) + grid.types.index(kind)
        keys = jax.random.split(jax.random.PRNGKey(lvl), 8)
        st = reset(jnp.concatenate([keys, jnp.full((8, 1), cell, keys.dtype)], -1))
        rows, alive = [], np.ones(8, bool)
        for t in range(steps):
            st = step(st, zero)
            q = np.asarray(st.data.qpos)
            h = np.asarray(env.unwrapped._field.height(jnp.asarray(q[:, 0]), jnp.asarray(q[:, 1])))
            alive &= ~(np.asarray(st.done) > 0)
            rows.append(q[:, 2] - h)
        r = np.stack(rows)
        at = lambda s: float(np.mean(r[min(int(s * 50), steps - 1)]))
        print(f"{kind:12s} {lvl:3d}  {at(0.5):.3f} / {at(1.0):.3f} / {at(2.0):.3f}      {r.min():.3f}    "
              f"{float((np.asarray(st.obs['privileged_state'])[:, 54 + 2] < -0.8).mean()):.2f}     {alive.mean():.2f}")
