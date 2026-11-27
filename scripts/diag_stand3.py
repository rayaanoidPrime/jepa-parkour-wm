"""Trace a standing robot that gets 'tilt'-terminated: gravity z, base height above ground, base velocities."""
import jax, jax.numpy as jnp, numpy as np
from jpw.config import EnvCfg
from jpw.envs import make_env
cfg = EnvCfg(name="Go1Tiles", num_envs=8, episode_length=1000, goal_task=True, use_heightmap=True, use_privileged=True)
env = make_env(cfg)
grid, field = env.unwrapped._grid, env.unwrapped._field
reset, step = jax.jit(env.reset), jax.jit(env.step)
zero = jnp.zeros((8, env.action_size))
kind, lvl, tgt = "flat", 4, 0
cell = lvl * len(grid.types) + grid.types.index(kind)
keys = jax.random.split(jax.random.PRNGKey(lvl), 8)
st = reset(jnp.concatenate([keys, jnp.full((8, 1), cell, keys.dtype)], -1))
for t in range(1, 60):
    st = step(st, zero)
    q = np.asarray(st.data.qpos[tgt]); v = np.asarray(st.data.qvel[tgt])
    why = np.asarray(st.info["why"][tgt])
    # upright from quaternion: z-component of body z axis in world
    w, x, y, z = q[3:7]
    up = 1 - 2 * (x * x + y * y)
    if t % 4 == 0 or why.any():
        print(f"t{t:3d} up={up:+.3f} z={q[2]:.3f} lin={np.round(v[:3],2)} ang={np.round(v[3:6],2)} why={why}")
    if why.any(): break
