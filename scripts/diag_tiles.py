"""Smoke / timing check of the Go1Tiles env (build, reset, step, fresh respawn on done).

  python scripts/diag_tiles.py [num_envs] [episode_length]
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from jpw.config import EnvCfg
from jpw.envs import get_obs, get_spec, make_env

n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
ep = int(sys.argv[2]) if len(sys.argv) > 2 else 20

t0 = time.time()
cfg = EnvCfg(name="Go1Tiles", num_envs=n, episode_length=ep, goal_task=True, use_heightmap=True, use_privileged=True)
env = make_env(cfg)
grid = env.unwrapped._grid
print(f"built env in {time.time() - t0:.1f}s; hfield {grid.nrow}x{grid.ncol}, {grid.num_cells} cells, "
      f"z range [{grid.h_min:.2f}, {grid.h_max:.2f}]", flush=True)

t0 = time.time()
st = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), n))
jax.block_until_ready(st)
spec = get_spec(st, cfg, env.action_size)
print(f"reset ok in {time.time() - t0:.1f}s; spec {spec}", flush=True)
print("cells at reset:", np.asarray(st.info["cell"]).tolist(), " goal dist:", np.round(np.asarray(st.info["goal_dist"]), 2).tolist())
cx, cy = grid.cell_centre(np.asarray(st.info["cell"]))
pos = np.asarray(st.data.qpos[:, :2])
print("spawn offset from tile centre (m):", np.round(np.abs(pos - np.stack([cx, cy], -1)).max(), 3))

step = jax.jit(env.step)
zero = jnp.zeros((n, env.action_size))
t0 = time.time()
st = step(st, zero)
jax.block_until_ready(st)
print(f"first step (compile) {time.time() - t0:.1f}s", flush=True)
t0 = time.time()
cells0 = np.asarray(st.info["cell"]).copy()
changed, ended_ok, tr_seen, steps_seen = 0, True, 0, []
for k in range(2, ep + 8):
    prev_cell = np.asarray(st.info["cell"])
    st = step(st, zero)
    done = np.asarray(st.done) > 0
    if done.any():
        ended_ok &= bool((np.asarray(st.info["ended_cell"])[done] == prev_cell[done]).all())
        changed += int((np.asarray(st.info["cell"])[done] != prev_cell[done]).sum())
        tr_seen += int((np.asarray(st.info["truncation"])[done] > 0).sum())
        steps_seen.append((k, int(done.sum())))
jax.block_until_ready(st)
print(f"{ep + 6} steps in {time.time() - t0:.1f}s; dones (step, count): {steps_seen}")
print(f"ended_cell matches the finished episode: {ended_ok}; respawned into a different tile: {changed}; truncations kept: {tr_seen}")
print("steps counter after respawn:", np.asarray(st.info["steps"]).tolist(), " extra[:,4] (time left):",
      np.round(np.asarray(st.obs["extra"][:, 4]), 3).tolist())
print("rewards finite:", bool(jnp.all(jnp.isfinite(st.reward))), " obs dim:", get_obs(st.obs, True).shape)
