"""What does the trained policy do on a stones tile? Per env: goal progress, feet over slits, first termination.
  python scripts/diag_stones.py CKPT [level] [type]"""
import sys

import jax, jax.numpy as jnp, numpy as np

from jpw import checkpoint
from jpw.envs import get_hmap, get_obs, make_env

ck, lvl, kind = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0, sys.argv[3] if len(sys.argv) > 3 else "stones"
cfg, spec, agent, ts, meta = checkpoint.load(ck)
N = 16
cfg.env.num_envs, cfg.env.domain_randomization, cfg.env.hmap_noise = N, False, 0.0
env = make_env(cfg.env)
u = env.unwrapped
grid, field, feet = u._grid, u._field, u._feet_site_id
cell = lvl * len(grid.types) + grid.types.index(kind)
keys = jax.random.split(jax.random.PRNGKey(1), N)
st = jax.jit(env.reset)(jnp.concatenate([keys, jnp.full((N, 1), cell, keys.dtype)], -1))
st = st.replace(info={**st.info, "cur_w": jnp.zeros((N, grid.num_cells)).at[:, cell].set(1.0)})
up, uh = cfg.env.use_privileged, cfg.env.use_heightmap


@jax.jit
def step(st):
    a = agent.act(ts, get_obs(st.obs, up), None, get_hmap(st.obs, uh), deterministic=True, ema=cfg.ac.ema > 0)
    n = env.step(st, a)
    fx = n.data.site_xpos[:, feet]                               # [N, 4, 3]
    gnd = field.height(fx[..., 0], fx[..., 1])
    slit = (gnd < -0.3)
    return n, slit, fx[..., 2] - gnd, n.data.qpos[:, 2] - field.height(n.data.qpos[:, 0], n.data.qpos[:, 1])


first = np.full(N, -1)
why = np.zeros((N, 5))
slit_steps = np.zeros(N)
foot_low = np.full(N, 9.0)
hb_min = np.full(N, 9.0)
gd0 = np.asarray(st.info["goal_dist"]).copy()
last_gd = gd0.copy()
for t in range(400):
    st, slit, fh, hb = step(st)
    live = first < 0
    slit_steps += live * np.asarray(slit).any(-1)
    foot_low = np.where(live, np.minimum(foot_low, np.asarray(fh).min(-1)), foot_low)
    hb_min = np.where(live, np.minimum(hb_min, np.asarray(hb)), hb_min)
    last_gd = np.where(live, np.asarray(st.info["goal_dist"]), last_gd)
    d = (np.asarray(st.done) > 0) & live
    first[d] = t
    why[d] = np.asarray(st.info["why"])[d]
    if (first >= 0).all():
        break
names = ("tilt", "belly", "fell", "left", "base")
print(f"{kind} level {lvl}: per-env  goal_d0 -> goal_d_end | steps_with_foot_over_slit | min base-above-ground | ended")
for e in range(N):
    end = "timeout" if first[e] < 0 else f"t={first[e]} " + ",".join(n for n, w in zip(names, why[e]) if w > 0)
    print(f"  env{e:2d} {gd0[e]:.2f} -> {last_gd[e]:.2f} | slit {int(slit_steps[e]):3d} | hb_min {hb_min[e]:.2f} | {end}")
print("SUMMARY reached<0.5m:", int(((last_gd < 0.5)).sum()), "/", N, " mean steps over slit:", slit_steps.mean())
