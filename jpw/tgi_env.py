"""Go1 goal-reaching on the TGI terrain grid (`Go1Tiles`): MJX, per-episode respawn, success-guided curriculum.

Terrain: `jpw.terrain_grid` (TGI's training terrains + stones + parkour families, one MuJoCo heightfield). Each
episode starts at the centre of a tile (a 2 m platform) with a random heading; the goal is 2.5-3.5 m away at a random
bearing and the episode lasts 8 s (episode_length 400), as in TGI-ANYmal. The reward is the AME2 goal reward
(`jpw.goal_task.GoalReward`) computed on the exact state; leaving the tile, tilting, the belly touching the terrain or
falling below z = -0.5 ends the episode.

Playground's auto-reset restores every env's *first* state, so a spawn would repeat forever. `FreshAutoReset` replaces
it: on done it resets the env for real, choosing the next tile from per-env curriculum weights (`info["cur_w"]`, which the
trainer updates from episode outcomes, see `jpw.curriculum`).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jpw import terrain_grid as tg
from jpw.parkour import _SCENE_XML as _PARKOUR_XML

FALL_Z = -0.5   # base below this: fell into a gap / off a stone (legit Go1 positions are > -0.4)

# the parkour scene with a free hfield pose: replace its fixed-pose geom
_SCENE_XML = _PARKOUR_XML.replace('pos="{posx} 0 0"', 'pos="{posx} {posy} {posz}"').replace(
    'size="{rx} {ry} {zmax} 0.1"', 'size="{rx} {ry} {zmax} 0.5"')


def build_env(seed: int = 0, goal_rate_weight: float = -0.01, hmap_step: float = 0.0):
    """Returns (unwrapped Go1Tiles env, TerrainGrid)."""
    from mujoco import mjx
    from mujoco_playground._src import mjx_env
    from mujoco_playground._src.locomotion.go1 import base as go1_base
    from mujoco_playground._src.locomotion.go1 import joystick as go1_joystick

    from jpw.goal_task import GoalReward

    grid = tg.TerrainGrid(seed)
    field = tg.TerrainField(grid)
    prior_cell = np.tile(grid.prior(), grid.num_levels).astype(np.float32)  # type prior spread over levels
    prior_cell = jnp.asarray(prior_cell / prior_cell.sum())

    class TilesJoystick(go1_joystick.Joystick):
        def __init__(self):
            sx, sy = (grid.ncol - 1) * tg.RES / 2, (grid.nrow - 1) * tg.RES / 2
            zmax = grid.h_max - grid.h_min
            xml = _SCENE_XML.format(nrow=grid.nrow, ncol=grid.ncol, rx=sx, ry=sy, zmax=zmax, posx=sx, posy=sy,
                                    posz=grid.h_min)
            path = Path(tempfile.gettempdir()) / f"jpw_tiles_{grid.nrow}x{grid.ncol}_{seed}.xml"
            path.write_text(xml)
            go1_base.Go1Env.__init__(self, xml_path=path.as_posix(), config=go1_joystick.default_config())
            hf = self._mj_model.hfield_adr[0]
            data = (grid.elev - grid.h_min) / zmax
            self._mj_model.hfield_data[hf:hf + grid.nrow * grid.ncol] = data.reshape(-1)
            self._mjx_model = mjx.put_model(self._mj_model)
            self._post_init()
            self._grid, self._field = grid, field
            self._hm_pose_fn = field.heightmap_pose_fn(hmap_step)
            self._goal_reward = GoalReward.from_env(
                self, field.height, pit_z=FALL_Z, weights={"rate": goal_rate_weight}, tile_size=tg.TILE)

        def sample_command(self, rng, x_k):
            return jnp.zeros(3)  # the goal drives the policy

        def reset(self, rng):
            """rng: a PRNG key [2], or [3] = (key, tile) to start in a given tile (tile = level * num_types + type;
            packed in the key array because Brax's wrappers expect an array)."""
            if rng.shape[-1] == 3:
                key, cell = rng[:2], rng[2]
            else:
                key, k_c = jax.random.split(rng)
                cell = jax.random.categorical(k_c, jnp.log(prior_cell))
            cell = jnp.asarray(cell, jnp.int32)
            cx, cy = grid.cell_centre(cell)
            key, k1, k2, k3, k4, k5, k6 = jax.random.split(key, 7)
            jit = jax.random.uniform(k1, (2,), minval=-0.2, maxval=0.2)
            sx, sy = cx + jit[0], cy + jit[1]
            yaw = jax.random.uniform(k2, (), minval=-jnp.pi, maxval=jnp.pi)
            qpos = self._init_q.at[0].set(sx).at[1].set(sy).at[2].set(0.29 + field.height(sx, sy))
            qpos = qpos.at[3:7].set(jnp.array([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)]))
            qvel = jnp.zeros(self.mjx_model.nv).at[0:6].set(jax.random.uniform(k3, (6,), minval=-0.1, maxval=0.1))
            data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])
            dist = jax.random.uniform(k4, (), minval=2.5, maxval=3.5)
            bearing = jax.random.uniform(k5, (), minval=0.0, maxval=2 * jnp.pi)
            goal = jnp.stack([sx + dist * jnp.cos(bearing), sy + dist * jnp.sin(bearing), bearing])
            info = {
                "rng": k6, "command": jnp.zeros(3),
                "steps_until_next_cmd": jnp.zeros((), jnp.int32),
                "last_act": jnp.zeros(self.mjx_model.nu), "last_last_act": jnp.zeros(self.mjx_model.nu),
                "feet_air_time": jnp.zeros(4), "last_contact": jnp.zeros(4, dtype=bool), "swing_peak": jnp.zeros(4),
                "steps_until_next_pert": jnp.zeros((), jnp.int32), "pert_duration_seconds": jnp.zeros(()),
                "pert_duration": jnp.zeros((), jnp.int32), "steps_since_last_pert": 0, "pert_steps": 0,
                "pert_dir": jnp.zeros(3), "pert_mag": jnp.zeros(()),
                "goal": goal, "goal_dist": dist, "cell": cell, "ended_cell": cell,
                "tile_xy": jnp.stack([cx, cy]), "cur_w": prior_cell, "why": jnp.zeros(5),
            }
            metrics = {f"reward/{k}": jnp.zeros(()) for k in self._config.reward_config.scales.keys()}
            metrics["swing_peak"] = jnp.zeros(())
            obs = self._get_obs(data, info)
            return mjx_env.State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

        def step(self, state, action):
            from jpw import go1_state as gs
            from jpw import goal_task as gt

            feet = self._feet_site_id
            a_prev, goal, tile = state.info["last_act"], state.info["goal"], state.info["tile_xy"]
            steps = state.info.get("steps", jnp.zeros(()))
            phys0 = gs.phys_targets(gs.obs_vector(state.obs), gs.aux_from_data(state.data, feet))
            out = super().step(state, action)
            phys1 = gs.phys_targets(gs.obs_vector(out.obs), gs.aux_from_data(out.data, feet))
            q = out.data.qpos
            pos, yaw = q[:3], gt.yaw_from_quat(q[3:7])
            t_left = gt.EPISODE_S - (steps + 1) * self.dt
            contact = out.info["last_contact"].astype(jnp.float32)
            r, _, _ = self._goal_reward.reward(
                gs.reward_inputs(phys0), gs.reward_inputs(phys1), pos, yaw, goal[:2], goal[2], t_left,
                action, a_prev, contact, tile_xy=tile)
            g_z = gs.split(phys1)["gravity"][2]
            hb = pos[2] - field.height(pos[0], pos[1])
            out_tile = jnp.any(jnp.abs(pos[:2] - tile) > tg.TILE / 2 - 0.05)
            tilt = -g_z < jnp.cos(self._goal_reward.term_angle)
            belly = hb < self._goal_reward.term_height
            fell = pos[2] < FALL_Z
            bad = tilt | belly | fell | out_tile
            out.info["why"] = jnp.stack([tilt, belly, fell, out_tile, out.done > 0]).astype(jnp.float32)
            done = jnp.maximum(out.done, bad.astype(out.done.dtype))
            out.info["goal_dist"] = jnp.sqrt(jnp.sum((goal[:2] - pos[:2]) ** 2) + 1e-8)
            return out.replace(reward=r, done=done)

    return TilesJoystick(), grid


def make_wrapped(env, episode_length: int, rand_fn=None):
    """Vmap (+ domain randomisation) -> Episode -> FreshAutoReset."""
    from brax.envs.wrappers import training as brax_training
    from mujoco_playground._src import wrapper

    base = (wrapper.BraxDomainRandomizationVmapWrapper(env, rand_fn) if rand_fn is not None
            else brax_training.VmapWrapper(env))
    return FreshAutoReset(brax_training.EpisodeWrapper(base, episode_length, 1))


class FreshAutoReset:
    """Auto-reset that really resets: finished envs get a freshly sampled spawn / goal, in a tile drawn from
    `info["cur_w"]` (per-env copy of the curriculum weights). The terminal step keeps `truncation`, `goal_dist` and
    `ended_cell` of the episode that just ended (the trainer needs them for the curriculum and for the replay)."""

    KEEP = ("truncation", "goal_dist", "cur_w", "episode_done", "episode_metrics", "why")

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["reset_key"] = rng[..., :2]
        return state

    def step(self, state, action):
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)
        k = jax.vmap(lambda key: jax.random.split(key, 3))(state.info["reset_key"])
        new_key, k_cell, k_res = k[:, 0], k[:, 1], k[:, 2]
        cell = jax.vmap(lambda kk, w: jax.random.categorical(kk, jnp.log(w + 1e-9)))(k_cell, state.info["cur_w"])
        fresh = self.env.reset(jnp.concatenate([k_res, cell[:, None].astype(k_res.dtype)], -1))
        done = state.done > 0

        def pick(f, s):
            d = done.reshape((-1,) + (1,) * (f.ndim - 1))
            return jnp.where(d, f, s)

        ended = state.info["cell"]
        info = {}
        for name, v in state.info.items():
            if name in self.KEEP or name not in fresh.info:
                info[name] = v
            else:
                info[name] = jax.tree_util.tree_map(pick, fresh.info[name], v)
        info["reset_key"], info["ended_cell"] = new_key, ended
        data = jax.tree_util.tree_map(pick, fresh.data, state.data)
        obs = jax.tree_util.tree_map(pick, fresh.obs, state.obs)
        return state.replace(data=data, obs=obs, info=info)
