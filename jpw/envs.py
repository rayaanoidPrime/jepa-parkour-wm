"""Batched, auto-resetting environments.

Every env here follows the Brax/Playground convention used by the rest of the code:
`reset(keys[N, 2]) -> state`, `step(state, actions[N, A]) -> state`, with `state.obs`,
`state.reward[N]`, `state.done[N]` and `state.info["truncation"][N]`, and auto-reset on done
(the returned obs after a done step is the first obs of the next episode).

Optional extras the models can use:
  obs["heightmap"]        [N, H, W]  heightmap / depth image (EnvCfg.use_heightmap)
  state.info["contact"]   [N, C]     contact flags *after* the step, e.g. one per foot
  state.info["aux"]       [N, X]     physical state at the current time (Go1: base pos 3 + quat 4 + feet
                                     world pos 12), added by PlaygroundAdapter for Go1 envs
"""
from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from flax import struct

from jpw.config import EnvCfg
from jpw.spec import Spec


def get_obs(obs, use_privileged: bool) -> jax.Array:
    """Flatten Playground dict observations into the world-model input vector."""
    if isinstance(obs, Mapping):
        x = obs["state"]
        if use_privileged and "privileged_state" in obs:
            x = jnp.concatenate([x, obs["privileged_state"]], axis=-1)
        return x
    return obs


def get_hmap(obs, use_heightmap: bool):
    """Heightmap / depth image [N, H, W] from a dict observation, or None."""
    if use_heightmap and isinstance(obs, Mapping) and "heightmap" in obs:
        return obs["heightmap"]
    return None


def get_contact(state):
    """Contact flags [N, C] (float32) from state.info, or None if the env doesn't report them."""
    c = state.info.get("contact") if hasattr(state, "info") else None
    return None if c is None else c.astype(jnp.float32)


def get_aux(state):
    """Aux physical state [N, X] (float32) from state.info, or None."""
    a = state.info.get("aux") if hasattr(state, "info") else None
    return None if a is None else a.astype(jnp.float32)


def get_spec(state, cfg: EnvCfg, act_dim: int) -> Spec:
    """Describe what an env produces, from one reset state."""
    prop = state.obs["state"].shape[-1] if isinstance(state.obs, Mapping) else None
    obs_dim = get_obs(state.obs, cfg.use_privileged).shape[-1]
    h = get_hmap(state.obs, cfg.use_heightmap)
    c = get_contact(state)
    ax = get_aux(state)
    if cfg.use_heightmap and h is None:
        raise ValueError(f"env.use_heightmap=true but {cfg.name!r} provides no obs['heightmap']")
    return Spec(
        obs_dim=obs_dim, prop_dim=obs_dim if prop is None else prop, act_dim=act_dim,
        hmap_shape=None if h is None else tuple(h.shape[-2:]),
        contact_dim=0 if c is None else int(c.shape[-1]),
        aux_dim=0 if ax is None else int(ax.shape[-1]),
    )


def make_env(cfg: EnvCfg):
    if cfg.name == "dummy":
        return PointMassEnv(episode_length=cfg.episode_length, heightmap=cfg.use_heightmap)
    # Imported lazily so the dummy preset runs without MuJoCo installed.
    from mujoco_playground import registry, wrapper

    hmap_fn = None
    if cfg.name == PARKOUR_ENV:
        from jpw import parkour

        env, elev = parkour.build_env(cfg.terrain_seed, cfg.terrain_difficulty,
                                      cfg.unclipped_reward, cfg.alive_bonus)
        if cfg.use_heightmap:
            hmap_fn = parkour.make_heightmap_fn(elev)
    elif cfg.unclipped_reward or cfg.alive_bonus:
        from mujoco_playground._src.locomotion.go1 import joystick as go1_joystick

        from jpw.go1_reward import unclipped

        tasks = {"Go1JoystickFlatTerrain": "flat_terrain", "Go1JoystickRoughTerrain": "rough_terrain"}
        if cfg.name not in tasks:
            raise ValueError(f"unclipped_reward / alive_bonus is only implemented for {sorted(tasks)}")
        env = unclipped(go1_joystick.Joystick, cfg.alive_bonus)(task=tasks[cfg.name])
    else:
        env = registry.load(cfg.name)
    rand_fn = None
    if cfg.domain_randomization:
        if not cfg.name.startswith("Go1"):
            raise ValueError(f"env.domain_randomization is only implemented for Go1 envs, not {cfg.name!r}")
        from jpw.randomize import make_randomization_fn

        rand_fn = make_randomization_fn(cfg.num_envs, seed=cfg.terrain_seed)
    env = wrapper.wrap_for_brax_training(env, episode_length=cfg.episode_length, action_repeat=1,
                                         randomization_fn=rand_fn)
    return PlaygroundAdapter(env, hmap_fn, cfg.hmap_noise)


PARKOUR_ENV = "Go1Parkour"


class PlaygroundAdapter:
    """Adds what the models consume on top of a (brax-wrapped) Playground env:

    info["contact"]     = info["last_contact"] as float32 (per-foot contact after the step)
    obs["heightmap"]    = hmap_fn(data.qpos), if given, plus Gaussian noise (std `hmap_noise`
                          metres per cell) and a per-sample height bias of the same std

    Extras are stripped before calling the wrapped env so its auto-reset sees the structure it
    created.
    """

    def __init__(self, env, hmap_fn=None, hmap_noise: float = 0.0):
        self.env, self.hmap_fn, self.hmap_noise = env, hmap_fn, hmap_noise
        self.action_size = env.action_size
        # Go1: expose base pose + feet world positions as aux physical state
        base = getattr(env, "unwrapped", None)
        self._feet_site = getattr(base, "_feet_site_id", None)

    def __getattr__(self, name):  # forward anything else (dt, mj_model, ...)
        return getattr(self.env, name)

    def _add(self, state):
        info = dict(state.info)
        if "last_contact" in info:
            info["contact"] = info["last_contact"].astype(jnp.float32)
        if self._feet_site is not None:
            d = state.data
            feet = d.site_xpos[:, self._feet_site, :].reshape(d.qpos.shape[0], -1)
            info["aux"] = jnp.concatenate([d.qpos[:, :7], feet], axis=-1)
        obs = state.obs
        if self.hmap_fn is not None:
            hm = self.hmap_fn(state.data.qpos)
            if self.hmap_noise > 0:
                def noise(key):  # key: this env's rng (advanced by the env every step)
                    k1, k2 = jax.random.split(jax.random.fold_in(key, 7))
                    return self.hmap_noise * (jax.random.normal(k1, hm.shape[-2:])
                                              + jax.random.normal(k2, ()))
                hm = hm + jax.vmap(noise)(state.info["rng"])
            obs = {**obs, "heightmap": hm}
        return state.replace(info=info, obs=obs)

    def _strip(self, state):
        info = {k: v for k, v in state.info.items() if k not in ("contact", "aux")}
        obs = state.obs
        if self.hmap_fn is not None:
            obs = {k: v for k, v in obs.items() if k != "heightmap"}
        return state.replace(info=info, obs=obs)

    def reset(self, rng):
        return self._add(self.env.reset(rng))

    def step(self, state, action):
        return self._add(self.env.step(self._strip(state), action))


# --------------------------------------------------------------------------------------
# Point-mass sanity env: reach a random goal in 2D. Lets the full loop run on CPU.
# Default obs is [pos, vel, goal - pos]. With heightmap=True the env mimics the asymmetric
# setup: proprio is [pos, vel] only, the goal is visible as a bump on an 8x8 "terrain" heightmap
# (and as privileged state goal - pos). It always reports 2 contact flags: pos pinned to a wall.
# --------------------------------------------------------------------------------------
@struct.dataclass
class PMState:
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict
    pos: jax.Array
    vel: jax.Array
    goal: jax.Array
    t: jax.Array
    rng: jax.Array


class PointMassEnv:
    action_size = 2

    HMAP = 8

    def __init__(self, episode_length: int = 200, dt: float = 0.05, heightmap: bool = False):
        self.episode_length = episode_length
        self.dt = dt
        self.heightmap = heightmap
        self._grid = jnp.linspace(-2.0, 2.0, self.HMAP)

    def _obs(self, pos, vel, goal):
        if not self.heightmap:
            return jnp.concatenate([pos, vel, goal - pos], axis=-1)
        gx = (self._grid[None, :, None] - goal[:, 0, None, None]) ** 2
        gy = (self._grid[None, None, :] - goal[:, 1, None, None]) ** 2
        return {
            "state": jnp.concatenate([pos, vel], axis=-1),
            "privileged_state": goal - pos,
            "heightmap": jnp.exp(-(gx + gy) / 0.5),
        }

    @staticmethod
    def _contact(pos):
        return (jnp.abs(pos) >= 1.99).astype(jnp.float32)

    def reset(self, rng: jax.Array) -> PMState:
        def one(k):
            k_next, k_pos, k_goal = jax.random.split(k, 3)
            pos = jax.random.uniform(k_pos, (2,), minval=-1.0, maxval=1.0)
            goal = jax.random.uniform(k_goal, (2,), minval=-1.0, maxval=1.0)
            return k_next, pos, goal

        rng, pos, goal = jax.vmap(one)(rng)
        n = pos.shape[0]
        vel = jnp.zeros_like(pos)
        zeros = jnp.zeros((n,), jnp.float32)
        return PMState(
            obs=self._obs(pos, vel, goal), reward=zeros, done=zeros,
            info={"truncation": zeros, "contact": self._contact(pos)}, pos=pos, vel=vel, goal=goal,
            t=jnp.zeros((n,), jnp.int32), rng=rng,
        )

    def step(self, s: PMState, action: jax.Array) -> PMState:
        action = jnp.clip(action, -1.0, 1.0)
        vel = 0.95 * s.vel + 2.0 * self.dt * action
        pos = jnp.clip(s.pos + self.dt * vel, -2.0, 2.0)
        contact = self._contact(pos)  # state after the step, before any auto-reset
        dist = jnp.linalg.norm(s.goal - pos, axis=-1)
        reward = -0.1 * dist - 0.01 * jnp.sum(action**2, -1) + 0.1 * (dist < 0.1)
        t = s.t + 1
        trunc = (t >= self.episode_length).astype(jnp.float32)
        done = trunc

        fresh = self.reset(s.rng)
        m = done[:, None] > 0
        pos = jnp.where(m, fresh.pos, pos)
        vel = jnp.where(m, fresh.vel, vel)
        goal = jnp.where(m, fresh.goal, s.goal)
        t = jnp.where(done > 0, 0, t)
        return PMState(
            obs=self._obs(pos, vel, goal), reward=reward, done=done,
            info={"truncation": trunc, "contact": contact}, pos=pos, vel=vel, goal=goal, t=t, rng=fresh.rng,
        )
