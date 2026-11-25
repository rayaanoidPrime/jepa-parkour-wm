"""Batched, auto-resetting environments.

Every env here follows the Brax/Playground convention used by the rest of the code:
`reset(keys[N, 2]) -> state`, `step(state, actions[N, A]) -> state`, with `state.obs`,
`state.reward[N]`, `state.done[N]` and `state.info["truncation"][N]`, and auto-reset on done
(the returned obs after a done step is the first obs of the next episode).
"""
from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from flax import struct

from jpw.config import EnvCfg


def get_obs(obs, use_privileged: bool) -> jax.Array:
    """Flatten Playground dict observations into the world-model input vector."""
    if isinstance(obs, Mapping):
        x = obs["state"]
        if use_privileged and "privileged_state" in obs:
            x = jnp.concatenate([x, obs["privileged_state"]], axis=-1)
        return x
    return obs


def make_env(cfg: EnvCfg):
    if cfg.name == "dummy":
        return PointMassEnv(episode_length=cfg.episode_length)
    # Imported lazily so the dummy preset runs without MuJoCo installed.
    from mujoco_playground import registry, wrapper

    env = registry.load(cfg.name)
    return wrapper.wrap_for_brax_training(
        env, episode_length=cfg.episode_length, action_repeat=1
    )


# --------------------------------------------------------------------------------------
# Point-mass sanity env: reach a random goal in 2D. Lets the full loop run on CPU.
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

    def __init__(self, episode_length: int = 200, dt: float = 0.05):
        self.episode_length = episode_length
        self.dt = dt

    @staticmethod
    def _obs(pos, vel, goal):
        return jnp.concatenate([pos, vel, goal - pos], axis=-1)

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
            info={"truncation": zeros}, pos=pos, vel=vel, goal=goal,
            t=jnp.zeros((n,), jnp.int32), rng=rng,
        )

    def step(self, s: PMState, action: jax.Array) -> PMState:
        action = jnp.clip(action, -1.0, 1.0)
        vel = 0.95 * s.vel + 2.0 * self.dt * action
        pos = jnp.clip(s.pos + self.dt * vel, -2.0, 2.0)
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
            info={"truncation": trunc}, pos=pos, vel=vel, goal=goal, t=t, rng=fresh.rng,
        )
