"""On-device, time-major replay buffer of shape [capacity, num_envs, ...].

Rows are written in time order, so any window of consecutive rows (starting no earlier than
the oldest row) is a contiguous trajectory per env. Row t holds (obs_t, a_t, r_t, done_t,
trunc_t); obs_{t+1} is in row t+1, and is the first obs of a new episode when done_t = 1.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Replay:
    obs: jax.Array    # [T, N, obs_dim]
    act: jax.Array    # [T, N, act_dim]
    rew: jax.Array    # [T, N]
    done: jax.Array   # [T, N]
    trunc: jax.Array  # [T, N]
    ptr: jax.Array    # int32 scalar: next row to write
    size: jax.Array   # int32 scalar: rows filled


def init(capacity: int, num_envs: int, obs_dim: int, act_dim: int) -> Replay:
    f = jnp.float32
    return Replay(
        obs=jnp.zeros((capacity, num_envs, obs_dim), f),
        act=jnp.zeros((capacity, num_envs, act_dim), f),
        rew=jnp.zeros((capacity, num_envs), f),
        done=jnp.zeros((capacity, num_envs), f),
        trunc=jnp.zeros((capacity, num_envs), f),
        ptr=jnp.zeros((), jnp.int32),
        size=jnp.zeros((), jnp.int32),
    )


def add(rb: Replay, obs, act, rew, done, trunc) -> Replay:
    """Append a chunk of S time steps (arrays shaped [S, N, ...])."""
    cap = rb.obs.shape[0]
    s = obs.shape[0]
    idx = (rb.ptr + jnp.arange(s)) % cap
    return rb.replace(
        obs=rb.obs.at[idx].set(obs),
        act=rb.act.at[idx].set(act),
        rew=rb.rew.at[idx].set(rew),
        done=rb.done.at[idx].set(done),
        trunc=rb.trunc.at[idx].set(trunc),
        ptr=(rb.ptr + s) % cap,
        size=jnp.minimum(rb.size + s, cap),
    )


def sample(rb: Replay, key: jax.Array, batch: int, length: int) -> dict:
    """Sample `batch` windows of `length` consecutive steps. Returns arrays shaped [B, L, ...]."""
    cap, n_envs = rb.obs.shape[0], rb.obs.shape[1]
    k_env, k_off = jax.random.split(key)
    env = jax.random.randint(k_env, (batch,), 0, n_envs)
    oldest = jnp.where(rb.size == cap, rb.ptr, 0)
    off = jax.random.randint(k_off, (batch,), 0, jnp.maximum(rb.size - length + 1, 1))
    t = (oldest + off[:, None] + jnp.arange(length)[None]) % cap  # [B, L]
    e = env[:, None]
    return {
        "obs": rb.obs[t, e],
        "act": rb.act[t, e],
        "rew": rb.rew[t, e],
        "done": rb.done[t, e],
        "trunc": rb.trunc[t, e],
    }
