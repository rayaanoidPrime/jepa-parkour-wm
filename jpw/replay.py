"""On-device, time-major replay buffer of shape [capacity, num_envs, ...].

Rows are written in time order, so any window of consecutive rows (starting no earlier than
the oldest row) is a contiguous trajectory per env. Row t holds (obs_t, a_t, r_t, done_t,
trunc_t, contact_t); obs_{t+1} is in row t+1, and is the first obs of a new episode when
done_t = 1. r_t, done_t and contact_t describe the state *after* applying a_t.

Optional fields (None when the env doesn't provide them):
  hmap     [T, N, H, W]  heightmap / depth image seen with obs_t, stored in bfloat16
  contact  [T, N, C]     contact flags after step t
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from jpw.spec import Spec

FIELDS = ("obs", "act", "rew", "done", "trunc", "hmap", "contact")


@struct.dataclass
class Replay:
    obs: jax.Array    # [T, N, obs_dim]
    act: jax.Array    # [T, N, act_dim]
    rew: jax.Array    # [T, N]
    done: jax.Array   # [T, N]
    trunc: jax.Array  # [T, N]
    ptr: jax.Array    # int32 scalar: next row to write
    size: jax.Array   # int32 scalar: rows filled
    hmap: jax.Array | None = None      # [T, N, H, W]
    contact: jax.Array | None = None   # [T, N, C]


def init(capacity: int, num_envs: int, spec: Spec) -> Replay:
    f = jnp.float32
    return Replay(
        obs=jnp.zeros((capacity, num_envs, spec.obs_dim), f),
        act=jnp.zeros((capacity, num_envs, spec.act_dim), f),
        rew=jnp.zeros((capacity, num_envs), f),
        done=jnp.zeros((capacity, num_envs), f),
        trunc=jnp.zeros((capacity, num_envs), f),
        ptr=jnp.zeros((), jnp.int32),
        size=jnp.zeros((), jnp.int32),
        hmap=None if spec.hmap_shape is None
        else jnp.zeros((capacity, num_envs, *spec.hmap_shape), jnp.bfloat16),
        contact=None if not spec.contact_dim
        else jnp.zeros((capacity, num_envs, spec.contact_dim), f),
    )


def add(rb: Replay, obs, act, rew, done, trunc, hmap=None, contact=None) -> Replay:
    """Append a chunk of S time steps (arrays shaped [S, N, ...])."""
    cap = rb.obs.shape[0]
    s = obs.shape[0]
    idx = (rb.ptr + jnp.arange(s)) % cap
    upd = dict(
        obs=rb.obs.at[idx].set(obs),
        act=rb.act.at[idx].set(act),
        rew=rb.rew.at[idx].set(rew),
        done=rb.done.at[idx].set(done),
        trunc=rb.trunc.at[idx].set(trunc),
        ptr=(rb.ptr + s) % cap,
        size=jnp.minimum(rb.size + s, cap),
    )
    if rb.hmap is not None:
        upd["hmap"] = rb.hmap.at[idx].set(hmap.astype(rb.hmap.dtype))
    if rb.contact is not None:
        upd["contact"] = rb.contact.at[idx].set(contact.astype(rb.contact.dtype))
    return rb.replace(**upd)


def add_chunk(rb: Replay, chunk: dict) -> Replay:
    return add(rb, **{k: chunk.get(k) for k in FIELDS})


def from_arrays(arrays: dict, capacity: int, spec: Spec) -> Replay:
    """Build a replay buffer from time-major numpy arrays (e.g. loaded exported rollouts).

    Keeps the most recent `capacity` rows if there are more.
    """
    t, n = arrays["obs"].shape[:2]
    rb = init(capacity, n, spec)
    for lo in range(max(0, t - capacity), t, 256):
        hi = min(lo + 256, t)
        rb = add_chunk(rb, {k: jnp.asarray(v[lo:hi]) for k, v in arrays.items() if k in FIELDS})
    return rb


def to_numpy(chunk: dict) -> dict:
    return {k: np.asarray(v) for k, v in chunk.items() if v is not None}


def sample(rb: Replay, key: jax.Array, batch: int, length: int) -> dict:
    """Sample `batch` windows of `length` consecutive steps. Returns arrays shaped [B, L, ...]."""
    cap, n_envs = rb.obs.shape[0], rb.obs.shape[1]
    k_env, k_off = jax.random.split(key)
    env = jax.random.randint(k_env, (batch,), 0, n_envs)
    oldest = jnp.where(rb.size == cap, rb.ptr, 0)
    off = jax.random.randint(k_off, (batch,), 0, jnp.maximum(rb.size - length + 1, 1))
    t = (oldest + off[:, None] + jnp.arange(length)[None]) % cap  # [B, L]
    e = env[:, None]
    out = {
        "obs": rb.obs[t, e],
        "act": rb.act[t, e],
        "rew": rb.rew[t, e],
        "done": rb.done[t, e],
        "trunc": rb.trunc[t, e],
    }
    if rb.hmap is not None:
        out["hmap"] = rb.hmap[t, e].astype(jnp.float32)
    if rb.contact is not None:
        out["contact"] = rb.contact[t, e]
    return out
