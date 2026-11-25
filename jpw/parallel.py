"""Data parallelism over the GPUs of one node.

  * envs and the replay buffer are sharded along the env axis (each GPU simulates its own envs
    and stores its own replay shard; nothing but gradients crosses devices),
  * parameters / optimiser state are replicated,
  * env stepping runs under jit with sharded inputs (XLA partitions the batched env),
  * the learner (`Agent.update` / `Agent.pretrain`) runs per device inside shard_map on its local
    replay shard, with gradients averaged over the mesh (see Agent._pmean).

Batch sizes in the config are *global*: each device samples batch // num_devices windows. Enable
with train.num_devices=N (0 = every visible device); num_envs and both batch sizes must be
divisible by it.
"""
from __future__ import annotations

import jax
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jpw import replay

AXIS = "d"


class Parallel:
    def __init__(self, num_devices: int):
        devs = jax.devices()
        if num_devices <= 0:
            num_devices = len(devs)
        if num_devices > len(devs):
            raise ValueError(f"train.num_devices={num_devices} but only {len(devs)} devices are visible")
        self.n = num_devices
        self.mesh = Mesh(np.array(devs[:num_devices]), (AXIS,))
        self.repl = NamedSharding(self.mesh, P())
        self.env = NamedSharding(self.mesh, P(AXIS))         # [N, ...]
        self.time = NamedSharding(self.mesh, P(None, AXIS))  # [T, N, ...]

    def check(self, num_envs: int, wm_batch: int, ac_batch: int) -> None:
        for name, v in (("env.num_envs", num_envs), ("wm.batch", wm_batch), ("ac.batch", ac_batch)):
            if v % self.n:
                raise ValueError(f"{name}={v} must be divisible by the number of devices ({self.n})")

    # ---- placement ----
    def replicate(self, tree):
        return jax.device_put(tree, self.repl)

    def put_env(self, x):
        return jax.device_put(x, self.env)

    def put_time(self, tree):
        return jax.device_put(tree, self.time)

    def _env_tree(self, shapes, n_envs: int):
        return jax.tree_util.tree_map(
            lambda s: self.env if (s.ndim >= 1 and s.shape[0] == n_envs) else self.repl, shapes)

    def constrain_env(self, tree, n_envs: int):
        """with_sharding_constraint on every leaf whose leading axis is the env axis."""
        return jax.tree_util.tree_map(
            lambda x: jax.lax.with_sharding_constraint(x, self.env)
            if (x.ndim >= 1 and x.shape[0] == n_envs) else x, tree)

    def constrain_time(self, tree):
        """... on every [T, N, ...] leaf (rank >= 2)."""
        return jax.tree_util.tree_map(
            lambda x: jax.lax.with_sharding_constraint(x, self.time) if x.ndim >= 2 else x, tree)

    def reset_env(self, env, keys, n_envs: int):
        shapes = jax.eval_shape(env.reset, keys)
        fn = jax.jit(env.reset, in_shardings=self.env, out_shardings=self._env_tree(shapes, n_envs))
        return fn(keys)

    # ---- replay ----
    def _replay_tree(self, rb, time, scalar):
        d = {k: (None if getattr(rb, k) is None else time) for k in replay.FIELDS}
        return rb.replace(**d, ptr=scalar, size=scalar)

    def init_replay(self, capacity: int, n_envs: int, spec):
        make = lambda: replay.init(capacity, n_envs, spec)
        shardings = self._replay_tree(jax.eval_shape(make), self.time, self.repl)
        return jax.jit(make, out_shardings=shardings)(), shardings

    def make_add(self, rb_shardings):
        return jax.jit(replay.add_chunk, donate_argnums=(0,), out_shardings=rb_shardings)

    # ---- learner ----
    def make_learner(self, agent, rb):
        """(update, pretrain): jitted shard_map versions of Agent.update / Agent.pretrain."""
        agent.axis, agent.ndev = AXIS, self.n
        specs = self._replay_tree(rb, P(None, AXIS), P())

        def wrap(fn):
            sm = shard_map(fn, mesh=self.mesh, in_specs=(P(), specs, P()), out_specs=(P(), P()),
                           check_rep=False)
            return jax.jit(sm, donate_argnums=(0,))

        return wrap(agent.update), wrap(agent.pretrain)


def replica_diff(tree) -> float:
    """Max abs difference between the per-device copies of every (replicated) leaf: 0 when in sync."""
    worst = 0.0
    for leaf in jax.tree_util.tree_leaves(tree):
        shards = [np.asarray(s.data) for s in leaf.addressable_shards]
        for s in shards[1:]:
            worst = max(worst, float(np.max(np.abs(s.astype(np.float64) - shards[0].astype(np.float64)))))
    return worst
