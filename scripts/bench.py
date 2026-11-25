"""Throughput benchmark, split into the parts that matter for an XLA-flag A/B:

  env_step   one batched env.step                           (MJX; GEMM-light)
  collect    policy + env for `collect_steps` steps         (what training does each iteration)
  update     agent.update on synthetic replay data          (GEMM-heavy: wm_updates + ac_updates steps)

The update is timed on random data of the right shapes, so it does not depend on the env being
numerically healthy (e.g. with XLA Triton GEMM on, MJX gives NaNs on ROCm, but the update timing
is still meaningful).

  python scripts/bench.py --preset parkour --set env.num_envs=2048 train.replay_capacity=200
"""
import argparse
import os
import time

import jax
import jax.numpy as jnp

from jpw import replay
from jpw.agent import Agent
from jpw.collect import make_collect
from jpw.config import build_config
from jpw.envs import get_spec, make_env


def timeit(fn, reps, warmup=2):
    for _ in range(warmup):
        out = fn()
    jax.block_until_ready(out)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="parkour")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--label", default=os.environ.get("BENCH_LABEL", ""))
    args = p.parse_args()

    cfg = build_config(args.preset, args.set)
    n, t = cfg.env.num_envs, cfg.train
    env = make_env(cfg.env)
    key = jax.random.PRNGKey(0)
    state = jax.jit(env.reset)(jax.random.split(key, n))
    spec = get_spec(state, cfg.env, env.action_size)
    agent = Agent(cfg, spec)
    ts = agent.init(key)
    print(f"BENCH label={args.label!r} env={cfg.env.name} num_envs={n} backend={jax.default_backend()} "
          f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}", flush=True)

    step = jax.jit(env.step)
    act = jnp.zeros((n, spec.act_dim))
    s = state
    def do_step():
        nonlocal s
        s = step(s, act)
        return s.reward
    dt = timeit(do_step, args.reps)
    nan = int(jnp.sum(~jnp.isfinite(s.reward)))
    print(f"BENCH env_step   {dt * 1e3:9.2f} ms  -> {n / dt:12.0f} env_sps   (nan rewards: {nan})", flush=True)

    collect = make_collect(env, agent, t.collect_steps, cfg.env.use_privileged, cfg.env.use_heightmap)
    cs = state
    ep_ret = jnp.zeros((n,))
    def do_collect():
        nonlocal cs
        cs, _, _, out = collect(ts, cs, key, ep_ret)
        return out["rew"]
    dt = timeit(do_collect, max(3, args.reps // 2))
    print(f"BENCH collect    {dt * 1e3:9.2f} ms  -> {n * t.collect_steps / dt:12.0f} env_sps   "
          f"({t.collect_steps} steps/iter incl. policy)", flush=True)

    rb = replay.init(t.replay_capacity, n, spec)
    k1, k2, k3 = jax.random.split(key, 3)
    cap = t.replay_capacity
    rb = replay.add(
        rb, jax.random.normal(k1, (cap, n, spec.obs_dim)), jax.random.uniform(k2, (cap, n, spec.act_dim)) * 2 - 1,
        jax.random.normal(k3, (cap, n)), jnp.zeros((cap, n)), jnp.zeros((cap, n)),
        None if spec.hmap_shape is None else jax.random.normal(k1, (cap, n, *spec.hmap_shape)),
        None if not spec.contact_dim else (jax.random.uniform(k2, (cap, n, spec.contact_dim)) > 0.5).astype(jnp.float32),
    )
    upd = jax.jit(agent.update, donate_argnums=(0,))
    box = [agent.init(key)]
    def do_update():
        box[0], m = upd(box[0], rb, key)
        return m["wm/loss"]
    dt = timeit(do_update, args.reps)
    print(f"BENCH update     {dt * 1e3:9.2f} ms  ({t.wm_updates} wm + {t.ac_updates} ac steps; "
          f"wm.batch={cfg.wm.batch} ac.batch={cfg.ac.batch} horizon={cfg.ac.horizon})", flush=True)


if __name__ == "__main__":
    main()
