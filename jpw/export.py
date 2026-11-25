"""Export simulator rollouts to disk, and load them back.

  python -m jpw.export --preset smoke --out data/go1 --steps 2000                 # random policy
  python -m jpw.export --ckpt runs/x/ckpt_0001000.pkl --out data/x --steps 2000   # trained policy

Output is a directory of `rollouts_0000.npz, ...` shards (time-major arrays [T, N, ...] in
replay layout: obs, act, rew, done, trunc, and hmap / contact if the env has them) plus
`meta.json` (config, spec). Load them with `load_rollouts`, or point `train.init_data` at the
directory to preload the replay buffer (optionally with `train.pretrain_updates` for offline
world-model pretraining).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from jpw import checkpoint, replay
from jpw.agent import Agent
from jpw.collect import make_collect
from jpw.config import apply_overrides, build_config
from jpw.envs import get_spec, make_env
from jpw.spec import Spec

ARRAY_KEYS = replay.FIELDS


def export_rollouts(cfg, spec: Spec, agent: Agent, ts, out_dir: str, steps: int, seed: int = 0,
                    random_policy: bool = False, chunk_steps: int = 256, env=None) -> int:
    """Run the policy for `steps` env steps per env and write shards. Returns shards written."""
    os.makedirs(out_dir, exist_ok=True)
    env = env or make_env(cfg.env)
    n = cfg.env.num_envs
    key = jax.random.PRNGKey(seed)
    key, k_env = jax.random.split(key)
    env_state = jax.jit(env.reset)(jax.random.split(k_env, n))
    collect = make_collect(env, agent, chunk_steps, cfg.env.use_privileged, cfg.env.use_heightmap,
                           random_policy=random_policy)
    ep_ret = jnp.zeros((n,), jnp.float32)
    shards = 0
    for lo in range(0, steps, chunk_steps):
        env_state, key, ep_ret, out = collect(ts, env_state, key, ep_ret)
        take = min(chunk_steps, steps - lo)
        arrs = {k: np.asarray(out[k])[:take] for k in ARRAY_KEYS if k in out}
        if "hmap" in arrs:
            arrs["hmap"] = arrs["hmap"].astype(np.float16)  # npz has no bfloat16
        np.savez(os.path.join(out_dir, f"rollouts_{shards:04d}.npz"), **arrs)
        shards += 1
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"cfg": cfg.to_dict(), "spec": spec.to_dict(), "steps": steps,
                   "num_envs": n, "random_policy": random_policy}, f, indent=2)
    return shards


def load_rollouts(path: str) -> tuple[dict, Spec]:
    """Load an exported directory -> (time-major numpy arrays, Spec)."""
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    files = sorted(glob.glob(os.path.join(path, "rollouts_*.npz")))
    if not files:
        raise FileNotFoundError(f"no rollouts_*.npz in {path}")
    parts: dict[str, list] = {}
    for fn in files:
        with np.load(fn) as z:
            for k in z.files:
                parts.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}, Spec.from_dict(meta["spec"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=1000, help="env steps per env")
    p.add_argument("--ckpt", default=None, help="policy checkpoint; omit for a random policy")
    p.add_argument("--preset", default=None)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.ckpt:
        cfg, spec, agent, ts, meta = checkpoint.load(args.ckpt)
        apply_overrides(cfg, args.set)  # e.g. env.num_envs=...
        print(f"loaded {args.ckpt} (it={meta['it']})", flush=True)
    else:
        cfg = build_config(args.preset, args.set)
        env = make_env(cfg.env)
        state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(args.seed), cfg.env.num_envs))
        spec = get_spec(state, cfg.env, env.action_size)
        agent = Agent(cfg, spec)
        ts = agent.init(jax.random.PRNGKey(args.seed))
    n = export_rollouts(cfg, spec, agent, ts, args.out, args.steps, args.seed,
                        random_policy=args.ckpt is None)
    print(f"wrote {n} shards ({args.steps} steps x {cfg.env.num_envs} envs) to {args.out}", flush=True)


if __name__ == "__main__":
    main()
