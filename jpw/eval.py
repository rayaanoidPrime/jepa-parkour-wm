"""Evaluate a checkpoint with the *deployed* actor, deterministically.

  python -m jpw.eval --ckpt runs/Go1Parkour-s0-123/ckpt_final.pkl --episodes 256
  python -m jpw.eval --ckpt ... --set env.domain_randomization=true env.hmap_noise=0.02

The actor acts on the student encoder (proprioception + heightmap only) when the checkpoint was
trained with ac.asymmetric=true, i.e. exactly what would run on the robot. Evaluation defaults to
no domain randomisation and no heightmap noise; pass --set to stress it. For Go1Parkour it also
reports mean forward progress (m) and the fraction of episodes that reach the end of the course.
"""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from jpw import checkpoint
from jpw.config import apply_overrides
from jpw.envs import PARKOUR_ENV, get_hmap, get_obs, get_spec, make_env


def evaluate(cfg, agent, ts, spec, episodes: int, seed: int = 0) -> dict:
    env = make_env(cfg.env)
    n = cfg.env.num_envs
    key = jax.random.PRNGKey(seed)
    key, k_env = jax.random.split(key)
    state = jax.jit(env.reset)(jax.random.split(k_env, n))
    got = get_spec(state, cfg.env, env.action_size)
    if got != spec:
        raise ValueError(f"checkpoint was made for {spec}, but this env produces {got}")
    parkour = cfg.env.name == PARKOUR_ENV
    if parkour:
        from jpw.parkour import FINISH_X

    use_priv, use_hmap = cfg.env.use_privileged, cfg.env.use_heightmap

    @jax.jit
    def step(state, ret, length, xmax):
        a = agent.act(ts, get_obs(state.obs, use_priv), None, get_hmap(state.obs, use_hmap),
                      deterministic=True)
        nxt = env.step(state, a)
        done = nxt.done > 0
        ret = ret + nxt.reward
        length = length + 1
        x = nxt.data.qpos[:, 0] if parkour else jnp.zeros(n)
        xmax = jnp.maximum(xmax, x)
        out = (jnp.sum(jnp.where(done, ret, 0.0)), jnp.sum(jnp.where(done, length, 0)),
               jnp.sum(done), jnp.sum(jnp.where(done, xmax, 0.0)),
               jnp.sum(done & (xmax >= (FINISH_X - 0.1 if parkour else jnp.inf))))
        return (nxt, jnp.where(done, 0.0, ret), jnp.where(done, 0, length),
                jnp.where(done, x, xmax)), out

    carry = (state, jnp.zeros(n), jnp.zeros(n, jnp.int32), jnp.zeros(n))
    tot = np.zeros(5)
    max_steps = cfg.env.episode_length * (-(-episodes // n) + 1)
    for _ in range(max_steps):
        carry, out = step(*carry)
        tot += np.array([float(o) for o in out])
        if tot[2] >= episodes:
            break
    ret, length, count, xmax, success = tot
    res = {"episodes": int(count), "return": ret / max(count, 1), "length": length / max(count, 1)}
    if parkour:
        res.update({"progress_m": xmax / max(count, 1), "success_rate": success / max(count, 1)})
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args(argv)

    cfg, spec, agent, ts, meta = checkpoint.load(args.ckpt)
    cfg.env.num_envs = args.num_envs
    cfg.env.domain_randomization = False
    cfg.env.hmap_noise = 0.0
    apply_overrides(cfg, args.set)
    print(f"loaded {args.ckpt} (it={meta['it']}, env_steps={meta['env_steps']}); "
          f"actor={'student (proprio+heightmap)' if cfg.ac.asymmetric else 'full latent'}", flush=True)
    res = evaluate(cfg, agent, ts, spec, args.episodes, args.seed)
    print("EVAL " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()))
    return res


if __name__ == "__main__":
    main()
