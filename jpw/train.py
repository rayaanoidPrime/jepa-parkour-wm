"""Train a JEPA world model and a policy inside it.

Each iteration:
  1. collect `collect_steps` steps from every simulator env with the current policy,
  2. append them to the on-device replay buffer,
  3. take `wm_updates` JEPA world-model steps and `ac_updates` imagination actor-critic steps.

Usage:
  python -m jpw.train --preset dummy                      # CPU sanity check, no MuJoCo
  python -m jpw.train --preset smoke                      # small MJX run
  python -m jpw.train --set env.num_envs=4096 ac.horizon=32
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp

from jpw import replay
from jpw.agent import Agent
from jpw.config import build_config
from jpw.envs import get_obs, make_env
from jpw.models import sample_tanh
from jpw.tracking import Tracker


def make_collect(env, agent: Agent, n_steps: int, use_priv: bool):
    def collect(ts, env_state, key, ep_ret):
        def body(carry, _):
            es, key, ep_ret = carry
            key, k = jax.random.split(key)
            obs = get_obs(es.obs, use_priv)
            z = agent.encode(ts.wm, obs)
            mean, std = agent.actor.apply(ts.actor, z)
            a, _ = sample_tanh(mean, std, k)
            nxt = env.step(es, a)
            done = nxt.done.astype(jnp.float32)
            trunc = nxt.info["truncation"].astype(jnp.float32)
            reward = nxt.reward.astype(jnp.float32)
            ep_ret = ep_ret + reward
            fin_sum = jnp.sum(jnp.where(done > 0, ep_ret, 0.0))
            fin_cnt = jnp.sum(done)
            ep_ret = jnp.where(done > 0, 0.0, ep_ret)
            return (nxt, key, ep_ret), (obs, a, reward, done, trunc, fin_sum, fin_cnt)

        (es, key, ep_ret), out = jax.lax.scan(body, (env_state, key, ep_ret), None, length=n_steps)
        return es, key, ep_ret, out

    return jax.jit(collect)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default=None)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--run-name", default=None)
    args = p.parse_args(argv)

    cfg = build_config(args.preset, args.set)
    t = cfg.train
    job = os.environ.get("SLURM_JOB_ID", "local")
    run_name = args.run_name or f"{cfg.env.name}-s{t.seed}-{job}"
    run_dir = os.path.join(t.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"JAX {jax.__version__} | backend={jax.default_backend()} | devices={jax.devices()}", flush=True)
    tracker = Tracker(cfg, run_name, run_dir)

    env = make_env(cfg.env)
    n = cfg.env.num_envs
    key = jax.random.PRNGKey(t.seed)
    key, k_env, k_agent = jax.random.split(key, 3)
    env_state = jax.jit(env.reset)(jax.random.split(k_env, n))
    obs_dim = get_obs(env_state.obs, cfg.env.use_privileged).shape[-1]
    act_dim = env.action_size
    print(f"env={cfg.env.name} num_envs={n} obs_dim={obs_dim} act_dim={act_dim}", flush=True)

    agent = Agent(cfg, obs_dim, act_dim)
    ts = agent.init(k_agent)
    rb = replay.init(t.replay_capacity, n, obs_dim, act_dim)

    collect = make_collect(env, agent, t.collect_steps, cfg.env.use_privileged)
    add = jax.jit(replay.add, donate_argnums=(0,))
    update = jax.jit(agent.update, donate_argnums=(0,))

    seq_len = cfg.wm.pred_horizon + 1
    filled = 0
    ep_ret = jnp.zeros((n,), jnp.float32)
    ep_sum, ep_cnt, env_steps = 0.0, 0.0, 0
    window_steps, window_t0 = 0, time.time()
    last_m: dict = {}

    for it in range(t.iters):
        env_state, key, ep_ret, out = collect(ts, env_state, key, ep_ret)
        obs_c, a_c, r_c, d_c, tr_c, fin_sum, fin_cnt = out
        rb = add(rb, obs_c, a_c, r_c, d_c, tr_c)
        filled = min(filled + t.collect_steps, t.replay_capacity)
        env_steps += t.collect_steps * n
        window_steps += t.collect_steps * n

        if it >= t.warmup_iters and filled >= seq_len:
            key, k = jax.random.split(key)
            ts, last_m = update(ts, rb, k)

        ep_sum += float(jnp.sum(fin_sum))
        ep_cnt += float(jnp.sum(fin_cnt))

        if it % t.log_every == 0 or it == t.iters - 1:
            jax.block_until_ready(ts.wm)
            dt = time.time() - window_t0
            m = {k: v for k, v in jax.device_get(last_m).items()}
            m.update({
                "env/reward_per_step": float(jnp.mean(r_c)),
                "env/done_rate": float(jnp.mean(d_c)),
                "env/episodes": ep_cnt,
                "perf/env_sps": window_steps / max(dt, 1e-6),
                "perf/iter": it,
            })
            if ep_cnt > 0:
                m["env/ep_return"] = ep_sum / ep_cnt
            tracker.log(m, step=env_steps)
            ep_sum, ep_cnt, window_steps, window_t0 = 0.0, 0.0, 0, time.time()

        if t.ckpt_every and it > 0 and it % t.ckpt_every == 0:
            path = os.path.join(run_dir, f"ckpt_{it:07d}.pkl")
            with open(path, "wb") as f:
                pickle.dump({"cfg": cfg.to_dict(), "it": it, "env_steps": env_steps,
                             "params": jax.device_get({"wm": ts.wm, "actor": ts.actor, "critic": ts.critic})}, f)
            print(f"saved {path}", flush=True)

    tracker.finish()


if __name__ == "__main__":
    main()
