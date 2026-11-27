"""Train a JEPA world model and a policy inside it.

Each iteration:
  1. collect `collect_steps` steps from every simulator env with the current policy,
  2. append them to the on-device replay buffer,
  3. take `wm_updates` JEPA world-model steps and `ac_updates` imagination actor-critic steps.

Usage:
  python -m jpw.train --preset dummy                      # CPU sanity check, no MuJoCo
  python -m jpw.train --preset dummy_full                 # + heightmap, contact head, student actor
  python -m jpw.train --preset smoke                      # small MJX run
  python -m jpw.train --set env.num_envs=4096 ac.horizon=32
  python -m jpw.train --set train.resume=runs/x/ckpt_0001000.pkl        # continue a run
  python -m jpw.train --set train.init_data=data/go1 train.pretrain_updates=500   # offline stage 1
  python -m jpw.train --preset parkour_mgpu               # data parallel over all allocated GPUs
"""
from __future__ import annotations

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from jpw import checkpoint, replay
from jpw.agent import Agent
from jpw.collect import make_collect
from jpw.config import build_config
from jpw.envs import get_spec, make_env
from jpw.parallel import Parallel, replica_diff


def _check_spec(what: str, got, want) -> None:
    if got != want:
        raise ValueError(f"{what} was made for {got}, but this env produces {want}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default=None)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--run-name", default=None)
    args = p.parse_args(argv)

    from jpw.tracking import Tracker  # after arg parsing so --help doesn't need wandb

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
    par = None
    if t.num_devices != 1:
        par = Parallel(t.num_devices)
        par.check(n, cfg.wm.batch, cfg.ac.batch)
        print(f"data parallel over {par.n} devices: {n // par.n} envs, wm.batch={cfg.wm.batch // par.n}, "
              f"ac.batch={cfg.ac.batch // par.n} per device", flush=True)
    key = jax.random.PRNGKey(t.seed)
    key, k_env, k_agent = jax.random.split(key, 3)
    env_keys = jax.random.split(k_env, n)
    env_state = par.reset_env(env, env_keys, n) if par else jax.jit(env.reset)(env_keys)
    spec = get_spec(env_state, cfg.env, env.action_size)
    print(f"env={cfg.env.name} num_envs={n} spec={spec}", flush=True)

    agent = Agent(cfg, spec)
    if cfg.ac.grounded:
        from jpw import parkour

        base_env = env.unwrapped
        agent.task = base_env._goal_reward
        agent.hm_fn = base_env._hm_pose_fn
    elif cfg.ac.analytic_reward:
        from jpw.go1_state import Go1Task

        agent.task = Go1Task.from_env(env, cfg.env.alive_bonus)
    ts = agent.init(k_agent)
    start_it, env_steps = 0, 0
    if t.resume:
        _, ck_spec, _, ts, meta = checkpoint.load(t.resume)
        _check_spec(f"checkpoint {t.resume}", ck_spec, spec)
        start_it, env_steps = meta["it"] + 1, meta["env_steps"]
        print(f"resumed from {t.resume} (it={meta['it']}, env_steps={env_steps})", flush=True)
    if par:
        ts = par.replicate(ts)

    seq_len = cfg.wm.pred_horizon + 1
    if t.init_data:
        from jpw.export import load_rollouts

        arrays, data_spec = load_rollouts(t.init_data)
        _check_spec(f"data {t.init_data}", data_spec, spec)
        if arrays["obs"].shape[1] != n:
            raise ValueError(f"data has {arrays['obs'].shape[1]} envs but env.num_envs={n}; "
                             "export with the same num_envs")
        if par:
            rb, rb_sh = par.init_replay(t.replay_capacity, n, spec)
            add_p = par.make_add(rb_sh)
            for chunk in replay.chunks(arrays, t.replay_capacity):
                rb = add_p(rb, par.put_time(chunk))
        else:
            rb = replay.from_arrays(arrays, t.replay_capacity, spec)
        filled = int(rb.size)
        print(f"preloaded {filled} replay rows from {t.init_data}", flush=True)
        if t.pretrain_updates:
            pretrain = (par.make_learner(agent, rb)[1] if par
                        else jax.jit(agent.pretrain, donate_argnums=(0,)))
            for i in range(t.pretrain_updates):
                key, k = jax.random.split(key)
                ts, pm = pretrain(ts, rb, k)
                if i % max(t.log_every, 1) == 0 or i == t.pretrain_updates - 1:
                    pm = jax.device_get(pm)
                    print(f"[pretrain {i}] jepa={pm['wm/jepa']:.4g} latent_std={pm['wm/latent_std']:.4g} "
                          f"reward={pm['wm/reward']:.4g}", flush=True)
    else:
        if par:
            rb, rb_sh = par.init_replay(t.replay_capacity, n, spec)
        else:
            rb = replay.init(t.replay_capacity, n, spec)
        filled = 0

    sampler = None
    if cfg.env.name == "Go1Tiles" and cfg.env.curriculum:
        from jpw.curriculum import SuccessGuidedSampler

        grid = env.unwrapped._grid
        e = cfg.env
        sampler = SuccessGuidedSampler(grid.num_levels, grid.types, grid.prior(), e.sgs_window, e.sgs_target,
                                       e.sgs_kappa, e.sgs_temperature, neighbor=e.sgs_neighbor)

        def set_weights(es, w):  # per-env copy of the curriculum weights, read by FreshAutoReset at every respawn
            cur = jnp.broadcast_to(jnp.asarray(w, jnp.float32), (n, len(w)))
            cur = par.put_env(cur) if par else cur
            return es.replace(info={**es.info, "cur_w": cur})

        env_state = set_weights(env_state, sampler.weights())

    collect = make_collect(env, agent, t.collect_steps, cfg.env.use_privileged, cfg.env.use_heightmap,
                           par=par)
    if par:
        add = par.make_add(rb_sh)
        update = par.make_learner(agent, rb)[0]
    else:
        add = jax.jit(replay.add_chunk, donate_argnums=(0,))
        update = jax.jit(agent.update, donate_argnums=(0,))

    ep_ret = jnp.zeros((n,), jnp.float32)
    if par:
        ep_ret = par.put_env(ep_ret)
    ep_sum, ep_cnt = 0.0, 0.0
    window_steps, window_t0 = 0, time.time()
    last_m: dict = {}

    def save_ckpt(it: int, tag: str) -> None:
        path = os.path.join(run_dir, f"ckpt_{tag}.pkl")
        checkpoint.save(path, cfg, spec, ts, it, env_steps)
        print(f"saved {path}", flush=True)

    last_it = start_it - 1
    for it in range(start_it, t.iters):
        last_it = it
        env_state, key, ep_ret, out = collect(ts, env_state, key, ep_ret)
        if sampler is not None:
            cell, done, trunc, gd = (np.asarray(x) for x in jax.device_get(
                (out["ep_cell"], out["done"], out["trunc"], out["ep_gd"])))
            fin = done > 0
            sampler.update(cell[fin].astype(np.int64), (trunc[fin] > 0) & (gd[fin] < 0.5))
            env_state = set_weights(env_state, sampler.weights())
        rb = add(rb, {k: v for k, v in out.items() if k in replay.FIELDS})
        filled = min(filled + t.collect_steps, t.replay_capacity)
        env_steps += t.collect_steps * n
        window_steps += t.collect_steps * n

        if it >= t.warmup_iters and filled >= seq_len:
            key, k = jax.random.split(key)
            ts, last_m = update(ts, rb, k)

        ep_sum += float(jnp.sum(out["fin_sum"]))
        ep_cnt += float(jnp.sum(out["fin_cnt"]))

        if it % t.log_every == 0 or it == t.iters - 1:
            jax.block_until_ready(ts.wm)
            dt = time.time() - window_t0
            m = {k: v for k, v in jax.device_get(last_m).items()}
            m.update({
                "env/reward_per_step": float(jnp.mean(out["rew"])),
                "env/done_rate": float(jnp.mean(out["done"])),
                "env/episodes": ep_cnt,
                "perf/env_sps": window_steps / max(dt, 1e-6),
                "perf/iter": it,
            })
            if ep_cnt > 0:
                m["env/ep_return"] = ep_sum / ep_cnt
            if sampler is not None:
                m.update(sampler.log_dict())
            tracker.log(m, step=env_steps)
            ep_sum, ep_cnt, window_steps, window_t0 = 0.0, 0.0, 0, time.time()

        if t.ckpt_every and it > 0 and it % t.ckpt_every == 0:
            save_ckpt(it, f"{it:07d}")

    if t.ckpt_every and last_it >= start_it:
        save_ckpt(last_it, "final")
    if par and last_it >= start_it:
        print(f"replica sync check: max param diff across devices = {replica_diff(ts):.3g}", flush=True)
    tracker.finish()
    return ts


if __name__ == "__main__":
    main()
