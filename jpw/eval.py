"""Evaluate a checkpoint with the *deployed* actor, deterministically.

  python -m jpw.eval --ckpt runs/Go1Parkour-s0-123/ckpt_final.pkl --episodes 256
  python -m jpw.eval --ckpt ... --set env.domain_randomization=true env.hmap_noise=0.02

The actor acts on the student encoder (proprioception + heightmap only) when the checkpoint was
trained with ac.asymmetric=true, i.e. exactly what would run on the robot. If the run used ac.ema > 0 the
EMA actor is evaluated (override with --set ac.ema=0 for the raw one). Evaluation defaults to
no domain randomisation and no heightmap noise; pass --set to stress it. For Go1Parkour it also
reports mean forward progress (m) and the fraction of episodes that reach the end of the course.
For Go1 envs it also prints the mean per-step contribution of every reward term (reward/<term> * dt),
which shows what the policy is paying for and what it is being penalised for.
"""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from jpw import checkpoint
from jpw.config import apply_overrides
from jpw.envs import PARKOUR_ENV, TILES_ENV, get_hmap, get_obs, get_spec, make_env


def evaluate(cfg, agent, ts, spec, episodes: int, seed: int = 0, only_types=None, only_levels=None) -> dict:
    env = make_env(cfg.env)
    n = cfg.env.num_envs
    key = jax.random.PRNGKey(seed)
    key, k_env = jax.random.split(key)
    state = jax.jit(env.reset)(jax.random.split(k_env, n))
    got = get_spec(state, cfg.env, env.action_size)
    if got != spec:
        raise ValueError(f"checkpoint was made for {spec}, but this env produces {got}")
    goal = cfg.env.goal_task
    tiles = cfg.env.name == TILES_ENV
    parkour = cfg.env.name == PARKOUR_ENV and not goal
    n_cells = env.unwrapped._grid.num_cells if tiles else 1
    if tiles:  # evaluate every tile (type x level) equally often, or only the chosen types / levels
        g = env.unwrapped._grid
        sel = np.ones((g.num_levels, len(g.types)), np.float32)
        if only_types:
            sel *= np.asarray([[t in only_types for t in g.types]], np.float32)
        if only_levels is not None:
            sel *= np.asarray([[lv in only_levels] for lv in range(g.num_levels)], np.float32)
        w = (sel / sel.sum()).reshape(-1)
        state = state.replace(info={**state.info, "cur_w": jnp.broadcast_to(jnp.asarray(w), (n, n_cells))})
    if parkour:
        from jpw.parkour import FINISH_X

    use_priv, use_hmap = cfg.env.use_privileged, cfg.env.use_heightmap

    @jax.jit
    def step(state, ret, length, xmax):
        a = agent.act(ts, get_obs(state.obs, use_priv), None, get_hmap(state.obs, use_hmap),
                      deterministic=True, ema=cfg.ac.ema > 0)
        nxt = env.step(state, a)
        done = nxt.done > 0
        ret = ret + nxt.reward
        length = length + 1
        x = nxt.data.qpos[:, 0] if parkour else jnp.zeros(n)
        xmax = jnp.maximum(xmax, x)
        if goal:  # final distance to the goal at episode end; success = reached (< 0.5 m) at the time limit
            gd, trunc = nxt.info["goal_dist"], nxt.info["truncation"] > 0
            prog = jnp.sum(jnp.where(done, gd, 0.0))
            succ = jnp.sum(done & trunc & (gd < 0.5))
            n_trunc = jnp.sum(done & trunc)
        else:
            prog = jnp.sum(jnp.where(done, xmax, 0.0))
            succ = jnp.sum(done & (xmax >= (FINISH_X - 0.1 if parkour else jnp.inf)))
            n_trunc = jnp.zeros(())
        out = (jnp.sum(jnp.where(done, ret, 0.0)), jnp.sum(jnp.where(done, length, 0)),
               jnp.sum(done), prog, succ, n_trunc)
        if goal and "obstacle" in nxt.info:  # split by terrain relief between spawn and goal (flat vs obstacle on the way)
            obst = nxt.info["obstacle"] > 0.03
            ok = done & trunc & (gd < 0.5)
            out = out + (jnp.sum(done & ~obst), jnp.sum(ok & ~obst), jnp.sum(jnp.where(done & ~obst, gd, 0.0)),
                         jnp.sum(done & obst), jnp.sum(ok & obst), jnp.sum(jnp.where(done & obst, gd, 0.0)))
        else:
            out = out + (jnp.zeros(()),) * 6
        terms = {k: jnp.sum(v) for k, v in getattr(nxt, "metrics", {}).items() if k.startswith("reward/")}
        if tiles:
            ok = (done & trunc & (gd < 0.5)).astype(jnp.float32)
            cells = (jnp.zeros(n_cells).at[nxt.info["ended_cell"]].add(done.astype(jnp.float32)),
                     jnp.zeros(n_cells).at[nxt.info["ended_cell"]].add(ok))
        else:
            cells = (jnp.zeros(1), jnp.zeros(1))
        return (nxt, jnp.where(done, 0.0, ret), jnp.where(done, 0, length),
                jnp.where(done, x, xmax)), out, terms, cells

    cell_n, cell_s = np.zeros(n_cells), np.zeros(n_cells)
    carry = (state, jnp.zeros(n), jnp.zeros(n, jnp.int32), jnp.zeros(n))
    tot = np.zeros(12)
    term_tot: dict[str, float] = {}
    steps_done = 0
    max_steps = cfg.env.episode_length * (-(-episodes // n) + 1)
    for _ in range(max_steps):
        carry, out, terms, cells = step(*carry)
        tot += np.array([float(o) for o in out])
        cell_n += np.asarray(cells[0])
        cell_s += np.asarray(cells[1])
        for k, v in terms.items():
            term_tot[k] = term_tot.get(k, 0.0) + float(v)
        steps_done += 1
        if tot[2] >= episodes:
            break
    ret, length, count, xmax, success, n_trunc, n_flat, s_flat, d_flat, n_obs, s_obs, d_obs = tot
    res = {"episodes": int(count), "return": ret / max(count, 1), "length": length / max(count, 1)}
    if parkour:
        res.update({"progress_m": xmax / max(count, 1), "success_rate": success / max(count, 1)})
    if goal:
        res.update({"final_dist_m": xmax / max(count, 1), "goal_success_rate": success / max(count, 1),
                    "timeout_frac": n_trunc / max(count, 1),
                    "flat_n": int(n_flat), "flat_success": s_flat / max(n_flat, 1), "flat_dist_m": d_flat / max(n_flat, 1),
                    "obst_n": int(n_obs), "obst_success": s_obs / max(n_obs, 1), "obst_dist_m": d_obs / max(n_obs, 1)})
    if tiles:
        g = env.unwrapped._grid
        nt = len(g.types)
        res["by_type"] = {name: (float(cell_n.reshape(-1, nt)[:, j].sum()), float(cell_s.reshape(-1, nt)[:, j].sum()))
                          for j, name in enumerate(g.types)}
        res["by_level"] = [(float(cell_n.reshape(-1, nt)[i].sum()), float(cell_s.reshape(-1, nt)[i].sum()))
                           for i in range(g.num_levels)]
    dt = float(getattr(env, "dt", 0.02))
    res["terms"] = {k[len("reward/"):]: dt * v / max(steps_done * n, 1) for k, v in term_tot.items()}
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--types", nargs="*", default=None, help="Go1Tiles: evaluate only these terrain types")
    p.add_argument("--levels", nargs="*", type=int, default=None, help="Go1Tiles: evaluate only these levels")
    args = p.parse_args(argv)

    cfg, spec, agent, ts, meta = checkpoint.load(args.ckpt)
    cfg.env.num_envs = args.num_envs
    cfg.env.domain_randomization = False
    cfg.env.hmap_noise = 0.0
    apply_overrides(cfg, args.set)
    print(f"loaded {args.ckpt} (it={meta['it']}, env_steps={meta['env_steps']}); "
          f"actor={'student (proprio+heightmap)' if cfg.ac.asymmetric else 'full latent'}", flush=True)
    res = evaluate(cfg, agent, ts, spec, args.episodes, args.seed, args.types, args.levels)
    terms = res.pop("terms", {})
    by_type, by_level = res.pop("by_type", None), res.pop("by_level", None)
    print("EVAL " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()))
    if terms:
        print("TERMS (mean per-step reward contribution, sorted by size; their sum is the task reward)")
        for k in sorted(terms, key=lambda k: -abs(terms[k])):
            print(f"TERM {k:16s} {terms[k]:+.5f}")
        print(f"TERM {'TOTAL':16s} {sum(terms.values()):+.5f}")
    if by_type:
        print("TYPES  " + "  ".join(f"{k}={s / max(n_, 1):.2f}({int(n_)})" for k, (n_, s) in by_type.items()))
        print("LEVELS " + "  ".join(f"L{i}={s / max(n_, 1):.2f}({int(n_)})" for i, (n_, s) in enumerate(by_level)))
    res["terms"] = terms
    return res


if __name__ == "__main__":
    main()
