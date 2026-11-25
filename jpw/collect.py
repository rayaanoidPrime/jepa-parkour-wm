"""Jitted rollout collection shared by training and rollout export."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from jpw.agent import Agent
from jpw.envs import get_contact, get_hmap, get_obs


def make_collect(env, agent: Agent, n_steps: int, use_priv: bool, use_hmap: bool = False,
                 random_policy: bool = False, par=None):
    """Returns collect(ts, env_state, key, ep_ret) -> (env_state, key, ep_ret, chunk).

    `chunk` is a dict of time-major arrays [n_steps, N, ...] in replay layout, plus per-step
    finished-episode return sums/counts ("fin_sum", "fin_cnt"). The acting latent is the one the
    deployed actor sees (student encoder when ac.asymmetric). With random_policy the actor is
    ignored and actions are uniform in [-1, 1] (use it to export data before any policy exists).
    With `par` (a jpw.parallel.Parallel) the envs / outputs are kept sharded over the mesh.
    """
    act_dim = agent.spec.act_dim

    def collect(ts, env_state, key, ep_ret):
        def body(carry, _):
            es, key, ep_ret = carry
            key, k = jax.random.split(key)
            obs = get_obs(es.obs, use_priv)
            hmap = get_hmap(es.obs, use_hmap)
            if random_policy:
                a = jax.random.uniform(k, (obs.shape[0], act_dim), minval=-1.0, maxval=1.0)
            else:
                a = agent.act(ts, obs, k, hmap)
            nxt = env.step(es, a)
            done = nxt.done.astype(jnp.float32)
            trunc = nxt.info["truncation"].astype(jnp.float32)
            reward = nxt.reward.astype(jnp.float32)
            ep_ret = ep_ret + reward
            fin_sum = jnp.sum(jnp.where(done > 0, ep_ret, 0.0))
            fin_cnt = jnp.sum(done)
            ep_ret = jnp.where(done > 0, 0.0, ep_ret)
            out = {"obs": obs, "act": a, "rew": reward, "done": done, "trunc": trunc,
                   "fin_sum": fin_sum, "fin_cnt": fin_cnt}
            if hmap is not None:
                out["hmap"] = hmap
            contact = get_contact(nxt)
            if contact is not None:
                out["contact"] = contact
            return (nxt, key, ep_ret), out

        (es, key, ep_ret), out = jax.lax.scan(body, (env_state, key, ep_ret), None, length=n_steps)
        if par is not None:
            n = ep_ret.shape[0]
            es, ep_ret = par.constrain_env((es, ep_ret), n)
            out = par.constrain_time(out)
        return es, key, ep_ret, out

    return jax.jit(collect)
