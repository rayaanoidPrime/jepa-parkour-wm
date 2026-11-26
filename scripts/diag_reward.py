"""How much of the Go1 reward does Playground's clip(sum * dt, 0, ...) throw away?

Reconstructs the unclipped reward from the per-term values in state.metrics and compares it with
the reward the env returns, for a zero-action (stand still) and a random-action policy.

  python scripts/diag_reward.py [env_name] [num_envs] [steps]
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

from jpw.config import EnvCfg
from jpw.envs import make_env

name = sys.argv[1] if len(sys.argv) > 1 else "Go1JoystickFlatTerrain"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 100

env = make_env(EnvCfg(name=name, num_envs=n, episode_length=1000))
dt = 0.02
reset, step = jax.jit(env.reset), jax.jit(env.step)
for mode in ("zero", "random"):
    st = reset(jax.random.split(jax.random.PRNGKey(0), n))
    key = jax.random.PRNGKey(1)
    rows, terms_acc, dones = [], {}, 0.0
    for t in range(steps):
        key, k = jax.random.split(key)
        a = jnp.zeros((n, env.action_size)) if mode == "zero" else \
            jax.random.uniform(k, (n, env.action_size), minval=-1.0, maxval=1.0)
        st = step(st, a)
        terms = {k2: np.asarray(v) for k2, v in st.metrics.items() if k2.startswith("reward/")}
        raw = dt * sum(terms.values())
        rows.append((np.asarray(st.reward), raw))
        dones += float(jnp.sum(st.done))
        for k2, v in terms.items():
            terms_acc[k2] = terms_acc.get(k2, 0.0) + v.mean() / steps
    clipped = np.stack([r[0] for r in rows])
    raw = np.stack([r[1] for r in rows])
    print(f"\n=== {name}, {mode} actions, {n} envs x {steps} steps, dones={int(dones)}")
    print(f"clipped reward: mean {clipped.mean():+.5f}   fraction exactly 0: {(clipped == 0).mean():.3f}")
    print(f"unclipped reward: mean {raw.mean():+.5f}   fraction < 0: {(raw < 0).mean():.3f}   "
          f"min {raw.min():+.4f}  max {raw.max():+.4f}")
    print("mean scaled term values (x dt for per-step reward):")
    for k2 in sorted(terms_acc, key=lambda k: -abs(terms_acc[k])):
        print(f"   {k2:28s} {terms_acc[k2]:+.5f}")
