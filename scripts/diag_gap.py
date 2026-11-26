"""Model-exploitation check on a checkpoint (flat Go1, decoder heads + analytic reward).

  python scripts/diag_gap.py runs/exp-stab-S-s0/ckpt_final.pkl [num_envs] [rollout_steps]

1. Roll the (deterministic) policy in the REAL env and record the real states and actions.
2. From several real start states, roll the SAME policy inside the world model for H=32 steps (it picks
   its own actions in the model) and compute the analytic reward of the decoded states.
3. Compare, term by term, the imagined per-step reward with the real per-step reward of the real
   32-step window starting at the same state (same formula on the exact physical state).
   A large imagined > real gap on a term means the actor believes it does better than it does.
4. Also report the open-loop prediction error of the decoded state when the model is driven by the REAL
   actions (how the model's physical error grows with the horizon).
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

from jpw import checkpoint
from jpw import go1_state as gs
from jpw.envs import get_aux, get_contact, get_obs, make_env
from jpw.models import WorldModel

H = 32
path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 128
T = int(sys.argv[3]) if len(sys.argv) > 3 else 200

cfg, spec, agent, ts, meta = checkpoint.load(path)
cfg.env.num_envs, cfg.env.domain_randomization = n, False
env = make_env(cfg.env)
task = gs.Go1Task.from_env(env, cfg.env.alive_bonus)
agent.task = task
ema = cfg.ac.ema > 0
actor_params = ts.actor_ema if ema else ts.actor
print(f"{path} (it={meta['it']}, ema={ema}); {n} envs, real rollout {T} steps, horizon {H}")

# ---- 1. real rollout
reset, step = jax.jit(env.reset), jax.jit(env.step)
act = jax.jit(lambda o: agent.act(ts, o, None, None, deterministic=True, ema=ema))
st = reset(jax.random.split(jax.random.PRNGKey(0), n))
obs_l, aux_l, act_l, con_l, done_l = [], [], [], [], []
for t in range(T + 1):
    obs = get_obs(st.obs, True)
    obs_l.append(obs), aux_l.append(get_aux(st))
    if t == T:
        break
    a = act(obs)
    st = step(st, a)
    act_l.append(a), con_l.append(get_contact(st)), done_l.append(st.done)
obs_a, aux_a = jnp.stack(obs_l), jnp.stack(aux_l)           # [T+1, N, .]
act_a, con_a, done_a = jnp.stack(act_l), jnp.stack(con_l), jnp.stack(done_l)
phys_a = gs.phys_targets(obs_a, aux_a)                       # exact physical state at each time

# ---- 2. imagined vs real from the same start states
wm_apply = lambda m, *a: agent.wm.apply(ts.wm, *a, method=m)


@jax.jit
def imagine(z0, a_prev0, cmd):
    def body(carry, _):
        z, a_prev = carry
        mean, _ = agent.actor.apply(actor_params, z)
        a = jnp.tanh(mean)
        z1 = wm_apply(WorldModel.step, z, a)
        c = jax.nn.sigmoid(wm_apply(WorldModel.contact_logit, z, a))
        phys = agent.decode_phys(ts.wm, z1)
        terms = task.terms(phys, a, a_prev, cmd, c)
        r, d = task.reward(phys, a, a_prev, cmd, c)
        return (z1, a), (r, d, terms)
    _, out = jax.lax.scan(body, (z0, a_prev0), None, length=H)
    return out


@jax.jit
def real_window(t0):
    def one(k):
        t = t0 + k
        phys1 = phys_a[t + 1]
        a, a_prev = act_a[t], obs_a[t][:, gs.LAST_ACT]
        cmd = obs_a[t0][:, gs.CMD]
        terms = task.terms(phys1, a, a_prev, cmd, con_a[t])
        r, d = task.reward(phys1, a, a_prev, cmd, con_a[t])
        return r, terms
    return jax.vmap(one)(jnp.arange(H))


@jax.jit
def open_loop_error(t0):  # model driven by the REAL actions, decoded state vs real state
    def body(z, k):
        z1 = wm_apply(WorldModel.step, z, act_a[t0 + k])
        return z1, agent.decode_phys(ts.wm, z1)
    z0 = agent.encode(ts.wm, obs_a[t0])
    _, ph = jax.lax.scan(body, z0, jnp.arange(H))
    return ph - phys_a[t0 + 1:t0 + 1 + H]


imag_r, real_r = [], []
imag_t, real_t = {}, {}
err = []
for t0 in range(0, T - H - 1, 24):
    alive = ~jnp.any(done_a[t0:t0 + H], axis=0)                    # real window without a reset
    if int(alive.sum()) < 4:
        continue
    z0 = agent.encode(ts.wm, obs_a[t0])
    r_i, _, tm_i = imagine(z0, obs_a[t0][:, gs.LAST_ACT], obs_a[t0][:, gs.CMD])
    r_r, tm_r = real_window(t0)
    w = alive.astype(jnp.float32)[None]
    imag_r.append(float((r_i * w).sum() / (w.sum() * H) * H) / H), real_r.append(float((r_r * w).sum() / (w.sum() * H)))
    for k in tm_i:
        imag_t.setdefault(k, []).append(float((tm_i[k] * w).sum() / (w.sum() * H)) * task.dt)
        real_t.setdefault(k, []).append(float((tm_r[k] * w).sum() / (w.sum() * H)) * task.dt)
    e = open_loop_error(t0)
    err.append(np.asarray(jnp.sqrt(((e * w[..., None]) ** 2).sum(1) / w.sum())))

print(f"\nstart states used: {len(imag_r)}")
print(f"mean per-step reward  imagined {np.mean(imag_r):+.5f}   real {np.mean(real_r):+.5f}   "
      f"gap {np.mean(imag_r) - np.mean(real_r):+.5f}")
print(f"\n{'term (x dt)':16s} {'imagined':>10s} {'real':>10s} {'gap':>10s}")
for k in sorted(imag_t, key=lambda k: -abs(np.mean(imag_t[k]) - np.mean(real_t[k]))):
    i, r = np.mean(imag_t[k]), np.mean(real_t[k])
    print(f"{k:16s} {i:+10.5f} {r:+10.5f} {i - r:+10.5f}")

err = np.mean(np.stack(err), 0)                                   # [H, P] RMS error vs step
print("\nopen-loop decoded-state RMS error (model driven by the real actions), raw units")
print(f"{'group':10s} " + " ".join(f"k={k:<5d}" for k in (1, 8, 16, 32)))
for i, name in enumerate(gs.NAMES):
    sl = slice(gs.OFFS[i], gs.OFFS[i + 1])
    rms = np.sqrt((err[:, sl] ** 2).mean(-1))
    print(f"{name:10s} " + " ".join(f"{rms[k - 1]:<7.3f}" for k in (1, 8, 16, 32)))
