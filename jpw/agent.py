"""JEPA world-model learning and actor-critic learning in imagination."""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import struct

from jpw import replay
from jpw.config import Config
from jpw.models import Actor, Critic, WorldModel, sample_tanh, symlog

sg = jax.lax.stop_gradient


@struct.dataclass
class TrainState:
    wm: Any
    wm_target: Any
    actor: Any
    critic: Any
    critic_target: Any
    wm_opt: Any
    actor_opt: Any
    critic_opt: Any
    ret_scale: jax.Array


def _vicreg(z: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Variance and covariance terms on a batch of latents z[n, d]."""
    z = z.reshape(-1, z.shape[-1])
    n, d = z.shape
    std = jnp.sqrt(jnp.var(z, axis=0) + 1e-4)
    var_loss = jnp.mean(jax.nn.relu(1.0 - std))
    zc = z - z.mean(0)
    cov = (zc.T @ zc) / (n - 1)
    off = cov - jnp.diag(jnp.diag(cov))
    cov_loss = jnp.sum(off**2) / d
    return var_loss, cov_loss


def lambda_returns(rew, cont, values, gamma, lam):
    """rew, cont: [H, B]; values: [H+1, B] -> returns [H, B]."""

    def body(nxt, xs):
        r, c, v_next = xs
        ret = r + gamma * c * ((1.0 - lam) * v_next + lam * nxt)
        return ret, ret

    _, rets = jax.lax.scan(body, values[-1], (rew, cont, values[1:]), reverse=True)
    return rets


class Agent:
    def __init__(self, cfg: Config, obs_dim: int, act_dim: int):
        self.cfg = cfg
        self.obs_dim, self.act_dim = obs_dim, act_dim
        w, a = cfg.wm, cfg.ac
        self.wm = WorldModel(w.latent_dim, w.hidden, w.depth)
        self.actor = Actor(act_dim, a.hidden, a.depth, a.min_std, a.max_std)
        self.critic = Critic(a.hidden, a.depth)
        self.wm_tx = optax.chain(optax.clip_by_global_norm(w.grad_clip), optax.adam(w.lr))
        self.actor_tx = optax.chain(optax.clip_by_global_norm(a.grad_clip), optax.adam(a.actor_lr))
        self.critic_tx = optax.chain(optax.clip_by_global_norm(a.grad_clip), optax.adam(a.critic_lr))

    # ------------------------------------------------------------------ init
    def init(self, key: jax.Array) -> TrainState:
        k1, k2, k3 = jax.random.split(key, 3)
        obs = jnp.zeros((1, self.obs_dim))
        act = jnp.zeros((1, self.act_dim))
        z = jnp.zeros((1, self.cfg.wm.latent_dim))
        wm = self.wm.init(k1, obs, act)
        actor = self.actor.init(k2, z)
        critic = self.critic.init(k3, z)
        return TrainState(
            wm=wm, wm_target=wm, actor=actor, critic=critic, critic_target=critic,
            wm_opt=self.wm_tx.init(wm), actor_opt=self.actor_tx.init(actor),
            critic_opt=self.critic_tx.init(critic), ret_scale=jnp.ones(()),
        )

    # -------------------------------------------------------- model helpers
    def encode(self, wm_params, obs):
        return self.wm.apply(wm_params, obs, method=WorldModel.encode)

    def act(self, ts: TrainState, obs, key):
        z = self.encode(ts.wm, obs)
        mean, std = self.actor.apply(ts.actor, z)
        a, _ = sample_tanh(mean, std, key)
        return a

    # ------------------------------------------------------- world model
    def wm_loss(self, wm_params, target_params, batch):
        c = self.cfg.wm
        obs, act = batch["obs"], batch["act"]
        done, trunc, rew = batch["done"], batch["trunc"], batch["rew"]
        k = obs.shape[1] - 1  # prediction steps

        z0 = self.encode(wm_params, obs[:, 0])
        target = sg(self.encode(target_params, obs[:, 1:]))  # [B, K, D]

        def body(z, a):
            z_next = self.wm.apply(wm_params, z, a, method=WorldModel.step)
            r = self.wm.apply(wm_params, z, a, method=WorldModel.reward_symlog)
            cl = self.wm.apply(wm_params, z, a, method=WorldModel.cont_logit)
            return z_next, (z_next, r, cl, z)

        acts = jnp.swapaxes(act[:, :k], 0, 1)
        _, (z_pred, r_pred, c_logit, z_in) = jax.lax.scan(body, z0, acts)
        z_pred, z_in = jnp.swapaxes(z_pred, 0, 1), jnp.swapaxes(z_in, 0, 1)  # [B, K, D]
        r_pred, c_logit = r_pred.T, c_logit.T                                   # [B, K]

        # Transition k is inside the episode if no done occurred at steps 0..k-1;
        # predicting z_{k+1} additionally requires done_k == 0.
        alive = jnp.cumprod(1.0 - done[:, :k], axis=1)
        head_mask = jnp.concatenate([jnp.ones_like(alive[:, :1]), alive[:, :-1]], axis=1)

        err = jnp.mean((z_pred - target) ** 2, axis=-1)                    # [B, K]
        jepa = jnp.sum(alive * err) / jnp.maximum(alive.sum(), 1.0)
        var_l, cov_l = _vicreg(z0)

        rew_l = jnp.sum(head_mask * (r_pred - symlog(rew[:, :k])) ** 2) / jnp.maximum(head_mask.sum(), 1.0)
        cont_target = 1.0 - done[:, :k] * (1.0 - trunc[:, :k])  # truncation is not termination
        cont_l = optax.sigmoid_binary_cross_entropy(c_logit, cont_target)
        cont_l = jnp.sum(head_mask * cont_l) / jnp.maximum(head_mask.sum(), 1.0)

        loss = jepa + c.var_coef * var_l + c.cov_coef * cov_l + c.reward_coef * rew_l + c.cont_coef * cont_l
        per_step = jnp.sum(alive * err, 0) / jnp.maximum(alive.sum(0), 1.0)
        metrics = {
            "wm/loss": loss, "wm/jepa": jepa, "wm/vicreg_var": var_l, "wm/vicreg_cov": cov_l,
            "wm/reward": rew_l, "wm/cont": cont_l,
            "wm/pred_err_k1": per_step[0], "wm/pred_err_kK": per_step[-1],
            "wm/latent_std": jnp.mean(jnp.std(z0, axis=0)),
        }
        return loss, metrics

    def wm_update(self, ts: TrainState, batch) -> tuple[TrainState, dict]:
        (_, m), g = jax.value_and_grad(self.wm_loss, has_aux=True)(ts.wm, ts.wm_target, batch)
        upd, opt = self.wm_tx.update(g, ts.wm_opt, ts.wm)
        wm = optax.apply_updates(ts.wm, upd)
        target = optax.incremental_update(wm, ts.wm_target, 1.0 - self.cfg.wm.target_ema)
        m["wm/grad_norm"] = optax.global_norm(g)
        return ts.replace(wm=wm, wm_target=target, wm_opt=opt), m

    # ------------------------------------------ actor-critic in imagination
    def ac_update(self, ts: TrainState, obs0, key) -> tuple[TrainState, dict]:
        c = self.cfg.ac
        z0 = sg(self.encode(ts.wm, obs0))

        def actor_loss(actor_params):
            keys = jax.random.split(key, c.horizon)

            def body(z, k):
                mean, std = self.actor.apply(actor_params, z)
                a, logp = sample_tanh(mean, std, k)
                r = self.wm.apply(ts.wm, z, a, method=WorldModel.reward)
                cont = jax.nn.sigmoid(self.wm.apply(ts.wm, z, a, method=WorldModel.cont_logit))
                z_next = self.wm.apply(ts.wm, z, a, method=WorldModel.step)
                return z_next, (z, r, cont, logp)

            # Gradients flow from returns back through the frozen world model into the actions
            # (analytic policy gradients, as in Dreamer / SHAC / SuperMimic).
            z_last, (zs, rs, cs, logps) = jax.lax.scan(body, z0, keys)
            z_all = jnp.concatenate([zs, z_last[None]], 0)
            values = self.critic.apply(ts.critic_target, z_all)
            rets = lambda_returns(rs, cs, values, c.gamma, c.lam)  # [H, B]
            disc = c.gamma * cs
            weight = sg(jnp.concatenate([jnp.ones_like(disc[:1]), jnp.cumprod(disc, 0)[:-1]], 0))
            spread = sg(jnp.percentile(rets, 95) - jnp.percentile(rets, 5))
            scale = jnp.maximum(1.0, ts.ret_scale)
            entropy = -logps
            loss = -jnp.mean(weight * rets) / scale - c.ent_coef * jnp.mean(weight * entropy)
            aux = {
                "zs": sg(zs), "rets": sg(rets), "weight": weight, "spread": spread,
                "entropy": jnp.mean(entropy), "ret_mean": jnp.mean(rets), "imag_reward": jnp.mean(rs),
                "imag_cont": jnp.mean(cs),
            }
            return loss, aux

        (a_loss, aux), g_a = jax.value_and_grad(actor_loss, has_aux=True)(ts.actor)
        upd, actor_opt = self.actor_tx.update(g_a, ts.actor_opt, ts.actor)
        actor = optax.apply_updates(ts.actor, upd)

        def critic_loss(critic_params):
            v = self.critic.apply(critic_params, aux["zs"])
            return jnp.mean(aux["weight"] * (v - aux["rets"]) ** 2)

        c_loss, g_c = jax.value_and_grad(critic_loss)(ts.critic)
        upd, critic_opt = self.critic_tx.update(g_c, ts.critic_opt, ts.critic)
        critic = optax.apply_updates(ts.critic, upd)
        critic_target = optax.incremental_update(critic, ts.critic_target, 1.0 - c.critic_ema)
        ret_scale = 0.99 * ts.ret_scale + 0.01 * aux["spread"]

        m = {
            "ac/actor_loss": a_loss, "ac/critic_loss": c_loss, "ac/entropy": aux["entropy"],
            "ac/imag_return": aux["ret_mean"], "ac/imag_reward": aux["imag_reward"],
            "ac/imag_cont": aux["imag_cont"], "ac/ret_scale": ret_scale,
            "ac/actor_grad_norm": optax.global_norm(g_a),
        }
        ts = ts.replace(actor=actor, actor_opt=actor_opt, critic=critic,
                        critic_opt=critic_opt, critic_target=critic_target, ret_scale=ret_scale)
        return ts, m

    # ------------------------------------------------------- fused update
    def update(self, ts: TrainState, rb: replay.Replay, key: jax.Array) -> tuple[TrainState, dict]:
        t = self.cfg.train
        k_wm, k_ac = jax.random.split(key)
        seq_len = self.cfg.wm.pred_horizon + 1

        def wm_body(ts, k):
            batch = replay.sample(rb, k, self.cfg.wm.batch, seq_len)
            return self.wm_update(ts, batch)

        def ac_body(ts, k):
            k_s, k_i = jax.random.split(k)
            obs0 = replay.sample(rb, k_s, self.cfg.ac.batch, 1)["obs"][:, 0]
            return self.ac_update(ts, obs0, k_i)

        ts, wm_m = jax.lax.scan(wm_body, ts, jax.random.split(k_wm, t.wm_updates))
        ts, ac_m = jax.lax.scan(ac_body, ts, jax.random.split(k_ac, t.ac_updates))
        metrics = jax.tree_util.tree_map(jnp.mean, {**wm_m, **ac_m})
        return ts, metrics
