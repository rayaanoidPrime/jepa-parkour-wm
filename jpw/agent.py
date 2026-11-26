"""JEPA world-model learning and actor-critic learning in imagination."""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import struct

from jpw import replay
from jpw.config import Config
from jpw import go1_state
from jpw.models import Actor, Critic, WorldModel, sample_tanh, symlog
from jpw.spec import Spec

sg = jax.lax.stop_gradient


@struct.dataclass
class TrainState:
    wm: Any
    wm_target: Any
    actor: Any
    actor_ema: Any
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
    def __init__(self, cfg: Config, spec: Spec):
        self.cfg, self.spec = cfg, spec
        # Data parallelism: set by jpw.parallel when update/pretrain run inside shard_map. Batch
        # sizes in the config are global; each device uses batch // ndev and gradients are
        # averaged over `axis`.
        self.axis: str | None = None
        self.ndev = 1
        w, a = cfg.wm, cfg.ac
        self.asymmetric = a.asymmetric
        self.wm = WorldModel(
            w.latent_dim, w.hidden, w.depth, obs_dim=spec.obs_dim, prop_dim=spec.prop_dim,
            use_hmap=spec.hmap_shape is not None, cnn_channels=w.cnn_channels,
            contact_dim=spec.contact_dim, asymmetric=a.asymmetric,
            phys_dim=go1_state.PHYS_DIM if w.decode_state else 0,
        )
        if w.decode_state:
            go1_state.check_layout(spec.obs_dim, spec.aux_dim)
            self._sd, self._nom = jnp.asarray(go1_state.SD), jnp.asarray(go1_state.NOM)
        if a.analytic_reward and not w.decode_state:
            raise ValueError("ac.analytic_reward needs wm.decode_state=true")
        self.task = None  # go1_state.Go1Task, set by the trainer when ac.analytic_reward
        self.actor = Actor(spec.act_dim, a.hidden, a.depth, a.min_std, a.max_std, a.fixed_std)
        self.critic = Critic(a.hidden, a.depth)
        t = cfg.train

        def lr(base, updates_per_iter):  # cosine decay to 0.1x over t.lr_decay_iters iterations
            if t.lr_decay_iters <= 0:
                return base
            return optax.cosine_decay_schedule(base, t.lr_decay_iters * max(updates_per_iter, 1), alpha=0.1)

        actor_clip = [optax.clip_by_global_norm(a.grad_clip)]
        if a.grad_clip_value > 0:
            actor_clip.append(optax.clip(a.grad_clip_value))
        self.wm_tx = optax.chain(optax.clip_by_global_norm(w.grad_clip), optax.adam(lr(w.lr, t.wm_updates)))
        self.actor_tx = optax.chain(*actor_clip, optax.adam(lr(a.actor_lr, t.ac_updates), b2=a.beta2))
        self.critic_tx = optax.chain(optax.clip_by_global_norm(a.grad_clip),
                                     optax.adam(lr(a.critic_lr, t.ac_updates)))

    # ------------------------------------------------------------------ init
    def init(self, key: jax.Array) -> TrainState:
        k1, k2, k3 = jax.random.split(key, 3)
        s = self.spec
        obs = jnp.zeros((1, s.obs_dim))
        act = jnp.zeros((1, s.act_dim))
        hmap = None if s.hmap_shape is None else jnp.zeros((1, *s.hmap_shape))
        z = jnp.zeros((1, self.cfg.wm.latent_dim))
        wm = self.wm.init(k1, obs, act, hmap)
        actor = self.actor.init(k2, z)
        critic = self.critic.init(k3, z)
        copy = lambda tree: jax.tree_util.tree_map(jnp.copy, tree)  # targets must not alias (donation)
        return TrainState(
            wm=wm, wm_target=copy(wm), actor=actor, actor_ema=copy(actor), critic=critic,
            critic_target=copy(critic),
            wm_opt=self.wm_tx.init(wm), actor_opt=self.actor_tx.init(actor),
            critic_opt=self.critic_tx.init(critic), ret_scale=jnp.ones(()),
        )

    def _pmean(self, x):
        return jax.lax.pmean(x, self.axis) if self.axis else x

    def _local(self, batch: int) -> int:
        if batch % self.ndev:
            raise ValueError(f"batch size {batch} must be divisible by the number of devices {self.ndev}")
        return batch // self.ndev

    # -------------------------------------------------------- model helpers
    def encode(self, wm_params, obs, hmap=None):
        return self.wm.apply(wm_params, obs, hmap, method=WorldModel.encode)

    def decode_phys(self, wm_params, z):
        """Latent -> exact physical state in raw units (see go1_state.GROUPS)."""
        return self.wm.apply(wm_params, z, method=WorldModel.decode) * self._sd

    def policy_latent(self, wm_params, obs, hmap=None):
        """Latent the *deployed* actor acts on: the student (proprio + heightmap only) if
        asymmetric, otherwise the full world-model encoder."""
        if self.asymmetric:
            return self.wm.apply(wm_params, obs, hmap, method=WorldModel.student)
        return self.encode(wm_params, obs, hmap)

    def act(self, ts: TrainState, obs, key, hmap=None, deterministic=False, ema=False):
        z = self.policy_latent(ts.wm, obs, hmap)
        mean, std = self.actor.apply(ts.actor_ema if ema else ts.actor, z)
        if deterministic:
            return jnp.tanh(mean)
        a, _ = sample_tanh(mean, std, key)
        return a

    # ------------------------------------------------------- world model
    def wm_loss(self, wm_params, target_params, batch):
        c = self.cfg.wm
        obs, act = batch["obs"], batch["act"]
        done, trunc, rew = batch["done"], batch["trunc"], batch["rew"]
        hmap, contact = batch.get("hmap"), batch.get("contact")
        has_contact = contact is not None and self.spec.contact_dim > 0
        k = obs.shape[1] - 1  # prediction steps

        z_all = self.encode(wm_params, obs, hmap)  # [B, K+1, D]
        z0 = z_all[:, 0]
        target = sg(self.encode(target_params, obs[:, 1:], None if hmap is None else hmap[:, 1:]))

        def body(z, a):
            z_next = self.wm.apply(wm_params, z, a, method=WorldModel.step)
            r = self.wm.apply(wm_params, z, a, method=WorldModel.reward_symlog)
            cl = self.wm.apply(wm_params, z, a, method=WorldModel.cont_logit)
            ct = (self.wm.apply(wm_params, z, a, method=WorldModel.contact_logit)
                  if has_contact else jnp.zeros(()))
            return z_next, (z_next, r, cl, ct)

        acts = jnp.swapaxes(act[:, :k], 0, 1)
        _, (z_pred, r_pred, c_logit, ct_logit) = jax.lax.scan(body, z0, acts)
        z_pred = jnp.swapaxes(z_pred, 0, 1)                                     # [B, K, D]
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
        extra = {}
        if has_contact:
            ct_logit = jnp.swapaxes(ct_logit, 0, 1)                             # [B, K, C]
            ct = contact[:, :k]
            prev = jnp.concatenate([ct[:, :1], ct[:, :-1]], axis=1)
            event = jnp.abs(ct - prev)                                          # touchdown / liftoff
            w = (1.0 + (c.contact_pos_weight - 1.0) * ct) * (1.0 + c.contact_event_weight * event)
            m = head_mask[..., None]
            bce = optax.sigmoid_binary_cross_entropy(ct_logit, ct)
            denom = jnp.maximum(m.sum() * ct.shape[-1], 1.0)
            contact_l = jnp.sum(m * w * bce) / denom
            hit = ((ct_logit > 0) == (ct > 0.5)).astype(jnp.float32)
            extra["wm/contact"] = contact_l
            extra["wm/contact_acc"] = jnp.sum(m * hit) / denom
            extra["wm/contact_event_acc"] = jnp.sum(m * event * hit) / jnp.maximum(jnp.sum(m * event), 1.0)
            loss = loss + c.contact_coef * contact_l
        if self.wm.phys_dim:
            tgt = go1_state.phys_targets(obs, batch["aux"])                      # [B, K+1, P]
            e_enc = jnp.clip((self.decode_phys(wm_params, z_all) - tgt) / self._nom, -10.0, 10.0)
            e_pred = jnp.clip((self.decode_phys(wm_params, z_pred) - tgt[:, 1:]) / self._nom, -10.0, 10.0)
            l_enc = jnp.mean(e_enc**2)
            l_pred = jnp.sum(alive[..., None] * e_pred**2) / (jnp.maximum(alive.sum(), 1.0) * e_pred.shape[-1])
            phys_l = l_enc + l_pred
            extra["wm/phys"] = phys_l
            m_last = alive[:, -1]
            for i, name in enumerate(go1_state.NAMES):  # rollout error at step K, in nominal-error units
                sl = e_pred[:, -1, go1_state.OFFS[i]:go1_state.OFFS[i + 1]]
                extra[f"wm/phys_{name}"] = jnp.sqrt(jnp.sum(m_last[:, None] * sl**2) /
                                                    (jnp.maximum(m_last.sum(), 1.0) * sl.shape[-1]))
            loss = loss + c.phys_coef * phys_l
        if self.asymmetric:
            z_s = self.wm.apply(wm_params, obs, hmap, method=WorldModel.student)
            student_l = jnp.mean((z_s - sg(z_all)) ** 2)
            extra["wm/student"] = student_l
            loss = loss + c.student_coef * student_l
        per_step = jnp.sum(alive * err, 0) / jnp.maximum(alive.sum(0), 1.0)
        metrics = {
            "wm/loss": loss, "wm/jepa": jepa, "wm/vicreg_var": var_l, "wm/vicreg_cov": cov_l,
            "wm/reward": rew_l, "wm/cont": cont_l,
            "wm/pred_err_k1": per_step[0], "wm/pred_err_kK": per_step[-1],
            "wm/latent_std": jnp.mean(jnp.std(z0, axis=0)),
            **extra,
        }
        return loss, metrics

    def wm_update(self, ts: TrainState, batch) -> tuple[TrainState, dict]:
        (_, m), g = jax.value_and_grad(self.wm_loss, has_aux=True)(ts.wm, ts.wm_target, batch)
        g, m = self._pmean(g), self._pmean(m)
        upd, opt = self.wm_tx.update(g, ts.wm_opt, ts.wm)
        wm = optax.apply_updates(ts.wm, upd)
        target = optax.incremental_update(wm, ts.wm_target, 1.0 - self.cfg.wm.target_ema)
        m["wm/grad_norm"] = optax.global_norm(g)
        return ts.replace(wm=wm, wm_target=target, wm_opt=opt), m

    # ------------------------------------------ actor-critic in imagination
    def ac_update(self, ts: TrainState, obs0, key, hmap0=None) -> tuple[TrainState, dict]:
        c = self.cfg.ac
        z0 = sg(self.encode(ts.wm, obs0, hmap0))
        analytic = c.analytic_reward
        if analytic:
            if self.task is None:
                raise ValueError("ac.analytic_reward: set agent.task = go1_state.Go1Task.from_env(env) first")
            cmd0 = obs0[..., go1_state.CMD]            # held fixed over the imagined rollout
            a0 = obs0[..., go1_state.LAST_ACT]         # previous action at the start state

        def actor_loss(actor_params):
            keys = jax.random.split(key, c.horizon)

            def body(carry, k):
                z, a_prev = carry
                mean, std = self.actor.apply(actor_params, z)
                a, logp = sample_tanh(mean, std, k)
                z_next = self.wm.apply(ts.wm, z, a, method=WorldModel.step)
                if self.spec.contact_dim:
                    c_prob = jax.nn.sigmoid(self.wm.apply(ts.wm, z, a, method=WorldModel.contact_logit))
                    p_ct = jnp.mean(c_prob)
                else:
                    c_prob, p_ct = jnp.zeros((z.shape[0], 4)), jnp.zeros(())
                if analytic:  # reward / termination by formula from the decoded exact state
                    r, d = self.task.reward(self.decode_phys(ts.wm, z_next), a, a_prev, cmd0, c_prob)
                    cont = 1.0 - d
                else:
                    r = self.wm.apply(ts.wm, z, a, method=WorldModel.reward)
                    cont = jax.nn.sigmoid(self.wm.apply(ts.wm, z, a, method=WorldModel.cont_logit))
                return (z_next, a), (z, r, cont, logp, p_ct)

            # Gradients flow from returns back through the frozen world model into the actions
            # (analytic policy gradients, as in Dreamer / SHAC / SuperMimic).
            a_init = a0 if analytic else jnp.zeros((z0.shape[0], self.spec.act_dim))
            (z_last, _), (zs, rs, cs, logps, p_cts) = jax.lax.scan(body, (z0, a_init), keys)
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
                "imag_cont": jnp.mean(cs), "imag_contact": jnp.mean(sg(p_cts)),
            }
            return loss, aux

        (a_loss, aux), g_a = jax.value_and_grad(actor_loss, has_aux=True)(ts.actor)
        g_a = self._pmean(g_a)
        aux["spread"] = self._pmean(aux["spread"])  # keeps ret_scale identical on every device
        upd, actor_opt = self.actor_tx.update(g_a, ts.actor_opt, ts.actor)
        actor = optax.apply_updates(ts.actor, upd)

        def critic_loss(critic_params):
            v = self.critic.apply(critic_params, aux["zs"])
            return jnp.mean(aux["weight"] * (v - aux["rets"]) ** 2)

        c_loss, g_c = jax.value_and_grad(critic_loss)(ts.critic)
        g_c = self._pmean(g_c)
        upd, critic_opt = self.critic_tx.update(g_c, ts.critic_opt, ts.critic)
        critic = optax.apply_updates(ts.critic, upd)
        critic_target = optax.incremental_update(critic, ts.critic_target, 1.0 - c.critic_ema)
        ret_scale = 0.99 * ts.ret_scale + 0.01 * aux["spread"]

        m = {
            "ac/actor_loss": a_loss, "ac/critic_loss": c_loss, "ac/entropy": aux["entropy"],
            "ac/imag_return": aux["ret_mean"], "ac/imag_reward": aux["imag_reward"],
            "ac/imag_cont": aux["imag_cont"], "ac/imag_contact": aux["imag_contact"],
            "ac/ret_scale": ret_scale,
            "ac/actor_grad_norm": optax.global_norm(g_a),
        }
        m = self._pmean(m)
        actor_ema = optax.incremental_update(actor, ts.actor_ema, 1.0 - c.ema) if c.ema > 0 else actor
        ts = ts.replace(actor=actor, actor_ema=actor_ema, actor_opt=actor_opt, critic=critic,
                        critic_opt=critic_opt, critic_target=critic_target, ret_scale=ret_scale)
        return ts, m

    # ------------------------------------------------------- fused update
    def update(self, ts: TrainState, rb: replay.Replay, key: jax.Array) -> tuple[TrainState, dict]:
        t = self.cfg.train
        if self.axis:  # same key on every device -> decorrelate the local samples
            key = jax.random.fold_in(key, jax.lax.axis_index(self.axis))
        k_wm, k_ac = jax.random.split(key)
        seq_len = self.cfg.wm.pred_horizon + 1
        wm_batch, ac_batch = self._local(self.cfg.wm.batch), self._local(self.cfg.ac.batch)

        def wm_body(ts, k):
            batch = replay.sample(rb, k, wm_batch, seq_len)
            return self.wm_update(ts, batch)

        def ac_body(ts, k):
            k_s, k_i = jax.random.split(k)
            b = replay.sample(rb, k_s, ac_batch, 1)
            hmap0 = b["hmap"][:, 0] if "hmap" in b else None
            return self.ac_update(ts, b["obs"][:, 0], k_i, hmap0)

        ts, wm_m = jax.lax.scan(wm_body, ts, jax.random.split(k_wm, t.wm_updates))
        ts, ac_m = jax.lax.scan(ac_body, ts, jax.random.split(k_ac, t.ac_updates))
        metrics = jax.tree_util.tree_map(jnp.mean, {**wm_m, **ac_m})
        return ts, metrics

    def pretrain(self, ts: TrainState, rb: replay.Replay, key: jax.Array) -> tuple[TrainState, dict]:
        """World-model-only updates (stage 1: offline pretraining on exported rollouts)."""
        if self.axis:
            key = jax.random.fold_in(key, jax.lax.axis_index(self.axis))
        seq_len = self.cfg.wm.pred_horizon + 1
        wm_batch = self._local(self.cfg.wm.batch)

        def wm_body(ts, k):
            return self.wm_update(ts, replay.sample(rb, k, wm_batch, seq_len))

        ts, m = jax.lax.scan(wm_body, ts, jax.random.split(key, self.cfg.train.wm_updates))
        return ts, jax.tree_util.tree_map(jnp.mean, m)
