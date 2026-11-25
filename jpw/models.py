"""Networks: JEPA world model (encoder, action-conditioned latent predictor, heads), actor, critic."""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


class MLP(nn.Module):
    hidden: int
    depth: int
    out: int
    ln_out: bool = False

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.Dense(self.hidden)(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
        x = nn.Dense(self.out)(x)
        if self.ln_out:
            x = nn.LayerNorm()(x)
        return x


class WorldModel(nn.Module):
    """Decoder-free JEPA world model.

    encode:   o_t -> z_t                      (trained against an EMA target encoder)
    step:     (z_t, a_t) -> z_{t+1}           (residual, layer-normed latent dynamics)
    reward:   (z_t, a_t) -> r_t               (predicted in symlog space)
    cont:     (z_t, a_t) -> P(episode continues after this step)
    """

    latent_dim: int
    hidden: int
    depth: int

    def setup(self):
        self.encoder = MLP(self.hidden, self.depth, self.latent_dim, ln_out=True)
        self.dynamics = MLP(self.hidden, self.depth, self.latent_dim)
        self.dyn_norm = nn.LayerNorm()
        self.reward_head = MLP(self.hidden, 2, 1)
        self.cont_head = MLP(self.hidden, 2, 1)

    def encode(self, obs):
        return self.encoder(obs)

    def step(self, z, a):
        return self.dyn_norm(z + self.dynamics(jnp.concatenate([z, a], -1)))

    def reward_symlog(self, z, a):
        return self.reward_head(jnp.concatenate([z, a], -1))[..., 0]

    def reward(self, z, a):
        return symexp(self.reward_symlog(z, a))

    def cont_logit(self, z, a):
        return self.cont_head(jnp.concatenate([z, a], -1))[..., 0]

    def __call__(self, obs, a):  # only used to initialise every submodule
        z = self.encode(obs)
        return self.step(z, a), self.reward_symlog(z, a), self.cont_logit(z, a)


class Actor(nn.Module):
    act_dim: int
    hidden: int
    depth: int
    min_std: float = 0.05
    max_std: float = 1.0

    @nn.compact
    def __call__(self, z):
        x = MLP(self.hidden, self.depth, 2 * self.act_dim)(z)
        mean, s = jnp.split(x, 2, axis=-1)
        std = self.min_std + (self.max_std - self.min_std) * nn.sigmoid(s + 2.0)
        return mean, std


class Critic(nn.Module):
    hidden: int
    depth: int

    @nn.compact
    def __call__(self, z):
        return MLP(self.hidden, self.depth, 1)(z)[..., 0]


def sample_tanh(mean, std, key):
    """Reparameterised tanh-Gaussian sample and its log-probability."""
    eps = jax.random.normal(key, mean.shape)
    u = mean + std * eps
    a = jnp.tanh(u)
    logp = jnp.sum(-0.5 * eps**2 - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi), -1)
    logp = logp - jnp.sum(jnp.log(1.0 - a**2 + 1e-6), -1)
    return a, logp
