"""Per-env domain randomisation for Go1 (needs mujoco_playground).

Builds on Playground's Go1 randomiser (floor friction, link masses, torso mass and COM, joint
friction / armature, nominal pose) and adds motor-strength randomisation: the position-servo
gain Kp (and its bias term) is scaled per env by U(1 - MOTOR_SPREAD, 1 + MOTOR_SPREAD).

Heightmap noise, the perception-side randomisation, lives in `jpw.envs.PlaygroundAdapter`.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

MOTOR_SPREAD = 0.15


def _randomize(model, rngs, motor_spread: float):
    from mujoco_playground._src.locomotion.go1 import randomize as go1_randomize

    model, in_axes = go1_randomize.domain_randomize(model, rngs)
    n = rngs.shape[0]
    key = jax.random.fold_in(rngs[0], 1234)
    scale = jax.random.uniform(key, (n,), minval=1.0 - motor_spread, maxval=1.0 + motor_spread)
    gain = jnp.broadcast_to(model.actuator_gainprm, (n, *model.actuator_gainprm.shape))
    bias = jnp.broadcast_to(model.actuator_biasprm, (n, *model.actuator_biasprm.shape))
    gain = gain.at[:, :, 0].multiply(scale[:, None])        # Kp
    bias = bias.at[:, :, 1].multiply(scale[:, None])        # -Kp
    model = model.tree_replace({"actuator_gainprm": gain, "actuator_biasprm": bias})
    in_axes = in_axes.tree_replace({"actuator_gainprm": 0, "actuator_biasprm": 0})
    return model, in_axes


def make_randomization_fn(num_envs: int, seed: int = 0, motor_spread: float = MOTOR_SPREAD):
    """randomization_fn for `wrapper.wrap_for_brax_training` (model -> (model_v, in_axes))."""
    rngs = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    return functools.partial(_randomize, rngs=rngs, motor_spread=motor_spread)
