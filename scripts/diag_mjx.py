"""Staged MJX diagnostic: prints a line before each step so a segfault shows where it happened.

  python scripts/diag_mjx.py [env_name] [num_envs]
"""
import faulthandler
import sys

faulthandler.enable()


def stage(msg):
    print(f"[diag] {msg}", flush=True)


name = sys.argv[1] if len(sys.argv) > 1 else "Go1JoystickRoughTerrain"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8

stage("import jax")
import jax
import jax.numpy as jnp

stage(f"jax {jax.__version__} backend={jax.default_backend()} devices={jax.devices()}")
x = jnp.ones((256, 256)) @ jnp.ones((256, 256))
stage(f"gpu matmul ok: {float(x[0, 0])}")

stage("import mujoco / mjx")
import mujoco
import mujoco.mjx as mjx

stage(f"mujoco {mujoco.__version__}")

stage("import mujoco_playground")
from mujoco_playground import registry, wrapper

stage(f"registry.load({name})")
env = registry.load(name)
stage(f"loaded; action_size={env.action_size}")

stage("wrap_for_brax_training")
wenv = wrapper.wrap_for_brax_training(env, episode_length=100, action_repeat=1)

stage(f"jit reset, n={n}")
state = jax.jit(wenv.reset)(jax.random.split(jax.random.PRNGKey(0), n))
jax.block_until_ready(state)
stage("reset ok")
stage(f"obs: {jax.tree_util.tree_map(lambda a: a.shape, state.obs)}")
stage(f"info keys: {list(state.info.keys())}")

def nans(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return sum(int(jnp.sum(~jnp.isfinite(x))) for x in leaves if jnp.issubdtype(x.dtype, jnp.floating))


stage(f"nan count after reset: obs={nans(state.obs)} reward={nans(state.reward)}")
stage("jit step")
step = jax.jit(wenv.step)
key = jax.random.PRNGKey(1)
for mode in ("zeros", "random"):
    s = state
    for i in range(30):
        key, k = jax.random.split(key)
        a = (jnp.zeros((n, wenv.action_size)) if mode == "zeros"
             else jax.random.uniform(k, (n, wenv.action_size), minval=-1.0, maxval=1.0))
        s = step(s, a)
        jax.block_until_ready(s)
        if i in (0, 1, 2, 9, 29):
            stage(f"{mode} step {i}: nan obs={nans(s.obs)} reward={nans(s.reward)} "
                  f"data.qpos={nans(s.data.qpos)} reward_mean={float(jnp.nanmean(s.reward)):.4f}")
stage("done")
