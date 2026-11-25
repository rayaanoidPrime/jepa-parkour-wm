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

stage("jit step")
step = jax.jit(wenv.step)
state = step(state, jnp.zeros((n, wenv.action_size)))
jax.block_until_ready(state)
stage(f"step ok, reward mean {float(jnp.mean(state.reward))}")
for _ in range(20):
    state = step(state, jnp.zeros((n, wenv.action_size)))
jax.block_until_ready(state)
stage("20 more steps ok")
