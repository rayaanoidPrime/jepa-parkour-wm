import jax, jax.numpy as jnp
from jpw.config import EnvCfg
from jpw.envs import make_env, get_spec
cfg = EnvCfg(name="Go1Tiles", num_envs=4, episode_length=400, goal_task=True, use_heightmap=True, use_privileged=True, hmap_step=0.05)
env = make_env(cfg)
st = env.reset(jax.random.split(jax.random.PRNGKey(0), 4))
print("HMAP", st.obs["heightmap"].shape if isinstance(st.obs, dict) else "obs", float(jnp.abs(st.obs["heightmap"]).max()))
print("SPEC", get_spec(env))
