"""Parkour env tests. The course tests are pure numpy; the env tests need MuJoCo Playground
(skipped if it isn't installed) and run on CPU."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jpw import parkour as pk


def test_course_has_obstacles_gaps_and_flat_start():
    for seed in range(5):
        h = pk.make_course(seed, 1.0)
        xs = pk.grid_x()
        assert np.all(h[xs < pk.START_FLAT - 1e-6] == 0.0), "spawn zone must be flat"
        assert h.min() <= -pk.G + 1e-6 or h.max() > 0.05, "course should not be flat"
        assert pk.G + h.max() < pk.ZMAX and pk.G + h.min() >= -1e-6, "must fit the hfield range"


def test_difficulty_scales_obstacles():
    easy = pk.make_course(3, 0.1)
    hard = pk.make_course(3, 1.0)
    assert hard.max() >= easy.max() and (hard < -1e-6).sum() >= (easy < -1e-6).sum()


def test_heightmap_is_relative_to_base_and_yaw_aligned():
    elev = pk.make_elevation(0, 1.0)
    fn = pk.make_heightmap_fn(elev)
    q = np.zeros(7, np.float32)
    q[:3] = [0.0, 0.0, pk.G + pk.NOMINAL_BASE_H]
    q[3] = 1.0
    hm = np.asarray(fn(jnp.asarray(q)))
    assert hm.shape == pk.HMAP_SHAPE
    np.testing.assert_allclose(hm, 0.0, atol=1e-5)  # flat start at nominal standing height
    # yaw by 90 deg: the sample points ahead now run along +y, so the map stays flat
    q[3], q[6] = np.cos(np.pi / 4), np.sin(np.pi / 4)
    np.testing.assert_allclose(np.asarray(fn(jnp.asarray(q))), 0.0, atol=1e-5)
    # a batch of poses works
    qb = jnp.tile(jnp.asarray(q), (5, 3, 1))
    assert fn(qb).shape == (5, 3, *pk.HMAP_SHAPE)


try:
    import mujoco  # noqa: F401
    import mujoco_playground  # noqa: F401
    HAVE_SIM = True
except Exception:
    HAVE_SIM = False
requires_sim = pytest.mark.skipif(not HAVE_SIM, reason="needs mujoco + mujoco_playground")


@pytest.fixture(scope="module")
def parkour_env():
    from jpw.config import EnvCfg
    from jpw.envs import make_env

    cfg = EnvCfg(name="Go1Parkour", num_envs=4, episode_length=50, use_heightmap=True,
                 use_privileged=True)
    return cfg, make_env(cfg)


@requires_sim
def test_hfield_matches_jax_ground_height():
    """The elevation written into MuJoCo and the one sampled for the heightmap agree."""
    import mujoco

    env, elev = pk.build_env(2, 1.0)
    model = env.mj_model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    prof = pk.G + pk.make_course(2, 1.0)
    xs = pk.grid_x()
    rng = np.random.default_rng(0)
    checked = 0
    floor = model.geom("floor").id
    for x in rng.uniform(1.0, 20.0, 400):  # beyond the robot, which sits at the origin
        i = int(np.argmin(np.abs(xs - x)))
        if i < 3 or i > len(xs) - 4 or np.ptp(prof[i - 3:i + 4]) > 1e-6:
            continue  # skip points near a step edge (bilinear vs triangle prisms differ there)
        y = rng.uniform(-1.0, 1.0)
        geomid = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(model, data, np.array([x, y, 5.0]), np.array([0.0, 0.0, -1.0]),
                             None, 1, -1, geomid)
        assert dist > 0 and geomid[0] == floor
        got = float(pk.ground_height(jnp.asarray(elev), jnp.float32(x), jnp.float32(y)))
        assert abs((5.0 - dist) - got) < 2e-3, (x, y, 5.0 - dist, got)
        checked += 1
    assert checked > 50


@requires_sim
def test_env_reset_step_outputs(parkour_env):
    from jpw.envs import get_contact, get_hmap, get_spec

    cfg, env = parkour_env
    st = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), cfg.num_envs))
    spec = get_spec(st, cfg, env.action_size)
    assert spec.hmap_shape == pk.HMAP_SHAPE and spec.contact_dim == 4 and spec.prop_dim == 48
    assert spec.obs_dim == 48 + 123
    step = jax.jit(env.step)
    for _ in range(3):
        st = step(st, jnp.zeros((cfg.num_envs, env.action_size)))
    assert get_hmap(st.obs, True).shape == (cfg.num_envs, *pk.HMAP_SHAPE)
    assert get_contact(st).shape == (cfg.num_envs, 4)
    assert bool(jnp.all(jnp.isfinite(st.reward)))
    # spawned on the start zone, facing forward, with a forward command
    assert float(jnp.abs(st.data.qpos[:, 1]).max()) < 0.6
    assert float(st.info["command"][:, 0].min()) >= 0.4 and float(jnp.abs(st.info["command"][:, 1:]).max()) == 0.0


@requires_sim
def test_leaving_the_lane_ends_the_episode(parkour_env):
    cfg, env = parkour_env
    st = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(1), cfg.num_envs))
    inner = env._strip(st)
    q = inner.data.qpos.at[:, 1].set(pk.LANE_HALF + 0.3)
    inner = inner.replace(data=inner.data.replace(qpos=q))
    out = jax.jit(env.env.step)(inner, jnp.zeros((cfg.num_envs, env.action_size)))
    assert float(out.done.min()) == 1.0


def _find(env, attr):
    w = env
    while not hasattr(w, attr):
        w = w.env
    return getattr(w, attr)


@requires_sim
def test_domain_randomization_varies_per_env():
    from jpw.config import EnvCfg
    from jpw.envs import make_env

    cfg = EnvCfg(name="Go1Parkour", num_envs=4, episode_length=20, use_heightmap=True,
                 domain_randomization=True)
    env = make_env(cfg)
    model_v = _find(env, "_mjx_model_v")
    assert model_v.geom_friction.shape[0] == 4 and model_v.actuator_gainprm.shape[0] == 4
    assert model_v.geom_friction[:, 0, 0].std() > 0 and model_v.actuator_gainprm[:, 0, 0].std() > 0
    base = _find(env, "unwrapped") if hasattr(_find(env, "unwrapped"), "mj_model") else None
    assert base is not None and base.mj_model.geom("floor").id == 0 and base.mj_model.body("trunk").id == 1
    st = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 4))
    st = jax.jit(env.step)(st, jnp.zeros((4, env.action_size)))
    assert bool(jnp.all(jnp.isfinite(st.reward)))


@requires_sim
def test_heightmap_noise_is_applied_and_optional():
    from jpw.config import EnvCfg
    from jpw.envs import make_env

    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    base = dict(name="Go1Parkour", num_envs=8, use_heightmap=True)
    clean = jax.jit(make_env(EnvCfg(**base)).reset)(keys).obs["heightmap"]
    noisy = jax.jit(make_env(EnvCfg(**base, hmap_noise=0.02)).reset)(keys).obs["heightmap"]
    diff = np.asarray(noisy - clean)
    assert 0.005 < diff.std() < 0.06, diff.std()
