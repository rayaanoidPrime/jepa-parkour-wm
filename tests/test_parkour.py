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


@requires_sim
def test_unclipped_reward_matches_terms_and_alive_bonus():
    from jpw.config import EnvCfg
    from jpw.envs import make_env

    def run(**kw):
        env = make_env(EnvCfg(name="Go1JoystickFlatTerrain", num_envs=16, episode_length=50, **kw))
        st = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 16))
        a = jax.random.uniform(jax.random.PRNGKey(1), (16, env.action_size), minval=-1.0, maxval=1.0)
        return jax.jit(env.step)(st, a)

    clipped, raw, bonus = run(), run(unclipped_reward=True), run(unclipped_reward=True, alive_bonus=0.05)
    terms = sum(v for k, v in raw.metrics.items() if k.startswith("reward/"))
    np.testing.assert_allclose(np.asarray(raw.reward), 0.02 * np.asarray(terms), rtol=1e-5, atol=1e-7)
    assert float(clipped.reward.min()) >= 0.0 and float(raw.reward.min()) < 0.0  # the clip removed signal
    np.testing.assert_allclose(np.asarray(bonus.reward - raw.reward), 0.05 * (1 - np.asarray(raw.done)),
                               atol=1e-6)


@requires_sim
def test_analytic_go1_reward_matches_playground_terms():
    """The reward recomputed from the exact physical state equals Playground's per-term reward."""
    from jpw import go1_state as gs
    from jpw.config import EnvCfg
    from jpw.envs import get_aux, make_env

    cfg = EnvCfg(name="Go1JoystickFlatTerrain", num_envs=16, episode_length=200, use_privileged=True,
                 unclipped_reward=True)
    env = make_env(cfg)
    task = gs.Go1Task.from_env(env)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    st = reset(jax.random.split(jax.random.PRNGKey(0), 16))
    assert st.obs["state"].shape[-1] + st.obs["privileged_state"].shape[-1] == gs.OBS_DIM
    key = jax.random.PRNGKey(1)
    worst = {}
    for t in range(12):
        key, k = jax.random.split(key)
        a = jax.random.uniform(k, (16, env.action_size), minval=-1.0, maxval=1.0)
        prev = st
        st = step(prev, a)
        obs = jnp.concatenate([st.obs["state"], st.obs["privileged_state"]], -1)
        phys = gs.phys_targets(obs, get_aux(st))
        mine = task.terms(phys, a, prev.info["last_act"], prev.info["command"], st.info["contact"])
        for k2, v in mine.items():
            real = st.metrics[f"reward/{k2}"]
            err = float(jnp.max(jnp.abs(v - real)) / (float(jnp.max(jnp.abs(real))) + 1e-3))
            worst[k2] = max(worst.get(k2, 0.0), err)
    exact = ["tracking_lin_vel", "tracking_ang_vel", "ang_vel_xy", "pose", "dof_pos_limits", "action_rate",
             "torques", "energy", "stand_still", "feet_slip", "feet_clearance"]
    for k2 in exact:
        assert worst[k2] < 2e-3, (k2, worst[k2])
    # approximations (body-frame gravity / velocity instead of the world-frame torso axis / velocity)
    assert worst["orientation"] < 0.5 and worst["lin_vel_z"] < 0.5, worst


@requires_sim
def test_decoder_heads_and_analytic_reward_train_on_go1(tmp_path, capfd):
    from jpw import train

    ts = train.main(["--run-name", "r", "--set", "env.name=Go1JoystickFlatTerrain", "env.num_envs=8",
                     "env.use_privileged=true", "env.unclipped_reward=true", "env.alive_bonus=0.05",
                     "wm.decode_state=true", "ac.analytic_reward=true", "ac.fixed_std=0.1", "ac.ent_coef=0.0",
                     "wm.latent_dim=32", "wm.hidden=64", "ac.hidden=64", "wm.batch=8", "ac.batch=8",
                     "wm.pred_horizon=3", "ac.horizon=4", "train.iters=6", "train.warmup_iters=1",
                     "train.collect_steps=4", "train.replay_capacity=24", "train.log_every=2",
                     "train.wandb_mode=disabled", f"train.out_dir={tmp_path}", "train.ckpt_every=3"])
    out = capfd.readouterr().out
    assert "wm/phys" in out or "phys=" in out
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(ts.wm))
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(ts.actor))


@requires_sim
def test_goal_task_env_features_reward_and_reset():
    from jpw import go1_state as gs
    from jpw import goal_task as gt
    from jpw.config import EnvCfg
    from jpw.envs import get_obs, get_spec, make_env

    cfg = EnvCfg(name="Go1Parkour", num_envs=8, episode_length=30, use_heightmap=True, use_privileged=True,
                 goal_task=True)
    env = make_env(cfg)
    st = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 8))
    spec = get_spec(st, cfg, env.action_size)
    assert spec.obs_dim == gs.GOAL_OBS_DIM and spec.prop_dim == 48 and spec.aux_dim == 19
    assert get_obs(st.obs, True).shape == (8, gs.GOAL_OBS_DIM)
    f = st.obs["extra"]
    assert f.shape == (8, gt.GOAL_FEAT_DIM)
    assert float(f[:, 0].min()) > 2.2 and float(f[:, 0].max()) < 3.8  # goal 2.5-3.5 m ahead (plus spawn jitter)
    np.testing.assert_allclose(np.asarray(f[:, 4]), 1.0, atol=1e-6)    # full time left
    step = jax.jit(env.step)
    zero = jnp.zeros((8, env.action_size))
    for k in range(1, 6):
        st = step(st, zero)
    assert bool(jnp.all(jnp.isfinite(st.reward))) and "goal_dist" in st.info
    np.testing.assert_allclose(np.asarray(st.obs["extra"][:, 4]), 1.0 - 5 * 0.02 / gt.EPISODE_S, atol=1e-4)
    for _ in range(40):  # past the 30-step episode: the timer and goal features restart with the new episode
        st = step(st, zero)
    assert float(st.obs["extra"][:, 4].max()) <= 1.0 and float(st.obs["extra"][:, 4].min()) > 0.9


def test_graded_course_ramps_difficulty_and_has_flat_goals():
    flat = pk.make_course(0, 1.0, ramp=False)
    ramp = pk.make_course(0, 1.0, ramp=True)
    xs = pk.grid_x()
    assert np.all(ramp[xs < pk.RAMP_START_FLAT - 1e-6] == 0.0)
    early, late = ramp[(xs > 6.0) & (xs < 10.0)], ramp[(xs > 13.0) & (xs < 19.0)]
    def step(w):  # largest single step, ignoring the fixed-depth pit edges
        s = np.abs(np.diff(w))
        return s[s < 0.249].max()

    assert step(early) < step(late) and ramp.max() <= flat.max() + 1e-9
    sp, p = pk.spawn_positions(0, 1.0, ramp=True, flat_goal_frac=0.3)
    assert abs(p.sum() - 1.0) < 1e-6 and len(sp) == len(p)
    prof = ramp
    is_flat = np.array([np.ptp(prof[(xs >= x) & (xs <= x + 3.5)]) < 0.03 for x in sp])
    assert is_flat.any() and abs(p[is_flat].sum() - 0.3) < 1e-6
    # the ungraded course has no flat-goal spawn at all (every goal needs an obstacle crossing)
    sp0, p0 = pk.spawn_positions(0, 1.0, ramp=False)
    prof0 = pk.make_course(0, 1.0, False)
    assert not any(np.ptp(prof0[(xs >= x) & (xs <= x + 3.5)]) < 0.03 for x in sp0)


@requires_sim
def test_tiles_env_respawns_fresh_tiles_and_keeps_terminal_info():
    from jpw.config import EnvCfg
    from jpw.envs import get_obs, get_spec, make_env

    cfg = EnvCfg(name="Go1Tiles", num_envs=4, episode_length=12, goal_task=True, use_heightmap=True,
                 use_privileged=True)
    env = make_env(cfg)
    grid = env.unwrapped._grid
    st = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 4))
    spec = get_spec(st, cfg, env.action_size)
    assert spec.obs_dim == 176 and spec.aux_dim == 19 and spec.hmap_shape == (12, 8)
    cx, cy = grid.cell_centre(np.asarray(st.info["cell"]))
    assert np.abs(np.asarray(st.data.qpos[:, :2]) - np.stack([cx, cy], -1)).max() < 0.25  # spawn on the tile centre
    d = np.asarray(st.info["goal_dist"])
    assert d.min() >= 2.5 and d.max() <= 3.5
    step = jax.jit(env.step)
    zero = jnp.zeros((4, env.action_size))
    seen_done = False
    for _ in range(14):
        prev = np.asarray(st.info["cell"])
        st = step(st, zero)
        if float(st.done.max()) > 0:
            seen_done = True
            assert (np.asarray(st.info["ended_cell"]) == prev).all()
            assert float(st.info["truncation"].min()) > 0           # timeout flag of the finished episode survives
    assert seen_done and bool(jnp.all(jnp.isfinite(st.reward)))
    assert float(st.obs["extra"][:, 4].min()) > 0.9                 # time-left feature restarted with the new episode
