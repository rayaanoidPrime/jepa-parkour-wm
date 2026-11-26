"""Geometry and reward sanity of jpw.goal_task (pure JAX, runs anywhere)."""
import jax.numpy as jnp
import numpy as np

from jpw import goal_task as gt


def _rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _random_tilted(rng, max_tilt=0.8):
    yaw, roll, pitch = rng.uniform(-3, 3), rng.uniform(-max_tilt, max_tilt), rng.uniform(-max_tilt, max_tilt)
    cy, sy, cr, sr, cp, sp = np.cos(yaw / 2), np.sin(yaw / 2), np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2)
    q = np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                  cr * cp * sy - sr * sp * cy])
    return q / np.linalg.norm(q)


def test_world_velocity_and_yaw_rate_match_true_rotation():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = _random_tilted(rng)
        R = _rot(q)
        g_b = R.T @ np.array([0.0, 0.0, -1.0])
        v_w, w_w = rng.normal(size=3), rng.normal(size=3)
        v_b, gyro = R.T @ v_w, R.T @ w_w
        yaw = float(gt.yaw_from_quat(jnp.asarray(q)))
        got = np.asarray(gt.world_velocity(jnp.asarray(v_b), jnp.asarray(g_b), yaw))
        np.testing.assert_allclose(got, v_w, atol=1e-5)
        assert abs(float(gt.yaw_rate(jnp.asarray(gyro), jnp.asarray(g_b))) - w_w[2]) < 1e-5


def test_pose_step_integrates_a_constant_world_velocity():
    q = _random_tilted(np.random.default_rng(1))
    R = _rot(q)
    g_b, v_w = R.T @ np.array([0, 0, -1.0]), np.array([0.4, -0.2, 0.05])
    pos, yaw = gt.pose_step(jnp.zeros(3), float(gt.yaw_from_quat(jnp.asarray(q))), jnp.asarray(R.T @ v_w),
                            jnp.asarray(g_b), jnp.zeros(3), 0.5)
    np.testing.assert_allclose(np.asarray(pos), v_w * 0.5, atol=1e-5)


def test_body_rates_from_gravity_recovers_roll_pitch_rate():
    rng = np.random.default_rng(2)
    for _ in range(20):
        g = np.array([0.1, -0.2, -1.0])
        g /= np.linalg.norm(g)
        w = rng.normal(size=3)
        dt = 1e-4
        g1 = g - np.cross(w, g) * dt
        got = np.asarray(gt.body_rates_from_gravity(jnp.asarray(g), jnp.asarray(g1), dt))
        want = w - g * (g @ w)  # the component perpendicular to gravity
        np.testing.assert_allclose(got, want, atol=2e-3)


def test_goal_features_roundtrip():
    rng = np.random.default_rng(3)
    pos = jnp.asarray(rng.uniform(-2, 2, (7, 3)))
    yaw = jnp.asarray(rng.uniform(-3, 3, 7))
    gxy = pos[:, :2] + jnp.asarray(rng.uniform(-3, 3, (7, 2)))
    gyaw = jnp.asarray(rng.uniform(-3, 3, 7))
    t = jnp.asarray(rng.uniform(0, 8, 7))
    f = gt.goal_features(pos, yaw, gxy, gyaw, t)
    assert f.shape == (7, gt.GOAL_FEAT_DIM)
    gxy2, gyaw2, t2 = gt.goal_from_features(pos, yaw, f)
    np.testing.assert_allclose(np.asarray(gxy2), np.asarray(gxy), atol=1e-4)
    np.testing.assert_allclose(np.asarray(gt.wrap_angle(gyaw2 - gyaw)), 0.0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(t2), np.asarray(t), atol=1e-4)


def _reward_fixture(**kw):
    ground = lambda x, y: jnp.zeros_like(x)
    return gt.GoalReward(np.zeros(12), -np.ones(12), np.ones(12), ground, **kw)


def _state(pos=(0.0, 0.0, 0.3), vel=(0.0, 0.0, 0.0), g=(0.0, 0.0, -1.0)):
    n = {"g": jnp.asarray(g)[None], "jpos": jnp.zeros((1, 12)), "linvel": jnp.asarray(vel)[None],
         "gyro": jnp.zeros((1, 3)), "feet_vel": jnp.zeros((1, 12))}
    return n, jnp.asarray(pos)[None]


def test_reward_prefers_moving_to_goal_and_standing_there_and_penalises_falling():
    rw = _reward_fixture()
    prev, pos = _state()
    z, c = jnp.zeros((1, 12)), jnp.ones((1, 4))
    goal, gyaw = jnp.array([[3.0, 0.0]]), jnp.zeros(1)

    def score(vel=(0, 0, 0), pos_=(0, 0, 0.3), g=(0, 0, -1.0), t_left=6.0):
        nxt, p = _state(pos_, vel, g)
        r, d, t = rw.reward(prev, nxt, p, jnp.zeros(1), goal, gyaw, jnp.asarray([t_left]), z, z, c)
        return float(r[0]), float(d[0]), t

    toward, _, tt = score(vel=(1.0, 0, 0))
    away, _, ta = score(vel=(-1.0, 0, 0))
    still, _, _ = score()
    assert toward > still > away - 1e-6 or toward > away
    assert float(tt["move"][0]) > float(ta["move"][0])
    at_goal, _, tg = score(pos_=(3.0, 0, 0.3), t_left=1.0)
    assert float(tg["stand"][0]) > 0.5 and float(tg["head"][0]) > 0.4
    far_late, _, tf = score(pos_=(0, 0, 0.3), t_left=1.0)
    assert float(tg["pos"][0]) > float(tf["pos"][0])
    _, d_up, _ = score()
    _, d_fall, _ = score(g=(0.0, 0.0, 1.0))
    assert d_up < 0.01 and d_fall > 0.99
    _, d_belly, _ = score(pos_=(0, 0, 0.05))
    assert d_belly > 0.9


def test_reward_is_differentiable_in_the_action_and_pose():
    import jax

    rw = _reward_fixture()
    prev, _ = _state()

    def f(a, p):
        nxt, _ = _state()
        r, _, _ = rw.reward(prev, nxt, p, jnp.zeros(1), jnp.array([[3.0, 0.0]]), jnp.zeros(1), jnp.asarray([3.0]),
                            a, jnp.zeros((1, 12)), jnp.ones((1, 4)))
        return r.sum()

    ga, gp = jax.grad(f, argnums=(0, 1))(jnp.full((1, 12), 0.3), jnp.array([[0.5, 0.2, 0.3]]))
    assert bool(jnp.all(jnp.isfinite(ga))) and bool(jnp.all(jnp.isfinite(gp)))
    assert float(jnp.abs(ga).max()) > 0 and float(jnp.abs(gp).max()) > 0
