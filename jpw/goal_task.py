"""Goal-reaching task for Go1 parkour: pose integration, goal features and the AME2 / TGI-ANYmal reward.

The ANYmal variant of Terrain-Grounded Imagination trains a position-goal task (Rudin et al. 2022): reach a
goal position and heading within the episode (8 s), with the AME2 reward (Zhang et al. 2026, Table I).
Everything here is a pure JAX function of the *physical state*, so the same code scores

  * real transitions (exact state from the simulator), and
  * imagined transitions (state decoded from the latent, pose integrated from decoded velocities),

with identical definitions. Penalties are built from quantities the world model predicts well (joint
positions, gravity direction, base velocity) and from finite differences of them, not from decoded
noisy velocities / forces, which a mean-regressing model underestimates (see docs/findings.md).

Frames: world z up; yaw = heading of the body x-axis projected on the horizontal plane; the heading frame is
that horizontal frame; gravity `g_b` is the unit gravity direction in the body frame (upright = (0, 0, -1)).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

EPISODE_S = 8.0
GOAL_FEAT_DIM = 5   # dx_heading, dy_heading, sin(dyaw), cos(dyaw), time_left / EPISODE_S
GOAL_CLIP = 4.0


def wrap_angle(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


def yaw_from_quat(q):
    """q = (w, x, y, z): heading angle of the body x-axis on the horizontal plane."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _unit(v, eps=1e-8):
    return v / jnp.sqrt(jnp.sum(v * v, -1, keepdims=True) + eps)


def heading_axes(g_b):
    """Heading-frame axes expressed in body coordinates: (x_h, y_h, up)."""
    up = -_unit(g_b)
    ex = jnp.zeros_like(up).at[..., 0].set(1.0)
    xh = _unit(ex - jnp.sum(ex * up, -1, keepdims=True) * up)
    yh = jnp.cross(up, xh)
    return xh, yh, up


def world_velocity(v_b, g_b, yaw):
    """Body-frame linear velocity -> world velocity (3) using the gravity direction and the yaw."""
    xh, yh, up = heading_axes(g_b)
    vx, vy, vz = jnp.sum(v_b * xh, -1), jnp.sum(v_b * yh, -1), jnp.sum(v_b * up, -1)
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.stack([c * vx - s * vy, s * vx + c * vy, vz], -1)


def yaw_rate(gyro, g_b):
    """World-frame yaw rate from the body-frame angular velocity and the gravity direction."""
    return jnp.sum(gyro * heading_axes(g_b)[2], -1)


def pose_step(pos, yaw, v_b, g_b, gyro, dt):
    """Semi-implicit pose update from the *next-state* velocities (as in MuJoCo's integrator)."""
    return pos + world_velocity(v_b, g_b, yaw) * dt, yaw + yaw_rate(gyro, g_b) * dt


def body_rates_from_gravity(g_prev, g_next, dt):
    """Roll / pitch angular velocity (body frame, component perpendicular to gravity) from two successive
    gravity directions: g' = -w x g  =>  w_perp = -g x g'."""
    return -jnp.cross(_unit(g_next), (g_next - g_prev) / dt)


def tile_centre(pos, tile_size):
    """Centre (x, y) of the tile (tiles are aligned at the origin) that contains the base position `pos`."""
    return (jnp.floor(pos[..., :2] / tile_size) + 0.5) * tile_size


def goal_features(pos, yaw, goal_xy, goal_yaw, t_left_s):
    """Goal in the heading frame + remaining time: [..., 5]."""
    d = goal_xy - pos[..., :2]
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    dx, dy = c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]
    dyaw = wrap_angle(goal_yaw - yaw)
    return jnp.stack([jnp.clip(dx, -GOAL_CLIP, GOAL_CLIP), jnp.clip(dy, -GOAL_CLIP, GOAL_CLIP), jnp.sin(dyaw),
                      jnp.cos(dyaw), jnp.clip(t_left_s, 0.0, EPISODE_S) / EPISODE_S], -1)


def goal_from_features(pos, yaw, feat):
    """Inverse of goal_features (exact while the goal is within GOAL_CLIP): world goal xy, goal yaw, time left (s)."""
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    dx, dy = feat[..., 0], feat[..., 1]
    gxy = pos[..., :2] + jnp.stack([c * dx - s * dy, s * dx + c * dy], -1)
    return gxy, wrap_angle(yaw + jnp.arctan2(feat[..., 2], feat[..., 3])), feat[..., 4] * EPISODE_S


class GoalReward:
    """AME2 goal-reaching reward (+ TGI-ANYmal shaping) for Go1, computed from physical-state quantities.

    Terms follow tgi_anymal/mdp/mdp.py `reward_ame2` / `shaping` / `soft_termination` (weights per second, x dt);
    ANYmal-specific pieces are dropped or approximated: no self-collision spheres, no foot-impact force, the leap
    event is omitted, "non-foot contact" is a belly-too-close-to-the-terrain event, and the floorless-gap
    termination is the course's pit rule (base below `pit_z`).
    """

    W = dict(pos=100.0, head=50.0, move=5.0, stand=5.0, term=-500.0, events=-1.0, roll=-0.1, joint=-0.001,
             rate=-0.01, qlim=-1000.0, qdlim=-1.0, taulim=-1.0, power=-0.002, face=2.0, lat=-1.0)

    def __init__(self, default_pose, soft_lo, soft_hi, ground_fn, dt=0.02, kp=35.0, kd=0.5, action_scale=0.5,
                 tau_limit=23.7, qd_limit=30.0, term_angle=1.4, term_height=0.12, pit_z=None, lane_half=None,
                 weights=None, tile_size=None):
        self.default_pose = jnp.asarray(default_pose, jnp.float32)
        self.lo, self.hi = jnp.asarray(soft_lo, jnp.float32), jnp.asarray(soft_hi, jnp.float32)
        self.ground, self.dt = ground_fn, dt
        self.kp, self.kd, self.scale = kp, kd, action_scale
        self.tau_limit, self.qd_limit = tau_limit, qd_limit
        self.term_angle, self.term_height, self.pit_z, self.lane_half = term_angle, term_height, pit_z, lane_half
        self.tile_size = tile_size  # tiled terrain: leaving the 8 m tile terminates (tile centre passed per call)
        self.w = dict(self.W, **(weights or {}))

    @classmethod
    def from_env(cls, env, ground_fn, **kw):
        u = env.unwrapped
        return cls(np.asarray(u._default_pose), np.asarray(u._soft_lowers), np.asarray(u._soft_uppers), ground_fn,
                   dt=float(u.dt), **kw)

    # ------------------------------------------------------------------
    def soft_termination(self, pos, g_b, tz=0.02, tt=0.03, tile_xy=None):
        """1 - P(alive): tilt, belly too close to the terrain, pit, lane, tile exit (all differentiable)."""
        sig = jax.nn.sigmoid
        hb = pos[..., 2] - self.ground(pos[..., 0], pos[..., 1])
        alive = sig((-g_b[..., 2] - jnp.cos(self.term_angle)) / tt) * sig((hb - self.term_height) / tz)
        if self.pit_z is not None:
            alive = alive * sig((pos[..., 2] - self.pit_z) / tz)
        if self.lane_half is not None:
            alive = alive * sig((self.lane_half - jnp.abs(pos[..., 1])) / 0.05)
        if self.tile_size is not None and tile_xy is not None:
            half = self.tile_size / 2 - 0.05
            off = jnp.abs(pos[..., :2] - tile_xy)
            alive = alive * jnp.prod(sig((half - off) / 0.05), -1)
        return 1.0 - alive

    def terms(self, prev, nxt, pos, yaw, goal_xy, goal_yaw, t_left_s, a, a_prev, contact, tile_xy=None):
        """prev / nxt: dicts with 'g' (gravity dir, body), 'jpos' (joint pos relative to default), and for nxt also
        'linvel' (body), 'gyro', 'feet_vel' (4 x 3). pos / yaw: base pose at the END of the step. Returns the
        per-second reward terms (before the weights) and the termination probability."""
        sig = jax.nn.sigmoid
        eps = 1e-8
        dt = self.dt
        g, v_b, gyro = nxt["g"], nxt["linvel"], nxt["gyro"]
        to_goal = goal_xy - pos[..., :2]
        d2 = jnp.sum(to_goal**2, -1)
        dxy = jnp.sqrt(d2 + eps)
        dyaw = jnp.abs(wrap_angle(yaw - goal_yaw))
        near = sig((0.5 - dxy) / 0.03)
        r_pos = (1.0 / (1.0 + 0.25 * d2)) * (t_left_s < 4.0) / 4.0
        r_head = (1.0 / (1.0 + dyaw**2)) * (t_left_s < 2.0) / 2.0 * near
        v_w = world_velocity(v_b, g, yaw)
        speed = jnp.sqrt(jnp.sum(v_b[..., :2] ** 2, -1) + eps)
        speed_w = jnp.sqrt(jnp.sum(v_w[..., :2] ** 2, -1) + eps)
        cos = jnp.sum(v_w[..., :2] * to_goal, -1) / (speed_w * dxy)
        moving = sig((cos - 0.5) / 0.05) * sig((speed - 0.3) / 0.03) * sig((2.0 - speed) / 0.03)
        r_move = near + (1 - near) * moving
        q = nxt["jpos"] + self.default_pose
        d_foot = jnp.mean(1.0 - contact, -1)
        d_g = 1.0 - g[..., 2] ** 2
        d_q = jnp.mean(jnp.abs(nxt["jpos"]), -1)
        r_stand = near * sig((0.5 - dyaw) / 0.05) * jnp.exp(-(d_foot + d_g + d_q + dxy) / 4.0)
        # events: spinning, belly near the terrain, foot slip (each -1)
        e_spin = sig((jnp.abs(yaw_rate(gyro, g)) - 2.0) / 0.1)
        hb = pos[..., 2] - self.ground(pos[..., 0], pos[..., 1])
        e_belly = sig((self.term_height + 0.03 - hb) / 0.02)
        fv = nxt["feet_vel"].reshape(*gyro.shape[:-1], 4, 3)
        e_slip = jnp.sum(contact * sig((jnp.sqrt(jnp.sum(fv[..., :2] ** 2, -1) + eps) - 0.1) / 0.03), -1)
        events = e_spin + e_belly + e_slip
        # smooth-quantity penalties: finite differences of the decoded joints / gravity
        qd = (nxt["jpos"] - prev["jpos"]) / dt
        w_perp = body_rates_from_gravity(prev["g"], g, dt)
        r_roll = w_perp[..., 0] ** 2
        q_prev = prev["jpos"] + self.default_pose
        tau_u = self.kp * (self.default_pose + self.scale * a - q_prev) - self.kd * qd
        tau = jnp.clip(tau_u, -self.tau_limit, self.tau_limit)
        r_joint = jnp.sum(qd**2, -1) + 0.01 * jnp.sum(tau**2, -1)
        r_rate = jnp.sum((a - a_prev) ** 2, -1)
        span = self.hi - self.lo
        r_qlim = jnp.sum(jax.nn.relu(q - (self.hi - 0.025 * span)) + jax.nn.relu((self.lo + 0.025 * span) - q), -1)
        r_qdlim = jnp.sum(jax.nn.relu(jnp.abs(qd) - 0.9 * self.qd_limit), -1)
        r_taulim = jnp.sum(jax.nn.relu(jnp.abs(tau_u) - 0.8 * self.tau_limit), -1)
        r_power = jnp.sum(jnp.sqrt((tau * qd) ** 2 + 1e-6), -1)
        # shaping (TGI-ANYmal): face the goal while away from it, no lateral sliding
        far = sig((dxy - 0.5) / 0.05)
        bearing = jnp.arctan2(to_goal[..., 1], to_goal[..., 0])
        r_face = far * jnp.cos(yaw - bearing)
        r_lat = v_b[..., 1] ** 2
        d = self.soft_termination(pos, g, tile_xy=tile_xy)
        t = dict(pos=r_pos, head=r_head, move=r_move, stand=r_stand, term=d, events=events, roll=r_roll,
                 joint=r_joint, rate=r_rate, qlim=r_qlim, qdlim=r_qdlim, taulim=r_taulim, power=r_power,
                 face=r_face, lat=r_lat)
        return t, d

    def reward(self, prev, nxt, pos, yaw, goal_xy, goal_yaw, t_left_s, a, a_prev, contact, tile_xy=None):
        t, d = self.terms(prev, nxt, pos, yaw, goal_xy, goal_yaw, t_left_s, a, a_prev, contact, tile_xy)
        r = sum(self.w[k] * v for k, v in t.items()) * self.dt
        return r, d, t
