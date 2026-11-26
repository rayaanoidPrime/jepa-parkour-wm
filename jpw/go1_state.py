"""Physical state of Go1 for decoder heads, and the analytic reward / termination computed from it.

The JEPA world model keeps its latent dynamics. A decoder head maps a latent to the *exact* physical
quantities the reward and termination need, and Playground's Go1 reward is recomputed from those
quantities by plain formulas (no learned reward head, no clipping). The decoder targets come from
what the replay already stores when env.use_privileged=true (the noise-free sensors in the
privileged observation) plus the `aux` vector (base pose, feet world positions).

Observation layout with use_privileged=true (obs_dim 171 = state 48 + privileged 123):
  state      [0:48]    linvel(3) gyro(3) gravity(3) joint_pos-default(12) joint_vel(12) last_act(12) cmd(3), noisy
  privileged [48:171]  state(48) gyro(3) accel(3) gravity(3) linvel(3) angvel(3) joint_pos-default(12)
                       joint_vel(12) actuator_force(12) contact(4) feet_vel(12) feet_air_time(4) ...
All of these are noise-free in the privileged part, which is what we decode.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

OBS_DIM = 171          # state 48 + privileged 123
GOAL_OBS_DIM = OBS_DIM + 5  # + goal features (jpw.goal_task) appended by the adapter in the goal task
# --- slices of the observation vector
GOAL = slice(OBS_DIM, GOAL_OBS_DIM)  # goal features (goal task), appended after the Playground observation
LAST_ACT = slice(33, 45)
CMD = slice(45, 48)
_GYRO, _GRAV, _LINVEL, _ANGVEL = slice(96, 99), slice(102, 105), slice(105, 108), slice(108, 111)
_JPOS, _JVEL, _FORCE, _FEETVEL = slice(111, 123), slice(123, 135), slice(135, 147), slice(151, 163)
AUX_FEET = slice(7, 19)  # aux = base pos(3) quat(4) feet world pos (4 x 3)

# name, dim, output scale (typical magnitude: keeps the head's output O(1)), nominal error (loss weight
# 1/nominal^2, in the spirit of SuperMimic's Bryson's-rule weights)
GROUPS = (
    ("linvel", 3, 0.5, 0.1),
    ("gyro", 3, 1.0, 0.3),
    ("angvel", 3, 1.0, 0.3),
    ("gravity", 3, 0.3, 0.05),
    ("jpos", 12, 0.3, 0.03),
    ("jvel", 12, 3.0, 1.0),
    ("force", 12, 5.0, 2.0),
    ("feet_vel", 12, 1.0, 0.5),
    ("feet_z", 4, 0.05, 0.01),
)
NAMES = tuple(g[0] for g in GROUPS)
DIMS = tuple(g[1] for g in GROUPS)
PHYS_DIM = int(sum(DIMS))
SD = np.concatenate([np.full(d, s, np.float32) for _, d, s, _ in GROUPS])      # head output scale
NOM = np.concatenate([np.full(d, n, np.float32) for _, d, _, n in GROUPS])     # loss scale
OFFS = np.concatenate([[0], np.cumsum(DIMS)])


def check_layout(obs_dim: int, aux_dim: int) -> None:
    if obs_dim not in (OBS_DIM, GOAL_OBS_DIM) or aux_dim != 19:
        raise ValueError(f"decode_state needs Go1 with env.use_privileged=true (obs_dim {OBS_DIM} or "
                         f"{GOAL_OBS_DIM} with goal features, aux 19); "
                         f"got obs_dim={obs_dim}, aux_dim={aux_dim}")


def phys_targets(obs, aux):
    """[..., obs_dim], [..., 19] -> exact physical state [..., PHYS_DIM] (raw units)."""
    feet_z = aux[..., AUX_FEET].reshape(*aux.shape[:-1], 4, 3)[..., 2]
    return jnp.concatenate([obs[..., _LINVEL], obs[..., _GYRO], obs[..., _ANGVEL], obs[..., _GRAV],
                            obs[..., _JPOS], obs[..., _JVEL], obs[..., _FORCE], obs[..., _FEETVEL], feet_z], -1)


def obs_vector(obs):
    """Playground dict observation -> the replay layout (state ++ privileged_state), no goal features."""
    return jnp.concatenate([obs["state"], obs["privileged_state"]], -1)


def aux_from_data(data, feet_site_id):
    """Base pose (pos 3 + quat 4) and feet world positions (12) of one env's mjx Data: the `aux` vector."""
    return jnp.concatenate([data.qpos[:7], data.site_xpos[feet_site_id, :].reshape(-1)], -1)


def reward_inputs(phys):
    """The decoded / exact physical state in the form goal_task.GoalReward expects."""
    s = split(phys)
    return {"g": s["gravity"], "jpos": s["jpos"], "linvel": s["linvel"], "gyro": s["gyro"],
            "feet_vel": s["feet_vel"]}


def split(phys):
    """[..., PHYS_DIM] -> dict of named groups."""
    return {n: phys[..., OFFS[i]:OFFS[i + 1]] for i, n in enumerate(NAMES)}


class Go1Task:
    """Playground's Go1 joystick reward and fall termination as functions of the physical state.

    Each term mirrors `mujoco_playground` Joystick (same weights and formulas). Approximations: the
    orientation cost uses the gravity vector in the body frame (equal to the torso z-axis tilt for
    small angles), lin_vel_z uses the body-frame velocity; the swing-peak (feet_height) and air-time
    terms need history and are left out. The reward is *not* clipped at 0.
    """

    SCALES = dict(tracking_lin_vel=1.0, tracking_ang_vel=0.5, lin_vel_z=-0.5, ang_vel_xy=-0.05,
                  orientation=-5.0, dof_pos_limits=-1.0, pose=0.5, stand_still=-1.0, torques=-0.0002,
                  action_rate=-0.01, energy=-0.001, feet_clearance=-2.0, feet_slip=-0.1)

    def __init__(self, default_pose, soft_lo, soft_hi, dt: float = 0.02, sigma: float = 0.25,
                 max_foot_height: float = 0.1, alive_bonus: float = 0.0, term_tau: float = 0.03):
        self.default_pose = jnp.asarray(default_pose, jnp.float32)
        self.soft_lo, self.soft_hi = jnp.asarray(soft_lo, jnp.float32), jnp.asarray(soft_hi, jnp.float32)
        self.dt, self.sigma, self.max_foot_height = dt, sigma, max_foot_height
        self.alive_bonus, self.term_tau = alive_bonus, term_tau
        self.pose_w = jnp.array([1.0, 1.0, 0.1] * 4, jnp.float32)

    @classmethod
    def from_env(cls, env, alive_bonus: float = 0.0) -> "Go1Task":
        u = env.unwrapped
        return cls(np.asarray(u._default_pose), np.asarray(u._soft_lowers), np.asarray(u._soft_uppers),
                   dt=float(u.dt), alive_bonus=alive_bonus)

    def terms(self, phys, a, a_prev, cmd, contact):
        """Scaled reward terms (before dt), same names / scaling as Playground's state.metrics."""
        g = split(phys)
        S = self.SCALES
        eps = 1e-8
        cmd_norm = jnp.sqrt(jnp.sum(cmd**2, -1) + eps)
        q = g["jpos"] + self.default_pose
        tau, qd = g["force"], g["jvel"]
        feet_v = g["feet_vel"].reshape(*phys.shape[:-1], 4, 3)
        vxy2 = jnp.sum(feet_v[..., :2] ** 2, -1)
        out_of_limits = -jnp.clip(q - self.soft_lo, None, 0.0) + jnp.clip(q - self.soft_hi, 0.0, None)
        slip = jnp.sum(vxy2 * contact, -1) * (cmd_norm > 0.01)
        clearance = jnp.sum(jnp.abs(g["feet_z"] - self.max_foot_height) * jnp.sqrt(jnp.sqrt(vxy2 + eps) + eps), -1)
        t = dict(
            tracking_lin_vel=jnp.exp(-jnp.sum((cmd[..., :2] - g["linvel"][..., :2]) ** 2, -1) / self.sigma),
            tracking_ang_vel=jnp.exp(-((cmd[..., 2] - g["gyro"][..., 2]) ** 2) / self.sigma),
            lin_vel_z=g["linvel"][..., 2] ** 2,
            ang_vel_xy=jnp.sum(g["angvel"][..., :2] ** 2, -1),
            orientation=jnp.sum(g["gravity"][..., :2] ** 2, -1),
            dof_pos_limits=jnp.sum(out_of_limits, -1),
            pose=jnp.exp(-jnp.sum(g["jpos"] ** 2 * self.pose_w, -1)),
            stand_still=jnp.sum(jnp.abs(g["jpos"]), -1) * (cmd_norm < 0.01),
            torques=jnp.sqrt(jnp.sum(tau**2, -1) + eps) + jnp.sum(jnp.abs(tau), -1),
            action_rate=jnp.sum((a - a_prev) ** 2, -1),
            energy=jnp.sum(jnp.abs(qd) * jnp.abs(tau), -1),
            feet_clearance=clearance,
            feet_slip=slip,
        )
        return {k: S[k] * v for k, v in t.items()}

    def term_prob(self, phys):
        """Soft fall probability: Playground terminates when the torso z-axis points down, i.e. when the
        body-frame gravity z component turns positive (upright: -1)."""
        return jax.nn.sigmoid(split(phys)["gravity"][..., 2] / self.term_tau)

    def reward(self, phys, a, a_prev, cmd, contact):
        """Per-step reward (scaled by dt, unclipped) and soft termination probability."""
        d = self.term_prob(phys)
        r = self.dt * (sum(self.terms(phys, a, a_prev, cmd, contact).values()) - 1.0 * d)  # termination cost
        return r + self.alive_bonus * (1.0 - d), d
