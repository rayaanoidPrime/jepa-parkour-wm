"""Go1 parkour: Playground's Go1 joystick task on a procedural obstacle course.

The course is a MuJoCo heightfield along +x (stairs, boxes, gaps) generated from a seed. The same
height array that is written into the MJX model is sampled in JAX to build the heightmap
observation, so the observation always matches the physics. Everything in this module needs
`mujoco_playground` (imported lazily by `jpw.envs.make_env`).

Terrain is shared by all envs (one MJX model); randomise per-env terrain later by sharding envs
over several seeds, or by swapping `make_course` for a per-env sampler.

Frames: terrain x runs from X0 to X0 + 2*RX; the robot spawns near x=0 on the flat start zone and
a run ends at FINISH_X. The ground sits at elevation G, so gaps can be pits (down to 0) and
obstacles rise above G (heightfields cannot go below z=0).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.ndimage as ndi
import numpy as np

# ---- terrain geometry (metres) -------------------------------------------------------------
G = 0.25            # ground elevation; pits go down to 0, obstacles up to ZMAX
ZMAX = 1.0          # hfield top (elevation data is normalised to [0, ZMAX])
RX, RY = 12.0, 3.0  # hfield half-extents
X0 = -2.0           # course start x; the hfield geom is centred at POSX
POSX = X0 + RX
RES = 0.04          # grid resolution
NCOL = int(round(2 * RX / RES)) + 1
NROW = int(round(2 * RY / RES)) + 1
START_FLAT = 2.0    # flat run before the first obstacle (spawn is at x in [-0.5, 0.5])
FINISH_X = 19.0     # reaching this x ends the episode (success)
LANE_HALF = RY - 0.5

# ---- task constants ------------------------------------------------------------------------
PIT_Z = G + 0.15    # base below this height means the robot fell into a gap
PROGRESS = 1.0      # reward per metre of forward progress
FALL_PENALTY = 1.0  # extra penalty for falling into a pit / leaving the lane
NOMINAL_BASE_H = 0.27
HM_X = np.linspace(-0.2, 1.2, 12)   # heightmap sample points ahead of the base (yaw frame)
HM_Y = np.linspace(-0.5, 0.5, 8)
HMAP_SHAPE = (len(HM_X), len(HM_Y))

_SCENE_XML = """<mujoco model="go1 parkour scene">
  <include file="go1_mjx_feetonly.xml"/>

  <statistic center="0 0 0.1" extent="0.8" meansize="0.04"/>

  <visual>
    <rgba force="1 0 0 1"/>
    <global azimuth="120" elevation="-20"/>
    <map force="0.01"/>
    <scale forcewidth="0.3" contactwidth="0.5" contactheight="0.2"/>
    <quality shadowsize="8192"/>
  </visual>

  <asset>
    <texture type="2d" name="groundplane" file="assets/rocky_texture.png"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance=".8"/>
    <hfield name="hfield" nrow="{nrow}" ncol="{ncol}" size="{rx} {ry} {zmax} 0.1"/>
  </asset>

  <worldbody>
    <geom name="floor" type="hfield" hfield="hfield" pos="{posx} 0 0" material="groundplane"
      contype="1" conaffinity="0" priority="1" friction="1.0"/>
  </worldbody>

  <keyframe>
    <key name="home" qpos="
    0 0 0.35
    1 0 0 0
    0.1 0.9 -1.8
    -0.1 0.9 -1.8
    0.1 0.9 -1.8
    -0.1 0.9 -1.8"
      ctrl="0.1 0.9 -1.8 -0.1 0.9 -1.8 0.1 0.9 -1.8 -0.1 0.9 -1.8"/>
    <key name="home_higher" qpos="0 0 0.35 1 0 0 0 0 0.82 -1.63 0 0.82 -1.63 0 0.82 -1.63 0 0.82 -1.63"
      ctrl="0 0.82 -1.63 0 0.82 -1.63 0 0.82 -1.63 0 0.82 -1.63"/>
  </keyframe>
</mujoco>
"""


# ---- course generation ---------------------------------------------------------------------
def grid_x() -> np.ndarray:
    """World x of each hfield column."""
    return POSX + np.linspace(-RX, RX, NCOL)


def make_course(seed: int = 0, difficulty: float = 1.0) -> np.ndarray:
    """Height profile along x, relative to the ground (m); shape [NCOL]. <0 is a pit.

    difficulty in (0, 1] scales step height / gap width. Segments: stairs up and down, boxes
    (step up, plateau, step down) and gaps, separated by flat runs.
    """
    rng = np.random.default_rng(seed)
    xs = grid_x()
    h = np.zeros(NCOL)
    d = float(np.clip(difficulty, 0.05, 1.0))

    def paint(lo, hi, val):
        h[(xs >= lo) & (xs < hi)] = val

    x = START_FLAT
    while x < FINISH_X - 2.5:
        kind = rng.choice(["stairs", "box", "gap"])
        if kind == "stairs":
            n = int(rng.integers(3, 5))
            rise = 0.06 + 0.08 * d
            tread = 0.32
            top = x + n * tread
            for i in range(n):  # up: each step paints through the plateau, later steps overwrite
                paint(x + i * tread, top + 0.6, rise * (i + 1))
            for i in range(n):  # down
                paint(top + 0.6 + i * tread, top + 0.6 + (i + 1) * tread, rise * (n - 1 - i))
            x = top + 0.6 + n * tread
        elif kind == "box":
            height = 0.08 + 0.12 * d
            length = rng.uniform(0.6, 1.0)
            paint(x, x + length, height)
            x += length
        else:
            width = 0.15 + 0.25 * d
            paint(x, x + width, -G)
            x += width
        x += rng.uniform(1.0, 1.6)
    return h


def make_elevation(seed: int = 0, difficulty: float = 1.0) -> np.ndarray:
    """Absolute ground height [NROW, NCOL] in metres (uniform across the lane)."""
    prof = G + make_course(seed, difficulty)
    return np.tile(prof[None, :], (NROW, 1))


def ground_height(elev: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
    """Bilinear ground height at world (x, y); any matching shapes. Clamped at the borders."""
    col = (x - POSX + RX) / (2 * RX) * (NCOL - 1)
    row = (y + RY) / (2 * RY) * (NROW - 1)
    return ndi.map_coordinates(elev, [row, col], order=1, mode="nearest")


def make_heightmap_fn(elev: np.ndarray):
    """qpos[..., :7] (base pos + wxyz quat) -> heightmap [..., 12, 8].

    Samples the ground ahead of the base in its yaw frame, as height relative to the base
    (0 on flat ground at the nominal standing height; positive = obstacle above the base line).
    """
    e = jnp.asarray(elev, jnp.float32)
    gx, gy = np.meshgrid(HM_X, HM_Y, indexing="ij")
    gx, gy = jnp.asarray(gx, jnp.float32), jnp.asarray(gy, jnp.float32)

    def fn(qpos):
        px, py, pz = qpos[..., 0], qpos[..., 1], qpos[..., 2]
        w, qx, qy, qz = qpos[..., 3], qpos[..., 4], qpos[..., 5], qpos[..., 6]
        yaw = jnp.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        c, s = jnp.cos(yaw)[..., None, None], jnp.sin(yaw)[..., None, None]
        wx = px[..., None, None] + c * gx - s * gy
        wy = py[..., None, None] + s * gx + c * gy
        return ground_height(e, wx, wy) - pz[..., None, None] + NOMINAL_BASE_H

    return fn


def spawn_positions(seed: int = 0, difficulty: float = 1.0) -> np.ndarray:
    """x positions on the course that are flat within +-0.8 m (safe places to spawn for the goal task)."""
    prof, xs = make_course(seed, difficulty), grid_x()
    cand = np.arange(-0.5, FINISH_X - 4.0, 0.25)
    ok = [x for x in cand if np.ptp(prof[(xs >= x - 0.8) & (xs <= x + 0.8)]) < 1e-6]
    return np.asarray(ok, np.float32)


# ---- environment ---------------------------------------------------------------------------
def build_env(seed: int = 0, difficulty: float = 1.0, unclipped_reward: bool = False,
              alive_bonus: float = 0.0, goal_task: bool = False, goal_rate_weight: float = -0.01):
    """Returns (unwrapped Playground env, elevation array)."""
    import mujoco
    from mujoco import mjx
    from mujoco_playground._src import mjx_env
    from mujoco_playground._src.locomotion.go1 import base as go1_base
    from mujoco_playground._src.locomotion.go1 import joystick as go1_joystick

    elev = make_elevation(seed, difficulty).astype(np.float32)
    elev_j = jnp.asarray(elev)
    spawn_xs = jnp.asarray(spawn_positions(seed, difficulty))
    ground_fn = lambda x, y: ground_height(elev_j, x, y)  # noqa: E731
    base = go1_joystick.Joystick
    if unclipped_reward or alive_bonus:
        from jpw.go1_reward import unclipped

        base = unclipped(base, alive_bonus)

    class ParkourJoystick(base):
        """Joystick task with forward-only commands, a spawn in the start zone and course logic."""

        def __init__(self):
            xml = _SCENE_XML.format(nrow=NROW, ncol=NCOL, rx=RX, ry=RY, zmax=ZMAX, posx=POSX)
            path = Path(tempfile.gettempdir()) / f"jpw_parkour_{NROW}x{NCOL}.xml"
            path.write_text(xml)
            go1_base.Go1Env.__init__(self, xml_path=path.as_posix(),
                                     config=go1_joystick.default_config())
            hf = self._mj_model.hfield_adr[0]
            self._mj_model.hfield_data[hf:hf + NROW * NCOL] = ((elev) / ZMAX).reshape(-1)
            self._mjx_model = mjx.put_model(self._mj_model)
            self._post_init()
            if goal_task:
                from jpw.goal_task import GoalReward

                self._goal_reward = GoalReward.from_env(
                    self, ground_fn, pit_z=PIT_Z, lane_half=LANE_HALF, weights={"rate": goal_rate_weight})

        def sample_command(self, rng, x_k):
            if goal_task:  # the goal (not a velocity command) drives the policy
                return jnp.zeros(3)
            vx = jax.random.uniform(rng, (), minval=0.4, maxval=1.0)
            return jnp.array([vx, 0.0, 0.0])

        def reset(self, rng):
            qpos = self._init_q.at[2].add(G)
            qvel = jnp.zeros(self.mjx_model.nv)
            rng, k1, k2, k3 = jax.random.split(rng, 4)
            qpos = qpos.at[0:2].add(jax.random.uniform(k1, (2,), minval=-0.5, maxval=0.5))
            if goal_task:  # spawn on a flat patch of the course, goal 2.5-3.5 m ahead
                rng, k_sp, k_gd = jax.random.split(rng, 3)
                sx = jax.random.choice(k_sp, spawn_xs)
                sy = jax.random.uniform(k1, (), minval=-0.5, maxval=0.5)
                qpos = qpos.at[0].set(sx).at[1].set(sy).at[2].set(0.35 + ground_fn(sx, sy))
                gx = jnp.minimum(sx + jax.random.uniform(k_gd, (), minval=2.5, maxval=3.5), FINISH_X - 0.3)
            yaw = jax.random.uniform(k2, (), minval=-0.15, maxval=0.15)
            qpos = qpos.at[3:7].set(jnp.array([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)]))
            qvel = qvel.at[0:6].set(jax.random.uniform(k3, (6,), minval=-0.5, maxval=0.5))
            data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

            rng, key, key_t = jax.random.split(rng, 3)
            cfg = self._config
            info = {
                "rng": rng,
                "command": self.sample_command(key, jnp.zeros(3)),
                "steps_until_next_cmd": jnp.round(
                    jax.random.exponential(key_t) * 5.0 / self.dt).astype(jnp.int32),
                "last_act": jnp.zeros(self.mjx_model.nu),
                "last_last_act": jnp.zeros(self.mjx_model.nu),
                "feet_air_time": jnp.zeros(4),
                "last_contact": jnp.zeros(4, dtype=bool),
                "swing_peak": jnp.zeros(4),
                "steps_until_next_pert": jnp.zeros((), jnp.int32),
                "pert_duration_seconds": jnp.zeros(()),
                "pert_duration": jnp.zeros((), jnp.int32),
                "steps_since_last_pert": 0,
                "pert_steps": 0,
                "pert_dir": jnp.zeros(3),
                "pert_mag": jnp.zeros(()),
            }
            if goal_task:
                info["command"] = jnp.zeros(3)
                info["goal"] = jnp.stack([gx, qpos[1], jnp.zeros(())])
                info["goal_dist"] = jnp.abs(gx - qpos[0])
            metrics = {f"reward/{k}": jnp.zeros(()) for k in cfg.reward_config.scales.keys()}
            metrics["swing_peak"] = jnp.zeros(())
            obs = self._get_obs(data, info)
            return mjx_env.State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

        def _goal_step(self, state, action):
            """Position-goal step: the AME2 reward on the exact state, goal-driven termination."""
            from jpw import go1_state as gs
            from jpw import goal_task as gt

            feet = self._feet_site_id
            a_prev, goal = state.info["last_act"], state.info["goal"]
            steps = state.info.get("steps", jnp.zeros(()))
            phys0 = gs.phys_targets(gs.obs_vector(state.obs), gs.aux_from_data(state.data, feet))
            out = super().step(state, action)
            phys1 = gs.phys_targets(gs.obs_vector(out.obs), gs.aux_from_data(out.data, feet))
            q = out.data.qpos
            pos, yaw = q[:3], gt.yaw_from_quat(q[3:7])
            t_left = gt.EPISODE_S - (steps + 1) * self.dt
            contact = out.info["last_contact"].astype(jnp.float32)
            r, _, _ = self._goal_reward.reward(
                gs.reward_inputs(phys0), gs.reward_inputs(phys1), pos, yaw, goal[:2], goal[2], t_left,
                action, a_prev, contact)
            g_z = gs.split(phys1)["gravity"][2]
            hb = pos[2] - ground_fn(pos[0], pos[1])
            bad = ((-g_z < jnp.cos(self._goal_reward.term_angle)) | (hb < self._goal_reward.term_height)
                   | (pos[2] < PIT_Z) | (jnp.abs(pos[1]) > LANE_HALF))
            done = jnp.maximum(out.done, bad.astype(out.done.dtype))
            out.info["goal_dist"] = jnp.sqrt(jnp.sum((goal[:2] - pos[:2]) ** 2) + 1e-8)
            return out.replace(reward=r, done=done)

        def step(self, state, action):
            if goal_task:
                return self._goal_step(state, action)
            x_prev = state.data.qpos[0]
            out = super().step(state, action)
            q = out.data.qpos
            fell = q[2] < PIT_Z
            off_lane = jnp.abs(q[1]) > LANE_HALF
            finished = q[0] > FINISH_X
            bad = fell | off_lane
            reward = out.reward + PROGRESS * (q[0] - x_prev) - FALL_PENALTY * bad
            done = jnp.maximum(out.done, (bad | finished).astype(out.done.dtype))
            return out.replace(reward=reward, done=done)

    return ParkourJoystick(), elev
