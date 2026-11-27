"""TGI-style terrain grid for Go1: 8 m tiles, rows = difficulty levels, columns = terrain types.

Ported from Terrain-Grounded Imagination (ETH, leggedrobotics/terrain-grounded-imagination): the `fpm` training
terrains (pyramid stairs up / down, random boxes, rough ground, pyramid slopes, flat, stepping stones), AME2's
"Stones-Everywhere" stones, and the ANYmal parkour families (climb-up, climb-down, gap). Every tile is a radial
terrain around a 2 m start platform at its centre, so a robot spawned at the centre can be asked to reach a goal
2.5-3.5 m away in any direction, exactly as in TGI.

Go1 is much smaller than ANYmal-D / G1, so lengths and heights are scaled (HEIGHT_SCALE, LENGTH_SCALE below; the TGI
values are kept in the comments). All heights are metres relative to the nominal ground z = 0.

The whole grid is one MuJoCo heightfield (shared by all envs); `TerrainField` samples the same array in JAX
(differentiable bilinear lookups), which is what the heightmap observation, the reward and the imagination use.
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.ndimage as ndi
import numpy as np

TILE = 8.0           # tile edge (m), as in TGI
RES = 0.05           # grid resolution (m); TGI hf terrains use 0.1
NS = int(round(TILE / RES))
NUM_LEVELS = 10      # difficulty levels (rows), as in TGI
PLATFORM = 2.0       # start platform width (m)
RADIUS = 3.7         # structures extend to this Chebyshev distance from the tile centre
HEIGHT_SCALE = 0.55  # Go1 vs the G1 / ANYmal stair, box and slope heights
LENGTH_SCALE = 0.65  # Go1 vs ANYmal stone sizes and gaps

TYPES = ("stairs_down", "stairs_up", "boxes", "rough", "slope_down", "slope_up", "flat", "stones",
         "climb_up", "climb_down", "gap")
# Share of each type before success-guided reattribution (TGI: stairs .15/.15, boxes .15, rough .05, slopes .075/.05,
# flat .025, stones .3, pit .05, + the parkour families).
PRIOR = {"stairs_down": 0.10, "stairs_up": 0.10, "boxes": 0.10, "rough": 0.05, "slope_down": 0.05, "slope_up": 0.05,
         "flat": 0.025, "stones": 0.25, "climb_up": 0.10, "climb_down": 0.10, "gap": 0.075}


def _lerp(lo, hi, u):
    return lo + u * (hi - lo)


def _coords():
    ax = (np.arange(NS) + 0.5) * RES - TILE / 2
    X, Y = np.meshgrid(ax, ax, indexing="xy")  # X varies along columns, Y along rows
    return X, Y, np.maximum(np.abs(X), np.abs(Y))


# ------------------------------------------------------------------------------ stepping stones (AME2 / TGI)
STONES = {"size": (0.92, 0.20), "gap": (0.08, 0.26), "shift": (0.036, 0.118), "height": (0.01, 0.10)}


def stones_everywhere_rects(level: int, size: tuple, rng: np.random.Generator, num_levels: int = NUM_LEVELS,
                            platform_width: float = PLATFORM, length_scale: float = LENGTH_SCALE,
                            height_scale: float = HEIGHT_SCALE):
    """TGI's `stones_everywhere_rects` (tgi_anymal/env/terrains.py) with Go1-scaled stone size / gap / shift / height.

    Returns (stone rectangles (x0, x1, y0, y1, top_z), platform half width). Every stone is complete: size, gap and
    shift are scaled together so a whole number of cells fills the tile, and the max shift stays below half the gap.
    """
    u = min(max(level, 0), num_levels - 1) / (num_levels - 1)
    P = {k: _lerp(v[0], v[1], u) for k, v in STONES.items()}
    for k in ("size", "gap", "shift"):
        P[k] *= length_scale
    P["height"] *= height_scale
    m = max(1, int(round(size[0] / (P["size"] + P["gap"]))))
    sc = size[0] / m / (P["size"] + P["gap"])
    w, g, s, h = P["size"] * sc, P["gap"] * sc, P["shift"] * sc, P["height"]
    s = min(s, g / 2 - 1e-3)
    pitch = size[0] / m
    cands = [n for n in range(1, m + 1) if n % 2 == m % 2]
    nb = min(cands, key=lambda n: abs(n * pitch - g - platform_width))
    rects = []
    for i in range(m):
        for j in range(m):
            if abs(i + 0.5 - m / 2) <= (nb - 1) / 2 + 1e-6 and abs(j + 0.5 - m / 2) <= (nb - 1) / 2 + 1e-6:
                continue
            x = (i + 0.5) * pitch + rng.uniform(-s, s)
            y = (j + 0.5) * pitch + rng.uniform(-s, s)
            rects.append((x - w / 2, x + w / 2, y - w / 2, y + w / 2, rng.uniform(-h, h)))
    return rects, nb * pitch / 2 - (pitch - w) / 2


# ------------------------------------------------------------------------------ tile generators
def _stairs(u, down: bool):
    X, Y, D = _coords()
    hs, w = _lerp(0.10, 0.25, u) * HEIGHT_SCALE, _lerp(0.25, 0.30, u)  # TGI: step height 0.1-0.25, width 0.25-0.3
    k = np.floor(np.clip(D - PLATFORM / 2, 0, None) / w)
    n = np.floor((RADIUS - PLATFORM / 2) / w)
    k = np.minimum(k, n)
    return hs * (n - k) if down else hs * k


def _slope(u, down: bool):
    X, Y, D = _coords()
    s = u * 0.44  # TGI slope 0 - 0.55 (rise / run); Go1 0 - 0.44
    run = np.clip(np.minimum(D, RADIUS) - PLATFORM / 2, 0, None)
    return s * (RADIUS - PLATFORM / 2 - run) if down else s * run


def _boxes(u, rng):
    X, Y, D = _coords()
    g = max(1, int(round(0.45 * LENGTH_SCALE / RES)))             # TGI grid width 0.45 m
    n = int(np.ceil(NS / g))
    hmax = _lerp(0.05, 0.30, u) * HEIGHT_SCALE                       # TGI grid height 0.05-0.3
    cells = rng.uniform(0.0, hmax, (n, n))
    h = np.kron(cells, np.ones((g, g)))[:NS, :NS]
    return np.where(D < PLATFORM / 2, 0.0, h)


def _rough(u, rng):
    X, Y, D = _coords()
    nz = _lerp(0.02, 0.10, u) * HEIGHT_SCALE                        # TGI noise 0.02-0.10, step 0.02
    c = int(round(0.1 / RES))
    m = int(np.ceil(NS / c))
    cells = np.round(rng.uniform(-nz, nz, (m, m)) / 0.01) * 0.01
    return np.kron(cells, np.ones((c, c)))[:NS, :NS]


def _stones(u, rng):
    level = int(round(u * (NUM_LEVELS - 1)))
    rects, a = stones_everywhere_rects(level, (TILE, TILE), rng)
    floor_z = -rng.uniform(0.8, 1.5)  # TGI: -1.5 .. -0.35; kept deep so a fall is unrecoverable (env terminates below z = -0.5)
    h = np.full((NS, NS), floor_z)
    ax = (np.arange(NS) + 0.5) * RES
    X, Y = np.meshgrid(ax, ax, indexing="xy")
    c = TILE / 2
    h[(abs(X - c) < a) & (abs(Y - c) < a)] = 0.0
    for x0, x1, y0, y1, top in rects:
        h[(X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)] = top
    return h


def _climb(u, up: bool):
    X, Y, D = _coords()
    depth = _lerp(0.10, 1.00, u) * 0.36                              # TGI climb 0.1-1.0 m (ANYmal); Go1 0.04-0.36
    half = (3.0 if up else 2.0) / 2                                  # TGI platform 3 m (pit) / 2 m (box)
    inside = D < half
    return np.where(inside, -depth, 0.0) if up else np.where(inside, depth, 0.0)


def _gap(u):
    X, Y, D = _coords()
    g = _lerp(0.10, 1.00, u) * 0.35                                  # TGI gap 0.1-1.0 m; Go1 0.035-0.35
    p = PLATFORM / 2
    return np.where(D < p, 0.0, np.where(D < p + g, -1.5, 0.0))


def make_tile(kind: str, u: float, rng: np.random.Generator) -> np.ndarray:
    """[NS, NS] heights (m) of one tile at difficulty u in [0, 1]; row index = y, column index = x."""
    if kind == "stairs_down":
        return _stairs(u, True)
    if kind == "stairs_up":
        return _stairs(u, False)
    if kind == "boxes":
        return _boxes(u, rng)
    if kind == "rough":
        return _rough(u, rng)
    if kind == "slope_down":
        return _slope(u, True)
    if kind == "slope_up":
        return _slope(u, False)
    if kind == "flat":
        return np.zeros((NS, NS))
    if kind == "stones":
        return _stones(u, rng)
    if kind == "climb_up":
        return _climb(u, True)
    if kind == "climb_down":
        return _climb(u, False)
    if kind == "gap":
        return _gap(u)
    raise KeyError(kind)


class TerrainGrid:
    """num_levels rows (y) x len(types) columns (x) of tiles in one heightfield."""

    def __init__(self, seed: int = 0, num_levels: int = NUM_LEVELS, types: tuple = TYPES):
        rng = np.random.default_rng(seed)
        self.types, self.num_levels = tuple(types), num_levels
        self.nrow, self.ncol = num_levels * NS, len(self.types) * NS
        self.elev = np.zeros((self.nrow, self.ncol), np.float32)
        self.centre_z = np.zeros((num_levels, len(self.types)), np.float32)
        for r in range(num_levels):
            for c, kind in enumerate(self.types):
                tile = make_tile(kind, (r + 0.5) / num_levels, rng)
                self.elev[r * NS:(r + 1) * NS, c * NS:(c + 1) * NS] = tile
                self.centre_z[r, c] = tile[NS // 2, NS // 2]
        self.h_min, self.h_max = float(self.elev.min()), float(self.elev.max())

    @property
    def num_cells(self) -> int:
        return self.num_levels * len(self.types)

    def cell_centre(self, cell):
        """cell = level * num_types + type_index -> (x, y) of the tile centre (works on numpy / jax arrays)."""
        nt = len(self.types)
        return (cell % nt) * TILE + TILE / 2, (cell // nt) * TILE + TILE / 2

    def group(self) -> np.ndarray:
        """Terrain type index of every cell (cell = level * num_types + type)."""
        return np.tile(np.arange(len(self.types)), self.num_levels)

    def prior(self) -> np.ndarray:
        p = np.asarray([PRIOR[t] for t in self.types], np.float64)
        return p / p.sum()


class TerrainField:
    """JAX view of the grid: differentiable bilinear height lookups and the heightmap observation."""

    def __init__(self, grid: TerrainGrid):
        self.e = jnp.asarray(grid.elev, jnp.float32)
        self.nrow, self.ncol = grid.nrow, grid.ncol

    def height(self, x, y):
        return ndi.map_coordinates(self.e, [y / RES, x / RES], order=1, mode="nearest")

    def heightmap_pose_fn(self, step: float = 0.0):
        """(pos [..., 3], yaw [...]) -> heightmap [..., H, W] relative to the base. step = 0: the coarse 12 x 8 layout
        of jpw.parkour; step > 0: the same 1.4 x 1.0 m window sampled every `step` metres (0.05 -> 29 x 21)."""
        from jpw.parkour import HM_X, HM_Y, NOMINAL_BASE_H

        if step > 0:
            HM_X = np.arange(-0.2, 1.2 + 1e-6, step)
            HM_Y = np.arange(-0.5, 0.5 + 1e-6, step)
        gx, gy = np.meshgrid(HM_X, HM_Y, indexing="ij")
        gx, gy = jnp.asarray(gx, jnp.float32), jnp.asarray(gy, jnp.float32)

        def fn(pos, yaw):
            c, s = jnp.cos(yaw)[..., None, None], jnp.sin(yaw)[..., None, None]
            wx = pos[..., 0, None, None] + c * gx - s * gy
            wy = pos[..., 1, None, None] + s * gx + c * gy
            return self.height(wx, wy) - pos[..., 2, None, None] + NOMINAL_BASE_H

        return fn

    def heightmap_fn(self, step: float = 0.0):
        """qpos[..., :7] (base pos + wxyz quat) -> heightmap, as jpw.parkour.make_heightmap_fn."""
        from jpw.goal_task import yaw_from_quat

        pose = self.heightmap_pose_fn(step)
        return lambda qpos: pose(qpos[..., :3], yaw_from_quat(qpos[..., 3:7]))
