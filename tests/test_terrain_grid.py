"""Geometry of the TGI-style terrain grid (numpy / jax only)."""
import jax.numpy as jnp
import numpy as np
import pytest

from jpw import terrain_grid as tg


@pytest.fixture(scope="module")
def grid():
    return tg.TerrainGrid(seed=0)


def _centre(tile, half=0.9):
    c = tg.NS // 2
    k = int(half / tg.RES)
    return tile[c - k:c + k, c - k:c + k]


def test_grid_shape_ranges_and_tile_layout(grid):
    assert grid.elev.shape == (tg.NUM_LEVELS * tg.NS, len(tg.TYPES) * tg.NS)
    assert grid.num_cells == tg.NUM_LEVELS * len(tg.TYPES)
    assert -1.51 <= grid.h_min < -0.7 and 0.3 < grid.h_max < 2.0
    assert np.isfinite(grid.elev).all()
    x, y = grid.cell_centre(np.arange(grid.num_cells))
    assert x.min() == tg.TILE / 2 and y.max() == (tg.NUM_LEVELS - 1) * tg.TILE + tg.TILE / 2
    assert abs(grid.prior().sum() - 1) < 1e-9 and len(grid.group()) == grid.num_cells


@pytest.mark.parametrize("kind", ["stairs_down", "stairs_up", "slope_down", "slope_up", "flat", "boxes", "stones",
                                  "climb_up", "climb_down", "gap"])
def test_start_platform_is_flat_at_every_level(kind):
    rng = np.random.default_rng(1)
    for u in (0.05, 0.5, 0.95):
        c = _centre(tg.make_tile(kind, u, rng), half=0.9)  # platform is 2 m wide (half 1.0)
        assert np.ptp(c) < 1e-6, (kind, u, np.ptp(c))


def test_stairs_have_the_right_direction_and_grow_with_difficulty():
    rng = np.random.default_rng(0)
    for u in (0.1, 0.9):
        down, up = tg.make_tile("stairs_down", u, rng), tg.make_tile("stairs_up", u, rng)
        c = tg.NS // 2
        assert down[c, c] > down[c, -2] and up[c, c] < up[c, -2]  # down: centre is the top; up: the bottom
    step = lambda u: np.abs(np.diff(tg.make_tile("stairs_up", u, rng)[tg.NS // 2])).max()
    assert step(0.1) < step(0.9) and step(0.9) < 0.15  # Go1-scaled step height


def test_stones_have_floor_gaps_and_complete_stones():
    rng = np.random.default_rng(0)
    easy, hard = tg.make_tile("stones", 0.0, rng), tg.make_tile("stones", 1.0, rng)
    for t in (easy, hard):
        assert t.min() < -0.7 and t.max() <= 0.1 + 1e-9
    gap_frac = lambda t: (t < -0.7).mean()
    assert gap_frac(hard) > gap_frac(easy)  # harder = smaller stones, wider gaps


def test_climb_and_gap_families():
    rng = np.random.default_rng(0)
    c = tg.NS // 2
    assert tg.make_tile("climb_up", 0.9, rng)[c, c] < -0.2 and tg.make_tile("climb_up", 0.9, rng)[c, 2] == 0.0
    assert tg.make_tile("climb_down", 0.9, rng)[c, c] > 0.2 and tg.make_tile("climb_down", 0.9, rng)[c, 2] == 0.0
    g = tg.make_tile("gap", 0.9, rng)
    assert g[c, c] == 0.0 and g.min() == -1.5 and g[c, 2] == 0.0
    assert (g == -1.5).sum() < (tg.make_tile("gap", 0.99, rng) == -1.5).sum() + 1  # wider at higher difficulty


def test_terrain_field_matches_the_array_and_heightmap_is_flat_on_flat_ground(grid):
    f = tg.TerrainField(grid)
    ii, jj = 700, 900
    got = float(f.height(jnp.float32(jj * tg.RES), jnp.float32(ii * tg.RES)))
    assert abs(got - grid.elev[ii, jj]) < 1e-5
    flat_col = tg.TYPES.index("flat")
    cx, cy = grid.cell_centre(np.array([3 * len(tg.TYPES) + flat_col]))
    pos = jnp.asarray([[cx[0], cy[0], 0.27]], jnp.float32)
    hm = f.heightmap_pose_fn()(pos, jnp.zeros(1))
    assert hm.shape == (1, 12, 8) and float(jnp.abs(hm).max()) < 1e-5
    # the qpos-based version agrees
    q = jnp.concatenate([pos, jnp.asarray([[1.0, 0, 0, 0]])], -1)
    np.testing.assert_allclose(np.asarray(f.heightmap_fn()(q)), np.asarray(hm), atol=1e-6)
