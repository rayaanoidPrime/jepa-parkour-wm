"""Success-guided sampler behaviour (numpy only)."""
import numpy as np

from jpw.curriculum import SuccessGuidedSampler


def _sampler(**kw):
    return SuccessGuidedSampler(6, ("a", "b"), np.array([0.7, 0.3]), **kw)


def test_weights_are_a_distribution_and_start_uniform_within_a_type():
    s = _sampler()
    w = s.weights().reshape(6, 2)
    assert abs(w.sum() - 1) < 1e-9
    np.testing.assert_allclose(w[:, 0], w[0, 0])                       # unvisited: uniform over levels
    np.testing.assert_allclose(w.sum(0), [0.7, 0.3], atol=1e-9)        # type shares follow the prior


def test_sampling_moves_to_levels_near_the_target_success():
    s = _sampler(target=0.8, window=4)
    cells = np.repeat(np.arange(12), 40)
    # type a: levels 0-2 always solved, 3 half, 4-5 never; type b: nothing solved
    succ = np.zeros(12 * 40)
    for lvl in range(6):
        p = [1.0, 1.0, 0.95, 0.5, 0.0, 0.0][lvl]
        succ[(cells == lvl * 2)] = (np.random.default_rng(lvl).uniform(size=40) < p)
    s.update(cells, succ)
    w = s.weights().reshape(6, 2)[:, 0]
    assert w[2] > w[0] and w[2] > w[5]   # near-target levels beat trivially easy and impossible ones
    assert w[3] > w[5]


def test_neighbour_bound_opens_harder_levels_once_easier_ones_are_solved():
    s = _sampler(neighbor=0.25)
    s.update(np.array([0] * 30), np.ones(30))          # level 0 of type a solved
    p = s.effective()[:, 0]
    assert p[0] > 0.9 and p[1] >= 0.25 * p[0] - 1e-9 and p[2] >= 0.25 * p[1] - 1e-9
    assert s.p[2] == 0.0                                # ...without any data for level 1


def test_ema_update_and_logging():
    s = _sampler(window=2)
    s.update(np.array([1, 1, 3]), np.array([1, 0, 1]))
    assert s.n[1] == 2 and s.n[3] == 1 and 0.0 < s.p[1] < 1.0 and s.p[3] == 1.0
    d = s.log_dict()
    assert "cur/mean_level" in d and "cur/success_a" in d and all(np.isfinite(list(d.values())))
