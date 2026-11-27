"""Success-guided terrain sampling (TGI, after Zhang et al., arXiv 2610.12465), host side.

Each configuration (terrain type x difficulty level = one tile of `jpw.terrain_grid`) keeps an exponential moving
average of its episode successes p_i (window H). Within a terrain type the next tile's level is drawn with

    w_i = max((p_i + eps)^(kappa t) (1 - p_i + eps)^(kappa (1 - t)), eps),   P(i) = softmax(log(w_i + eps) / T)

so levels whose success is near the target t are favoured (not too easy, not impossible); a level's success counts as
at least `neighbor` x that of the level one step easier (TGI's lower bound, so unvisited harder levels open up as easier
ones are solved). Terrain types keep their prior shares. Cell index = level * num_types + type.

The trainer feeds finished episodes (cell, success) and writes the resulting weights into the environments
(`info["cur_w"]`, read by `jpw.tgi_env.FreshAutoReset` at every respawn).
"""
from __future__ import annotations

import numpy as np


class SuccessGuidedSampler:
    def __init__(self, num_levels: int, type_names: tuple, prior: np.ndarray, window: int = 16, target: float = 0.8,
                 kappa: float = 5.0, temperature: float = 2.0, eps: float = 1e-8, neighbor: float = 0.25):
        self.L, self.T, self.names = num_levels, len(type_names), tuple(type_names)
        self.prior = np.asarray(prior, np.float64) / np.sum(prior)
        self.alpha, self.target, self.kappa = 1.0 / window, target, kappa
        self.temp, self.eps, self.neighbor = max(temperature, 1.0), eps, neighbor
        self.p = np.zeros(num_levels * self.T)       # success EMA per cell
        self.n = np.zeros(num_levels * self.T)       # episodes seen per cell

    @property
    def num_cells(self) -> int:
        return self.L * self.T

    def update(self, cells: np.ndarray, success: np.ndarray) -> None:
        """cells: int array of finished episodes' tiles; success: bool/float array, same length."""
        if len(cells) == 0:
            return
        k = self.num_cells
        cnt = np.bincount(cells, minlength=k).astype(np.float64)
        suc = np.bincount(cells, weights=np.asarray(success, np.float64), minlength=k)
        seen = cnt > 0
        keep = (1.0 - self.alpha) ** cnt
        mean = np.where(seen, suc / np.maximum(cnt, 1), 0.0)
        first = seen & (self.n == 0)
        self.p = np.where(first, mean, np.where(seen, keep * self.p + (1.0 - keep) * mean, self.p))
        self.n += cnt

    def effective(self) -> np.ndarray:
        """[L, T] success with the neighbour lower bound (a level is at least `neighbor` x the easier one)."""
        p = self.p.reshape(self.L, self.T).copy()
        for lvl in range(1, self.L):
            p[lvl] = np.maximum(p[lvl], self.neighbor * p[lvl - 1])
        return p

    def weights(self) -> np.ndarray:
        """[num_cells] sampling probabilities (sum to 1)."""
        p = self.effective()
        e, t, kp = self.eps, self.target, self.kappa
        w = np.maximum((p + e) ** (kp * t) * (1.0 - p + e) ** (kp * (1.0 - t)), e)
        logits = np.log(w + e) / self.temp
        logits -= logits.max(0, keepdims=True)
        sm = np.exp(logits)
        sm /= sm.sum(0, keepdims=True)                 # softmax over levels, per type
        out = sm * self.prior[None, :]
        return (out / out.sum()).reshape(-1)

    def log_dict(self) -> dict:
        """Per-type mean success and mean sampled level, plus the overall mean sampled level."""
        p, w = self.effective(), self.weights().reshape(self.L, self.T)
        lv = np.arange(self.L)[:, None]
        d = {"cur/mean_level": float((w * lv).sum())}
        for j, name in enumerate(self.names):
            share = w[:, j] / max(w[:, j].sum(), 1e-12)
            d[f"cur/success_{name}"] = float((share * p[:, j]).sum())
            d[f"cur/level_{name}"] = float((share * np.arange(self.L)).sum())
        return d
