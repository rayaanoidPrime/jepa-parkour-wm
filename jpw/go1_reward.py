"""Unclipped Go1 reward (needs mujoco_playground).

Playground's Go1 joystick task returns clip(sum(terms) * dt, 0, 1e4). With an untrained policy the
penalty terms dominate, so the clipped reward is exactly 0 on ~99.7% of steps (see
scripts/diag_reward.py): a world-model / analytic-gradient learner then sees no signal at all.
`unclipped` rebuilds the reward from the per-term values the env already stores in
state.metrics["reward/<term>"] (scaled, before dt) and optionally adds a per-step alive bonus so
that surviving is never worse than terminating (the unclipped sum is mostly negative early on).
"""
from __future__ import annotations


def unclipped(base, alive_bonus: float = 0.0):
    """Subclass of a Playground Go1 Joystick-style env with the clip removed."""

    class Unclipped(base):
        def step(self, state, action):
            out = super().step(state, action)
            terms = [v for k, v in out.metrics.items() if k.startswith("reward/")]
            reward = self.dt * sum(terms) + alive_bonus * (1.0 - out.done)
            return out.replace(reward=reward)

    Unclipped.__name__ = f"Unclipped{base.__name__}"
    return Unclipped
