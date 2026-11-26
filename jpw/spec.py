"""Shapes that describe what an env produces; shared by the replay buffer, models and checkpoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Spec:
    obs_dim: int                          # world-model input vector (proprio, then privileged if enabled)
    prop_dim: int                         # leading slice of obs the deployed actor may use
    act_dim: int
    hmap_shape: tuple[int, int] | None = None   # (H, W) heightmap / depth image, or None
    contact_dim: int = 0                  # number of contact flags (e.g. feet), 0 = none
    aux_dim: int = 0                      # extra physical state kept in replay (Go1: base pose 7 + feet pos 12)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Spec":
        d = dict(d)
        if d.get("hmap_shape") is not None:
            d["hmap_shape"] = tuple(d["hmap_shape"])
        return cls(**d)
