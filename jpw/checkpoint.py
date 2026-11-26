"""Full-state checkpoints (params, EMA targets, optimiser states) that can be resumed or loaded
for evaluation / export. The replay buffer is not saved."""
from __future__ import annotations

import os
import pickle

import jax
from flax import serialization

from jpw.agent import Agent, TrainState
from jpw.config import Config
from jpw.spec import Spec

VERSION = 2


def save(path: str, cfg: Config, spec: Spec, ts: TrainState, it: int, env_steps: int) -> None:
    blob = {
        "version": VERSION, "cfg": cfg.to_dict(), "spec": spec.to_dict(),
        "it": int(it), "env_steps": int(env_steps),
        "ts": serialization.to_state_dict(jax.device_get(ts)),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(blob, f)
    os.replace(tmp, path)


def load(path: str) -> tuple[Config, Spec, Agent, TrainState, dict]:
    """Rebuild (cfg, spec, agent, train_state, meta) from a checkpoint written by `save`."""
    with open(path, "rb") as f:
        blob = pickle.load(f)
    if blob.get("version") != VERSION:
        raise ValueError(f"{path}: unsupported checkpoint version {blob.get('version')!r} (want {VERSION})")
    cfg = Config.from_dict(blob["cfg"])
    spec = Spec.from_dict(blob["spec"])
    agent = Agent(cfg, spec)
    template = agent.init(jax.random.PRNGKey(0))
    state = dict(blob["ts"])
    state.setdefault("actor_ema", state["actor"])  # checkpoints from before the EMA actor
    ts = serialization.from_state_dict(template, state)
    return cfg, spec, agent, ts, {"it": blob["it"], "env_steps": blob["env_steps"]}
