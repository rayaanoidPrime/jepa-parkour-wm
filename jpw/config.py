"""Configuration dataclasses, presets and `--set a.b=value` overrides."""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field


@dataclass
class EnvCfg:
    # Any MuJoCo Playground env name, or "dummy" for a CPU-friendly point-mass sanity check.
    name: str = "Go1JoystickRoughTerrain"
    num_envs: int = 2048
    episode_length: int = 1000
    # Concatenate privileged sim state into the world-model input. Policies trained this way
    # cannot be deployed directly; see README ("Deployment caveat").
    use_privileged: bool = False


@dataclass
class WMCfg:
    latent_dim: int = 256
    hidden: int = 512
    depth: int = 3
    pred_horizon: int = 8          # K: open-loop latent prediction steps for the JEPA loss
    target_ema: float = 0.995      # EMA rate of the target encoder
    var_coef: float = 1.0          # VICReg variance term (anti-collapse)
    cov_coef: float = 0.04         # VICReg covariance term
    reward_coef: float = 1.0
    cont_coef: float = 1.0
    lr: float = 3e-4
    batch: int = 512
    grad_clip: float = 1.0


@dataclass
class ACCfg:
    hidden: int = 512
    depth: int = 3
    horizon: int = 32              # imagination horizon (SuperMimic uses 32)
    gamma: float = 0.99
    lam: float = 0.95
    ent_coef: float = 3e-4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch: int = 1024              # imagined start states per update
    critic_ema: float = 0.98
    min_std: float = 0.05
    max_std: float = 1.0
    grad_clip: float = 1.0


@dataclass
class TrainCfg:
    seed: int = 0
    iters: int = 20000
    collect_steps: int = 24        # env steps per env per iteration
    wm_updates: int = 8            # world-model gradient steps per iteration
    ac_updates: int = 8            # actor-critic gradient steps per iteration
    replay_capacity: int = 2000    # time steps kept per env
    warmup_iters: int = 10         # collect-only iterations before updates start
    log_every: int = 10
    ckpt_every: int = 1000
    out_dir: str = "runs"
    wandb_project: str = "jepa-parkour-wm"
    wandb_entity: str = ""
    wandb_mode: str = "online"     # online | offline | disabled
    wandb_group: str = ""
    wandb_tags: str = ""           # comma-separated


@dataclass
class Config:
    env: EnvCfg = field(default_factory=EnvCfg)
    wm: WMCfg = field(default_factory=WMCfg)
    ac: ACCfg = field(default_factory=ACCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, list[str]] = {
    # Seconds-long CPU/GPU check that the whole loop runs (no MuJoCo needed).
    "dummy": [
        "env.name=dummy", "env.num_envs=64", "env.episode_length=200",
        "wm.latent_dim=64", "wm.hidden=128", "wm.batch=64",
        "ac.hidden=128", "ac.batch=128", "ac.horizon=16",
        "train.iters=60", "train.collect_steps=16", "train.wm_updates=2",
        "train.ac_updates=2", "train.replay_capacity=256", "train.warmup_iters=2",
        "train.log_every=5", "train.ckpt_every=0",
    ],
    # Small real-simulator run to validate MJX + ROCm + W&B on one GPU.
    "smoke": [
        "env.num_envs=256", "train.iters=50", "train.replay_capacity=400",
        "train.warmup_iters=5", "train.log_every=5", "train.ckpt_every=0",
    ],
}


def _coerce(cur, raw: str):
    try:
        val = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        val = raw
    if isinstance(cur, bool):
        return val if isinstance(val, bool) else str(val).lower() in ("1", "true", "yes", "on")
    if isinstance(cur, float) and isinstance(val, int):
        return float(val)
    if isinstance(cur, str):
        return str(val)
    return val


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        if not hasattr(obj, parts[-1]):
            raise KeyError(f"Unknown config key: {key}")
        setattr(obj, parts[-1], _coerce(getattr(obj, parts[-1]), raw))
    return cfg


def build_config(preset: str | None, overrides: list[str]) -> Config:
    cfg = Config()
    if preset:
        if preset not in PRESETS:
            raise KeyError(f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}")
        apply_overrides(cfg, PRESETS[preset])
    return apply_overrides(cfg, overrides)
