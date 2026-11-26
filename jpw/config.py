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
    # Feed the env's heightmap / depth observation (obs["heightmap"], [H, W]) to the models.
    use_heightmap: bool = False
    # Go1Parkour only: obstacle course seed and difficulty in (0, 1] (step height, gap width).
    terrain_seed: int = 0
    terrain_difficulty: float = 1.0
    # Go1 envs: per-env dynamics randomisation (friction, masses, COM, joint friction/armature,
    # nominal pose, motor strength), and Gaussian heightmap noise std in metres (0 = off).
    domain_randomization: bool = False
    hmap_noise: float = 0.0
    # Go1 envs: use the unclipped reward (Playground clips sum(terms) at 0, which zeroes the reward on
    # ~99.7% of steps for an untrained policy) plus a per-step alive bonus (see jpw.go1_reward).
    unclipped_reward: bool = False
    alive_bonus: float = 0.0


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
    contact_coef: float = 1.0      # per-foot contact prediction (needs env info["contact"])
    contact_pos_weight: float = 1.0    # upweight "in contact" flags in the BCE
    contact_event_weight: float = 4.0  # extra weight on touchdown / liftoff transitions
    student_coef: float = 1.0      # deployable-encoder distillation (only if ac.asymmetric)
    # Decoder head from the latent to the exact physical state (Go1, env.use_privileged=true); trained
    # on the encoded states and on the open-loop predicted latents (jpw.go1_state).
    decode_state: bool = False
    phys_coef: float = 1.0
    lr: float = 3e-4
    batch: int = 512
    grad_clip: float = 1.0
    cnn_channels: int = 32         # heightmap CNN width (first conv; second is 2x)


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
    # > 0: constant action noise std instead of a learned one (SuperMimic uses 0.1). Set ent_coef=0 with it.
    fixed_std: float = 0.0
    # Stabilisers used by SuperMimic / TGI. ema > 0: the deployed (evaluated) actor is an exponential moving
    # average of the trained one with this decay (0.995 in the papers); data collection uses the raw actor.
    # grad_clip_value > 0: element-wise gradient clip after the global-norm clip (0.1 in SuperMimic).
    ema: float = 0.0
    grad_clip_value: float = 0.0
    beta2: float = 0.999           # Adam beta2 of the actor (0.95 in SuperMimic)
    # Imagined reward and termination computed by formula from the decoded physical state (the Go1
    # Playground terms, unclipped) instead of the learned reward / continue heads. Needs wm.decode_state.
    analytic_reward: bool = False
    grad_clip: float = 1.0
    # Deployable actor: acts on a student encoder of proprioception (+ heightmap) only, trained to
    # regress the privileged world-model latent. Imagination still uses the privileged latent.
    asymmetric: bool = False


@dataclass
class TrainCfg:
    seed: int = 0
    # Cosine learning-rate decay to 0.1x over this many iterations for the world model, actor and critic
    # (TGI: 100k). 0 = constant learning rate.
    lr_decay_iters: int = 0
    # GPUs for data parallelism (see jpw.parallel). 1 = single device, 0 = all visible devices.
    # Batch sizes (wm.batch, ac.batch) and env.num_envs are global and must divide evenly.
    num_devices: int = 1
    iters: int = 20000
    collect_steps: int = 24        # env steps per env per iteration
    wm_updates: int = 8            # world-model gradient steps per iteration
    ac_updates: int = 8            # actor-critic gradient steps per iteration
    replay_capacity: int = 2000    # time steps kept per env
    warmup_iters: int = 10         # collect-only iterations before updates start
    log_every: int = 10
    ckpt_every: int = 1000
    resume: str = ""               # checkpoint to continue from (full train state)
    init_data: str = ""            # exported rollout dir to preload into the replay buffer
    pretrain_updates: int = 0      # world-model-only iterations on init_data before the loop
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

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(**{k: getattr(cls(), k).__class__(**v) for k, v in d.items()})


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
    # Dummy env with every new piece on: heightmap, privileged WM input, contact head, student actor.
    "dummy_full": [
        "env.name=dummy", "env.num_envs=64", "env.episode_length=200",
        "env.use_heightmap=true", "env.use_privileged=true", "ac.asymmetric=true",
        "wm.latent_dim=64", "wm.hidden=128", "wm.batch=64", "wm.cnn_channels=8",
        "ac.hidden=128", "ac.batch=128", "ac.horizon=16",
        "train.iters=60", "train.collect_steps=16", "train.wm_updates=2",
        "train.ac_updates=2", "train.replay_capacity=256", "train.warmup_iters=2",
        "train.log_every=5", "train.ckpt_every=0",
    ],
    # Go1 parkour with everything on: heightmap, privileged WM input, contact head, student actor.
    "parkour_smoke": [
        "env.name=Go1Parkour", "env.num_envs=256", "env.use_heightmap=true",
        "env.use_privileged=true", "ac.asymmetric=true",
        "env.domain_randomization=true", "env.hmap_noise=0.01",
        "train.iters=50", "train.replay_capacity=400", "train.warmup_iters=5",
        "train.log_every=5", "train.ckpt_every=0",
    ],
    "parkour": [
        "env.name=Go1Parkour", "env.num_envs=2048", "env.use_heightmap=true",
        "env.use_privileged=true", "ac.asymmetric=true",
        "env.domain_randomization=true", "env.hmap_noise=0.01",
    ],
    # Multi-GPU parkour: run with e.g.  sbatch --gres=gpu:4 --cpus-per-task=32 --mem=256G
    # slurm/train.sbatch --preset parkour_mgpu   (uses every allocated GPU; 2048 envs and
    # 512 / 1024 wm / ac batch per GPU)
    "parkour_mgpu": [
        "env.name=Go1Parkour", "env.use_heightmap=true", "env.use_privileged=true",
        "ac.asymmetric=true", "env.domain_randomization=true", "env.hmap_noise=0.01",
        "train.num_devices=0", "env.num_envs=8192", "wm.batch=2048", "ac.batch=4096",
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
