# jepa-parkour-wm

Train a **JEPA-style (decoder-free) world model** of a legged robot from simulator data, then train a
**locomotion / parkour policy inside it** with imagined rollouts. The stack is JAX + MuJoCo MJX
(MuJoCo Playground). It runs on AMD MI300X under ROCm with no container, uses Slurm for jobs and
Weights & Biases for tracking.

## Method

| Piece | What it does | Lineage |
| --- | --- | --- |
| Encoder `o_t → z_t` | MLP + LayerNorm, trained against an **EMA target encoder** | V-JEPA, TD-MPC2 |
| Predictor `(z_t, a_t) → z_{t+1}` | Residual latent dynamics, rolled out open-loop for `K` steps | JEPA multi-step prediction |
| Anti-collapse | VICReg variance + covariance terms on `z` | VICReg / JEPA |
| Heads | reward (symlog) and continue, trained on *predicted* latents | Dreamer |
| Policy learning | Actor backpropagates λ-returns through `H`-step imagined rollouts of the frozen model; the critic regresses the returns | Dreamer, SHAC, SuperMimic |
| Data | Rollouts in MJX are added to an on-device replay buffer, and the model is refit every iteration | MBPO, SuperMimic |

The code is in `jpw/`: `models.py` (networks), `agent.py` (losses and updates), `replay.py`,
`envs.py` and `train.py` (loop).

The default env is Playground's `Go1JoystickRoughTerrain`. Parkour terrain (gaps, boxes, stairs as
heightfields) is the next step and will plug in through `jpw/envs.py`.

**Deployment caveat:** with `env.use_privileged=true`, the world model, and therefore the actor, sees
privileged sim state. For hardware you'll want an asymmetric setup: a privileged world model and an
actor that uses only proprioception and depth.

## One-time setup

1. On your machine, make sure `ssh iisc_amd` works, then run `cp cluster.env.example cluster.env` and
   fill in the partition, account, modules and the ROCm `jax` extra.
2. Build the bare-metal virtualenv on the cluster with `scripts/remote.sh setup`.
3. Log in to W&B once on the cluster: `ssh iisc_amd`, then `source ~/venvs/jpw/bin/activate && wandb login`.

## Running

```bash
scripts/remote.sh submit --preset dummy      # whole loop on a point-mass env (no MuJoCo)
scripts/remote.sh submit --preset smoke      # 256 MJX envs, 50 iterations
scripts/remote.sh submit --set env.num_envs=4096 train.iters=20000 train.wandb_group=go1-rough
scripts/remote.sh status | logs [JOB_ID] | cancel JOB_ID
scripts/remote.sh wandb-sync                 # if compute nodes ran W&B offline
```

Any config field can be overridden with `--set section.key=value` (see `jpw/config.py`).

Key W&B metrics:

- `wm/jepa`, `wm/pred_err_k1` and `wm/pred_err_kK`: latent prediction error.
- `wm/latent_std`: should stay well above 0, otherwise the latents are collapsing.
- `ac/imag_return`.
- `env/ep_return` and `env/reward_per_step`: the policy's real simulator performance.
- `perf/env_sps`.
