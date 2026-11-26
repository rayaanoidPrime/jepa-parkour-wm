# Findings from the exploration phase

Everything learned before the large parkour run. Written at the end of the 6-hour autonomous exploration
window. Numbers come from short runs on one MI300X unless stated; most comparisons are 1-2 seeds, and single
evals are noisy (the same checkpoint varied by 10-45% in episode length), so read them as strong hints, not
proof. No large run has been launched with the new recipe.

## TL;DR

1. **Playground's Go1 reward is clipped at 0.** For an untrained policy it is exactly 0 on 99.7% of steps. A
   world-model learner (learned reward head / analytic gradients) then gets no signal. This is why the first
   100M-step parkour run learned nothing. Unclipped reward + a per-step alive bonus (0.05) was the biggest win.
2. **What works on flat Go1 so far** (flat-terrain survival + a positive task reward early in training):
   unclipped reward, alive bonus 0.05, fixed action noise 0.1 with no entropy bonus, decoder heads that decode
   the exact physical state from the JEPA latent with the reward and termination computed by formula, and
   stabilisers (EMA actor, cosine LR decay, element-wise gradient clip).
3. **Not solved:** the gait. Velocity tracking never improves (about 24% of its maximum) and the penalties grow;
   the model-gap diagnostic shows the imagined penalties are 3-8x too small, so the actor drifts to jerky,
   energetic motion.
4. **Parkour is not validated.** All of the above was tested on flat terrain. The grounded goal-reaching agent
   (TGI-ANYmal style) is only partly built (see "Status and what is left").
5. **Speed:** an iteration costs about 2.2 s at 2048 envs (collection about 1.25 s, update about 1 s). XLA Triton GEMM
   must be off on this ROCm stack (it produces NaNs) and that costs nothing. 2-GPU data parallel works.

## What was built (code map)

| piece | where |
|---|---|
| JEPA world model, actor, critic, heads | `jpw/models.py`, `jpw/agent.py` |
| Replay with heightmap / contact / aux (pose, feet) | `jpw/replay.py` |
| Unclipped Go1 reward + alive bonus | `jpw/go1_reward.py` (`env.unclipped_reward`, `env.alive_bonus`) |
| Decoder heads + analytic reward / termination | `jpw/go1_state.py`, `wm.decode_state`, `ac.analytic_reward` |
| Stabilisers (EMA actor, LR decay, element-wise clip, beta2) | `ac.ema`, `train.lr_decay_iters`, `ac.grad_clip_value`, `ac.beta2` |
| Fixed action noise | `ac.fixed_std` (set `ac.ent_coef=0` with it) |
| Go1 parkour env: procedural stairs / boxes / gaps hfield, heightmap obs | `jpw/parkour.py` |
| Goal-reaching task (AME2 reward, pose integration, goal features) | `jpw/goal_task.py`, `env.goal_task`, `env.episode_length=400` |
| Domain randomisation + heightmap noise | `jpw/randomize.py`, `env.domain_randomization`, `env.hmap_noise` |
| Multi-GPU data parallel | `jpw/parallel.py`, `train.num_devices` |
| Rollout export, full checkpoints, resume, eval | `jpw/export.py`, `jpw/checkpoint.py`, `jpw/eval.py` |
| Diagnostics | `scripts/diag_reward.py`, `diag_mjx.py`, `diag_std.py`, `diag_gap.py`, `bench.py` |
| Experiment jobs | `slurm/exp_*.sbatch`, `slurm/bench_*.sbatch`, `slurm/diag*.sbatch` |

## Findings, in order of importance

### 1. The reward clip (root cause of the failed 100M parkour run)
`diag_reward.py`: Playground returns `clip(sum(terms) * dt, 0, 1e4)`. On a random policy the clipped reward is 0
on 99.7% of steps (mean +0.00002); the unclipped reward has mean -0.091 and range -0.69..+0.02. The clip is
harmless for a standing robot (3.8% zeros), which is why PPO survives it. The first 100M-step parkour run
(old recipe) ended with eval episodes of 69 steps, 0.96 m progress, 0% success.

### 2. Alive bonus
Unclipped reward with no bonus: the policy learns to terminate (episode length 10). Bonus 0.05: length 849-935;
bonus 0.10: 298-552 (noisy). Use 0.05 as the starting point.

### 3. Fixed action noise
The learned std was already small (0.167 after 12k updates, 0.105 after 48k), so I predicted fixed noise would
change nothing. Wrong: fixed std 0.1 / 0.2 with `ent_coef=0` reached full 1000-step survival and the best task
reward of the learned-std family (about -0.004 vs best -0.0095), but unstable across checkpoints
(F0: 28 / 47 / 1000 steps, F1: 816 / 1000 / 29).

### 4. Decoder heads + analytic reward (same privileged observations in both arms, 2 seeds)
Eval episode length (of 1000) at 12k / 24k / 24.5k updates:

| arm | 12k | 24k | final |
|---|---|---|---|
| G seed 0 (decoder + analytic reward) | 1000 | 861 | 665 |
| G seed 1 | 951 | 798 | 966 |
| B seed 0 (learned reward / continue heads) | 21 | 68 | 991 |
| B seed 1 | 11 | 657 | 73 |

G survives far more reliably (mean 874 vs 304 steps; wins 5 of 6 pairings) and gave the first positive task
rewards (+0.017 and +0.006 per step at 12k updates; standing still is +0.013). Task reward then drifts to
-0.003..-0.010 by 24k updates. Decoder rollout error after 8 open-loop steps: linvel about 0.05 m/s, gravity about
0.02, joint pos about 0.05 rad; high-frequency quantities are poor even at step 1 (joint vel 1.6 rad/s, gyro 0.36,
actuator force 1.6).

### 5. Stabilisers (EMA 0.995, cosine LR decay to 0.1x, element-wise clip 0.1, Adam beta2 0.95), EMA actor evaluated
S seed 0: 1000 / 1000 / 967, S seed 1: 993 / 934 / 920 (G without them: min 665). Raw vs EMA at the final
checkpoint: 816 vs 967 and 877 vs 920. Task reward unchanged (about -0.007): they fix falling, not the gait.

### 6. Why the gait does not improve
- Per-term breakdown (S seed 0, 12k updates -> final): tracking_lin_vel +0.0048 -> +0.0047 (flat), energy -0.0010 ->
  -0.0065, feet_clearance -0.0048 -> -0.0079, orientation -0.0005 -> -0.0029, body roll/pitch rate -0.0005 -> -0.0027.
  The policy moves more without tracking better.
- `diag_gap.py` (imagined vs real reward of the same policy from the same start states): tracking terms agree
  (imagined +0.0035 vs real +0.0032), but the imagined penalties are far too small: energy -0.0012 vs -0.0068,
  ang_vel_xy -0.0004 vs -0.0030, feet_slip -0.0004 vs -0.0023, orientation -0.0010 vs -0.0028. Explanation: a
  deterministic model / decoder regresses to the mean on noisy high-frequency quantities, so squared penalties
  built on them are underestimated and the actor exploits that (hypothesis consistent with all data, not proven).
- Fix designed into `jpw/goal_task.py`: build penalties from smooth, well-predicted quantities (PD-law torque from
  decoded joint positions + the known action, joint velocity from finite differences of joint positions, roll/pitch
  rate from finite differences of the gravity direction).
- Also untested: action-rate weight (TGI-G1 uses -0.1, Playground -0.01), Muon.

### 7. Update vs data ratio (flat, clipped reward, so all weak; useful for speed only)
Same number of updates with 11x less simulator data (2 steps per iteration instead of 24) was no worse and 35%
faster. Updates cost about 60 ms per (world-model + actor) step; more updates per sample did not help by
themselves.

### 8. Speed and systems
- `bench.py`, 2048 envs, parkour: env step 55.8 ms (about 37k steps/s); 24-step collection 1255 ms; update (8 wm + 8
  actor steps) 970 ms. Triton GEMM on: update 1043 ms and NaN rewards on 1024 of 2048 envs; `highest` matmul
  precision: 913 ms. Keep Triton GEMM off (`--xla_gpu_enable_triton_gemm=false`, already in `slurm/*.sbatch`).
- `MUJOCO_GL=egl` segfaults on import here; use `disable`. XLA command buffers are disabled (AMD MJX guidance).
- 2-GPU data parallel works (replicas bit-identical). A 100M-step parkour run (old recipe, 2 GPUs, 4096 envs)
  took 1 h 10 min; throughput fell from about 31k to 12-17k steps/s late in the run (cause not diagnosed).
- Untested speed levers: Muon, bf16 / TF32 matmuls, enabling XLA command buffers for the learner, fusing an
  iteration into one program, overlapping collection with updates, 4+ GPUs.

### 9. What the reference work does (read in full: SuperMimic paper + code, BlinkMimic paper, TGI code)
- **SuperMimic / BlinkMimic:** physical-state dynamics (predict accelerations, integrate), reward and termination
  by formula, deterministic actor with action noise 0.1, no entropy bonus, no EMA critic, Muon (BlinkMimic ablation:
  without it the target is never reached), 4x policy step size decaying to 1x, element-wise grad clip 0.1, running
  normalisation of inputs, CUDA-graph captured updates, bf16, 10 world-model steps of 2048 windows per update. Task:
  dense tracking reward, reference-state resets, flat ground only. The paper's table says Adam but the released
  code defaults to Muon + Adam.
- **TGI (ETH):** terrain is not learned. The world model predicts the physical state; every terrain question is
  answered by a differentiable lookup in the exact heightmap at the predicted pose; reward, termination, critic and
  perception use the same lookups. Gradient cut at imagined touchdowns on unsupported terrain; a small
  likelihood-ratio term; success-guided terrain curriculum; foothold prediction model; policy EMA. Budget: 8192
  envs x 40k iterations (about 820M steps, 7.5 h on an RTX PRO 6000) for G1; 4096 envs x 30k iterations (5.6 h) for
  ANYmal-D.
- **ANYmal variant reward** (the template we chose): goal reaching (AME2): reach a goal 2.5-3.5 m away within 8 s;
  100 x position closeness (last 4 s) + 50 x heading (last 2 s) + 5 x moving-to-goal + 5 x standing-at-goal;
  events (spin, belly contact, slip) -1; roll, joint, action-rate, joint-limit, power penalties; facing the goal +2;
  lateral velocity -1; foot impact -3; termination -500 per second.

## Status and what is left

**Done and tested** (CPU tests + short GPU runs): everything in the code map. The goal-reaching *environment*
(`env.goal_task=true`, `env.episode_length=400`) is built and tested on Go1 CPU: spawn on flat patches, goal 2.5-3.5 m
ahead, goal features (5) appended to the observation (obs_dim 176), the AME2 reward on the exact state, goal-driven
termination, timer reset on auto-reset (each env keeps one spawn and goal, because Playground's auto-reset restores
the first state).

**Not built** (the grounded agent; about 2-3 h of work plus tests):
1. World-model dynamics step conditioned on terrain features (heightmap at the pose), trained from the replay
   heightmap.
2. Actor and critic inputs = latent + goal features + heightmap scan, computed analytically at the (integrated)
   pose in imagination, and from the true pose in the simulator.
3. Imagination loop that carries the pose (`goal_task.pose_step` from decoded velocity / gravity / gyro) and scores
   decoded states with `GoalReward`; `aux0` passed to `ac_update`.
4. Goal-task evaluation (success = timeout with final distance < 0.5 m), terrain tiling + success-guided curriculum
   (currently one fixed course).
5. Optional: force head for the foot-impact term, leap event, Muon, the touchdown gradient cut.

## Recommendation for the long run

Honest state: there is no recipe validated on parkour yet. Options:

- **A. Validate on the existing parkour env first (about 15 minutes).** `slurm/train.sbatch` with the flat recipe
  (unclipped reward, alive 0.05, fixed noise 0.1, stabilisers, easier terrain `env.terrain_difficulty=0.5`) was
  launched as `exp-pk-val` (jobs 6435 / 6436); its result is appended below if it finished in time.
- **B. Finish the grounded goal agent (steps 1-4 above), then run.** This is the TGI-style route and the one most
  likely to learn obstacle crossing, but it is not built.
- **Suggested first long run settings once a recipe is chosen** (2 GPUs, `--gres=gpu:2`): `env.num_envs=4096`,
  `wm.batch=1024 ac.batch=2048`, `train.collect_steps=2`, `train.wm_updates=8 train.ac_updates=8`,
  `train.lr_decay_iters` = total iterations, `ac.ema=0.995`, `ac.fixed_std=0.1 ac.ent_coef=0`, W&B group named for the
  run. Time one 100-iteration job first; flat Go1 ran at about 0.55 s per iteration at 1024 envs on one GPU, so
  20k iterations is roughly 3-5 hours but the parkour figure is unmeasured.

## Open questions for you
1. Is path B (finish the grounded goal agent, about 2-3 h) the plan, with the validation run only as a sanity check?
2. Success definition for parkour: goal reached within 0.5 m at the end of the 8 s episode (TGI) vs distance along
   the corridor (our old metric)?
3. One fixed course vs a tiled terrain curriculum (needs engineering) for the long run?
