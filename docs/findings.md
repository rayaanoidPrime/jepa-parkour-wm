# Findings from the exploration phase

Everything learned before the large parkour run, from two autonomous exploration windows. Numbers come from
short runs on one MI300X unless stated; flat-terrain comparisons are 1-2 seeds and the parkour goal-task results are
a single seed, and single evals are noisy (the same checkpoint varied by 10-45% in episode length), so read them as
strong hints, not proof. No large run has been launched.

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
4. **The terrain-grounded goal-reaching agent (TGI-ANYmal style) works on our parkour course.** Single seed, 1 GPU,
   46 minutes of training (4000 iterations, 8.2M env steps): **80% goal success** (within 0.5 m of a goal 2.5-3.5 m
   ahead at the 8 s limit; 99% on flat goals, 69-76% across obstacles), and **82-97% on three held-out courses** it never
   trained on. 78% was reached after about 2000 iterations (about 23 minutes). See finding 11 and 12.
5. **Speed:** about 0.69 s per iteration at 1024 envs on one GPU; two GPUs with twice the envs and batch cost the same
   per iteration (tested, replicas in sync), so a 20k-iteration run is about 3.8 h. XLA Triton GEMM must be off on this
   ROCm stack (it produces NaNs) and that costs nothing.
6. **Still missing for the full plan:** a deployable actor (proprioception + depth, no privileged heightmap / state),
   a success-guided curriculum, harder / more varied terrain (stepping stones, higher stairs), multiple seeds.

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

### 10. Validation of the flat recipe on the existing parkour env (jobs 6435 / 6436, 1 GPU, 11 min, 2.2M env steps)
Setup: `Go1Parkour` (velocity command + progress reward, difficulty 0.5, no DR, no student), unclipped reward,
alive bonus 0.05, fixed noise 0.1, stabilisers, learned heads. Eval of the EMA actor: **episode length 997 of 1000, but progress
only 0.13 m and 0% success.** The policy survives the whole episode by standing still: the alive bonus (0.05 per step,
about 50 per episode) is worth far more than the progress reward (1.0 per metre, about 0.02 per step at 1 m/s) and the
tracking terms (about 0.005 per step). This is reward hacking by the bonus, and it likely explains the flat-terrain
results too: tracking never improved because it is a 10x smaller signal than the bonus. Hypothesis (untested): use a
much smaller bonus (e.g. 0.005-0.01) or none with a strong termination penalty instead, as AME2 / TGI do (-500 per
second on termination, no alive bonus), and keep the progress / goal terms large relative to any survival term.

### 11. Terrain-grounded goal-reaching agent (window 2; single seed; Go1Parkour goal task)
Built: world-model dynamics conditioned on the heightmap at the pose; actor / critic input = latent + goal features +
heightmap computed at the pose integrated in imagination; imagined states scored with the AME2 goal reward (no alive
bonus, termination penalty -500/s); goal-task eval (success = within 0.5 m at the 8 s time limit).

- **g1 (course with obstacles everywhere, 3000 iterations, 1024 envs, 36 min):** survival solved (episodes run to the time
  limit, 99-100% timeouts, vs 69 steps for the first parkour run) but 0% goal success, final distance 1.8 m of about 3.
  A CPU eval split showed why: in that course *every* spawn-to-goal path crossed an obstacle (flat_n = 0 of 128), and the
  course had only 8 safe spawn positions. There were no easy goals to learn from or to measure walking on.
- **Fix:** graded course (`env.course_ramp=true`: flat start of 6 m, obstacle difficulty ramps from 0.05 to 1.0 along
  the course) and a fixed mixture with 30% flat-goal spawns (`env.flat_goal_frac=0.3`); finer spawn grid (61 safe
  positions instead of 21). Each env keeps one spawn for the whole run (Playground's auto-reset restores the first
  state), so this fixed mixture is the curriculum; a success-guided curriculum like TGI's would need custom reset logic.
- **g3 (graded course, 4000 iterations, 1024 envs, 46 min training, 8.2M env steps, 1 seed):**

| checkpoint | goal success | flat goals | obstacle goals | final dist | episode length (of 400) |
|---|---|---|---|---|---|
| 1000 it | 24% | 54% | 11% | 0.71 m | 223 |
| 2000 it | 78% | 99% | 64% | 0.30 m | 347 |
| 3000 it | 83% | 94% | 76% | 0.24 m | 356 |
| 4000 it (final) | 80% | 99% | 69% | 0.36 m | 339 |

  Training episode return rose from 11 to about 140-160 (the goal-proximity reward). About 80% of eval episodes reach the
  time limit; the rest end early (fall, pit, lane).

### 12. Held-out courses, speed versus the reference work
`slurm/eval_heldout.sbatch` on the g3 final checkpoint (other terrain seeds = other random obstacle sequences from the
same generator and difficulty ramp):

| course | goal success | flat goals | obstacle goals | final distance |
|---|---|---|---|---|
| training course (seed 0) | 78% | 96% | 66% | 0.31 m |
| held-out seed 1 | 97% | 99% | 95% | 0.15 m |
| held-out seed 2 | 85% | 98% | 77% | 0.26 m |
| held-out seed 3 | 82% | 98% | 72% | 0.34 m |

It does at least as well on unseen courses as on the training one (which happens to be among the harder layouts), so it is
reading the terrain, not memorising a layout. It has not been tested on obstacle types or sizes outside the generator.

Two-GPU smoke (200 iterations, 2048 envs, wm.batch 1024 / ac.batch 2048, domain randomisation + heightmap noise on):
runs, finite, replicas bit-identical, about 5.9k env steps/s, i.e. 0.69 s per iteration (same as one GPU at 1024 envs).

How this compares with the reference work (not a like-for-like comparison):
- TGI-ANYmal: 30k iterations at 4096 envs, 5.6 h, to its final policy on a much harder multi-terrain suite (stepping
  stones, pits, boxes, stairs) with a success-guided curriculum that aims for 80% success. We reach 80% success in about
  0.4-0.8 h on one GPU, but on a single corridor of stairs / boxes / gaps with forward goals only.
- SuperMimic: 1.2 h on one H100 to a tracking return of 0.82 on LAFAN1; BlinkMimic 3.4 min on 40 clips. Different task and
  metric.
- Our iteration rate (1.45 it/s) is about TGI's (1.5 it/s: 40k iterations in 7.5 h); the difference is how many iterations
  each needs, and TGI's tasks are harder.
Caveats: single seed; evaluation uses the exact heightmap and exact physical state (privileged, simulation only); no
deployable (proprioception + depth) actor; the obstacle success is not monotonic across checkpoints (64% / 76% / 69%);
about 20% of evaluation episodes end early (fall, pit, lane).

## Status and what is left

**Done and tested:** everything in the code map, including the grounded goal agent (terrain-conditioned dynamics, actor /
critic with goal + heightmap features at the integrated pose, pose-carrying imagination, AME2 reward, goal-task eval with
flat / obstacle split), the graded course + flat-goal mixture, held-out evaluation, and two-GPU training with domain
randomisation (smoke). 24 unit / integration tests (CPU) plus short GPU runs.

**Not done:**
1. **Deployable actor.** The grounded actor sees the exact heightmap and the privileged latent. The plan's deployable actor
   (proprioception + depth only) is not integrated with the grounded agent (`ac.grounded` requires `ac.asymmetric=false`).
2. **Success-guided curriculum.** The mixture is fixed (each env keeps one spawn because Playground's auto-reset restores
   the first state); TGI adapts terrain difficulty from success rates.
3. **Harder terrain** (stepping stones, bigger stairs, wider gaps) and held-out obstacle *types*.
4. **Seeds.** Every parkour result is one seed.
5. Optional: foot-impact force head, leap event, Muon, touchdown gradient cut, larger world-model batches (TGI: 2048 windows).

## Recommendation for the long run

Use the recipe that produced g3 (+ g4, see below), at the scale that was smoke-tested:

```bash
sbatch --gres=gpu:2 --cpus-per-task=32 --mem=256G slurm/train.sbatch --preset goal --run-name goal-long --set \
  train.num_devices=2 env.num_envs=2048 wm.batch=1024 ac.batch=2048 \
  train.iters=20000 train.lr_decay_iters=20000 train.ckpt_every=2000 train.log_every=50 \
  env.course_ramp=true env.flat_goal_frac=0.3 env.domain_randomization=true env.hmap_noise=0.01 \
  train.wandb_group=goal-long
# afterwards, per checkpoint:  sbatch slurm/eval_heldout.sbatch runs/goal-long/ckpt_<N>.pkl 1 2 3 4
```
Estimated 3.8 h (0.69 s per iteration). The `goal` preset already sets: grounded actor, decoder heads, fixed action noise 0.1
with no entropy bonus, EMA actor 0.995, element-wise gradient clip 0.1, Adam beta2 0.95, 2 collection steps and 8 + 8
updates per iteration. Things you may want to decide differently:
- A longer horizon is only worthwhile if success keeps rising: g3 plateaued around 78-83% from 2000 iterations at 1024
  envs, so the long run is mostly about harder terrain and robustness, not about reaching 80% faster.
- If you want a harder test, raise `env.terrain_difficulty` (the ramp's maximum, 1.0 now) is already at its cap; the next
  lever is new obstacle types (needs code).
- The 24-test suite and `docs/findings.md` are the reference for what is verified.

## Open questions for you
1. Is the corridor-with-goals setting (one terrain family, forward goals) the right target for the first long run, or should
   it include stepping stones / pits first?
2. Do you want the deployable (proprioception + depth) actor integrated before the long run, or after we see the
   privileged-agent long run?
3. Is 0.5 m at the 8 s limit the success definition you want (TGI uses it for its curriculum)?
