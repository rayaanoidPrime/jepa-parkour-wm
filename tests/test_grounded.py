"""The grounded goal-task agent on synthetic Go1-shaped data (no MuJoCo): update runs, is finite, the actor
gets gradient through pose integration + heightmap lookups, and the terrain features really enter the model."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jpw import go1_state as gs
from jpw import goal_task as gt
from jpw import replay
from jpw.agent import Agent
from jpw.config import build_config
from jpw.spec import Spec

N, T = 8, 24


def _spec():
    return Spec(obs_dim=gs.GOAL_OBS_DIM, prop_dim=48, act_dim=12, hmap_shape=(12, 8), contact_dim=4, aux_dim=19)


def _cfg(extra=()):
    return build_config(None, ["wm.latent_dim=16", "wm.hidden=32", "wm.depth=2", "ac.hidden=32", "ac.depth=2",
                               "wm.batch=8", "ac.batch=8", "wm.pred_horizon=3", "ac.horizon=5", "wm.cnn_channels=4",
                               "wm.decode_state=true", "ac.grounded=true", "ac.fixed_std=0.1", "ac.ent_coef=0.0",
                               "train.wm_updates=2", "train.ac_updates=2", "train.lr_decay_iters=50",
                               "ac.grad_clip_value=0.1", "ac.beta2=0.95", "ac.ema=0.99", *extra])


def _agent(cfg=None):
    spec = _spec()
    agent = Agent(cfg or _cfg(), spec)
    ground = lambda x, y: 0.1 * jnp.sin(x)  # a bumpy test terrain
    agent.task = gt.GoalReward(np.zeros(12), -np.ones(12), np.ones(12), ground, pit_z=-5.0, lane_half=50.0)

    def hm_fn(pos, yaw):  # heightmap of the bumpy terrain at the pose, same layout as the real one
        gx, gy = np.meshgrid(np.linspace(-0.2, 1.2, 12), np.linspace(-0.5, 0.5, 8), indexing="ij")
        c, s = jnp.cos(yaw)[..., None, None], jnp.sin(yaw)[..., None, None]
        wx = pos[..., 0, None, None] + c * gx - s * gy
        return ground(wx, 0.0) - pos[..., 2, None, None] + 0.27

    agent.hm_fn = hm_fn
    return agent, spec


def _buffer(spec, seed=0):
    rng = np.random.default_rng(seed)
    rb = replay.init(T, N, spec)
    obs = rng.normal(size=(T, N, spec.obs_dim)).astype(np.float32) * 0.3
    obs[..., 102:105] = np.array([0.0, 0.0, -1.0]) + 0.05 * rng.normal(size=(T, N, 3))  # gravity ~ upright
    yaw = rng.uniform(-0.2, 0.2, (T, N))
    obs[..., gs.GOAL] = np.stack([rng.uniform(2, 3, (T, N)), rng.uniform(-0.3, 0.3, (T, N)), np.sin(yaw) * 0,
                                  np.ones((T, N)), rng.uniform(0.2, 1.0, (T, N))], -1)
    aux = np.zeros((T, N, 19), np.float32)
    aux[..., :3] = np.stack([rng.uniform(0, 2, (T, N)), rng.uniform(-0.5, 0.5, (T, N)), np.full((T, N), 0.35)], -1)
    aux[..., 3], aux[..., 6] = np.cos(yaw / 2), np.sin(yaw / 2)
    aux[..., 9:19:3] = rng.uniform(0, 0.05, (T, N, 4))
    rb = replay.add(rb, jnp.asarray(obs), jnp.asarray(rng.uniform(-1, 1, (T, N, 12)), jnp.float32),
                    jnp.zeros((T, N)), jnp.zeros((T, N)), jnp.zeros((T, N)),
                    jnp.asarray(rng.normal(size=(T, N, 12, 8)) * 0.05, jnp.float32),
                    jnp.asarray((rng.uniform(size=(T, N, 4)) > 0.5), jnp.float32), jnp.asarray(aux))
    return rb


def test_grounded_update_is_finite_and_trains_the_actor():
    agent, spec = _agent()
    ts = agent.init(jax.random.PRNGKey(0))
    rb = _buffer(spec)
    ts2, m = jax.jit(agent.update)(ts, rb, jax.random.PRNGKey(1))
    for k in ("wm/loss", "wm/phys", "ac/actor_loss", "ac/imag_return", "ac/actor_grad_norm", "wm/contact"):
        assert k in m and np.isfinite(float(m[k])), k
    assert float(m["ac/actor_grad_norm"]) > 0
    moved = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                for a, b in zip(jax.tree_util.tree_leaves(ts.actor), jax.tree_util.tree_leaves(ts2.actor)))
    assert moved > 0


def test_actor_input_has_goal_and_terrain_features():
    agent, spec = _agent()
    assert agent.ext_dim == gt.GOAL_FEAT_DIM + 96
    ts = agent.init(jax.random.PRNGKey(0))
    obs = jnp.zeros((3, spec.obs_dim)).at[:, gs.GOAL].set(jnp.array([2.0, 0.0, 0.0, 1.0, 0.5]))
    hm0, hm1 = jnp.zeros((3, 12, 8)), jnp.ones((3, 12, 8)) * 0.3
    a0 = agent.act(ts, obs, None, hm0, deterministic=True)
    a1 = agent.act(ts, obs, None, hm1, deterministic=True)
    a2 = agent.act(ts, obs.at[:, gs.GOAL].set(jnp.array([0.0, 2.0, 0.5, 0.9, 0.1])), None, hm0, deterministic=True)
    assert a0.shape == (3, 12)
    assert float(jnp.abs(a0 - a1).max()) > 0 and float(jnp.abs(a0 - a2).max()) > 0  # both inputs matter


def test_dynamics_step_depends_on_the_terrain_features():
    agent, spec = _agent()
    ts = agent.init(jax.random.PRNGKey(0))
    from jpw.models import WorldModel

    z, a = jnp.ones((2, 16)), jnp.zeros((2, 12))
    s0 = agent.wm.apply(ts.wm, z, a, jnp.zeros((2, 96)), method=WorldModel.step)
    s1 = agent.wm.apply(ts.wm, z, a, jnp.ones((2, 96)) * 0.5, method=WorldModel.step)
    # zero-initialised? the dynamics output layer is not, so the two must differ once trained; at init allow tiny
    ts2, _ = jax.jit(agent.update)(ts, _buffer(spec), jax.random.PRNGKey(2))
    t0 = agent.wm.apply(ts2.wm, z, a, jnp.zeros((2, 96)), method=WorldModel.step)
    t1 = agent.wm.apply(ts2.wm, z, a, jnp.ones((2, 96)) * 0.5, method=WorldModel.step)
    assert float(jnp.abs(t0 - t1).max()) > 0 or float(jnp.abs(s0 - s1).max()) > 0


def test_pose_gradient_reaches_the_actor_through_the_heightmap():
    """The return must depend on the action through (action -> velocity -> pose -> terrain lookup)."""
    agent, spec = _agent()
    ts = agent.init(jax.random.PRNGKey(0))
    rb = _buffer(spec)
    b = replay.sample(rb, jax.random.PRNGKey(3), 8, 1)
    obs0, aux0, hm0 = b["obs"][:, 0], b["aux"][:, 0], b["hmap"][:, 0]
    z0 = agent.encode(ts.wm, obs0, hm0)

    def f(actor_params):
        zs, zl, rs, cs, lp, pc = agent._rollout_grounded(ts, actor_params, z0, obs0, aux0,
                                                         jax.random.split(jax.random.PRNGKey(4), 5))
        return rs.sum()

    g = jax.grad(f)(ts.actor)
    gn = float(jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(g))))
    assert np.isfinite(gn) and gn > 0


def test_grounded_requires_decoder_and_goal_obs():
    with pytest.raises(ValueError):
        Agent(build_config(None, ["ac.grounded=true"]), _spec())
    bad = Spec(obs_dim=171, prop_dim=48, act_dim=12, hmap_shape=(12, 8), contact_dim=4, aux_dim=19)
    with pytest.raises(ValueError):
        Agent(_cfg(), bad)
