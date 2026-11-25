"""CPU tests on the point-mass env: every new piece (heightmap, contact head, student actor,
rollout export, checkpoint load / resume, replay preload) runs end to end."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jpw import checkpoint, replay, train
from jpw.agent import Agent
from jpw.config import build_config
from jpw.envs import get_spec, make_env
from jpw.export import export_rollouts, load_rollouts

FAST = ["train.iters=8", "train.warmup_iters=1", "train.log_every=4", "train.wandb_mode=disabled",
        "train.replay_capacity=96", "train.collect_steps=8", "env.num_envs=16", "wm.batch=16",
        "ac.batch=16", "wm.pred_horizon=4", "ac.horizon=4"]


def _run(tmp_path, preset, extra=()):
    args = ["--preset", preset, "--set", *FAST, f"train.out_dir={tmp_path}", *extra,
            "--run-name", "r"]
    train.main(args)
    return tmp_path / "r"


def _setup(preset, extra=()):
    cfg = build_config(preset, [*FAST, *extra])
    env = make_env(cfg.env)
    st = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), cfg.env.num_envs))
    spec = get_spec(st, cfg.env, env.action_size)
    agent = Agent(cfg, spec)
    return cfg, env, spec, agent, agent.init(jax.random.PRNGKey(1))


def test_specs():
    _, _, s, _, _ = _setup("dummy")
    assert (s.obs_dim, s.prop_dim, s.hmap_shape, s.contact_dim) == (6, 6, None, 2)
    _, _, s, _, _ = _setup("dummy_full")
    assert (s.obs_dim, s.prop_dim, s.hmap_shape, s.contact_dim) == (6, 4, (8, 8), 2)


@pytest.mark.parametrize("preset", ["dummy", "dummy_full"])
def test_train_runs_and_metrics(tmp_path, preset, capfd):
    _run(tmp_path, preset)
    assert "[step" in capfd.readouterr().out


def test_wm_metrics_present_and_finite():
    cfg, env, spec, agent, ts = _setup("dummy_full")
    rb = replay.init(96, 16, spec)
    from jpw.collect import make_collect
    st = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 16))
    collect = make_collect(env, agent, 48, True, True)
    _, _, _, out = collect(ts, st, jax.random.PRNGKey(2), jnp.zeros(16))
    rb = replay.add_chunk(rb, {k: v for k, v in out.items() if k in replay.FIELDS})
    assert rb.hmap.dtype == jnp.bfloat16 and rb.contact.shape == (96, 16, 2)
    ts, m = jax.jit(agent.update)(ts, rb, jax.random.PRNGKey(3))
    for k in ("wm/contact", "wm/contact_acc", "wm/contact_event_acc", "wm/student", "ac/imag_contact"):
        assert k in m and np.isfinite(float(m[k])), k


def test_student_sees_only_proprio():
    """Changing privileged state must not change the student latent; it must change the encoder's."""
    cfg, env, spec, agent, ts = _setup("dummy_full")
    obs = jnp.ones((3, spec.obs_dim))
    hm = jnp.ones((3, 8, 8))
    obs2 = obs.at[:, spec.prop_dim:].set(5.0)
    s1 = agent.policy_latent(ts.wm, obs, hm)
    s2 = agent.policy_latent(ts.wm, obs2, hm)
    np.testing.assert_allclose(np.asarray(s1), np.asarray(s2), atol=1e-6)
    assert not np.allclose(agent.encode(ts.wm, obs, hm), agent.encode(ts.wm, obs2, hm), atol=1e-4)


def test_contact_flags_follow_walls():
    env = make_env(build_config("dummy", []).env)
    st = env.reset(jax.random.split(jax.random.PRNGKey(0), 8))
    for _ in range(80):  # push toward +x wall
        st = env.step(st, jnp.tile(jnp.array([[1.0, 0.0]]), (8, 1)))
    assert float(st.info["contact"][:, 0].min()) == 1.0 and float(st.info["contact"][:, 1].max()) == 0.0


def test_export_load_and_preload(tmp_path):
    cfg, env, spec, agent, ts = _setup("dummy_full")
    d = str(tmp_path / "data")
    n = export_rollouts(cfg, spec, agent, ts, d, steps=40, chunk_steps=16, random_policy=True)
    assert n == 3
    arrays, spec2 = load_rollouts(d)
    assert spec2 == spec and arrays["obs"].shape == (40, 16, 6) and arrays["hmap"].shape == (40, 16, 8, 8)
    rb = replay.from_arrays(arrays, 96, spec)
    assert int(rb.size) == 40
    # offline pretrain on the exported data, then continue online
    _run(tmp_path, "dummy_full", [f"train.init_data={d}", "train.pretrain_updates=2"])


def test_checkpoint_roundtrip_and_resume(tmp_path):
    cfg, env, spec, agent, ts = _setup("dummy_full")
    path = str(tmp_path / "c.pkl")
    checkpoint.save(path, cfg, spec, ts, it=7, env_steps=123)
    cfg2, spec2, agent2, ts2, meta = checkpoint.load(path)
    assert cfg2 == cfg and spec2 == spec and meta == {"it": 7, "env_steps": 123}
    for a, b in zip(jax.tree_util.tree_leaves(ts), jax.tree_util.tree_leaves(ts2)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # train writes a final ckpt; resuming from it continues past that iteration
    run = _run(tmp_path, "dummy_full", ["train.ckpt_every=4"])
    final = run / "ckpt_final.pkl"
    assert final.exists()
    _run(tmp_path / "again", "dummy_full", [f"train.resume={final}", "train.iters=12", "train.ckpt_every=4"])
    _, _, _, _, meta = checkpoint.load(str(tmp_path / "again" / "r" / "ckpt_final.pkl"))
    assert meta["it"] == 11


def test_export_from_checkpoint_uses_policy(tmp_path):
    cfg, env, spec, agent, ts = _setup("dummy_full")
    path = str(tmp_path / "c.pkl")
    checkpoint.save(path, cfg, spec, ts, it=0, env_steps=0)
    from jpw import export
    export.main(["--ckpt", path, "--out", str(tmp_path / "d"), "--steps", "16"])
    arrays, _ = load_rollouts(str(tmp_path / "d"))
    assert arrays["act"].shape == (16, 16, 2) and np.abs(arrays["act"]).max() <= 1.0


def test_spec_mismatch_is_rejected(tmp_path):
    cfg, env, spec, agent, ts = _setup("dummy_full")
    path = str(tmp_path / "c.pkl")
    checkpoint.save(path, cfg, spec, ts, it=0, env_steps=0)
    with pytest.raises(ValueError, match="was made for"):
        _run(tmp_path / "x", "dummy", [f"train.resume={path}"])


def test_eval_runs_on_checkpoint(tmp_path, capsys):
    from jpw import eval as jeval

    run = _run(tmp_path, "dummy_full", ["train.ckpt_every=4"])
    res = jeval.main(["--ckpt", str(run / "ckpt_final.pkl"), "--num-envs", "16", "--episodes", "16",
                      "--set", "env.episode_length=20"])
    assert res["episodes"] >= 16 and res["length"] == 20 and np.isfinite(res["return"])
    assert "EVAL" in capsys.readouterr().out
