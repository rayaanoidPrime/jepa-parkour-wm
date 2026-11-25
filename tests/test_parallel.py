"""Data-parallel training on 4 simulated CPU devices (separate process: the device count has to
be set before JAX starts)."""
import os
import subprocess
import sys
import textwrap

import pytest

FAST = ("train.iters=8 train.warmup_iters=1 train.log_every=4 train.wandb_mode=disabled "
        "train.replay_capacity=96 train.collect_steps=8 env.num_envs=16 wm.batch=16 ac.batch=16 "
        "wm.pred_horizon=4 ac.horizon=4").split()


def _run(code, tmp_path):
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=4", JAX_PLATFORMS="cpu")
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env, cwd=tmp_path,
                          capture_output=True, text=True, timeout=900)


@pytest.mark.parametrize("preset", ["dummy", "dummy_full"])
def test_multi_device_training_stays_in_sync(tmp_path, preset):
    code = f"""
        import jax
        assert jax.device_count() == 4
        from jpw import train, parallel
        ts = train.main(["--preset", "{preset}", "--run-name", "r", "--set", *{FAST!r},
                         "train.num_devices=4", "train.ckpt_every=4", "train.out_dir={tmp_path.as_posix()}"])
        print("DIFF", parallel.replica_diff(ts))
    """
    p = _run(code, tmp_path)
    assert p.returncode == 0, p.stderr[-3000:]
    assert "data parallel over 4 devices" in p.stdout
    diff = float([l for l in p.stdout.splitlines() if l.startswith("DIFF")][0].split()[1])
    assert diff == 0.0, f"replicas diverged by {diff}"
    assert (tmp_path / "r" / "ckpt_final.pkl").exists()


def test_indivisible_batch_is_rejected(tmp_path):
    code = f"""
        from jpw import train
        train.main(["--preset", "dummy", "--set", *{FAST!r}, "train.num_devices=4", "wm.batch=18",
                    "train.out_dir={tmp_path.as_posix()}"])
    """
    p = _run(code, tmp_path)
    assert p.returncode != 0 and "must be divisible" in p.stderr
