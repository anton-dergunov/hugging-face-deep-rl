import os
import pytest
import numpy as np

import gymnasium as gym

from rlcourse import gymnasium_utils


# -------------------------------
# with_dummy_video_driver
# -------------------------------
def test_with_dummy_video_driver_sets_and_restores(monkeypatch):
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)  # start unset
    called = {}

    @gymnasium_utils.with_dummy_video_driver
    def fn():
        called["ran"] = True
        assert os.environ["SDL_VIDEODRIVER"] == "dummy"

    fn()
    assert called["ran"]
    assert "SDL_VIDEODRIVER" not in os.environ  # restored


# -------------------------------
# describe_space
# -------------------------------
def test_describe_basic_spaces():
    assert "Discrete(n=3)" in gymnasium_utils.describe_space(gym.spaces.Discrete(3))
    box = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    assert "Box(shape=(2," in gymnasium_utils.describe_space(box)
    md = gym.spaces.MultiDiscrete([2, 3])
    assert "MultiDiscrete(nvec=" in gymnasium_utils.describe_space(md)
    mb = gym.spaces.MultiBinary(4)
    assert "MultiBinary(n=4)" in gymnasium_utils.describe_space(mb)


def test_describe_composite_spaces():
    tup = gym.spaces.Tuple([gym.spaces.Discrete(2), gym.spaces.Discrete(3)])
    desc = gymnasium_utils.describe_space(tup)
    assert "Tuple(" in desc
    assert "Discrete(n=2)" in desc

    dic = gym.spaces.Dict({"a": gym.spaces.Discrete(2)})
    desc = gymnasium_utils.describe_space(dic)
    assert "Dict(" in desc
    assert "a:" in desc


def test_describe_unknown_space_type():
    class FakeSpace:
        pass
    desc = gymnasium_utils.describe_space(FakeSpace())
    assert "Unknown space type" in desc


# -------------------------------
# describe_environment
# -------------------------------
def test_describe_environment_prints(capsys):
    env = gym.make("CartPole-v1")
    gymnasium_utils.describe_environment(env)
    env.close()
    out = capsys.readouterr().out
    assert "Observation Space:" in out
    assert "Action Space:" in out
    # Flexible check: at least one of reward range or max steps must be printed
    assert ("Reward range:" in out) or ("Max episode steps:" in out)


# -------------------------------
# evaluate_model
# -------------------------------
class DummyModel:
    def predict(self, obs, **kwargs):
        if isinstance(obs, np.ndarray) and obs.ndim > 1:
            # Likely vectorized env (DummyVecEnv -> evaluate_policy)
            return np.array([0]), None
        else:
            # Raw env (record_agent_video)
            return 0, None


def test_evaluate_model_with_dummy_model():
    model = DummyModel()
    result = gymnasium_utils.evaluate_model(model, "CartPole-v1", n_eval_episodes=1)
    assert set(result.keys()) == {"mean_reward", "std_reward", "n_eval_episodes", "seed"}
    assert result["n_eval_episodes"] == 1


def test_stable_evaluate_confidence_interval(monkeypatch):
    """Test that stable_evaluate computes confidence intervals correctly."""

    # Return controlled results for three seeds
    fake_results = [
        {"mean_reward": 200.0, "std_reward": 30.0, "n_eval_episodes": 30, "seed": 0},
        {"mean_reward": 210.0, "std_reward": 40.0, "n_eval_episodes": 30, "seed": 1},
        {"mean_reward": 190.0, "std_reward": 20.0, "n_eval_episodes": 30, "seed": 2},
    ]

    monkeypatch.setattr(gymnasium_utils, "evaluate_model", lambda *args, **kwargs: fake_results.pop(0))

    model = DummyModel()
    result = gymnasium_utils.stable_evaluate(model, "CartPole-v1", seeds=(0,1,2), n_eval_episodes=30)

    assert pytest.approx(result["mean_reward"], rel=1e-6) == 200.0
    assert pytest.approx(result["std_reward"], rel=1e-2) == 31.091
    assert pytest.approx(result["ci_low"], rel=1e-2) == 193.576
    assert pytest.approx(result["ci_high"], rel=1e-2) == 206.424
    assert result["n_total"] == 90


# -------------------------------
# record_agent_video
# -------------------------------
def test_record_agent_video_creates_file(tmp_path):
    env_id = "CartPole-v1"

    video_path = tmp_path / "test.mp4"
    gymnasium_utils.record_agent_video(DummyModel(), env_id, str(video_path), steps=5)

    assert video_path.exists()
    assert video_path.stat().st_size > 0
