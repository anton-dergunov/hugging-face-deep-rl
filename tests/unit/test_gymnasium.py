import os
import pytest
import numpy as np
import gymnasium as gym
from rlcourse import gymnasium_utils


def test_describe_discrete_space():
    space = gym.spaces.Discrete(5)
    desc = gymnasium_utils.describe_space(space)
    assert "Discrete(n=5)" in desc


def test_describe_box_space():
    space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
    desc = gymnasium_utils.describe_space(space)
    assert "Box(shape=(3,)" in desc
    assert "low=-1.0" in desc
    assert "high=1.0" in desc


def test_with_dummy_video_driver_restores(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "original")
    called = {}

    @gymnasium_utils.with_dummy_video_driver
    def dummy_func():
        called["ran"] = True
        assert os.environ["SDL_VIDEODRIVER"] == "dummy"

    dummy_func()
    assert called["ran"]
    assert os.environ["SDL_VIDEODRIVER"] == "original"


def test_describe_environment_prints(capsys):
    env = gym.make("CartPole-v1")
    gymnasium_utils.describe_environment(env)
    env.close()
    output = capsys.readouterr().out
    assert "Observation Space:" in output
    assert "Action Space:" in output


# def test_evaluate_model_dummy(monkeypatch):
#     env_id = "CartPole-v1"
#     env = gym.make(env_id)

#     class DummyModel:
#         def predict(self, obs, state, episode_start, deterministic=True):
#             return env.action_space.sample(), None
#     model = DummyModel()

#     result = gymnasium_utils.evaluate_model(model, env_id, n_eval_episodes=1)
#     assert "mean_reward" in result
#     assert "std_reward" in result
