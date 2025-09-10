import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict
from stable_baselines3.common.vec_env import VecEnv
import matplotlib.pyplot as plt
import os
import warnings
import numpy as np


def setup_ignore_warnings():
    # This filter handles the "pkg_resources is deprecated" warning from Pygame
    warnings.filterwarnings("ignore", category=DeprecationWarning, module='pygame')
    warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

    # This filter handles all the "Deprecated call to `pkg_resources.declare_namespace`" warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module='pkg_resources')


def show_environment(env_id, steps=20):
    # Set the video driver to a dummy one to prevent window creation
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = gym.make(env_id, render_mode="rgb_array")
    env.reset()

    # Progress the environment for several steps so that the image is more interesting
    for _ in range(steps):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()

    frame = env.render()

    plt.imshow(frame)
    plt.axis("off")
    plt.show()

    env.close()


def describe_space(space, indent=0):
    """
    Helper function to describe a gymnasium space in detail.
    Details about spaces: https://gymnasium.farama.org/api/spaces/
    """
    prefix = " " * indent
    if isinstance(space, Box):
        return (f"{prefix}Box(shape={space.shape}, dtype={space.dtype}, "
                f"low={np.min(space.low)}, high={np.max(space.high)})")
    elif isinstance(space, Discrete):
        return f"{prefix}Discrete(n={space.n})"
    elif isinstance(space, MultiDiscrete):
        return f"{prefix}MultiDiscrete(nvec={space.nvec})"
    elif isinstance(space, MultiBinary):
        return f"{prefix}MultiBinary(n={space.n})"
    elif isinstance(space, Tuple):
        descs = [describe_space(s, indent+2) for s in space.spaces]
        return f"{prefix}Tuple(\n" + "\n".join(descs) + f"\n{prefix})"
    elif isinstance(space, Dict):
        descs = [f"{prefix}{k}: {describe_space(v, indent+2)}" for k, v in space.spaces.items()]
        return f"{prefix}Dict(\n" + "\n".join(descs) + f"\n{prefix})"
    else:
        return f"{prefix}Unknown space type: {space}"


def describe_environment(env):
    # If this is a VecEnv, unwrap the first env
    base_env = None
    if isinstance(env, VecEnv):
        base_env = env.envs[0]
    else:
        base_env = env

    print("Observation Space:")
    print(describe_space(env.observation_space, indent=2))
    try:
        print("  Example observation:", env.observation_space.sample())
    except Exception as e:
        print("  Could not sample observation:", e)

    print("\nAction Space:")
    print(describe_space(env.action_space, indent=2))
    try:
        print("  Example action:", env.action_space.sample())
    except Exception as e:
        print("  Could not sample action:", e)

    # Rewards
    if hasattr(base_env, "reward_range"):
        print("\nReward range:", base_env.reward_range)

    # Max episode steps (from base spec)
    if getattr(base_env, "spec", None) and hasattr(base_env.spec, "max_episode_steps"):
        print("\nMax episode steps:", base_env.spec.max_episode_steps)

