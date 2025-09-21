import os
import cv2
import imageio
from functools import wraps

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt


def with_dummy_video_driver(func):
    """Set the video driver to a dummy one to prevent window creation"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        original_sdl = os.environ.get("SDL_VIDEODRIVER")
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        try:
            return func(*args, **kwargs)
        finally:
            if original_sdl is None:
                os.environ.pop("SDL_VIDEODRIVER", None)
            else:
                os.environ["SDL_VIDEODRIVER"] = original_sdl
    return wrapper


@with_dummy_video_driver
def show_environment(env_id, steps=20):
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


def evaluate_model(model, env_id, n_eval_episodes=10):
    env = Monitor(gym.make(env_id))
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    env.close()
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "n_eval_episodes": n_eval_episodes
    }


@with_dummy_video_driver
def record_agent_video(model, env_id, video_path, steps=1000):
    # Create env with video recording enabled
    env = gym.make(env_id, render_mode="rgb_array")
    
    try:
        frames = []
        obs, info = env.reset()
        episode, step = 0, 0
        total_reward = 0

        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Render the frame from the environment
            frame = env.render()

            # --- Overlay text (small, anti-aliased) ---
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (155, 105), (255, 255, 255), -1)
            alpha = 0.5  # 50% opacity
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.putText(frame, f"Ep: {episode}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Step: {step}", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Reward: {total_reward:.2f}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Delta: {reward:.2f}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

            frames.append(frame)
            step += 1

            if terminated or truncated:
                episode += 1
                step = 0
                total_reward = 0
                obs, info = env.reset()
        
        # Save the collected frames to a single video file
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        fps = env.metadata.get("render_fps", 30)
        imageio.mimsave(video_path, frames, fps=fps, macro_block_size=None)
        print(f"Saved video to {video_path}")

    finally:
        # Ensure the environment is closed to release all resources
        env.close()
