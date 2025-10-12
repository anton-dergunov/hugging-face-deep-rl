import os
import cv2
import imageio
from functools import wraps
from tqdm.auto import tqdm

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback


class TQDMProgressCallback(BaseCallback):
    """Custom progress class for SB3 which integrates better with Jupyter Notebook"""
    def __init__(self, total_timesteps: int, title: str = "Training"):
        super().__init__()
        self.total_timesteps = total_timesteps
        # update every 0.1% of total timesteps
        self.update_freq = max(1, total_timesteps // 1000)
        self.progress = tqdm(
            total=total_timesteps,
            unit="timesteps",
            desc=title,
            dynamic_ncols=True,
        )

    def _on_step(self) -> bool:
        if self.num_timesteps % self.update_freq == 0:
            self.progress.update(self.update_freq)
        return True

    def _on_training_end(self) -> None:
        self.progress.close()


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
def show_environment(env, steps=20):
    """Displays the environment to get an idea about it"""
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


def make_subproc_env(env_id, n_proc, seed, log_dir=None):
    """
    Creates vectorized environment with multiprocessing.

    :param env_id: the environment ID
    :param n_proc: number of subprocess
    :param seed: the initial seed for RNG
    :param log_dir: directory to save Monitor logs
    """
    def make_env(env_id: str, rank: int, seed: int = 0, log_dir: str = None):
        def _init():
            # Silence warnings inside the subprocess
            import warnings
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="pygame")
            warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

            env = gym.make(env_id)
            env.reset(seed=seed + rank)
            if log_dir is not None:
                env = Monitor(env, filename=os.path.join(log_dir, f"proc_{rank}"))
            return env

        set_random_seed(seed)
        return _init

    return SubprocVecEnv([make_env(env_id, i, seed=seed, log_dir=log_dir) for i in range(n_proc)])


def evaluate_model(model, env_id, n_eval_episodes=50, seed=42, env_kwargs={}):
    """
    Evaluate the model for a given seed and return summary statistics.
    """
    env = Monitor(gym.make(env_id, **env_kwargs))
    env.reset(seed=seed)
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    env.close()
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "n_eval_episodes": n_eval_episodes,
        "seed": seed,
    }


def stable_evaluate(model, env_id, seeds=(0, 42, 99), n_eval_episodes=30, env_kwargs={}):
    """
    Evaluate the model across multiple seeds and compute a 95% confidence interval
    for the mean reward estimate.
    """
    results = [evaluate_model(model, env_id, n_eval_episodes, s, env_kwargs) for s in seeds]

    mean_rewards = np.array([r["mean_reward"] for r in results])
    std_rewards = np.array([r["std_reward"] for r in results])

    # Average mean reward across seeds
    mean_reward = mean_rewards.mean()

    # Combine standard deviations (root mean of variances)
    std_reward = np.sqrt(np.mean(std_rewards**2))

    # Effective sample size (episodes * number of seeds)
    n_total = len(seeds) * n_eval_episodes

    # Standard error of the mean
    sem = std_reward / np.sqrt(n_total)

    # 95% confidence interval using normal approximation (z=1.96)
    z = 1.96
    ci_low = mean_reward - z * sem
    ci_high = mean_reward + z * sem

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_total": n_total,
    }


def _annotate_frame(frame, episode, step, total_reward, reward_delta):
    """Return annotated frame copy."""
    out = frame.copy()
    cv2.rectangle(out, (5, 5), (155, 105), (255, 255, 255), -1)
    alpha = 0.5
    out = cv2.addWeighted(out, 1 - alpha, frame, alpha, 0)  # subtle transparent box
    cv2.putText(out, f"Ep: {episode}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(out, f"Step: {step}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(out, f"Reward: {total_reward:.2f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(out, f"Delta: {reward_delta:.2f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _render_and_annotate(env, episode, step, total_reward, reward_delta):
    frame = env.render()
    return _annotate_frame(frame, episode, step, total_reward, reward_delta)


@with_dummy_video_driver
def record_agent_video(model, env_id, video_path, steps=1000, env_kwargs={}):
    # Create env with video recording enabled
    env = gym.make(env_id, **env_kwargs, render_mode="rgb_array")
    
    try:
        frames = []
        obs, info = env.reset()
        episode, step = 0, 0
        total_reward = 0

        # Record initial frame right after reset
        frames.append(_render_and_annotate(env, episode, step, total_reward, 0.0))

        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
                    
            # Render the frame from the environment
            frames.append(_render_and_annotate(env, episode, step, total_reward, reward))

            if terminated or truncated:
                episode += 1
                step = 0
                total_reward = 0
                obs, info = env.reset()

                # Immediately record new episode's initial frame
                frames.append(_render_and_annotate(env, episode, step, total_reward, 0.0))

        # Save the collected frames to a single video file
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        fps = env.metadata.get("render_fps", 30)
        imageio.mimsave(video_path, frames, fps=fps, macro_block_size=None)
        print(f"Saved video to {video_path}")

    finally:
        # Ensure the environment is closed to release all resources
        env.close()
