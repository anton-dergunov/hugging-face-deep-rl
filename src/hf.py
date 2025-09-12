from huggingface_hub import HfApi, whoami, notebook_login, login, create_repo, hf_hub_download
from stable_baselines3 import PPO  # TODO Clean up this list of imports
from pathlib import Path
import hashlib
import json
import tempfile
import os


def ensure_hf_login(interactive=True):
    """
    Ensure Hugging Face login is available.
    
    - If already logged in, does nothing.
    - If not, asks the user to log in (via notebook or CLI).
    - Works safely in Jupyter or scripts.
    """
    try:
        whoami()
        print("✅ Hugging Face credentials already available.")
    except Exception:
        if interactive:
            try:
                # Notebook-specific login prompt
                notebook_login()
            except Exception:
                # CLI fallback
                login()
        else:
            raise RuntimeError(
                "No Hugging Face credentials found. "
                "Run this function with interactive=True to log in."
            )


def generate_readme(repo_name, env_id, algo, metrics):
    yaml_metadata = f"""---
library_name: stable-baselines3
tags:
- reinforcement-learning
- deep-reinforcement-learning
- sb3
- {env_id}
model-index:
- name: {repo_name}
  results:
  - task:
      type: reinforcement-learning
      name: Reinforcement Learning
    dataset:
      name: {env_id}
      type: gym
    metrics:
    - type: mean_reward
      value: {metrics['mean_reward']:.2f}
      name: Mean Reward
    - type: std_reward
      value: {metrics['std_reward']:.2f}
      name: Std Reward
---
"""

    readme = f"""{yaml_metadata}

# {repo_name}

This is a reinforcement learning agent trained with **{algo}** on **{env_id}** using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

## Usage

```python
import gymnasium as gym
from stable_baselines3 import PPO
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="{repo_name}", filename="model.zip")
model = PPO.load(model_path, device="cpu")

env = gym.make("{env_id}")
obs, info = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
```
"""

    return readme


def upload_sb3_model(model, repo_id, model_name, model_architecture, env_id,
                     metrics, commit_message="Upload SB3 model", video_path=None):
    api = HfApi()

    def upload_file(path_or_fileobj, path_in_repo, repo_id, commit_message):
        print(f"Uploading {path_in_repo}...")
        api.upload_file(
            path_or_fileobj=path_or_fileobj,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            commit_message=commit_message
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(os.path.abspath(tmpdir))

        # 1. Save model
        model_path = tmpdir / "model.zip"
        print(f"Saving the model to {model_path}...")
        model.save(model_path)

        # Output the size and hash of the saved model
        model_size = os.path.getsize(model_path)
        with open(model_path, "rb") as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()
        print(f"Model size: {model_size / 1024:.2f} KB")
        print(f"Model SHA256: {model_hash}")

        # Try loading the file we just created as a verification step
        try:
            PPO.load(model_path, device="cpu")
        except Exception as e:
            print(f"❌ Verification failed. The saved file is corrupted: {e}")
            # Do not proceed if verification fails
            raise

        # 2. Create metadata
        metadata = {
            "model_name": model_name,
            "model_architecture": model_architecture,
            "env_id": env_id,
        }
        metadata_path = tmpdir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # 3. Create README (model card)
        readme_path = tmpdir / "README.md"
        readme_text = generate_readme(repo_id, env_id, model_architecture, metrics)
        with open(readme_path, "w") as f:
            f.write(readme_text)

        # 4. Create repo if missing
        try:
            create_repo(repo_id, exist_ok=True)
        except Exception as e:
            print(f"Repo already exists or could not be created: {e}")

        # 5. Upload all files
        for file in [model_path, metadata_path, readme_path]:
            upload_file(
                path_or_fileobj=file,
                path_in_repo=os.path.basename(file),
                repo_id=repo_id,
                commit_message=commit_message,
            )

        # 6. Upload video separately
        if video_path:
            upload_file(
                path_or_fileobj=video_path,
                path_in_repo=f"replay.mp4",
                repo_id=repo_id,
                commit_message="Add demo video"
            )

    print(f"\n✅ Upload complete! View the model at:")
    print(f"https://huggingface.co/{repo_id}")


def load_model_from_hub(repo_id, filename, device="cpu"):
    print(f"Loading model {filename} from repo {repo_id}...")
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    
    model_size = os.path.getsize(model_path)
    with open(model_path, "rb") as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Model size: {model_size / 1024:.2f} KB")
    print(f"Model SHA256: {model_hash}")
    
    return PPO.load(model_path, device=device)
