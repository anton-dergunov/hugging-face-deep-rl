"""
Utilities for uploading/downloading model files to/from the Hugging Face Hub.

Class: HuggingFaceModelHub
- login_if_needed(interactive=True)
- upload_model_file(...)
- download_model_file(...)
- delete_repo(...)

TODO Make sure that the format of Model Cards is correct
https://huggingface.co/docs/hub/en/model-cards
"""

import json
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union

from huggingface_hub import HfApi, whoami, notebook_login, login, create_repo, hf_hub_download, delete_repo


class HuggingFaceModelHub:
    """Thin helper wrapper around huggingface_hub for uploading/downloading model files.
    """

    def __init__(self):
        self.api = HfApi()

    # ------------------ Authentication ------------------
    def login_if_needed(self, interactive: bool = True) -> None:
        """Ensure Hugging Face credentials are available.

        If credentials are present, do nothing. Otherwise - if interactive - try
        notebook_login() then login() as a fallback. If interactive is False and
        no credentials are found, raises RuntimeError.
        """
        try:
            whoami()
            print("✅ Hugging Face credentials already available.")
        except Exception:
            if interactive:
                try:
                    notebook_login()
                except Exception:
                    login()
            else:
                raise RuntimeError(
                    "No Hugging Face credentials found. Run login_if_needed(interactive=True) to log in."
                )

    # ------------------ Upload / Download ------------------
    def upload_model_file(
        self,
        repo_id: str,
        file: Union[str, Path],
        filename: Optional[str] = None,
        model_name: Optional[str] = None,
        library: Optional[str] = None,
        algo: Optional[str] = None,
        env_id: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        video_file: Optional[Union[str, Path]] = None,
        readme_extra: Optional[str] = None,
        commit_message: str = "Upload model",
    ) -> None:
        """Upload a pre-saved model file to the Hugging Face Hub.

        Parameters
        ----------
        repo_id: str
            Repo identifier, e.g. "username/repo".
        file: str | Path
            Path to the file to upload.
        filename: Optional[str]
            Name to use inside the repo. If None, basename(file) is used.
        model_name, library, algo, env_id: Optional[str]
            Metadata fields written to metadata.json.
            List of libraries: https://huggingface.co/docs/hub/en/models-libraries
        metrics: Optional[dict]
            Dictionary of metric name -> value to include in metadata/readme.
        video_file: Optional[str | Path]
            Path to the video file demonstrating the trained model in this environment.
        readme_extra: Optional[str]
            Extra text (markdown) appended into the README. Useful for library-specific
            loading examples. The default README contains a small pseudocode snippet.
        commit_message: str
            Commit message for file uploads.
        """
        file = Path(file)
        if not file.exists():
            raise FileNotFoundError(f"Model file not found: {file}")

        self.login_if_needed(interactive=False)

        # Create repo if missing
        try:
            create_repo(repo_id, exist_ok=True)
        except Exception as e:
            # create_repo may throw if networking/auth fails or repo exists; ignore if exists
            print(f"Warning: create_repo returned error (ignored): {e}")

        filename_in_repo = filename or file.name

        # Metadata
        metadata = {
            "model_name": model_name or repo_id.split("/")[-1],
            "library": library or "unknown",
            "algo": algo or "unknown",
            "env_id": env_id or "unknown",
        }
        if metrics:
            metadata["metrics"] = metrics

        # README
        readme_text = self._default_readme(
            repo_id=repo_id,
            env_id=env_id or "",
            algo=algo or "",
            library=library or "",
            metrics=metrics or {},
            readme_extra=readme_extra,
        )

        # Upload model file.
        # Print size and checksum for convenience
        size = os.path.getsize(file)
        with open(file, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        print(f"Uploading {filename_in_repo} to {repo_id} ... | size={size / 1024:.2f} KB | sha256={sha}")
        self.api.upload_file(
            path_or_fileobj=str(file),
            path_in_repo=filename_in_repo,
            repo_id=repo_id,
            commit_message=commit_message,
        )

        # Upload metadata.json
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            metadata_path = tmpdir / "metadata.json"
            with open(metadata_path, "w") as fh:
                json.dump(metadata, fh, indent=2)

            self.api.upload_file(
                path_or_fileobj=str(metadata_path),
                path_in_repo="metadata.json",
                repo_id=repo_id,
                commit_message=commit_message,
            )

        # Upload README.md
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            readme_path = tmpdir / "README.md"
            with open(readme_path, "w") as fh:
                fh.write(readme_text)

            self.api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message=commit_message,
            )

        # Upload video file is provided
        if video_file:
            self.api.upload_file(
                path_or_fileobj=str(video_file),
                path_in_repo="replay.mp4",
                repo_id=repo_id,
                commit_message=commit_message,
            )

        print(f"\n✅ Upload complete! View the model at:")
        print(f"https://huggingface.co/{repo_id}")

    def download_model_file(self, repo_id: str, filename: str, cache_dir: Optional[Union[str, Path]] = None) -> Path:
        """Download a file from a HF repo and return the local path.
        """
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)

        # Print size and checksum for convenience
        size = os.path.getsize(local_path)
        with open(local_path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        print(f"Downloaded {local_path} | size={size / 1024:.2f} KB | sha256={sha}")

        return Path(local_path)

    # ------------------ Repo manipulation ------------------
    def delete_repo(self, repo_id: str, repo_type: str = "model") -> None:
        """Delete a repo. Requires appropriate permissions.

        Use with care; primarily designed for test cleanup.
        """
        self.login_if_needed(interactive=False)
        delete_repo(repo_id=repo_id, repo_type=repo_type)

    # ------------------ Helpers ------------------
    @staticmethod
    def _default_readme(
        repo_id: str,
        env_id: str,
        algo: str,
        library: str,
        metrics: Dict[str, Any],
        readme_extra: Optional[str] = None,
    ) -> str:
        """Generate a default README.md content with a placeholder for a loading snippet.

        readme_extra can be a markdown string containing library-specific code examples
        (e.g. `python` fenced code block showing how to load the file with Stable-Baselines3).
        """
        yaml_metadata = f"""---
library_name: {library}
tags:
- reinforcement-learning
- deep-reinforcement-learning
- {algo or ''}
- {env_id or ''}
model-index:
- name: {repo_id}
  results:
  - task:
      type: reinforcement-learning
      name: Reinforcement Learning
    dataset:
      name: {env_id or 'unknown'}
      type: environment
    metrics:
    - type: mean_reward
      value: {metrics.get('mean_reward', 'NA') if isinstance(metrics, dict) else 'NA'}
      name: Mean Reward
    - type: std_reward
      value: {metrics.get('std_reward', 'NA') if isinstance(metrics, dict) else 'NA'}
      name: Std Reward
---
"""

        snippet = (
            readme_extra
            or (
                "(Replace the snippet below with your library-specific loading code.)\n\n"
                "```python\n# Example (pseudocode):\nfrom huggingface_hub import hf_hub_download\n\nmodel_path = hf_hub_download(repo_id=\"{repo_id}\", filename=\"model.zip\")\n# load with your library, e.g. MyLib.load(model_path)\n```\n"
            )
        )

        readme = f"""{yaml_metadata}
# {repo_id}

This model was trained with **{algo}** using **{library}** on **{env_id}**.

## Usage

{snippet}
"""
        return readme


# ------------------ Pytest-friendly quick test ------------------
# This test will be skipped if no HF credentials are present. Save this file as
# src/hf.py and run `pytest src/hf.py::test_hub_roundtrip -q` to execute it.


def _compute_sha256(path: Union[str, Path]) -> str:
    path = Path(path)
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_hub_roundtrip(tmp_path: Path):
    """Small integration test: upload a tiny file, download it back and verify checksum.

    Skips if the user is not authenticated (so it is safe to run in CI / locally without
    the token).
    """
    import pytest

    hub = HuggingFaceModelHub()

    # If not logged in, skip the test
    try:
        whoami()
    except Exception:
        pytest.skip("No Hugging Face credentials available - skipping hub roundtrip test")

    # Prepare a tiny file
    content = b"hello-hf-test\n"
    local_file = tmp_path / "model.bin"
    local_file.write_bytes(content)
    sha_before = _compute_sha256(local_file)

    # Repo id used for test - use your namespace if desired
    username = whoami().get("name") or whoami().get("user") or "test-user"
    repo_id = f"{username}/hf-model-utils-test"

    try:
        # upload
        hub.upload_model_file(
            repo_id=repo_id,
            file=local_file,
            filename="model.bin",
            model_name="hf-utils-test",
            library="none",
            algo="none",
            env_id="none",
            metrics={"mean_reward": 0.0},
            readme_extra="Test artifact uploaded by automated pytest",
        )

        # download
        downloaded = hub.download_model_file(repo_id=repo_id, filename="model.bin")
        sha_after = _compute_sha256(downloaded)

        assert sha_before == sha_after, "Downloaded file checksum mismatch"

    finally:
        # cleanup
        try:
            hub.delete_repo(repo_id)
        except Exception as e:
            print(f"Cleanup failed (you may need to remove the repo manually): {e}")


if __name__ == "__main__":
    # Quick manual smoke test
    h = HuggingFaceModelHub()
    try:
        h.login_if_needed(interactive=True)
        print("You can now call h.upload_model_file(...) or h.download_model_file(...)")
    except Exception as exc:
        print(f"Login failed or aborted: {exc}")
