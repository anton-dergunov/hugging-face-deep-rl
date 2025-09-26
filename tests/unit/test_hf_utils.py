import hashlib
from pathlib import Path
from typing import Union
import pytest

from rlcourse import hf_utils as hf


def _compute_sha256(path: Union[str, Path]) -> str:
    path = Path(path)
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_upload_model_file_monkeypatch(monkeypatch, tmp_path):
    """Upload should call API.upload_file for model, metadata.json and README.md."""
    uploaded = []

    def fake_upload_file(**kw):
        # huggingface_hub.HfApi.upload_file is called with path_or_fileobj, path_in_repo, repo_id, commit_message
        uploaded.append(kw.get("path_in_repo"))

    # Prevent real network repo creation
    monkeypatch.setattr(hf, "create_repo", lambda repo_id, exist_ok=True: None)
    # Replace HfApi.upload_file instance method
    monkeypatch.setattr(hf.HfApi, "upload_file", lambda self, **kw: fake_upload_file(**kw))
    # Pretend we're logged in
    monkeypatch.setattr(hf, "whoami", lambda: {"name": "me"})

    hub = hf.HuggingFaceModelHub()
    local = tmp_path / "dummy.bin"
    local.write_bytes(b"hello")

    hub.upload_model_file(repo_id="me/test", file=local, filename="model.bin")

    # check expected uploads took place
    assert "model.bin" in uploaded
    assert "metadata.json" in uploaded
    assert "README.md" in uploaded


def test_upload_model_file_file_not_found(monkeypatch, tmp_path):
    """Missing model file should raise FileNotFoundError."""
    monkeypatch.setattr(hf, "whoami", lambda: {"name": "me"})
    monkeypatch.setattr(hf, "create_repo", lambda repo_id, exist_ok=True: None)

    hub = hf.HuggingFaceModelHub()
    missing = tmp_path / "does-not-exist.bin"
    with pytest.raises(FileNotFoundError):
        hub.upload_model_file(repo_id="me/test", file=missing)


def test_default_readme_contains_expected_fields():
    """_default_readme should include repo id and usage snippet or readme_extra."""
    rd = hf.HuggingFaceModelHub._default_readme(
        repo_id="me/repo",
        env_id="CartPole-v1",
        algo="PPO",
        library="stable-baselines3",
        metrics={"mean_reward": 1.23, "std_reward": 0.45},
        readme_extra=None,
    )
    assert "me/repo" in rd
    assert "stable-baselines3" in rd
    assert "mean_reward" in rd or "Mean Reward" in rd


def test_download_model_file_monkeypatch(monkeypatch, tmp_path):
    """download_model_file should return the local path returned by hf_hub_download."""
    # Create a local file to pretend it was downloaded
    f = tmp_path / "model.bin"
    f.write_bytes(b"abc123")
    monkeypatch.setattr(hf, "hf_hub_download", lambda repo_id, filename, cache_dir=None: str(f))
    hub = hf.HuggingFaceModelHub()

    out = hub.download_model_file(repo_id="me/repo", filename="model.bin")
    assert Path(out).exists()
    assert _compute_sha256(out) == _compute_sha256(f)


def test_login_if_needed_when_already_logged_in(monkeypatch):
    """If whoami() returns info, login_if_needed should do nothing."""
    monkeypatch.setattr(hf, "whoami", lambda: {"name": "me"})
    # make sure interactive login functions raise if called (they shouldn't)
    monkeypatch.setattr(hf, "notebook_login", lambda: (_ for _ in ()).throw(RuntimeError("not expected")))
    monkeypatch.setattr(hf, "login", lambda: (_ for _ in ()).throw(RuntimeError("not expected")))

    hub = hf.HuggingFaceModelHub()
    # Should not raise
    hub.login_if_needed(interactive=False)


def test_login_if_needed_not_logged_noninteractive(monkeypatch):
    """If not logged in and non-interactive, raise RuntimeError."""
    monkeypatch.setattr(hf, "whoami", lambda: (_ for _ in ()).throw(Exception("no auth")))
    hub = hf.HuggingFaceModelHub()
    with pytest.raises(RuntimeError):
        hub.login_if_needed(interactive=False)


def test_delete_repo_monkeypatch(monkeypatch):
    """delete_repo should call the module-level delete_repo function."""
    called = {}

    def fake_delete_repo(repo_id, repo_type="model"):
        called["repo_id"] = repo_id
        called["repo_type"] = repo_type

    monkeypatch.setattr(hf, "delete_repo", fake_delete_repo)
    monkeypatch.setattr(hf, "whoami", lambda: {"name": "me"})  # avoid login error

    hub = hf.HuggingFaceModelHub()
    hub.delete_repo("me/test")
    assert called["repo_id"] == "me/test"
