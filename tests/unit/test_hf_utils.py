import hashlib
import pytest
from pathlib import Path
from typing import Union

from huggingface_hub import whoami

from rlcourse import hf_utils as hf


def _compute_sha256(path: Union[str, Path]) -> str:
    path = Path(path)
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.mark.integration
def test_hub_roundtrip(tmp_path: Path):
    """Small integration test: upload a tiny file, download it back and verify checksum.

    Skips if the user is not authenticated (so it is safe to run in CI / locally without
    the token).
    """
    import pytest

    hub = hf.HuggingFaceModelHub()

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


def test_upload_model_file_monkeypatch(monkeypatch, tmp_path):
    uploaded = []
    def fake_upload_file(path_or_fileobj, path_in_repo, repo_id, commit_message):
        uploaded.append(path_in_repo)
    monkeypatch.setattr(hf.HfApi, 'upload_file', lambda self, **kw: fake_upload_file(**kw))
    monkeypatch.setattr(hf, 'create_repo', lambda repo_id, exist_ok=True: None)

    hub = hf.HuggingFaceModelHub()
    local = tmp_path / "dummy.bin"
    local.write_bytes(b"hello")
    hub.upload_model_file(repo_id="me/test", file=local, filename="model.bin")
    assert "model.bin" in uploaded
