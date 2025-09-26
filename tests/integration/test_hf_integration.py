import hashlib
from pathlib import Path
from typing import Union

import pytest
from huggingface_hub import whoami

from rlcourse import hf_utils as hf


def _compute_sha256(path: Union[str, Path]) -> str:
    path = Path(path)
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.mark.integration
def test_hub_roundtrip(tmp_path: Path):
    """Upload a tiny file to the hub, download it back and verify checksum.
    Skips if no HF credentials are present.
    """
    hub = hf.HuggingFaceModelHub()

    # If not logged in, skip
    try:
        whoami()
    except Exception:
        pytest.skip("No Hugging Face credentials available - skipping hub roundtrip test")

    content = b"hello-hf-test\n"
    local_file = tmp_path / "model.bin"
    local_file.write_bytes(content)
    sha_before = _compute_sha256(local_file)

    username = whoami().get("name") or whoami().get("user") or "test-user"
    repo_id = f"{username}/hf-model-utils-test"

    try:
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
            commit_message="pytest integration test upload",
        )

        downloaded = hub.download_model_file(repo_id=repo_id, filename="model.bin")
        sha_after = _compute_sha256(downloaded)
        assert sha_before == sha_after
    finally:
        # best-effort cleanup
        try:
            hub.delete_repo(repo_id)
        except Exception as e:
            print(f"Cleanup failed: {e}")
