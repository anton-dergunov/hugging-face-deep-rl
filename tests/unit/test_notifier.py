"""
Unit tests for notifier.py that avoid hitting the real Telegram API by mocking requests.
"""

import json
import types
import sys
from unittest import mock

from rlcourse import notifier


def test_format_duration_exact():
    assert notifier._format_duration(0.0005) == "0.5ms"
    assert notifier._format_duration(0.5) == "500.0ms"
    assert notifier._format_duration(2) == "2s"
    assert notifier._format_duration(65) == "1m 5s"
    assert notifier._format_duration(3661.2) == "1h 1m 1.2s"


def test_build_message_contains_expected_fields():
    msg = notifier._build_message(
        "train_model", 120.3, True, env_desc="CI", source="script.py", extra_text="extra"
    )
    assert "train_model" in msg
    assert "Succeeded" in msg
    assert "120.3" not in msg  # should be formatted, not raw
    assert "2m 0.3s" in msg
    assert "Env: <code>CI</code>" in msg
    assert "script.py" in msg
    assert "extra" in msg


def test_get_notebook_or_script_name_notebook(monkeypatch):
    # Simulate ipynbname returning a notebook path
    fake_ipynb = types.SimpleNamespace(path=lambda: types.SimpleNamespace(name="test_nb.ipynb"))
    monkeypatch.setitem(sys.modules, "ipynbname", fake_ipynb)
    assert notifier.get_notebook_or_script_name() == "test_nb.ipynb"


def test_get_notebook_or_script_name_script(monkeypatch):
    # Force ipynbname import to fail
    monkeypatch.setitem(sys.modules, "ipynbname", None)
    # Simulate inspect.stack returning a frame with a filename
    monkeypatch.setattr(notifier.inspect, "stack", lambda: [types.SimpleNamespace(filename="script.py")])
    assert notifier.get_notebook_or_script_name() == "script.py"


def test_get_notebook_or_script_name_unknown(monkeypatch):
    # Break ipynbname
    monkeypatch.setitem(sys.modules, "ipynbname", None)
    # Break inspect.stack
    monkeypatch.setattr(notifier.inspect, "stack", lambda: [types.SimpleNamespace(filename="<stdin>")])
    # Simulate empty sys.argv
    monkeypatch.setattr(notifier.sys, "argv", [])
    assert notifier.get_notebook_or_script_name() is None


@mock.patch("rlcourse.notifier.requests.post")
@mock.patch("rlcourse.notifier.requests.get")
def test_get_or_fetch_chat_id_and_send(mock_get, mock_post, tmp_path):
    # prepare a fake getUpdates payload that contains a chat id
    payload = {"ok": True, "result": [{"update_id": 1, "message": {"chat": {"id": 99999}}}]}
    mock_get.return_value.json.return_value = payload
    mock_get.return_value.status_code = 200

    # use a temporary cache file to avoid touching user's home dir
    cache_file = tmp_path / "cfg.json"
    bot_token = "TEST_TOKEN_123"
    # call get_or_fetch_chat_id
    cid = notifier.get_or_fetch_chat_id(bot_token, cache_file=str(cache_file))
    assert cid == 99999
    # ensure cache written
    data = json.loads(cache_file.read_text())
    assert data["chat_id"] == 99999

    # now test send_telegram_message will call requests.post
    mock_post.return_value.json.return_value = {"ok": True}
    resp = notifier.send_telegram_message("hi", bot_token=bot_token, chat_id=cid)
    assert mock_post.called
    assert resp.get("ok") is True


def monkey_patch_send_messages(monkeypatch):
    sent_messages = []

    # stub send_telegram_message so we don't call network
    def fake_send(*args, **kwargs):
        text = kwargs.get("text", args[0] if args else None)
        sent_messages.append(text)
        return {"ok": True}

    monkeypatch.setattr(notifier, "send_telegram_message", fake_send)
    return sent_messages


def test_notify_context_manager_success(monkeypatch):
    sent_messages = monkey_patch_send_messages(monkeypatch)

    # use the context manager
    with notifier.Notify(job_name="t1", bot_token="DUMMY", min_duration=0.0, extra_text="details"):
        # quick operation
        x = 1+1
    assert "Succeeded: t1" in sent_messages[0]


def test_notify_context_manager_failure(monkeypatch):
    sent_messages = monkey_patch_send_messages(monkeypatch)

    try:
        with notifier.Notify(job_name="t_err", bot_token="DUMMY", min_duration=0.0):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert "Failed: t_err" in sent_messages[0]
    assert "Error" in sent_messages[0]


def test_notify_decorator_success(monkeypatch, tmp_path):
    sent_messages = monkey_patch_send_messages(monkeypatch)

    # use the decorator
    @notifier.Notify(job_name="t1", bot_token="DUMMY", min_duration=0.0, extra_text="details")
    def job():
        # quick operation
        x = 1+1
    job()
    assert "Succeeded: t1" in sent_messages[0]


def test_notify_decorator_failure(monkeypatch):
    sent_messages = monkey_patch_send_messages(monkeypatch)

    @notifier.Notify(job_name="t_err", bot_token="DUMMY", min_duration=0.0)
    def job():
        raise RuntimeError("boom")
    try:
        job()
    except RuntimeError:
        pass
    assert "Failed: t_err" in sent_messages[0]
    assert "Error" in sent_messages[0]


def test_min_duration_filter(monkeypatch):
    sent_messages = monkey_patch_send_messages(monkeypatch)

    # Very short job (< min_duration=1.0s)
    with notifier.Notify("TooFastJob", "dummy", "dummy", min_duration=1.0):
        pass

    assert sent_messages == [], "No messages should be sent for too short jobs"
