
"""
Unit tests for notifier.py that avoid hitting the real Telegram API by mocking requests.
Run with pytest.
"""

import os
import json
import tempfile
import builtins
import requests
import notifier
from unittest import mock

def test_format_duration():
    assert notifier._format_duration(0.001)  # returns ms
    assert notifier._format_duration(0.5).endswith("ms") or notifier._format_duration(0.5).endswith("s")
    assert "h" in notifier._format_duration(3600)

def test_build_message_contains_job_and_env():
    msg = notifier._build_message("my_job", 1.23, True, env_desc="TestEnv", extra_text="hey")
    assert "my_job" in msg
    assert "TestEnv" in msg
    assert "hey" in msg

@mock.patch("notifier.requests.post")
@mock.patch("notifier.requests.get")
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

def test_notify_context_manager_success(monkeypatch, tmp_path):
    # stub send_telegram_message so we don't call network
    sent = {}
    def fake_send(text, bot_token=None, chat_id=None, parse_mode="HTML", extra=None):
        sent['text'] = text
        return {"ok": True}
    monkeypatch.setattr(notifier, "send_telegram_message", fake_send)

    # use the context manager
    with notifier.Notify(job_name="t1", min_duration=0.0, extra_text="details"):
        # quick operation
        x = 1+1
    assert "Job: t1" in sent['text']

def test_notify_context_manager_failure(monkeypatch):
    sent = {}
    def fake_send(text, bot_token=None, chat_id=None, parse_mode="HTML", extra=None):
        sent['text'] = text
        return {"ok": True}
    monkeypatch.setattr(notifier, "send_telegram_message", fake_send)

    try:
        with notifier.Notify(job_name="t_err", min_duration=0.0):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert "t_err" in sent['text']
    assert ("Error" in sent['text'] or "❌" in sent['text'])
