"""
Notifier library for sending Telegram messages on job completion.

Features:
- Auto-detects runtime environment (local Jupyter, Colab, Kaggle, script).
- Retrieves bot token from env, Colab secrets, or Kaggle secrets.
- Formats informative notifications (job name, duration, success/failure).
- Safe: silently skips notifications if no bot token is configured.
- Provides:
  * send_telegram_message(message,..)
  * Notify decorator and context manager

Example Usage:
- Usage as context manager:
    with Notify(job_name="training"):
        long_running_work()
- Usage as decorator:
    @Notify(job_name="do_stuff")
    def my_fn(...):
        ...
- Sending Telegram message manually
    send_telegram_message(message)

TODO Describe Bot setup
TODO Also add start/finish time
"""

import os
import sys
import time
import json
import platform
import traceback
import socket
import getpass
from contextlib import ContextDecorator
from typing import Optional, Callable, Any, Dict
import inspect
import requests
import urllib.parse


TG_BOT_TOKEN_NAME = "TG_BOT_TOKEN"

# Default cache location for chat_id and other simple settings
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".jup_notif")
DEFAULT_CACHE_FILE = os.path.join(DEFAULT_CACHE_DIR, "config.json")


# -----------------------------
# Helper: runtime/source detection
# -----------------------------

def detect_environment() -> str:
    """Return a short human-friendly description of the execution environment."""
    # Kaggle detection (Kaggle sets this env var in kernels)
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None or os.environ.get("KAGGLE_URL_BASE") is not None:
        return "Kaggle"
    # Google Colab detection
    try:
        import google.colab  # type: ignore
        return "Google Colab"
    except Exception:
        pass
    # General local machine
    user = getpass.getuser()
    host = socket.gethostname()
    system = platform.system()
    machine = platform.machine()
    return f"{user}@{host} ({system}/{machine})"


def get_notebook_or_script_name() -> str:
    """
    Try to return a meaningful name for the current execution context.
    Works for local Jupyter, Colab, Kaggle, or plain Python scripts.

    Returns:
        str: The notebook/script name or None if unknown.
    """
    # --- Try ipynbname (local Jupyter or Colab) ---
    try:
        import ipynbname

        nb_path = ipynbname.path()
        nb_name = str(nb_path)

        if "fileId=" in nb_name:  # Colab pseudo-path
            parsed = urllib.parse.unquote(nb_name)
            # Extract after last '/' or '='
            return parsed.split("/")[-1].split("=")[-1]
        else:
            return nb_path.name
    except Exception:
        pass

    # --- Script detection ---
    try:
        frame = inspect.stack()[-1]
        filename = frame.filename
        if filename and filename != "<stdin>" and filename != "<frozen runpy>":
            return os.path.basename(filename)
    except Exception:
        pass

    # --- sys.argv fallback ---
    if len(sys.argv) > 0:
        return os.path.basename(sys.argv[0])

    return None


# -----------------------------
# Helper: read secrets to get TG_BOT_TOKEN
# -----------------------------

def _read_token_from_colab() -> str | None:
    try:
        from google.colab import userdata  # type: ignore
        return userdata.get(TG_BOT_TOKEN_NAME)
    except Exception:
        return None


def _read_token_from_kaggle() -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        user_secrets = UserSecretsClient()
        return user_secrets.get_secret(TG_BOT_TOKEN_NAME)
    except Exception:
        return None


def get_bot_token_from_env_or_secrets() -> str | None:
    """Try to load TG_BOT_TOKEN from env, Colab, or Kaggle secrets."""
    token = os.getenv(TG_BOT_TOKEN_NAME)
    if token:
        return token

    token = _read_token_from_colab()
    if token:
        return token

    token = _read_token_from_kaggle()
    if token:
        return token

    return None


# -----------------------------
# Helper: fetch TG_CHAT_ID
# -----------------------------

def ensure_cache_dir(path: str = DEFAULT_CACHE_DIR):
    os.makedirs(path, exist_ok=True)


def _read_cache(cache_file: str = DEFAULT_CACHE_FILE) -> Dict[str, Any]:
    try:
        with open(cache_file, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_cache(data: Dict[str, Any], cache_file: str = DEFAULT_CACHE_FILE):
    ensure_cache_dir(os.path.dirname(cache_file))
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


def get_or_fetch_chat_id(bot_token: str,
                         cache_file: str = DEFAULT_CACHE_FILE,
                         force_refresh: bool = False) -> Optional[int]:
    """
    Return a chat_id to use with the bot. Strategy:
      - If env var TG_CHAT_ID exists, use it (no network call)
      - Else, if cache exists and not force_refresh, use cached chat_id
      - Else, call getUpdates once, inspect result for a chat id, cache it and return.
    Note: This requires that the user has already sent at least one message to the bot
    (or the bot was added to a group and saw a message there).
    """
    # 1) env var override
    env_chat = os.environ.get("TG_CHAT_ID")
    if env_chat:
        try:
            return int(env_chat)
        except Exception:
            # Fall through to other methods
            pass

    cache = _read_cache(cache_file)
    if not force_refresh and cache.get("chat_id"):
        try:
            return int(cache["chat_id"])
        except Exception:
            pass

    # No cached id -> try to fetch via getUpdates
    if not bot_token:
        raise ValueError("bot_token is required to fetch chat_id if not cached and TG_CHAT_ID not set.")

    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    try:
        r = requests.get(url, timeout=10)
        j = r.json()
    except Exception as e:
        raise RuntimeError(f"Failed to call Telegram getUpdates: {e}")

    # navigate results for a chat id
    results = j.get("result", [])
    for item in results[::-1]:  # walk from newest to oldest
        # message may live in different subkeys (message, edited_message, channel_post)
        for key in ("message", "edited_message", "channel_post"):
            msg = item.get(key)
            if not msg:
                continue
            chat = msg.get("chat", {})
            cid = chat.get("id")
            if cid:
                # cache and return
                cache["chat_id"] = int(cid)
                cache["bot_token_sample"] = str(bot_token)[:8]  # not storing full token
                _write_cache(cache, cache_file)
                return int(cid)

    # If no chat found:
    raise RuntimeError("No chat_id found in bot updates. Make sure you have sent a message to the bot at least once.")


# -----------------------------
# Telegram API
# -----------------------------

def send_telegram_message(text: str,
                          bot_token: Optional[str] = None,
                          chat_id: Optional[int] = None,
                          parse_mode: str = "HTML") -> Dict[str, Any]:
    """
    Send a message via Telegram bot API.
    - bot_token: if None, read from env TG_BOT_TOKEN
    - chat_id: if None, try env TG_CHAT_ID or cached value via get_or_fetch_chat_id
    - parse_mode: 'HTML' recommended by default to avoid escaping issues with MarkdownV2
    Returns parsed JSON response.
    """
    bot_token = bot_token or get_bot_token_from_env_or_secrets()
    if not bot_token:
        raise ValueError("bot_token not provided and TG_BOT_TOKEN not set.")

    if chat_id is None:
        # try env or cached; may raise if it cannot find any chat id
        chat_id = get_or_fetch_chat_id(bot_token)

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode
    }

    # Use a short timeout so notebook doesn't hang if Telegram is unreachable
    resp = requests.post(url, json=payload, timeout=10)
    try:
        return resp.json()
    except Exception:
        return {"ok": False, "status_code": resp.status_code, "text": resp.text}


# -----------------------------
# Notification wrapper
# -----------------------------

def _format_duration(seconds: float) -> str:
    # human friendly duration like "1h 2m 3.45s" or "0.32s"
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h}h {m}m {round(s,2)}s"
    if seconds >= 60:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}m {round(s,2)}s"
    if seconds >= 1:
        return f"{round(seconds,2)}s"
    return f"{round(seconds*1000,1)}ms"


def _build_message(job_name: str,
                   elapsed: float,
                   success: bool,
                   env_desc: Optional[str] = None,
                   source: Optional[str] = None,
                   extra_text: Optional[str] = None,
                   error_short: Optional[str] = None) -> str:
    """
    Build a nicely formatted HTML message for Telegram.
    """
    status_emoji = "✅" if success else "❌"
    lines = []
    lines.append(f"<b>{status_emoji} Job: {job_name}</b>")
    lines.append(f"Elapsed: {_format_duration(elapsed)}")
    lines.append(f"Env: <code>{env_desc}</code>")
    if source:
        lines.append(f"Source: <code>{source}</code>")
    if error_short:
        # short one-line error summary
        lines.append(f"<pre>Error: {error_short}</pre>")
    if extra_text:
        # allow user provided multi-line text. We place inside <pre> to keep formatting.
        lines.append("<pre>")
        lines.append(extra_text)
        lines.append("</pre>")
    return "\n".join(lines)


class Notify(ContextDecorator):
    """
    Context manager and decorator to notify via Telegram when a block finishes.
    Usage as context manager:
        with Notify(job_name="training", min_duration=0.1):
            long_running_work()
    Usage as decorator:
        @Notify(job_name="do_stuff")
        def my_fn(...):
            ...
    Parameters:
      - job_name: short name printed in the notification
      - bot_token/chat_id: optional overrides
      - min_duration: don't notify for jobs shorter than this (in seconds)
      - on_success: optional callback to call with (text, payload) right before sending (can be used to also email)
      - on_failure: optional callback used on exception
    """
    def __init__(self,
                 job_name: str = "job",
                 bot_token: Optional[str] = None,
                 chat_id: Optional[int] = None,
                 min_duration: float = 0.1,
                 cache_file: str = DEFAULT_CACHE_FILE,
                 on_success: Optional[Callable[[str, Dict[str,Any]], Any]] = None,
                 on_failure: Optional[Callable[[str, Dict[str,Any]], Any]] = None,
                 extra_text: Optional[str] = None):
        self.job_name = job_name
        self.bot_token = bot_token or os.environ.get("TG_BOT_TOKEN")
        self.chat_id = chat_id or (os.environ.get("TG_CHAT_ID") and int(os.environ.get("TG_CHAT_ID")))
        self.min_duration = float(min_duration)
        self.cache_file = cache_file
        self.on_success = on_success
        self.on_failure = on_failure
        self.extra_text = extra_text

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.bot_token:
            # no-op if not configured
            return False

        elapsed = time.time() - self._start
        success = exc_type is None
        # skip notifications for very short runs
        if elapsed < self.min_duration:
            # However, still call failure hook if there was an exception
            if not success and self.on_failure:
                self.on_failure("skipped_due_to_short_duration", {})
            return False  # do not suppress exceptions

        # build message
        error_short = None
        if not success:
            error_short = str(exc)
        text = _build_message(self.job_name, elapsed, success,
                              env_desc=detect_environment(),
                              source=get_notebook_or_script_name(),
                              extra_text=self.extra_text,
                              error_short=error_short)
        # allow hooks
        payload = {"job_name": self.job_name, "elapsed": elapsed, "success": success}
        if success and self.on_success:
            try:
                self.on_success(text, payload)
            except Exception:
                pass
        if (not success) and self.on_failure:
            try:
                self.on_failure(text, payload)
            except Exception:
                pass
        # finally try to send via Telegram (best effort; errors are printed but not raised)
        try:
            send_telegram_message(text, bot_token=self.bot_token, chat_id=self.chat_id, parse_mode="HTML")
        except Exception as e:
            print("Failed to send telegram notification:", e)
        # do not suppress exceptions
        return False


# -----------------------------
# Manual integration test (not CI)
# -----------------------------

def send_and_verify_integration(
    bot_token: str, chat_id: str | None = None, test_message: str = "🔔 Test message"
):
    """
    Send a test message and verify it appears in getUpdates.
    For manual use only (not suitable for CI).
    """
    resp = send_telegram_message(test_message, bot_token, chat_id)

    # Fetch recent updates
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    updates = requests.get(url, timeout=10).json()
    texts = [r["message"]["text"] for r in updates.get("result", []) if "message" in r]

    return test_message in texts
