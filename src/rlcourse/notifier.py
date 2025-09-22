"""
Small, dependency-light notification helper for notebooks and scripts.

Design highlights:
- Uses BOT token from env var or explicit argument.
- Chat ID is resolved automatically (calls getUpdates once) and cached locally in a small JSON file
  under ~/.jup_notif/config.json. The cache can be overridden by env var TG_CHAT_ID or via parameter.
- Provides:
  * send_telegram_message(text, bot_token=None, chat_id=None, parse_mode="HTML")
  * get_or_fetch_chat_id(bot_token, cache_path=...)
  * notify_on_completion decorator and Notify context manager
- Message format includes job name, elapsed time, hostname/environment, success/failure, and optional custom text.
- Uses HTML parse_mode for Telegram messages to avoid complex escaping for MarkdownV2.
- By default, skips notifications for very short jobs (min_duration=0.1s); this is configurable.
"""

import os
import json
import time
import platform
import socket
import getpass
from contextlib import ContextDecorator
from typing import Optional, Callable, Any, Dict
import requests

# Default cache location for chat_id and other simple settings
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".jup_notif")
DEFAULT_CACHE_FILE = os.path.join(DEFAULT_CACHE_DIR, "config.json")

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

def send_telegram_message(text: str,
                          bot_token: Optional[str] = None,
                          chat_id: Optional[int] = None,
                          parse_mode: str = "HTML",
                          extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Send a message via Telegram bot API.
    - bot_token: if None, read from env TG_BOT_TOKEN
    - chat_id: if None, try env TG_CHAT_ID or cached value via get_or_fetch_chat_id
    - parse_mode: 'HTML' recommended by default to avoid escaping issues with MarkdownV2
    - extra: dict with additional fields to pass through to Telegram API (e.g. disable_web_page_preview)
    Returns parsed JSON response.
    """
    bot_token = bot_token or os.environ.get("TG_BOT_TOKEN")
    if not bot_token:
        raise ValueError("bot_token not provided and TG_BOT_TOKEN not set.")

    if chat_id is None:
        # try env or cached
        env_chat = os.environ.get("TG_CHAT_ID")
        if env_chat:
            try:
                chat_id = int(env_chat)
            except Exception:
                chat_id = None

    if chat_id is None:
        # may raise if it cannot find any chat id
        chat_id = get_or_fetch_chat_id(bot_token)

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode
    }
    if extra:
        payload.update(extra)
    # Use a short timeout so notebook doesn't hang if Telegram is unreachable
    resp = requests.post(url, json=payload, timeout=10)
    try:
        return resp.json()
    except Exception:
        return {"ok": False, "status_code": resp.status_code, "text": resp.text}


def _format_duration(seconds: float) -> str:
    # human friendly duration like "1h2m3.45s" or "0.32s"
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h}h{m}m{round(s,2)}s"
    if seconds >= 60:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}m{round(s,2)}s"
    if seconds >= 1:
        return f"{round(seconds,2)}s"
    return f"{round(seconds*1000,1)}ms"


def _build_message(job_name: str,
                   elapsed: float,
                   success: bool,
                   env_desc: Optional[str] = None,
                   extra_text: Optional[str] = None,
                   error_short: Optional[str] = None) -> str:
    """
    Build a nicely formatted HTML message for Telegram.
    Example output (HTML mode):
    <b>✅ Job: my_training</b>
    <pre>Elapsed: 12.34s</pre>
    <i>Env: Google Colab</i>
    <pre>Note: custom text...</pre>
    """
    status_emoji = "✅" if success else "❌"
    env_desc = env_desc or detect_environment()
    lines = []
    lines.append(f"<b>{status_emoji} Job: {job_name}</b>")
    lines.append(f"Elapsed: {_format_duration(elapsed)}")
    lines.append(f"Env: <code>{env_desc}</code>")
    if error_short:
        # short one-line error summary
        lines.append(f"<pre>Error: {error_short}</pre>")
    if extra_text:
        # allow user provided multi-line text. We place inside <pre> to keep formatting.
        lines.append("<pre>")
        lines.append(extra_text)
        lines.append("</pre>")
    # small footer with machine info
    try:
        lines.append(f"Host: {socket.gethostname()} | User: {getpass.getuser()}")
    except Exception:
        pass
    return "\n".join(lines)


class Notify(ContextDecorator):
    """
    Context manager and decorator to notify via Telegram (and future backends) when a block finishes.
    Usage as context manager:
        with Notify(job_name="training", bot_token=..., min_duration=0.1):
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
      - extra_text: additional custom text (string) to include in message
      - parse_mode: Telegram parse_mode, default "HTML"
    """
    def __init__(self,
                 job_name: str = "job",
                 bot_token: Optional[str] = None,
                 chat_id: Optional[int] = None,
                 min_duration: float = 0.1,
                 cache_file: str = DEFAULT_CACHE_FILE,
                 on_success: Optional[Callable[[str, Dict[str,Any]], Any]] = None,
                 on_failure: Optional[Callable[[str, Dict[str,Any]], Any]] = None,
                 extra_text: Optional[str] = None,
                 parse_mode: str = "HTML"):
        self.job_name = job_name
        self.bot_token = bot_token or os.environ.get("TG_BOT_TOKEN")
        self.chat_id = chat_id or (os.environ.get("TG_CHAT_ID") and int(os.environ.get("TG_CHAT_ID")))
        self.min_duration = float(min_duration)
        self.cache_file = cache_file
        self.on_success = on_success
        self.on_failure = on_failure
        self.extra_text = extra_text
        self.parse_mode = parse_mode

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
            send_telegram_message(text, bot_token=self.bot_token, chat_id=self.chat_id, parse_mode=self.parse_mode)
        except Exception as e:
            print("Failed to send telegram notification:", e)
        # do not suppress exceptions
        return False
