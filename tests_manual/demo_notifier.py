"""
Manual integration test for notifier.py.
Run this file directly (python demo_notifier.py) after setting:

    export TG_BOT_TOKEN="123456:ABC..."

This will send test notifications to your Telegram chat.
"""

import os
import time
from rlcourse.notifier import send_telegram_message, Notify


BOT_TOKEN = os.getenv("TG_BOT_TOKEN")


def test_direct_send():
    print("\n🚀 [1/5] Direct send test\n" + "-" * 40)
    response = send_telegram_message("🔔 Manual test message", BOT_TOKEN)
    print("Response:", response)


def test_notify_success():
    print("\n✅ [2/5] Notify context manager success test\n" + "-" * 40)
    with Notify("ManualSuccessJob", BOT_TOKEN):
        time.sleep(0.2)
        print("Finished successfully.")


def test_notify_failure():
    print("\n💥 [3/5] Notify context manager failure test\n" + "-" * 40)
    try:
        with Notify("ManualFailureJob", BOT_TOKEN):
            time.sleep(0.2)
            raise ValueError("This is a manual test error")
    except ValueError:
        print("Failure processed successfully.")


def test_decorator_success():
    print("\n✅ [4/5] Notify decorator success test\n" + "-" * 40)
    @Notify(job_name="do_stuff")
    def job():
        time.sleep(0.2)
        print("Finished successfully.")
    job()


def test_decorator_failure():
    print("\n✅ [5/5] Notify decorator failure test\n" + "-" * 40)
    @Notify(job_name="do_stuff")
    def job():
        time.sleep(0.2)
        raise ValueError("This is a manual test error")
    try:
        job()
    except ValueError:
        print("Failure processed successfully.")


if __name__ == "__main__":
    if not BOT_TOKEN:
        print("Please set TG_BOT_TOKEN env vars before running.")
    else:
        test_direct_send()
        test_notify_success()
        test_notify_failure()
        test_decorator_success()
        test_decorator_failure()
