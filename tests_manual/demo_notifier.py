"""
Manual integration test for notifier.py.
Run this file directly (python demo_notifier.py) after setting:

    export TG_BOT_TOKEN="123456:ABC..."

This will send test notifications to your Telegram chat.
"""

import time
from rlcourse import notifier


def run_manual_tests():
    # 1) Simple direct message
    print("Sending direct Telegram message...")
    resp = notifier.send_telegram_message("🔔 Manual test message")
    print("Response:", resp)

    # 2) Notify context manager (success)
    print("Running Notify success test...")
    with notifier.Notify(job_name="manual_success", min_duration=0.1):
        time.sleep(0.2)
        print("Finished successfully.")

    # 3) Notify context manager (failure)
    print("Running Notify failure test...")
    try:
        with notifier.Notify(job_name="manual_failure", min_duration=0.1):
            time.sleep(0.2)
            raise RuntimeError("Deliberate test failure 🚨")
    except RuntimeError:
        print("Failure triggered, notification should have been sent.")


if __name__ == "__main__":
    run_manual_tests()
