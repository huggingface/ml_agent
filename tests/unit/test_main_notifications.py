"""Tests for CLI attention notifications in agent/main.py."""

import sys
from pathlib import Path
from unittest.mock import patch

# Add repo root to sys.path so `agent` imports resolve in tests.
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from agent.main import _desktop_notification_command, _notify_attention_needed  # noqa: E402


def test_desktop_notification_command_uses_osascript_on_macos():
    with patch("agent.main.sys.platform", "darwin"), patch(
        "agent.main.which", return_value="/usr/bin/osascript"
    ):
        cmd = _desktop_notification_command("Need input", "Approve item")

    assert cmd is not None
    assert cmd[:2] == ["osascript", "-e"]
    assert "display notification" in cmd[2]


def test_desktop_notification_command_uses_notify_send_on_linux():
    with patch("agent.main.sys.platform", "linux"), patch(
        "agent.main.which", return_value="/usr/bin/notify-send"
    ):
        cmd = _desktop_notification_command("Need input", "Approve item")

    assert cmd == ["notify-send", "Need input", "Approve item"]


def test_notify_attention_needed_falls_back_to_bell_when_auto_has_no_desktop_command():
    with patch("agent.main._desktop_notification_command", return_value=None), patch(
        "agent.main._ring_terminal_bell"
    ) as bell:
        _notify_attention_needed(enabled=True, method="auto")

    bell.assert_called_once()


def test_notify_attention_needed_uses_desktop_command_when_available():
    with patch(
        "agent.main._desktop_notification_command",
        return_value=["notify-send", "title", "msg"],
    ), patch("agent.main.subprocess.run") as run, patch(
        "agent.main._ring_terminal_bell"
    ) as bell:
        _notify_attention_needed(enabled=True, method="desktop")

    run.assert_called_once()
    bell.assert_not_called()
