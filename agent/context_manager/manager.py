"""
Context management for conversation history
"""

import logging
import os
import zoneinfo
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template
from litellm import Message, acompletion

logger = logging.getLogger(__name__)

_HF_WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
_HF_WHOAMI_TIMEOUT = 5  # seconds


def _get_hf_username(hf_token: str | None = None) -> str:
    """Return the HF username for the given token.

    Uses subprocess + curl to avoid Python HTTP client IPv6 issues that
    cause 40+ second hangs (httpx/urllib try IPv6 first which times out
    at OS level before falling back to IPv4 — the "Happy Eyeballs" problem).
    """
    import json
    import subprocess
    import time as _t

    if not hf_token:
        logger.warning("No hf_token provided, using 'unknown' as username")
        return "unknown"

    t0 = _t.monotonic()
    try:
        result = subprocess.run(
            [
                "curl",
                "-s",
                "-4",  # force IPv4
                "-m",
                str(_HF_WHOAMI_TIMEOUT),  # max time
                "-H",
                f"Authorization: Bearer {hf_token}",
                _HF_WHOAMI_URL,
            ],
            capture_output=True,
            text=True,
            timeout=_HF_WHOAMI_TIMEOUT + 2,
        )
        t1 = _t.monotonic()
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            username = data.get("name", "unknown")
            logger.info(f"HF username resolved to '{username}' in {t1 - t0:.2f}s")
            return username
        else:
            logger.warning(
                f"curl whoami failed (rc={result.returncode}) in {t1 - t0:.2f}s"
            )
            return "unknown"
    except Exception as e:
        t1 = _t.monotonic()
        logger.warning(f"HF whoami failed in {t1 - t0:.2f}s: {e}")
        return "unknown"


class ContextManager:
    """Manages conversation context and message history for the agent"""

    def __init__(
        self,
        max_context: int = 180_000,
        compact_size: float = 0.1,
        untouched_messages: int = 5,
        tool_specs: list[dict[str, Any]] | None = None,
        prompt_file_suffix: str = "system_prompt_v3.yaml",
        hf_token: str | None = None,
    ):
        self.system_prompt = self._load_system_prompt(
            tool_specs or [],
            prompt_file_suffix="system_prompt_v3.yaml",
            hf_token=hf_token,
        )
        self.max_context = max_context
        self.compact_size = int(max_context * compact_size)
        self.context_length = len(self.system_prompt) // 4
        self.untouched_messages = untouched_messages
        self.items: list[Message] = [Message(role="system", content=self.system_prompt)]

    def _load_system_prompt(
        self,
        tool_specs: list[dict[str, Any]],
        prompt_file_suffix: str = "system_prompt.yaml",
        hf_token: str | None = None,
    ):
        """Load and render the system prompt from YAML file with Jinja2"""
        prompt_file = Path(__file__).parent.parent / "prompts" / f"{prompt_file_suffix}"

        with open(prompt_file, "r") as f:
            prompt_data = yaml.safe_load(f)
            template_str = prompt_data.get("system_prompt", "")

        # Get current date and time
        tz = zoneinfo.ZoneInfo("Europe/Paris")
        now = datetime.now(tz)
        current_date = now.strftime("%d-%m-%Y")
        current_time = now.strftime("%H:%M:%S.%f")[:-3]
        current_timezone = f"{now.strftime('%Z')} (UTC{now.strftime('%z')[:3]}:{now.strftime('%z')[3:]})"

        # Get HF user info from OAuth token
        hf_user_info = _get_hf_username(hf_token)

        template = Template(template_str)
        return template.render(
            tools=tool_specs,
            num_tools=len(tool_specs),
            current_date=current_date,
            current_time=current_time,
            current_timezone=current_timezone,
            hf_user_info=hf_user_info,
        )

    def add_message(self, message: Message, token_count: int = None) -> None:
        """Add a message to the history"""
        if token_count:
            self.context_length = token_count
        self.items.append(message)

    def get_messages(self) -> list[Message]:
        """Get all messages for sending to LLM.

        Defensive gateway — fixes context corruption before every LLM call:
        1. Sanitizes malformed tool_call arguments (truncated JSON)
        2. Adds stub results for dangling tool_calls (missing tool results)
        """
        self._sanitize_tool_calls()
        self._patch_dangling_tool_calls()
        return self.items

    @staticmethod
    def _tc_id(tc) -> str:
        """Get tool_call id from either a ToolCall object or a dict."""
        if isinstance(tc, dict):
            return tc.get("id", "")
        return tc.id

    @staticmethod
    def _tc_func_name(tc) -> str:
        """Get function name from either a ToolCall object or a dict."""
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            return fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
        return tc.function.name

    @staticmethod
    def _tc_func_args(tc) -> str:
        """Get function arguments string from either a ToolCall object or a dict."""
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            return fn.get("arguments", "{}") if isinstance(fn, dict) else getattr(fn, "arguments", "{}")
        return tc.function.arguments

    @staticmethod
    def _tc_set_func_args(tc, value: str) -> None:
        """Set function arguments on either a ToolCall object or a dict."""
        if isinstance(tc, dict):
            fn = tc.get("function")
            if isinstance(fn, dict):
                fn["arguments"] = value
            else:
                fn.arguments = value
        else:
            tc.function.arguments = value

    def _sanitize_tool_calls(self) -> None:
        """Fix malformed tool_call arguments across all assistant messages.

        When output is truncated (finish_reason=length), tool call arguments
        can be incomplete JSON.  LLM APIs reject messages with invalid JSON
        in function.arguments, breaking all subsequent calls.
        """
        import json

        for msg in self.items:
            if getattr(msg, "role", None) != "assistant":
                continue
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                continue
            for tc in tool_calls:
                try:
                    json.loads(self._tc_func_args(tc))
                except (json.JSONDecodeError, TypeError, ValueError):
                    logger.warning(
                        "Sanitizing malformed arguments for tool_call %s (%s)",
                        self._tc_id(tc), self._tc_func_name(tc),
                    )
                    self._tc_set_func_args(tc, "{}")

    def _patch_dangling_tool_calls(self) -> None:
        """Add stub tool results for any tool_calls that lack a matching result.

        Scans backwards to find the last assistant message with tool_calls,
        which may not be items[-1] if some tool results were already added.
        """
        if not self.items:
            return

        # Find the last assistant message with tool_calls
        assistant_msg = None
        for i in range(len(self.items) - 1, -1, -1):
            msg = self.items[i]
            if getattr(msg, "role", None) == "assistant" and getattr(msg, "tool_calls", None):
                assistant_msg = msg
                break
            # Stop scanning once we hit a user message — anything before
            # that belongs to a previous (complete) turn.
            if getattr(msg, "role", None) == "user":
                break

        if not assistant_msg:
            return

        answered_ids = {
            getattr(m, "tool_call_id", None)
            for m in self.items
            if getattr(m, "role", None) == "tool"
        }
        for tc in assistant_msg.tool_calls:
            tc_id = self._tc_id(tc)
            if tc_id not in answered_ids:
                self.items.append(
                    Message(
                        role="tool",
                        content="Tool was not executed (interrupted or error).",
                        tool_call_id=tc_id,
                        name=self._tc_func_name(tc),
                    )
                )

    def undo_last_turn(self) -> bool:
        """Remove the last complete turn (user msg + all assistant/tool msgs that follow).

        Pops from the end until the last user message is removed, keeping the
        tool_use/tool_result pairing valid. Never removes the system message.

        Returns True if a user message was found and removed.
        """
        if len(self.items) <= 1:
            return False

        while len(self.items) > 1:
            msg = self.items.pop()
            if getattr(msg, "role", None) == "user":
                return True

        return False

    async def compact(
        self, model_name: str, tool_specs: list[dict] | None = None
    ) -> None:
        """Remove old messages to keep history under target size"""
        if (self.context_length <= self.max_context) or not self.items:
            return

        system_msg = (
            self.items[0] if self.items and self.items[0].role == "system" else None
        )

        # Don't summarize a certain number of just-preceding messages
        # Walk back to find a user message to make sure we keep an assistant -> user ->
        # assistant general conversation structure
        idx = len(self.items) - self.untouched_messages
        while idx > 1 and self.items[idx].role != "user":
            idx -= 1

        recent_messages = self.items[idx:]
        messages_to_summarize = self.items[1:idx]

        # improbable, messages would have to very long
        if not messages_to_summarize:
            return

        messages_to_summarize.append(
            Message(
                role="user",
                content="Please provide a concise summary of the conversation above, focusing on key decisions, the 'why' behind the decisions, problems solved, and important context needed for developing further. Your summary will be given to someone who has never worked on this project before and they will be have to be filled in.",
            )
        )

        hf_key = os.environ.get("INFERENCE_TOKEN")
        response = await acompletion(
            model=model_name,
            messages=messages_to_summarize,
            max_completion_tokens=self.compact_size,
            tools=tool_specs,
            api_key=hf_key
            if hf_key and model_name.startswith("huggingface/")
            else None,
        )
        summarized_message = Message(
            role="assistant", content=response.choices[0].message.content
        )

        # Reconstruct: system + summary + recent messages (includes tools)
        if system_msg:
            self.items = [system_msg, summarized_message] + recent_messages
        else:
            self.items = [summarized_message] + recent_messages

        self.context_length = (
            len(self.system_prompt) // 4 + response.usage.completion_tokens
        )
