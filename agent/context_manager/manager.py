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
        self.max_context = max_context - 10000
        self.compact_size = int(max_context * compact_size)
        self.context_length = max_context
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

        Automatically recovers malformed tool_call arguments and patches
        any dangling tool_calls (assistant messages with tool_calls that
        have no matching tool-result message).  Both can happen after
        errors or cancellations and would cause the LLM API to reject the
        request.
        """
        self.recover_malformed_tool_calls()
        self._patch_dangling_tool_calls()
        return self.items

    @staticmethod
    def _normalize_tool_calls(msg: Message) -> None:
        """Ensure msg.tool_calls contains proper ToolCall objects, not dicts.

        litellm's Message has validate_assignment=False (Pydantic v2 default),
        so direct attribute assignment (e.g. inside litellm's streaming handler)
        can leave raw dicts.  Re-assigning via the constructor fixes this.
        """
        from litellm import ChatCompletionMessageToolCall as ToolCall

        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            return
        needs_fix = any(isinstance(tc, dict) for tc in tool_calls)
        if not needs_fix:
            return
        msg.tool_calls = [
            tc if not isinstance(tc, dict) else ToolCall(**tc) for tc in tool_calls
        ]

    def recover_malformed_tool_calls(self) -> set[str]:
        """Sanitize malformed tool_call arguments and inject error results.

        Handles two classes of corruption:
        - **Empty/missing IDs**: Stripped from the assistant message entirely
          (common when streaming is interrupted mid-tool-call).
        - **Malformed JSON arguments**: Replaced with ``"{}"`` and an error
          tool-result is injected asking the agent to retry.

        This method is idempotent — safe to call from both the agent loop
        (before tool execution) and from :meth:`get_messages` (safety net).

        Returns:
            Set of tool_call IDs that had malformed arguments.
        """
        import json

        malformed_ids: set[str] = set()

        for msg in self.items:
            if getattr(msg, "role", None) != "assistant":
                continue
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                continue
            self._normalize_tool_calls(msg)

            # 1. Strip tool_calls with empty/missing IDs (cannot be repaired)
            valid_tcs = []
            for tc in msg.tool_calls:
                if not getattr(tc, "id", None):
                    logger.warning(
                        "Stripping tool_call with empty ID (name=%s) — likely interrupted stream",
                        getattr(tc.function, "name", "?"),
                    )
                    continue
                valid_tcs.append(tc)
            if len(valid_tcs) != len(msg.tool_calls):
                msg.tool_calls = valid_tcs or None

            if not msg.tool_calls:
                continue

            # 2. Fix malformed JSON arguments
            for tc in msg.tool_calls:
                try:
                    json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    logger.warning(
                        "Malformed arguments for tool_call %s (%s): %s",
                        tc.id,
                        tc.function.name,
                        e,
                    )
                    tc.function.arguments = "{}"
                    malformed_ids.add(tc.id)

        if not malformed_ids:
            return malformed_ids

        # 3. Inject error results for malformed calls that don't have one yet
        answered_ids = {
            getattr(m, "tool_call_id", None)
            for m in self.items
            if getattr(m, "role", None) == "tool"
        }
        for msg in self.items:
            if getattr(msg, "role", None) != "assistant":
                continue
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                continue
            for tc in msg.tool_calls:
                if tc.id in malformed_ids and tc.id not in answered_ids:
                    self.items.append(
                        Message(
                            role="tool",
                            content=(
                                f"ERROR: Your tool call to '{tc.function.name}' had malformed "
                                f"JSON arguments and was NOT executed. This usually happens "
                                f"when the content is too large and gets truncated. "
                                f"Please retry with smaller content — for 'write', split the "
                                f"file into multiple smaller writes using 'edit' to build up "
                                f"the file incrementally."
                            ),
                            tool_call_id=tc.id,
                            name=tc.function.name,
                        )
                    )
                    answered_ids.add(tc.id)

        return malformed_ids

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
            if getattr(msg, "role", None) == "assistant" and getattr(
                msg, "tool_calls", None
            ):
                assistant_msg = msg
                break
            # Stop scanning once we hit a user message — anything before
            # that belongs to a previous (complete) turn.
            if getattr(msg, "role", None) == "user":
                break

        if not assistant_msg:
            return

        self._normalize_tool_calls(assistant_msg)
        answered_ids = {
            getattr(m, "tool_call_id", None)
            for m in self.items
            if getattr(m, "role", None) == "tool"
        }
        for tc in assistant_msg.tool_calls:
            if tc.id not in answered_ids:
                self.items.append(
                    Message(
                        role="tool",
                        content="Tool was not executed (interrupted or error).",
                        tool_call_id=tc.id,
                        name=tc.function.name,
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
