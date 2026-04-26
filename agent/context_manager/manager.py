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

from agent.core.prompt_caching import with_prompt_caching

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


_COMPACT_PROMPT = (
    "Please provide a concise summary of the conversation above, focusing on "
    "key decisions, the 'why' behind the decisions, problems solved, and "
    "important context needed for developing further. Your summary will be "
    "given to someone who has never worked on this project before and they "
    "will be have to be filled in."
)

# Used when seeding a brand-new session from prior browser-cached messages.
# Here we're writing a note to *ourselves* — so preserve the tool-call trail,
# files produced, and planned next steps in first person. Optimized for
# continuity, not brevity.
_RESTORE_PROMPT = (
    "You're about to be restored into a fresh session with no memory of the "
    "conversation above. Write a first-person note to your future self so "
    "you can continue right where you left off. Include:\n"
    "  • What the user originally asked for and what progress you've made.\n"
    "  • Every tool you called, with arguments and a one-line result summary.\n"
    "  • Any code, files, scripts, or artifacts you produced (with paths).\n"
    "  • Key decisions and the reasoning behind them.\n"
    "  • What you were planning to do next.\n\n"
    "Don't be cute. Be specific. This is the only context you'll have."
)


async def summarize_messages(
    messages: list[Message],
    model_name: str,
    hf_token: str | None = None,
    max_tokens: int = 2000,
    tool_specs: list[dict] | None = None,
    prompt: str = _COMPACT_PROMPT,
) -> tuple[str, int]:
    """Run a summarization prompt against a list of messages.

    ``prompt`` defaults to the compaction prompt (terse, decision-focused).
    Callers seeding a new session after a restart should pass ``_RESTORE_PROMPT``
    instead — it preserves the tool-call trail so the agent can answer
    follow-up questions about what it did.

    Returns ``(summary_text, completion_tokens)``.
    """
    from agent.core.llm_params import _resolve_llm_params

    prompt_messages = list(messages) + [Message(role="user", content=prompt)]
    llm_params = _resolve_llm_params(model_name, hf_token, reasoning_effort="high")
    prompt_messages, tool_specs = with_prompt_caching(
        prompt_messages, tool_specs, llm_params.get("model")
    )
    response = await acompletion(
        messages=prompt_messages,
        max_completion_tokens=max_tokens,
        tools=tool_specs,
        **llm_params,
    )
    summary = response.choices[0].message.content or ""
    completion_tokens = response.usage.completion_tokens if response.usage else 0
    return summary, completion_tokens


class ContextManager:
    """Manages conversation context and message history for the agent"""

    def __init__(
        self,
        model_max_tokens: int = 180_000,
        compact_size: float = 0.1,
        untouched_messages: int = 5,
        tool_specs: list[dict[str, Any]] | None = None,
        prompt_file_suffix: str = "system_prompt_v3.yaml",
        hf_token: str | None = None,
        local_mode: bool = False,
    ):
        self.system_prompt = self._load_system_prompt(
            tool_specs or [],
            prompt_file_suffix="system_prompt_v3.yaml",
            hf_token=hf_token,
            local_mode=local_mode,
        )
        # The model's real input-token ceiling (from litellm.get_model_info).
        # Compaction triggers at _COMPACT_THRESHOLD_RATIO below it — see
        # the compaction_threshold property.
        self.model_max_tokens = model_max_tokens
        self.compact_size = int(model_max_tokens * compact_size)
        # Running count of tokens the last LLM call reported. Drives the
        # compaction gate; updated in add_message() with each response's
        # usage.total_tokens.
        self.running_context_usage = 0
        self.untouched_messages = untouched_messages
        self.items: list[Message] = [Message(role="system", content=self.system_prompt)]

    def _load_system_prompt(
        self,
        tool_specs: list[dict[str, Any]],
        prompt_file_suffix: str = "system_prompt.yaml",
        hf_token: str | None = None,
        local_mode: bool = False,
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
        static_prompt = template.render(
            tools=tool_specs,
            num_tools=len(tool_specs),
        )

        # CLI-specific context for local mode
        if local_mode:
            import os
            cwd = os.getcwd()
            local_context = (
                f"\n\n# CLI / Local mode\n\n"
                f"You are running as a local CLI tool on the user's machine. "
                f"There is NO sandbox — bash, read, write, and edit operate directly "
                f"on the local filesystem.\n\n"
                f"Working directory: {cwd}\n"
                f"Use absolute paths or paths relative to the working directory. "
                f"Do NOT use /app/ paths — that is a sandbox convention that does not apply here.\n"
                f"The sandbox_create tool is NOT available. Run code directly with bash."
            )
            static_prompt += local_context

        return (
            f"{static_prompt}\n\n"
            f"[Session context: Date={current_date}, Time={current_time}, "
            f"Timezone={current_timezone}, User={hf_user_info}, "
            f"Tools={len(tool_specs)}]"
        )

    def add_message(self, message: Message, token_count: int = None) -> None:
        """Add a message to the history"""
        if token_count:
            self.running_context_usage = token_count
        # Eagerly normalize tool_calls so every downstream model_dump()
        # (trajectories, compaction, token counting) works without
        # PydanticSerializationUnexpectedValue warnings.
        self._normalize_tool_calls(message)
        self.items.append(message)

    def get_messages(self) -> list[Message]:
        """Get all messages for sending to LLM.

        Patches any dangling tool_calls (assistant messages with tool_calls
        that have no matching tool-result message) so the LLM API doesn't
        reject the request.
        """
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

    def _patch_dangling_tool_calls(self) -> None:
        """Add stub tool results for any tool_calls that lack a matching result.

        Ensures each assistant message's tool_calls are followed immediately
        by matching tool-result messages. This has to work across the whole
        history, not just the most recent turn, because a cancelled tool use
        in an earlier turn can still poison the next provider request.
        """
        if not self.items:
            return

        i = 0
        while i < len(self.items):
            msg = self.items[i]
            if getattr(msg, "role", None) != "assistant" or not getattr(msg, "tool_calls", None):
                i += 1
                continue

            self._normalize_tool_calls(msg)

            # Consume the contiguous tool-result block that immediately follows
            # this assistant message. Any missing tool ids must be inserted
            # before the next non-tool message to satisfy provider ordering.
            j = i + 1
            immediate_ids: set[str | None] = set()
            while j < len(self.items) and getattr(self.items[j], "role", None) == "tool":
                immediate_ids.add(getattr(self.items[j], "tool_call_id", None))
                j += 1

            missing: list[Message] = []
            for tc in msg.tool_calls:
                if tc.id not in immediate_ids:
                    missing.append(
                        Message(
                            role="tool",
                            content="Tool was not executed (interrupted or error).",
                            tool_call_id=tc.id,
                            name=tc.function.name,
                        )
                    )

            if missing:
                self.items[j:j] = missing
                j += len(missing)

            i = j

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

    def truncate_to_user_message(self, user_message_index: int) -> bool:
        """Truncate history to just before the Nth user message (0-indexed).

        Removes that user message and everything after it.
        System message (index 0) is never removed.

        Returns True if the target user message was found and removed.
        """
        count = 0
        for i, msg in enumerate(self.items):
            if i == 0:
                continue  # skip system message
            if getattr(msg, "role", None) == "user":
                if count == user_message_index:
                    self.items = self.items[:i]
                    return True
                count += 1
        return False

    # Compaction fires at 90% of model_max_tokens so there's headroom for
    # the next turn's prompt + response before we actually hit the ceiling.
    _COMPACT_THRESHOLD_RATIO = 0.9

    @property
    def compaction_threshold(self) -> int:
        """Token count at which `compact()` kicks in."""
        return int(self.model_max_tokens * self._COMPACT_THRESHOLD_RATIO)

    @property
    def needs_compaction(self) -> bool:
        return self.running_context_usage > self.compaction_threshold and bool(self.items)

    async def compact(
        self,
        model_name: str,
        tool_specs: list[dict] | None = None,
        hf_token: str | None = None,
    ) -> None:
        """Remove old messages to keep history under target size"""
        if not self.needs_compaction:
            return

        system_msg = (
            self.items[0] if self.items and self.items[0].role == "system" else None
        )

        # Preserve the first user message (task prompt) — never summarize it
        first_user_msg = None
        first_user_idx = 1
        for i in range(1, len(self.items)):
            if getattr(self.items[i], "role", None) == "user":
                first_user_msg = self.items[i]
                first_user_idx = i
                break

        # Don't summarize a certain number of just-preceding messages
        # Walk back to find a user message to make sure we keep an assistant -> user ->
        # assistant general conversation structure
        idx = len(self.items) - self.untouched_messages
        while idx > 1 and self.items[idx].role != "user":
            idx -= 1

        recent_messages = self.items[idx:]
        messages_to_summarize = self.items[first_user_idx + 1:idx]

        # improbable, messages would have to very long
        if not messages_to_summarize:
            return

        summary, completion_tokens = await summarize_messages(
            messages_to_summarize,
            model_name=model_name,
            hf_token=hf_token,
            max_tokens=self.compact_size,
            tool_specs=tool_specs,
            prompt=_COMPACT_PROMPT,
        )
        summarized_message = Message(role="assistant", content=summary)

        # Reconstruct: system + first user msg + summary + recent messages
        head = [system_msg] if system_msg else []
        if first_user_msg:
            head.append(first_user_msg)
        self.items = head + [summarized_message] + recent_messages

        # Count the actual post-compact context — system prompt + first user
        # turn + summary + the preserved tail all contribute, not just the
        # summary. litellm.token_counter uses the model's real tokenizer.
        from litellm import token_counter

        try:
            self.running_context_usage = token_counter(
                model=model_name,
                messages=[m.model_dump() for m in self.items],
            )
        except Exception as e:
            logger.warning("token_counter failed post-compact (%s); falling back to rough estimate", e)
            self.running_context_usage = len(self.system_prompt) // 4 + completion_tokens
