"""
Context management for conversation history
"""

import zoneinfo
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template
from litellm import Message, acompletion


class ContextManager:
    """Manages conversation context and message history for the agent"""

    def __init__(
        self,
        max_context: int = 180_000,
        compact_size: float = 0.1,
        untouched_messages: int = 5,
        tool_specs: list[dict[str, Any]] | None = None,
        prompt_file_suffix: str = "system_prompt.yaml",
    ):
        self.system_prompt = self._load_system_prompt(
            tool_specs or [], prompt_file_suffix="system_prompt.yaml"
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

        template = Template(template_str)
        return template.render(
            tools=tool_specs,
            num_tools=len(tool_specs),
            current_date=current_date,
            current_time=current_time,
            current_timezone=current_timezone,
        )

    def add_message(self, message: Message, token_count: int = None) -> None:
        """Add a message to the history"""
        if token_count:
            self.context_length = token_count
        self.items.append(message)

    def get_messages(self) -> list[Message]:
        """Get all messages for sending to LLM"""
        return self.items

    async def compact(self, model_name: str) -> None:
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
                content="Please provide a concise summary of the conversation above, focusing on key decisions, code changes, problems solved, and important context needed for future turns.",
            )
        )

        response = await acompletion(
            model=model_name,
            messages=messages_to_summarize,
            max_completion_tokens=self.compact_size,
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
