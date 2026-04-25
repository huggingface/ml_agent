import asyncio
import json
import logging
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from agent.config import Config
from agent.context_manager.manager import ContextManager
from agent.messaging.gateway import NotificationGateway
from agent.messaging.models import NotificationRequest

logger = logging.getLogger(__name__)

# Local max-token overrides — used for models not yet in the litellm catalog
# or when we want to pin a specific window.
# NOTE: Ollama models not listed here will fall back to _DEFAULT_MAX_TOKENS (200k).
_MAX_TOKENS_MAP: dict[str, int] = {
    "ollama/qwen3.6": 128_000,
}
_DEFAULT_MAX_TOKENS = 200_000
_TURN_COMPLETE_NOTIFICATION_CHARS = 39000


def _get_max_tokens_safe(model_name: str) -> int:
    """Return the max input-context tokens for a model.

    Primary source: ``litellm.get_model_info(model)['max_input_tokens']`` —
    LiteLLM maintains an upstream catalog that knows Claude Opus 4.6 is
    1M, GPT-5 is 272k, Sonnet 4.5 is 200k, and so on. Strips any HF routing
    suffix / huggingface/ prefix so tagged ids ('moonshotai/Kimi-K2.6:cheapest')
    look up the bare model. Falls back to a conservative 200k default for
    models not in the catalog (typically HF-router-only models).
    """
    # Check local overrides first
    if model_name in _MAX_TOKENS_MAP:
        return _MAX_TOKENS_MAP[model_name]

    from litellm import get_model_info

    candidates = [model_name]
    stripped = model_name.removeprefix("huggingface/").split(":", 1)[0]
    if stripped != model_name:
        candidates.append(stripped)
    for candidate in candidates:
        try:
            info = get_model_info(candidate)
            max_input = info.get("max_input_tokens") if info else None
            if isinstance(max_input, int) and max_input > 0:
                return max_input
        except Exception:
            continue
    logger.info(
        "No litellm.get_model_info entry for %s, falling back to %d",
        model_name, _DEFAULT_MAX_TOKENS,
    )
    return _DEFAULT_MAX_TOKENS


class OpType(Enum):
    USER_INPUT = "user_input"
    EXEC_APPROVAL = "exec_approval"
    INTERRUPT = "interrupt"
    UNDO = "undo"
    COMPACT = "compact"
    SHUTDOWN = "shutdown"


@dataclass
class Event:
    event_type: str
    data: Optional[dict[str, Any]] = None
    seq: Optional[int] = None


class Session:
    """
    Maintains agent session state
    Similar to Session in codex-rs/core/src/codex.rs
    """

    def __init__(
        self,
        event_queue: asyncio.Queue,
        config: Config,
        tool_router=None,
        context_manager: ContextManager | None = None,
        hf_token: str | None = None,
        local_mode: bool = False,
        stream: bool = True,
        notification_gateway: NotificationGateway | None = None,
        notification_destinations: list[str] | None = None,
        defer_turn_complete_notification: bool = False,
        session_id: str | None = None,
        user_id: str | None = None,
        hf_username: str | None = None,
        persistence_store: Any | None = None,
    ):
        self.hf_token: Optional[str] = hf_token
        self.user_id: Optional[str] = user_id
        self.hf_username: Optional[str] = hf_username
        self.persistence_store = persistence_store
        self.tool_router = tool_router
        self.stream = stream
        if config is None:
            raise ValueError("Session requires a Config")
        tool_specs = tool_router.get_tool_specs_for_llm() if tool_router else []
        self.context_manager = context_manager or ContextManager(
            model_max_tokens=_get_max_tokens_safe(config.model_name),
            compact_size=0.1,
            untouched_messages=5,
            tool_specs=tool_specs,
            hf_token=hf_token,
            local_mode=local_mode,
        )
        self.event_queue = event_queue
        self.session_id = session_id or str(uuid.uuid4())
        self.config = config
        self.is_running = True
        self._cancelled = asyncio.Event()
        self.pending_approval: Optional[dict[str, Any]] = None
        self.sandbox = None
        self.sandbox_hardware: Optional[str] = None
        self.sandbox_preload_task: Optional[asyncio.Task] = None
        self.sandbox_preload_error: Optional[str] = None
        self.sandbox_preload_cancel_event: Any | None = None
        self._running_job_ids: set[str] = set()  # HF job IDs currently executing
        self.notification_gateway = notification_gateway
        self.notification_destinations = list(notification_destinations or [])
        self.defer_turn_complete_notification = defer_turn_complete_notification
        self.auto_approval_enabled: bool = False
        self.auto_approval_cost_cap_usd: float | None = None
        self.auto_approval_estimated_spend_usd: float = 0.0

        # Session trajectory logging
        self.logged_events: list[dict] = []
        self.session_start_time = datetime.now().isoformat()
        self.turn_count: int = 0
        self.last_auto_save_turn: int = 0
        # Stable local save path so heartbeat saves overwrite one file instead
        # of spamming session_logs/. ``_last_heartbeat_ts`` is owned by
        # ``agent.core.telemetry.HeartbeatSaver`` and lazily initialised there.
        self._local_save_path: Optional[str] = None
        self._last_heartbeat_ts: Optional[float] = None

        # Per-model probed reasoning-effort cache. Populated by the probe
        # on /model switch, read by ``effective_effort_for`` below. Keys are
        # raw model ids (including any ``:tag``). Values:
        #   str  → the effort level to send (may be a downgrade from the
        #          preference, e.g. "high" when user asked for "max")
        #   None → model rejected all efforts in the cascade; send no
        #          thinking params at all
        # Key absent → not probed yet; fall back to the raw preference.
        self.model_effective_effort: dict[str, str | None] = {}
        self.context_manager.on_message_added = self._schedule_trace_message

    async def send_event(self, event: Event) -> None:
        """Send event back to client and log to trajectory"""
        # Log event to trajectory
        self.logged_events.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event_type": event.event_type,
                "data": event.data,
            }
        )
        if self.persistence_store is not None:
            try:
                event.seq = await self.persistence_store.append_event(
                    self.session_id, event.event_type, event.data
                )
            except Exception as e:
                logger.debug("Event persistence failed for %s: %s", self.session_id, e)

        await self.event_queue.put(event)
        await self._enqueue_auto_notification_requests(event)

        # Mid-turn heartbeat flush (owned by telemetry module).
        from agent.core.telemetry import HeartbeatSaver

        HeartbeatSaver.maybe_fire(self)

    def _schedule_trace_message(self, message: Any) -> None:
        """Best-effort append-only trace save for SFT/KPI export."""
        if self.persistence_store is None:
            return
        try:
            payload = message.model_dump(mode="json")
        except Exception:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        source = str(payload.get("role") or "message")
        loop.create_task(
            self.persistence_store.append_trace_message(
                self.session_id, payload, source=source
            )
        )

    def set_notification_destinations(self, destinations: list[str]) -> None:
        """Replace the session's opted-in auto-notification destinations."""
        deduped: list[str] = []
        seen: set[str] = set()
        for destination in destinations:
            if destination not in seen:
                deduped.append(destination)
                seen.add(destination)
        self.notification_destinations = deduped

    async def send_deferred_turn_complete_notification(self, event: Event) -> None:
        if event.event_type != "turn_complete":
            return
        await self._enqueue_auto_notification_requests(
            event,
            include_deferred_turn_complete=True,
        )

    async def _enqueue_auto_notification_requests(
        self,
        event: Event,
        include_deferred_turn_complete: bool = False,
    ) -> None:
        if self.notification_gateway is None:
            return
        if not self.notification_destinations:
            return
        auto_events = set(self.config.messaging.auto_event_types)
        if event.event_type not in auto_events:
            return
        if (
            self.defer_turn_complete_notification
            and event.event_type == "turn_complete"
            and not include_deferred_turn_complete
        ):
            return

        requests = self._build_auto_notification_requests(event)
        for request in requests:
            await self.notification_gateway.enqueue(request)

    def _build_auto_notification_requests(
        self, event: Event
    ) -> list[NotificationRequest]:
        metadata = {
            "session_id": self.session_id,
            "model": self.config.model_name,
            "event_type": event.event_type,
        }

        title: str | None = None
        message: str | None = None
        severity = "info"
        data = event.data or {}
        if event.event_type == "approval_required":
            tools = data.get("tools", [])
            tool_names = []
            for tool in tools if isinstance(tools, list) else []:
                if isinstance(tool, dict):
                    tool_name = str(tool.get("tool") or "").strip()
                    if tool_name and tool_name not in tool_names:
                        tool_names.append(tool_name)
            count = len(tools) if isinstance(tools, list) else 0
            title = "Agent approval required"
            message = (
                f"Session {self.session_id} is waiting for approval "
                f"for {count} tool call(s)."
            )
            if tool_names:
                message += " Tools: " + ", ".join(tool_names)
            severity = "warning"
        elif event.event_type == "error":
            title = "Agent error"
            error = str(data.get("error") or "Unknown error")
            message = f"Session {self.session_id} hit an error.\n{error[:500]}"
            severity = "error"
        elif event.event_type == "turn_complete":
            title = "Agent task complete"
            summary = str(data.get("final_response") or "").strip()
            if summary:
                summary = summary[:_TURN_COMPLETE_NOTIFICATION_CHARS]
                message = (
                    f"Session {self.session_id} completed successfully.\n"
                    f"{summary}"
                )
            else:
                message = f"Session {self.session_id} completed successfully."
            severity = "success"

        if message is None:
            return []

        requests: list[NotificationRequest] = []
        for destination in self.notification_destinations:
            if not self.config.messaging.can_auto_send(destination):
                continue
            requests.append(
                NotificationRequest(
                    destination=destination,
                    title=title,
                    message=message,
                    severity=severity,
                    metadata=metadata,
                    event_type=event.event_type,
                )
            )
        return requests

    def cancel(self) -> None:
        """Signal cancellation to the running agent loop."""
        self._cancelled.set()

    def reset_cancel(self) -> None:
        """Clear the cancellation flag before a new run."""
        self._cancelled.clear()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def update_model(self, model_name: str) -> None:
        """Switch the active model and update the context window limit."""
        self.config.model_name = model_name
        self.context_manager.model_max_tokens = _get_max_tokens_safe(model_name)

    def set_auto_approval_policy(
        self, *, enabled: bool, cost_cap_usd: float | None
    ) -> None:
        self.auto_approval_enabled = bool(enabled)
        self.auto_approval_cost_cap_usd = cost_cap_usd

    def add_auto_approval_estimated_spend(self, amount_usd: float | None) -> None:
        if amount_usd is None or amount_usd <= 0:
            return
        self.auto_approval_estimated_spend_usd = round(
            self.auto_approval_estimated_spend_usd + float(amount_usd), 4
        )

    @property
    def auto_approval_remaining_usd(self) -> float | None:
        if self.auto_approval_cost_cap_usd is None:
            return None
        return round(
            max(
                0.0,
                self.auto_approval_cost_cap_usd
                - self.auto_approval_estimated_spend_usd,
            ),
            4,
        )

    def auto_approval_policy_summary(self) -> dict[str, Any]:
        return {
            "enabled": self.auto_approval_enabled,
            "cost_cap_usd": self.auto_approval_cost_cap_usd,
            "estimated_spend_usd": round(self.auto_approval_estimated_spend_usd, 4),
            "remaining_usd": self.auto_approval_remaining_usd,
        }

    def effective_effort_for(self, model_name: str) -> str | None:
        """Resolve the effort level to actually send for ``model_name``.

        Returns the probed result when we have one (may be ``None`` meaning
        "model doesn't do thinking, strip it"), else the raw preference.
        Unknown-model case falls back to the preference so a stale cache
        from a prior ``/model`` can't poison research sub-calls that use a
        different model id.
        """
        if model_name in self.model_effective_effort:
            return self.model_effective_effort[model_name]
        return self.config.reasoning_effort

    def increment_turn(self) -> None:
        """Increment turn counter (called after each user interaction)"""
        self.turn_count += 1

    async def auto_save_if_needed(self) -> None:
        """Check if auto-save should trigger and save if so (completely non-blocking)"""
        if not self.config.save_sessions:
            return

        interval = self.config.auto_save_interval
        if interval <= 0:
            return

        turns_since_last_save = self.turn_count - self.last_auto_save_turn
        if turns_since_last_save >= interval:
            logger.info(f"Auto-saving session (turn {self.turn_count})...")
            # Fire-and-forget save - returns immediately
            self.save_and_upload_detached(self.config.session_dataset_repo)
            self.last_auto_save_turn = self.turn_count

    def get_trajectory(self) -> dict:
        """Serialize complete session trajectory for logging"""
        tools: list = []
        if self.tool_router is not None:
            try:
                tools = self.tool_router.get_tool_specs_for_llm() or []
            except Exception:
                tools = []
        # Sum per-call cost from llm_call events so analyzers don't have to
        # walk the events array themselves. Each `llm_call` event already
        # carries cost_usd from `agent.core.telemetry.record_llm_call`.
        total_cost_usd = sum(
            float((e.get("data") or {}).get("cost_usd") or 0.0)
            for e in self.logged_events
            if e.get("event_type") == "llm_call"
        )
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "hf_username": self.hf_username,
            "session_start_time": self.session_start_time,
            "session_end_time": datetime.now().isoformat(),
            "model_name": self.config.model_name,
            "total_cost_usd": total_cost_usd,
            "messages": [msg.model_dump() for msg in self.context_manager.items],
            "events": self.logged_events,
            "tools": tools,
        }

    def save_trajectory_local(
        self,
        directory: str = "session_logs",
        upload_status: str = "pending",
        dataset_url: Optional[str] = None,
    ) -> Optional[str]:
        """
        Save trajectory to local JSON file as backup with upload status

        Args:
            directory: Directory to save logs (default: "session_logs")
            upload_status: Status of upload attempt ("pending", "success", "failed")
            dataset_url: URL of dataset if upload succeeded

        Returns:
            Path to saved file if successful, None otherwise
        """
        try:
            log_dir = Path(directory)
            log_dir.mkdir(parents=True, exist_ok=True)

            trajectory = self.get_trajectory()

            # Scrub secrets at save time so session_logs/ never holds raw
            # tokens on disk — a log aggregator, crash dump, or filesystem
            # snapshot between heartbeats would otherwise leak them.
            try:
                from agent.core.redact import scrub
                for key in ("messages", "events", "tools"):
                    if key in trajectory:
                        trajectory[key] = scrub(trajectory[key])
            except Exception as _e:
                logger.debug("Redact-on-save failed (non-fatal): %s", _e)

            # Add upload metadata
            trajectory["upload_status"] = upload_status
            trajectory["upload_url"] = dataset_url
            trajectory["last_save_time"] = datetime.now().isoformat()

            # Reuse one stable path per session so heartbeat saves overwrite
            # the same file instead of creating a new timestamped file every
            # minute. The timestamp in the filename is kept for first-save
            # ordering; subsequent saves just rewrite that file.
            if self._local_save_path and Path(self._local_save_path).parent == log_dir:
                filepath = Path(self._local_save_path)
            else:
                filename = (
                    f"session_{self.session_id}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                filepath = log_dir / filename
                self._local_save_path = str(filepath)

            # Atomic-ish write: stage to .tmp then rename so a crash mid-write
            # doesn't leave a truncated JSON that breaks the retry scanner.
            tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
            with open(tmp_path, "w") as f:
                json.dump(trajectory, f, indent=2)
            tmp_path.replace(filepath)

            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save session locally: {e}")
            return None

    def update_local_save_status(
        self, filepath: str, upload_status: str, dataset_url: Optional[str] = None
    ) -> bool:
        """Update the upload status of an existing local save file"""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            data["upload_status"] = upload_status
            data["upload_url"] = dataset_url
            data["last_save_time"] = datetime.now().isoformat()

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            return True
        except Exception as e:
            logger.error(f"Failed to update local save status: {e}")
            return False

    def _personal_trace_repo_id(self) -> Optional[str]:
        """Resolve the per-user trace repo id from config + HF username.

        Returns ``None`` when sharing is disabled, the user is anonymous,
        or the template is missing — caller skips the personal upload in
        those cases.
        """
        if not getattr(self.config, "share_traces", False):
            return None
        hf_user = self.hf_username or self.user_id
        if not hf_user:
            return None
        template = getattr(self.config, "personal_trace_repo_template", None)
        if not template:
            return None
        try:
            return template.format(hf_user=hf_user)
        except (KeyError, IndexError):
            logger.debug("personal_trace_repo_template format failed: %r", template)
            return None

    def _spawn_uploader(
        self,
        action: str,
        target: str,
        repo_id: str,
        *,
        format: str,
        token_env: Optional[str],
        private: bool,
        token_value: Optional[str] = None,
    ) -> None:
        """Fire-and-forget spawn of ``session_uploader.py`` with the given args."""
        try:
            uploader_script = Path(__file__).parent / "session_uploader.py"
            cmd = [
                sys.executable,
                str(uploader_script),
                action,
                target,
                repo_id,
                "--format",
                format,
                "--private",
                "true" if private else "false",
            ]
            if token_env:
                cmd.extend(["--token-env", token_env])

            env = os.environ.copy()
            if token_value:
                env["_ML_INTERN_PERSONAL_TOKEN"] = token_value

            subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                start_new_session=True,  # Detach from parent
            )
        except Exception as e:
            logger.warning(f"Failed to spawn upload subprocess: {e}")

    def save_and_upload_detached(self, repo_id: str) -> Optional[str]:
        """
        Save session locally and spawn detached subprocess(es) for upload
        (fire-and-forget).

        Always uploads to the shared org dataset (``repo_id``) in the
        single-row format used by the KPI scheduler. When
        ``config.share_traces`` is enabled and a username is known, also
        uploads to the user's personal private dataset in Claude Code JSONL
        format so the HF Agent Trace Viewer auto-renders it.

        Args:
            repo_id: HuggingFace dataset repo ID for the org/KPI upload.

        Returns:
            Path to local save file
        """
        local_path = self.save_trajectory_local(upload_status="pending")
        if not local_path:
            return None

        self._spawn_uploader(
            "upload",
            local_path,
            repo_id,
            format="row",
            token_env=None,  # default org token chain
            private=False,
        )

        personal_repo = self._personal_trace_repo_id()
        if personal_repo:
            # User's own HF_TOKEN write-scoped to their namespace.
            self._spawn_uploader(
                "upload",
                local_path,
                personal_repo,
                format="claude_code",
                token_env="HF_TOKEN",
                token_value=self.hf_token,
                private=True,
            )

        return local_path

    @staticmethod
    def retry_failed_uploads_detached(
        directory: str = "session_logs",
        repo_id: Optional[str] = None,
        *,
        personal_repo_id: Optional[str] = None,
    ) -> None:
        """
        Spawn detached subprocess(es) to retry failed/pending uploads
        (fire-and-forget).

        Args:
            directory: Directory containing session logs
            repo_id: Target dataset repo ID for the shared org/KPI upload.
            personal_repo_id: Per-user dataset for Claude-Code-format
                retries. ``None`` skips the personal retry pass.
        """
        if not repo_id and not personal_repo_id:
            return

        try:
            uploader_script = Path(__file__).parent / "session_uploader.py"

            if repo_id:
                subprocess.Popen(
                    [
                        sys.executable,
                        str(uploader_script),
                        "retry",
                        directory,
                        repo_id,
                        "--format",
                        "row",
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )

            if personal_repo_id:
                subprocess.Popen(
                    [
                        sys.executable,
                        str(uploader_script),
                        "retry",
                        directory,
                        personal_repo_id,
                        "--format",
                        "claude_code",
                        "--token-env",
                        "HF_TOKEN",
                        "--private",
                        "true",
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
        except Exception as e:
            logger.warning(f"Failed to spawn retry subprocess: {e}")
