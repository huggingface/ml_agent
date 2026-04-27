"""
Notification service for ML Intern.

Provides pluggable notification providers:
- email: SMTP email notifications
- pushbullet: Pushbullet mobile notifications
- telegram: Telegram bot notifications
- slack: Slack webhook notifications
- discord: Discord webhook notifications
- system: System notifications (platform-specific)
"""

import asyncio
import httpx
import logging
import platform
import smtplib
from abc import ABC, abstractmethod
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

logger = logging.getLogger(__name__)


class NotificationProviderBase(ABC):
    """Base class for notification providers."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.events = config.get("events", [
            "approval_required",
            "waiting",
            "job_complete",
            "job_failed",
            "error",
            "session_saved",
        ])

    def should_notify(self, event_type: str) -> bool:
        """Check if this provider should send for the given event type."""
        return event_type in self.events

    @abstractmethod
    async def send(self, event_type: str, title: str, message: str, metadata: dict[str, Any] | None = None) -> bool:
        """Send a notification. Returns True on success."""
        pass


class EmailProvider(NotificationProviderBase):
    """SMTP email notification provider."""

    async def send(self, event_type: str, title: str, message: str, metadata: dict[str, Any] | None = None) -> bool:
        try:
            smtp_host = self.config.get("smtp_host")
            smtp_port = self.config.get("smtp_port", 587)
            smtp_user = self.config.get("smtp_user")
            smtp_password = self.config.get("smtp_password")
            email_to = self.config.get("email_to")
            email_from = self.config.get("email_from", smtp_user)

            # Unwrap SecretStr if needed
            if hasattr(smtp_password, "get_secret_value"):
                smtp_password = smtp_password.get_secret_value()

            if not all([smtp_host, smtp_user, smtp_password, email_to]):
                logger.warning("Email provider: incomplete config, skipping notification")
                return False

            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[ML Intern] {title}"
            msg["From"] = email_from
            msg["To"] = email_to

            text_body = f"{message}\n\nEvent: {event_type}"
            if metadata:
                text_body += f"\nMetadata: {metadata}"
            msg.attach(MIMEText(text_body, "plain"))

            def _send():
                with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                    server.starttls()
                    server.login(smtp_user, smtp_password)
                    server.send_message(msg)

            await asyncio.to_thread(_send)
            logger.info("Email notification sent successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False


class PushbulletProvider(NotificationProviderBase):
    """Pushbullet mobile notification provider."""

    async def send(self, event_type: str, title: str, message: str, metadata: dict[str, Any] | None = None) -> bool:
        try:
            api_key = self.config.get("pushbullet_api_key")
            if api_key is None:
                logger.warning("Pushbullet provider: no API key configured")
                return False

            # Unwrap SecretStr if needed
            if hasattr(api_key, "get_secret_value"):
                api_key = api_key.get_secret_value()

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.pushbullet.com/v2/pushes",
                    headers={"Access-Token": api_key},
                    json={
                        "type": "note",
                        "title": f"[ML Intern] {title}",
                        "body": message,
                    },
                )
                if resp.status_code == 200:
                    logger.info("Pushbullet notification sent successfully")
                    return True
                else:
                    logger.error(f"Pushbullet API error: {resp.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Failed to send Pushbullet notification: {e}")
            return False


class TelegramProvider(NotificationProviderBase):
    """Telegram bot notification provider."""

    async def send(self, event_type: str, title: str, message: str, metadata: dict[str, Any] | None = None) -> bool:
        try:
            bot_token = self.config.get("telegram_bot_token")
            chat_id = self.config.get("telegram_chat_id")

            if not all([bot_token, chat_id]):
                logger.warning("Telegram provider: incomplete config")
                return False

            # Unwrap SecretStr if needed
            if hasattr(bot_token, "get_secret_value"):
                bot_token = bot_token.get_secret_value()

            text = f"*[ML Intern] {title}*\n\n{message}"
            if metadata:
                text += f"\n\nMetadata: {metadata}"

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"https://api.telegram.org/bot{bot_token}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": text,
                        "parse_mode": "Markdown",
                    },
                )
                if resp.status_code == 200:
                    logger.info("Telegram notification sent successfully")
                    return True
                else:
                    logger.error(f"Telegram API error: {resp.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False


class SlackProvider(NotificationProviderBase):
    """Slack webhook notification provider."""

    async def send(self, event_type: str, title: str, message: str, metadata: dict[str, Any] | None = None) -> bool:
        try:
            webhook_url = self.config.get("slack_webhook_url")
            if not webhook_url:
                logger.warning("Slack provider: no webhook URL configured")
                return False

            text = f"*[ML Intern] {title}*\n>{message}"
            if metadata:
                text += f"\n```json\n{metadata}\n```"

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    webhook_url,
                    json={"text": text},
                )
                if resp.status_code == 200:
                    logger.info("Slack notification sent successfully")
                    return True
                else:
                    logger.error(f"Slack webhook error: {resp.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False


class DiscordProvider(NotificationProviderBase):
    """Discord webhook notification provider."""

    async def send(self, event_type: str, title: str, message: str, metadata: dict[str, Any] | None = None) -> bool:
        try:
            webhook_url = self.config.get("discord_webhook_url")
            if not webhook_url:
                logger.warning("Discord provider: no webhook URL configured")
                return False

            embed = {
                "title": f"[ML Intern] {title}",
                "description": message,
                "color": 0x6366F1,  # Indigo color
            }
            if metadata:
                embed["fields"] = [
                    {"name": key, "value": str(value), "inline": True}
                    for key, value in metadata.items()
                ]

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    webhook_url,
                    json={"embeds": [embed]},
                )
                if resp.status_code == 200 or resp.status_code == 204:
                    logger.info("Discord notification sent successfully")
                    return True
                else:
                    logger.error(f"Discord webhook error: {resp.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False


class SystemProvider(NotificationProviderBase):
    """Platform-specific system notification provider."""

    async def send(self, event_type: str, title: str, message: str, metadata: dict[str, Any] | None = None) -> bool:
        try:
            system = platform.system()

            if system == "Darwin":  # macOS
                import subprocess

                # Escape special characters to prevent command injection
                safe_msg = message.replace("\\", "\\\\").replace('"', '\\"')
                safe_title = title.replace("\\", "\\\\").replace('"', '\\"')
                script = f'display notification "{safe_msg}" with title "[ML Intern] {safe_title}"'
                await asyncio.to_thread(
                    subprocess.run, ["osascript", "-e", script], check=False, capture_output=True
                )
                return True

            elif system == "Windows":
                # Use PowerShell for Windows notifications
                import subprocess

                # Escape double-quotes to prevent PowerShell injection
                safe_title = title.replace('"', '`"').replace("'", "''")
                safe_msg = message.replace('"', '`"').replace("'", "''")
                await asyncio.to_thread(
                    subprocess.run,
                    [
                        "powershell",
                        "-Command",
                        f'[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null; '
                        f'$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02); '
                        f'$template.GetElementsByTagName("text")[0].AppendChild($template.CreateTextNode("[ML Intern] {safe_title}")) | Out-Null; '
                        f'$template.GetElementsByTagName("text")[1].AppendChild($template.CreateTextNode("{safe_msg}")) | Out-Null; '
                        f'$toast = [Windows.UI.Notifications.ToastNotification]::new($template); '
                        f'[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("ML Intern").Show($toast)',
                    ],
                    check=False,
                    capture_output=True,
                )
                return True

            elif system == "Linux":
                # Try notify-send (most Linux distros)
                import subprocess

                await asyncio.to_thread(
                    subprocess.run,
                    ["notify-send", f"[ML Intern] {title}", message],
                    check=False,
                    capture_output=True,
                )
                return True

            else:
                logger.warning(f"System notifications not supported on {system}")
                return False

        except Exception as e:
            logger.error(f"Failed to send system notification: {e}")
            return False


_PROVIDER_CLASSES: dict[str, type[NotificationProviderBase]] = {
    "email": EmailProvider,
    "pushbullet": PushbulletProvider,
    "telegram": TelegramProvider,
    "slack": SlackProvider,
    "discord": DiscordProvider,
    "system": SystemProvider,
}


class NotificationService:
    """
    Central notification service that manages multiple providers.

    Usage:
        service = NotificationService(config.notifications)
        await service.notify("approval_required", "Approval Needed", "ML Intern wants to run a job", {...})
    """

    def __init__(self, provider_configs: list[dict]):
        self.providers: list[NotificationProviderBase] = []
        for cfg in provider_configs:
            provider_type = cfg.get("provider", "").lower()
            enabled = cfg.get("enabled", True)
            if not enabled:
                continue
            provider_cls = _PROVIDER_CLASSES.get(provider_type)
            if provider_cls:
                self.providers.append(provider_cls(cfg))
                logger.info(f"Notification provider initialized: {provider_type}")
            else:
                logger.warning(f"Unknown notification provider: {provider_type}")

    async def _safe_send_provider(self, coro):
        """Execute provider coroutine, logging any errors."""
        try:
            await coro
        except Exception as e:
            logger.error(f"Notification provider error: {e}")

    async def notify(
        self,
        event_type: str,
        title: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Send notification to all providers that want this event type.
        Non-blocking - fire and forget.
        """
        if not self.providers:
            return

        # Fire all providers concurrently, don't block
        for provider in self.providers:
            if provider.should_notify(event_type):
                asyncio.create_task(self._safe_send_provider(
                    provider.send(event_type, title, message, metadata)
                ))

    async def notify_approval_required(
        self,
        tools: list[dict],
        count: int,
        session_id: str | None = None,
    ) -> None:
        """Notify that ML Intern needs approval for tool execution."""
        tool_names = [t.get("tool", "unknown") for t in tools]
        await self.notify(
            "approval_required",
            "Approval Required",
            f"ML Intern needs your approval to: {', '.join(tool_names)}",
            {"count": count, "tools": tool_names, "session_id": session_id},
        )

    async def notify_waiting(self, message: str, session_id: str | None = None) -> None:
        """Notify that ML Intern is waiting for user input."""
        await self.notify(
            "waiting",
            "Waiting for Input",
            message,
            {"session_id": session_id},
        )

    async def notify_job_complete(
        self,
        job_id: str,
        status: str,
        session_id: str | None = None,
    ) -> None:
        """Notify that a HuggingFace job has completed."""
        await self.notify(
            "job_complete",
            "Job Complete",
            f"Job {job_id} finished with status: {status}",
            {"job_id": job_id, "status": status, "session_id": session_id},
        )

    async def notify_job_failed(
        self,
        job_id: str,
        error: str,
        session_id: str | None = None,
    ) -> None:
        """Notify that a HuggingFace job has failed."""
        await self.notify(
            "job_failed",
            "Job Failed",
            f"Job {job_id} failed: {error}",
            {"job_id": job_id, "error": error, "session_id": session_id},
        )

    async def notify_error(
        self,
        error: str,
        session_id: str | None = None,
    ) -> None:
        """Notify that an error occurred."""
        await self.notify(
            "error",
            "Error Occurred",
            f"ML Intern encountered an error: {error}",
            {"error": error, "session_id": session_id},
        )

    async def notify_session_saved(
        self,
        session_id: str,
        repo_id: str | None = None,
    ) -> None:
        """Notify that session trajectory was saved."""
        await self.notify(
            "session_saved",
            "Session Saved",
            f"Session {session_id} trajectory saved",
            {"session_id": session_id, "repo_id": repo_id},
        )