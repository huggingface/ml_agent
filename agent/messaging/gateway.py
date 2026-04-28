import asyncio
import logging
from collections.abc import Iterable

import httpx

from agent.messaging.base import (
    NotificationError,
    NotificationProvider,
    RetryableNotificationError,
)
from agent.messaging.models import (
    MessagingConfig,
    NotificationRequest,
    NotificationResult,
)
from agent.messaging.slack import SlackProvider

logger = logging.getLogger(__name__)

_RETRY_DELAYS = (1, 2, 4)


class NotificationGateway:
    def __init__(self, config: MessagingConfig):
        self.config = config
        self._providers: dict[str, NotificationProvider] = {
            "slack": SlackProvider(),
        }
        self._queue: asyncio.Queue[NotificationRequest] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._client: httpx.AsyncClient | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    async def start(self) -> None:
        if not self.enabled or self._worker_task is not None:
            return
        self._client = httpx.AsyncClient(timeout=10.0)
        self._worker_task = asyncio.create_task(self._worker(), name="notification-gateway")

    async def flush(self) -> None:
        if not self.enabled:
            return
        await self._queue.join()

    async def close(self) -> None:
        if not self.enabled:
            return
        await self.flush()
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send(self, request: NotificationRequest) -> NotificationResult:
        if not self.enabled:
            return NotificationResult(
                destination=request.destination,
                ok=False,
                provider="disabled",
                error="Messaging is disabled",
            )

        destination = self.config.get_destination(request.destination)
        if destination is None:
            return NotificationResult(
                destination=request.destination,
                ok=False,
                provider="unknown",
                error=f"Unknown destination '{request.destination}'",
            )

        provider = self._providers.get(destination.provider)
        if provider is None:
            return NotificationResult(
                destination=request.destination,
                ok=False,
                provider=destination.provider,
                error=f"No provider implementation for '{destination.provider}'",
            )
        return await self._send_with_retries(provider, request.destination, destination, request)

    async def send_many(
        self, requests: Iterable[NotificationRequest]
    ) -> list[NotificationResult]:
        results: list[NotificationResult] = []
        for request in requests:
            results.append(await self.send(request))
        return results

    async def enqueue(self, request: NotificationRequest) -> bool:
        if not self.enabled or self._worker_task is None:
            return False
        await self._queue.put(request)
        return True

    async def _worker(self) -> None:
        while True:
            request = await self._queue.get()
            try:
                result = await self.send(request)
                if not result.ok:
                    logger.warning(
                        "Notification delivery failed for %s: %s",
                        request.destination,
                        result.error,
                    )
            except Exception:
                logger.exception("Unexpected notification worker failure")
            finally:
                self._queue.task_done()

    async def _send_with_retries(
        self,
        provider: NotificationProvider,
        destination_name: str,
        destination,
        request: NotificationRequest,
    ) -> NotificationResult:
        client = self._client or httpx.AsyncClient(timeout=10.0)
        owns_client = self._client is None
        try:
            for attempt in range(len(_RETRY_DELAYS) + 1):
                try:
                    return await provider.send(client, destination_name, destination, request)
                except RetryableNotificationError as exc:
                    if attempt >= len(_RETRY_DELAYS):
                        return NotificationResult(
                            destination=destination_name,
                            ok=False,
                            provider=provider.provider_name,
                            error=str(exc),
                        )
                    delay = _RETRY_DELAYS[attempt]
                    logger.warning(
                        "Retrying notification to %s in %ss after transient error: %s",
                        destination_name,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                except NotificationError as exc:
                    return NotificationResult(
                        destination=destination_name,
                        ok=False,
                        provider=provider.provider_name,
                        error=str(exc),
                    )
            return NotificationResult(
                destination=destination_name,
                ok=False,
                provider=provider.provider_name,
                error="Notification delivery exhausted retries",
            )
        finally:
            if owns_client:
                await client.aclose()
