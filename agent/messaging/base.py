from abc import ABC, abstractmethod

import httpx

from agent.messaging.models import (
    DestinationConfig,
    NotificationRequest,
    NotificationResult,
)


class NotificationError(Exception):
    """Delivery failed and should not be retried."""


class RetryableNotificationError(NotificationError):
    """Delivery failed transiently and can be retried."""


class NotificationProvider(ABC):
    provider_name: str

    @abstractmethod
    async def send(
        self,
        client: httpx.AsyncClient,
        destination_name: str,
        destination: DestinationConfig,
        request: NotificationRequest,
    ) -> NotificationResult:
        """Deliver a notification to one destination."""
