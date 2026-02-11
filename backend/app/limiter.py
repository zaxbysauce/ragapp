"""Shared slowapi limiter with health check whitelist."""

import hmac
import logging
from typing import Any, Callable, Optional

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.config import settings

logger = logging.getLogger("rate_limit")


def _should_whitelist(request: Request) -> bool:
    """Check if request should be whitelisted from rate limiting.

    Returns True if the request has a valid X-API-Key header matching the
    configured health_check_api_key, causing the request to bypass rate limits.
    """
    key = request.headers.get("X-API-Key")
    if key and hmac.compare_digest(key, settings.health_check_api_key):
        logger.info(
            "Whitelist hit", extra={
                "client_ip": request.client.host if request.client else None,
                "request_id": request.headers.get("X-Request-ID"),
                "reason": "health-check whitelist",
            }
        )
        return True
    return False


class WhitelistLimiter(Limiter):
    """Custom limiter that exempts health check requests from rate limiting."""

    def _check_request_limit(
        self,
        request: Request,
        endpoint_func: Optional[Callable[..., Any]],
        in_middleware: bool = True,
    ) -> None:
        """Skip rate limiting if the request is whitelisted."""
        if _should_whitelist(request):
            return
        super()._check_request_limit(request, endpoint_func, in_middleware)


# Create the limiter instance
limiter = WhitelistLimiter(key_func=get_remote_address)
