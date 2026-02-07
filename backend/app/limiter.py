"""Shared slowapi limiter with whitelist filter."""

import logging
from functools import wraps

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.config import settings

logger = logging.getLogger("rate_limit")


def _should_whitelist(request: Request) -> bool:
    """Check if request should be whitelisted from rate limiting."""
    key = request.headers.get("X-API-Key")
    if key and key == settings.health_check_api_key:
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
    """Custom limiter that supports health-check whitelisting."""

    def _check_whitelist(self, request: Request) -> bool:
        """Check if request should skip rate limiting."""
        return _should_whitelist(request)


# Create the limiter instance
limiter = WhitelistLimiter(key_func=get_remote_address)


def exempt_when_health_check(func):
    """
    Decorator to exempt health check requests from rate limiting.

    Use this decorator on endpoint functions to check for health check API key
    before applying rate limits. If the X-API-Key header matches the configured
    health_check_api_key, rate limiting is skipped.
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Extract request from args/kwargs
        request = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        if not request:
            request = kwargs.get('request')

        if request and _should_whitelist(request):
            return await func(*args, **kwargs)
        # Continue to rate limit check
        return await func(*args, **kwargs)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        # Extract request from args/kwargs
        request = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        if not request:
            request = kwargs.get('request')

        if request and _should_whitelist(request):
            return func(*args, **kwargs)
        # Continue to rate limit check
        return func(*args, **kwargs)

    # Return appropriate wrapper based on whether func is async
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
