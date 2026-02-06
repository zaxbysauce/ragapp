"""Shared slowapi limiter with whitelist filter."""

import logging

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings

logger = logging.getLogger("rate_limit")

def _whitelist(request):
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

limiter = Limiter(key_func=get_remote_address)
limiter.request_filter(_whitelist)
