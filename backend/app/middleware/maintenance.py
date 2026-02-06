"""Middleware that blocks writes during maintenance."""

from typing import Callable

from fastapi import FastAPI, Request, Response

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.services.maintenance import MaintenanceService


class MaintenanceMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, service: MaintenanceService) -> None:
        super().__init__(app)
        self.service = service

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        flag = self.service.get_flag()
        if flag.enabled and request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
            return Response(
                content="maintenance", status_code=503, media_type="application/json"
            )
        return await call_next(request)
