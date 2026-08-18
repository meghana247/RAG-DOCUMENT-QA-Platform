"""Cross-cutting HTTP middleware: correlation IDs, access logs, rate limiting."""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.logging import new_request_id, request_id_var, user_id_var

logger = logging.getLogger("app.access")

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request id and logs one line per request.

    An inbound X-Request-ID is honoured so a trace can be followed across a
    gateway or a frontend; otherwise one is generated. It is echoed on the
    response so a user reporting a problem can quote it.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or new_request_id()
        token = request_id_var.set(request_id)
        user_token = user_id_var.set(None)

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - started) * 1000
            # The exception handlers produce the response body; this records the
            # access-log line that would otherwise be missing for a 500.
            logger.exception(
                "request errored",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration_ms, 1),
                },
            )
            raise
        finally:
            request_id_var.reset(token)
            user_id_var.reset(user_token)

        duration_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(duration_ms, 1),
            },
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window-per-caller limiter.

    Deliberately simple and in-process: it protects a single instance from a
    runaway client, and costs nothing. It is *not* a distributed limit — with more
    than one replica each holds its own counters, so the effective limit is
    `replicas x limit`. Move the counters to Redis before relying on this for
    abuse prevention or billing protection.
    """

    def __init__(self, app, requests_per_minute: int, exempt_paths: frozenset[str]) -> None:
        super().__init__(app)
        self._limit = requests_per_minute
        self._exempt = exempt_paths
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _identify(self, request: Request) -> str:
        # Bearer tokens are per-user, so the token prefix is a stable key that
        # does not require decoding the JWT in middleware. Falls back to the peer
        # address for unauthenticated calls.
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return f"token:{auth[7:47]}"
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        return f"ip:{request.client.host if request.client else 'unknown'}"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if self._limit <= 0 or request.url.path in self._exempt:
            return await call_next(request)

        key = self._identify(request)
        now = time.monotonic()
        window = self._hits[key]

        while window and now - window[0] > 60.0:
            window.popleft()

        if len(window) >= self._limit:
            retry_after = int(60.0 - (now - window[0])) + 1
            logger.warning("rate limit exceeded", extra={"caller": key})
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "rate_limited",
                        "message": f"Rate limit of {self._limit} requests/minute exceeded",
                    }
                },
                headers={"Retry-After": str(retry_after)},
            )

        window.append(now)

        # Unbounded growth guard: drop keys that have aged out entirely.
        if len(self._hits) > 10_000:
            for stale in [k for k, v in self._hits.items() if not v]:
                del self._hits[stale]

        return await call_next(request)
