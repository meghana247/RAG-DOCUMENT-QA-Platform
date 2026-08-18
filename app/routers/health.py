from fastapi import APIRouter, Response, status

from app.core.database import check_connection
from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])

VERSION = "2.0.0"


@router.get("/health", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Liveness: is the process up? Deliberately touches no dependency.

    A liveness probe that checks the database will restart a healthy application
    during a brief database blip, turning a recoverable outage into a crash loop.
    """
    return HealthResponse(status="ok", version=VERSION, database="not_checked")


@router.get("/health/ready", response_model=HealthResponse)
async def readiness(response: Response) -> HealthResponse:
    """Readiness: can this instance serve traffic? Probes the database.

    Returns 503 when the database is unreachable so the load balancer stops
    routing here without the process being killed.
    """
    connected = await check_connection()
    if not connected:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if connected else "degraded",
        version=VERSION,
        database="up" if connected else "down",
    )
