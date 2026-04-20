"""Health check, readiness, and system status endpoints."""

import logging

from fastapi import APIRouter

from src.api.schemas.routes import HealthResponse
from src.api.dependencies import get_app_state, get_signal_manager, get_refresh_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness check — confirms the service process is running."""
    state = get_app_state()
    return HealthResponse(
        status="healthy",
        version="0.5.0",
        model_loaded=state.get("model_loaded", False),
        graph_nodes=state.get("graph_nodes", 0),
        graph_edges=state.get("graph_edges", 0),
        vector_db_status=state.get("vector_db_status", "unknown"),
        realtime_enabled=state.get("realtime_enabled", False),
    )


@router.get("/ready")
async def readiness_check():
    """Readiness check — confirms all dependencies are available for serving traffic.

    Returns 200 if ready, 503 if not.
    """
    state = get_app_state()
    checks = {
        "model_loaded": state.get("model_loaded", False),
        "graph_available": state.get("graph_nodes", 0) > 0,
        "vector_db": state.get("vector_db_status", "disconnected") != "disconnected",
        "realtime": state.get("realtime_enabled", False),
    }

    # Check refresh manager health
    rm = get_refresh_manager()
    if rm:
        freshness = rm.get_freshness()
        error_sources = [
            name for name, src in freshness.get("sources", {}).items()
            if src.get("status") == "error"
        ]
        checks["refresh_jobs"] = len(error_sources) == 0
        if error_sources:
            checks["refresh_errors"] = error_sources
    else:
        checks["refresh_jobs"] = False

    all_ready = checks["model_loaded"] and checks["graph_available"]
    status_code = 200 if all_ready else 503

    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content={
            "ready": all_ready,
            "checks": checks,
            "version": "0.5.0",
        },
    )


@router.get("/")
async def root():
    """API root — basic service info for discovery."""
    sm = get_signal_manager()
    active_incidents = 0
    if sm:
        try:
            status = sm.get_network_status()
            active_incidents = status["incidents"]["active"]
        except Exception:
            pass
    return {
        "name": "Urban Mobility Disruption Copilot",
        "version": "0.5.0",
        "description": "Disruption-aware route intelligence with ST-GAT forecasting",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "realtime": sm is not None,
        "active_incidents": active_incidents,
    }
