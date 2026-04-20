"""Network status and real-time signal endpoints."""

from fastapi import APIRouter

from src.api.dependencies import get_signal_manager

router = APIRouter(prefix="/api/v1", tags=["network"])


@router.get("/network-status")
def get_network_status():
    """Current network disruption state: active incidents, weather, alerts."""
    sm = get_signal_manager()
    if sm is None:
        return {"error": "Signal manager not initialized", "incidents": {"active": 0},
                "weather": {"condition": "unknown"}, "alerts": []}
    return sm.get_network_status()


@router.post("/inject-incident")
def inject_incident(incident_type: str = "accident", corridor: str | None = None,
                    severity: float | None = None):
    """Manually inject a disruption (for demo / testing)."""
    sm = get_signal_manager()
    if sm is None:
        return {"error": "Signal manager not initialized"}
    inc = sm.inject_incident(incident_type, corridor, severity)
    return {"injected": inc}


@router.post("/refresh-signals")
def refresh_signals():
    """Force a signal refresh tick (normally happens every 30s)."""
    sm = get_signal_manager()
    if sm is None:
        return {"error": "Signal manager not initialized"}
    sm.simulator._tick()
    return sm.get_network_status()
