"""API router for data freshness monitoring."""

import logging

from fastapi import APIRouter, HTTPException

from src.api.dependencies import get_refresh_manager

router = APIRouter(prefix="/api/v1", tags=["freshness"])
logger = logging.getLogger(__name__)


@router.get(
    "/data-freshness",
    summary="Get freshness status of all data sources",
    description="Returns last refresh time, next refresh, and status for each data source.",
)
def get_data_freshness():
    refresh_manager = get_refresh_manager()
    if refresh_manager is None:
        return {
            "timestamp": None,
            "sources": {},
            "note": "Refresh manager not initialized — data freshness tracking unavailable.",
        }
    return refresh_manager.get_freshness()


@router.post(
    "/data-freshness/{source}/refresh",
    summary="Manually trigger a data source refresh",
)
def trigger_refresh(source: str):
    refresh_manager = get_refresh_manager()
    if refresh_manager is None:
        raise HTTPException(status_code=503, detail="Refresh manager not initialized")
    success = refresh_manager.trigger_refresh(source)
    if not success:
        raise HTTPException(status_code=404, detail=f"Unknown data source: {source}")
    return {"status": "ok", "source": source, "message": f"Refresh triggered for {source}"}
