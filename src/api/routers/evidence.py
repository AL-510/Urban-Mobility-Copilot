"""Endpoints for retrieving advisories and evidence."""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_retriever

router = APIRouter(prefix="/api/v1", tags=["evidence"])
logger = logging.getLogger(__name__)


@router.get("/evidence")
async def search_evidence(
    query: str = Query(..., description="Search query for advisories"),
    area: str | None = Query(None, description="Filter by area"),
    incident_type: str | None = Query(None, description="Filter by incident type"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results"),
    lat: float | None = Query(None, description="Center latitude for spatial filter"),
    lon: float | None = Query(None, description="Center longitude for spatial filter"),
    retriever=Depends(get_retriever),
):
    if retriever is None:
        return {"results": [], "message": "Vector DB not available"}

    results = retriever.retrieve(
        query=query,
        top_k=top_k,
        area=area,
        incident_type=incident_type,
        lat=lat,
        lon=lon,
    )
    return {"results": results, "count": len(results)}


@router.get("/alerts")
async def get_active_alerts(
    area: str | None = Query(None),
    top_k: int = Query(10, ge=1, le=50),
    retriever=Depends(get_retriever),
):
    if retriever is None:
        return {"results": [], "message": "Vector DB not available"}

    results = retriever.retrieve_active_alerts(
        timestamp=datetime.now(), area=area, top_k=top_k
    )
    return {"results": results, "count": len(results)}
