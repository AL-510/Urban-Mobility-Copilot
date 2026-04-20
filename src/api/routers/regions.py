"""API router for forecast region information."""

import logging

from fastapi import APIRouter, Depends

from src.api.dependencies import get_region_registry

router = APIRouter(prefix="/api/v1", tags=["regions"])
logger = logging.getLogger(__name__)


@router.get(
    "/regions",
    summary="List all forecast regions",
    description="Returns configured forecast regions with center, radius, and model info.",
)
def list_regions(registry=Depends(get_region_registry)):
    if registry is None:
        return {"regions": []}
    return {"regions": registry.get_all_regions()}


@router.get(
    "/regions/coverage",
    summary="Check forecast coverage for a point",
    description="Check which forecast regions cover a given lat/lon coordinate.",
)
def check_coverage(lat: float, lon: float, registry=Depends(get_region_registry)):
    if registry is None:
        return {"regions": [], "covered": False}
    matching = registry.find_regions_for_point(lat, lon)
    return {
        "lat": lat,
        "lon": lon,
        "covered": len(matching) > 0,
        "regions": [
            {"name": r.name, "model_name": r.model_name, "radius_km": r.radius_km}
            for r in matching
        ],
    }
