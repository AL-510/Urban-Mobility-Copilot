"""Geocoding router — address search, autocomplete, and reverse geocoding via Nominatim (OSM).

Hardened for production: all external calls wrapped in error handling,
global search supported alongside optional service-area bias.
"""

import logging

import httpx
from fastapi import APIRouter, HTTPException, Query

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["geocoding"])

NOMINATIM_URL = "https://nominatim.openstreetmap.org"
HEADERS = {"User-Agent": "UrbanMobilityCopilot/0.5.0 (research project)"}


def _get_viewbox() -> str | None:
    """Build viewbox from settings service area, or None for global search."""
    settings = get_settings()
    if not settings.service_area_enabled:
        return None
    return (
        f"{settings.service_area_min_lon},{settings.service_area_min_lat},"
        f"{settings.service_area_max_lon},{settings.service_area_max_lat}"
    )


@router.get("/geocode", summary="Forward geocode — address/place to coordinates")
async def geocode(
    q: str = Query(..., min_length=1, description="Address, place name, or landmark"),
    limit: int = Query(8, ge=1, le=15),
    bias_to_service_area: bool = Query(False, description="Bias results toward service area"),
):
    """Search for places by name/address. Works globally by default."""
    params: dict = {
        "q": q,
        "format": "jsonv2",
        "limit": limit,
        "addressdetails": 1,
        "extratags": 1,
    }

    viewbox = _get_viewbox()
    if bias_to_service_area and viewbox:
        params["viewbox"] = viewbox
        params["bounded"] = 0

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{NOMINATIM_URL}/search",
                params=params,
                headers=HEADERS,
            )
            resp.raise_for_status()
            results = resp.json()
    except httpx.TimeoutException:
        logger.warning(f"Geocode timeout for query: {q}")
        raise HTTPException(status_code=504, detail="Geocoding service timed out. Please try again.")
    except httpx.HTTPStatusError as e:
        logger.warning(f"Geocode HTTP error: {e.response.status_code}")
        raise HTTPException(status_code=502, detail="Geocoding service returned an error.")
    except Exception as e:
        logger.warning(f"Geocode failed: {e}")
        raise HTTPException(status_code=502, detail="Geocoding service unavailable.")

    return [
        {
            "display_name": r.get("display_name", ""),
            "short_name": _short_name(r),
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "type": r.get("type", ""),
            "category": r.get("category", ""),
            "importance": r.get("importance", 0),
            "place_id": r.get("place_id"),
            "address": r.get("address", {}),
        }
        for r in results
    ]


@router.get("/autocomplete", summary="Autocomplete place search")
async def autocomplete(
    q: str = Query(..., min_length=1, description="Partial query for autocomplete"),
    limit: int = Query(6, ge=1, le=10),
):
    """Fast autocomplete for place/address search. Global by default."""
    params: dict = {
        "q": q,
        "format": "jsonv2",
        "limit": limit,
        "addressdetails": 1,
    }

    # No viewbox bias for autocomplete — users search globally
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{NOMINATIM_URL}/search",
                params=params,
                headers=HEADERS,
            )
            resp.raise_for_status()
            results = resp.json()
    except httpx.TimeoutException:
        logger.warning(f"Autocomplete timeout for query: {q}")
        return []  # Graceful degradation — empty results, not error
    except Exception as e:
        logger.warning(f"Autocomplete failed: {e}")
        return []

    return [
        {
            "display_name": r.get("display_name", ""),
            "short_name": _short_name(r),
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "type": r.get("type", ""),
            "category": r.get("category", ""),
        }
        for r in results
    ]


@router.get("/reverse-geocode", summary="Reverse geocode — coordinates to address")
async def reverse_geocode(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    """Convert coordinates to a human-readable address."""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{NOMINATIM_URL}/reverse",
                params={"lat": lat, "lon": lon, "format": "jsonv2", "addressdetails": 1},
                headers=HEADERS,
            )
            resp.raise_for_status()
            result = resp.json()
    except httpx.TimeoutException:
        logger.warning(f"Reverse geocode timeout for ({lat}, {lon})")
        # Return coordinate-based fallback instead of crashing
        return {
            "display_name": f"{lat:.4f}, {lon:.4f}",
            "short_name": f"{lat:.4f}, {lon:.4f}",
            "lat": lat,
            "lon": lon,
            "type": "coordinate",
            "address": {},
        }
    except Exception as e:
        logger.warning(f"Reverse geocode failed: {e}")
        return {
            "display_name": f"{lat:.4f}, {lon:.4f}",
            "short_name": f"{lat:.4f}, {lon:.4f}",
            "lat": lat,
            "lon": lon,
            "type": "coordinate",
            "address": {},
        }

    return {
        "display_name": result.get("display_name", f"{lat:.4f}, {lon:.4f}"),
        "short_name": _short_name(result),
        "lat": float(result.get("lat", lat)),
        "lon": float(result.get("lon", lon)),
        "type": result.get("type", ""),
        "address": result.get("address", {}),
    }


def _short_name(result: dict) -> str:
    """Extract a concise name from a Nominatim result."""
    addr = result.get("address", {})
    name = result.get("name") or addr.get("amenity") or addr.get("shop") or ""
    road = addr.get("road", "")
    city = addr.get("city") or addr.get("town") or addr.get("village") or ""

    if name and city:
        return f"{name}, {city}"
    if name and road:
        return f"{name}, {road}"
    if road and city:
        return f"{road}, {city}"

    # Fallback: first 2 parts of display_name
    parts = result.get("display_name", "").split(",")
    return ", ".join(p.strip() for p in parts[:2])
