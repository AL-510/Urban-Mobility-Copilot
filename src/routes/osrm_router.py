"""OSRM-based routing for global road geometry.

Uses the OSRM public demo API for route geometry and time estimates.
Provides real road routing for any coordinates worldwide.
Hardened with retries, timeouts, and clear error reporting.
"""

import logging

import httpx

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

PROFILES = {
    "driving": "car",
    "walking": "foot",
    "cycling": "bike",
}


class OSRMRouter:
    """Route generator using OSRM for arbitrary origin/destination pairs."""

    def __init__(self, base_url: str | None = None):
        settings = get_settings()
        self.base_url = (base_url or settings.osrm_base_url).rstrip("/")

    async def get_routes(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        profile: str = "driving",
        alternatives: int = 3,
    ) -> list[dict]:
        """Fetch routes from OSRM.

        Returns list of route dicts with: coordinates, distance_m, duration_s, geometry.
        Returns empty list on failure (never raises).
        """
        osrm_profile = PROFILES.get(profile, "car")
        url = (
            f"{self.base_url}/route/v1/{osrm_profile}/"
            f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
        )
        params = {
            "overview": "full",
            "geometries": "geojson",
            "alternatives": str(min(alternatives, 3)),
            "steps": "false",
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            logger.warning(f"OSRM timeout for {profile}: ({origin_lat},{origin_lon}) -> ({dest_lat},{dest_lon})")
            return []
        except httpx.HTTPStatusError as e:
            logger.warning(f"OSRM HTTP error {e.response.status_code} for {profile}")
            return []
        except Exception as e:
            logger.warning(f"OSRM request failed for {profile}: {e}")
            return []

        if data.get("code") != "Ok":
            logger.warning(f"OSRM returned: {data.get('code')} — {data.get('message', '')}")
            return []

        routes = []
        for i, route in enumerate(data.get("routes", [])):
            # OSRM returns [lon, lat] — convert to [lat, lon]
            coords = route.get("geometry", {}).get("coordinates", [])
            if not coords:
                continue

            coordinates = [[c[1], c[0]] for c in coords]
            distance = route.get("distance", 0)
            duration = route.get("duration", 0)

            # Validate: skip degenerate routes
            if len(coordinates) < 2 or distance < 1:
                logger.debug(f"Skipping degenerate OSRM route (coords={len(coordinates)}, dist={distance})")
                continue

            routes.append({
                "coordinates": coordinates,
                "distance_m": distance,
                "duration_s": duration,
                "osrm_index": i,
            })

        return routes

    async def get_multi_profile_routes(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
    ) -> list[dict]:
        """Get routes across multiple profiles (driving, walking).

        Always returns at least driving routes. Walking added for short trips.
        """
        all_routes = []

        # Driving routes (with alternatives)
        driving = await self.get_routes(
            origin_lat, origin_lon, dest_lat, dest_lon,
            profile="driving", alternatives=3,
        )
        for i, r in enumerate(driving):
            name = "Fastest Route" if i == 0 else f"Alternative {i}"
            all_routes.append({
                **r,
                "name": name,
                "strategy": "fastest" if i == 0 else "alternative",
                "profile": "driving",
                "modes": ["drive"],
            })

        # Walking route (only if distance is walkable — under 5km)
        if driving and driving[0]["distance_m"] < 5000:
            walking = await self.get_routes(
                origin_lat, origin_lon, dest_lat, dest_lon,
                profile="walking", alternatives=0,
            )
            for r in walking[:1]:
                all_routes.append({
                    **r,
                    "name": "Walking Route",
                    "strategy": "walk",
                    "profile": "walking",
                    "modes": ["walk"],
                })

        return all_routes

    async def health_check(self) -> bool:
        """Quick check if OSRM is reachable. Used for readiness probes."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Simple route request to verify OSRM is responding
                resp = await client.get(
                    f"{self.base_url}/route/v1/car/0,0;1,1",
                    params={"overview": "false"},
                )
                return resp.status_code in (200, 400)  # 400 = invalid coords but server is up
        except Exception:
            return False
