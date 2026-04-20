"""Forecast region registry — manages trained ML regions and coverage analysis."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ForecastRegion:
    """A geographic region with a trained disruption forecast model."""
    name: str
    center_lat: float
    center_lon: float
    radius_km: float
    model_name: str
    graph_path: str  # Relative to project root
    checkpoint_path: str  # Relative to project root
    enabled: bool = True

    @property
    def radius_deg(self) -> float:
        return self.radius_km / 111.0

    def contains(self, lat: float, lon: float, margin_km: float = 2.0) -> bool:
        """Check if a point is within this region (with margin)."""
        margin_deg = margin_km / 111.0
        cos_lat = np.cos(np.radians(self.center_lat))
        dlat = abs(lat - self.center_lat)
        dlon = abs(lon - self.center_lon) * cos_lat
        dist_deg = np.sqrt(dlat**2 + dlon**2)
        return dist_deg <= (self.radius_deg + margin_deg)


class RegionRegistry:
    """Registry of trained forecast regions.

    Currently loads Portland from settings. Designed to support
    multiple regions via config in the future.
    """

    def __init__(self):
        self.regions: list[ForecastRegion] = []
        self._load_from_settings()

    def _load_from_settings(self):
        """Load the default Portland region from settings."""
        settings = get_settings()
        portland = ForecastRegion(
            name=settings.city_name,
            center_lat=settings.city_center_lat,
            center_lon=settings.city_center_lon,
            radius_km=settings.city_radius_km,
            model_name=settings.model_name,
            graph_path=str(settings.processed_dir / "transport_graph_train.pkl"),
            checkpoint_path=str(settings.checkpoint_dir / f"{settings.model_name}_best.pt"),
        )
        self.regions.append(portland)
        logger.info(f"Registered forecast region: {portland.name} "
                     f"({portland.center_lat}, {portland.center_lon}, r={portland.radius_km}km)")

    def find_regions_for_point(self, lat: float, lon: float) -> list[ForecastRegion]:
        """Find all regions containing a point."""
        return [r for r in self.regions if r.enabled and r.contains(lat, lon)]

    def analyze_route_coverage(
        self,
        coordinates: list[list[float]],
    ) -> tuple[str, float, str, list[ForecastRegion]]:
        """Analyze what forecast regions a route passes through.

        Args:
            coordinates: List of [lat, lon] pairs along the route

        Returns:
            (tier, coverage_pct, note, matching_regions)
        """
        if not coordinates:
            return "base", 0.0, "No route coordinates to analyze.", []

        # Sample points along route (at most 20 for efficiency)
        n = len(coordinates)
        step = max(1, n // 20)
        sample_points = coordinates[::step]
        if coordinates[-1] not in sample_points:
            sample_points.append(coordinates[-1])

        # Check each sample point against all regions
        covered_count = 0
        matching_regions: set[str] = set()
        all_matching: list[ForecastRegion] = []

        for point in sample_points:
            lat, lon = point[0], point[1]
            regions = self.find_regions_for_point(lat, lon)
            if regions:
                covered_count += 1
                for r in regions:
                    if r.name not in matching_regions:
                        matching_regions.add(r.name)
                        all_matching.append(r)

        coverage_pct = (covered_count / len(sample_points)) * 100 if sample_points else 0

        if coverage_pct >= 80:
            tier = "full"
            note = ""
        elif coverage_pct > 0:
            tier = "partial"
            region_names = ", ".join(matching_regions)
            note = (
                f"Route partially covered by forecast region(s): {region_names}. "
                f"{coverage_pct:.0f}% of route has predictive intelligence."
            )
        else:
            tier = "base"
            note = (
                "Route is outside all trained forecast regions. "
                "Showing time/distance estimates with heuristic risk scoring."
            )

        return tier, round(coverage_pct, 1), note, all_matching

    def get_all_regions(self) -> list[dict]:
        """Return region info for API/frontend consumption."""
        return [
            {
                "name": r.name,
                "center_lat": r.center_lat,
                "center_lon": r.center_lon,
                "radius_km": r.radius_km,
                "model_name": r.model_name,
                "enabled": r.enabled,
            }
            for r in self.regions
        ]
