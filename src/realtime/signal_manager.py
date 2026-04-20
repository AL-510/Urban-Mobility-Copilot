"""Network signal manager — single source of truth for current conditions.

Bridges the DisruptionSimulator (or real APIs in the future) with the
route recommendation pipeline.  Provides a clean interface for:
  - Current weather features (for FeatureEngine)
  - Current incidents DataFrame (for FeatureEngine + label gen)
  - Network status summary (for frontend dashboard)
  - Signal freshness timestamps
"""

import logging
from datetime import datetime

from src.realtime.simulator import DisruptionSimulator

logger = logging.getLogger(__name__)


class SignalManager:
    """Centralises all real-time (or simulated) network signals."""

    def __init__(self, simulator: DisruptionSimulator | None = None):
        self.simulator = simulator or DisruptionSimulator()
        self._manual_weather_override: dict | None = None

    def start(self):
        """Start the underlying simulator."""
        self.simulator.start()

    def stop(self):
        self.simulator.stop()

    # ── query interface (used by RouteService / scorers) ────────

    def get_weather_features(self) -> dict:
        """Weather dict compatible with FeatureEngine / Predictor."""
        if self._manual_weather_override:
            return self._manual_weather_override
        return self.simulator.get_weather_features()

    def get_incidents_df(self):
        """Active incidents as a pandas DataFrame."""
        return self.simulator.get_incidents_df()

    def get_service_alerts(self) -> list[dict]:
        """Current service alerts for the frontend."""
        state = self.simulator.get_state()
        return state.get("service_alerts", [])

    # ── network status (for dashboard / health) ─────────────────

    def get_network_status(self) -> dict:
        """Full network status snapshot for /api/v1/network-status."""
        state = self.simulator.get_state()
        weather = self.get_weather_features()
        return {
            "timestamp": datetime.now().isoformat(),
            "last_signal_update": state["last_tick"],
            "tick_count": state["tick_count"],
            "incidents": {
                "active": state["incident_count_active"],
                "resolving": state["incident_count_resolving"],
                "details": state["active_incidents"],
            },
            "weather": {
                "condition": state["weather"].get("name", "unknown"),
                "severity": round(weather.get("weather_severity", 0), 3),
                "precip_mm": round(weather.get("precip_mm", 0), 1),
                "wind_speed_kmh": round(weather.get("wind_speed_kmh", 0), 1),
                "visibility_m": round(weather.get("visibility_m", 10000)),
            },
            "alerts": state["service_alerts"],
            "signal_source": "simulated",
        }

    # ── manual controls (for demo) ──────────────────────────────

    def inject_incident(self, incident_type: str = "accident",
                        corridor: str | None = None,
                        severity: float | None = None) -> dict:
        """Manually inject an incident and return it."""
        return self.simulator.force_inject(incident_type, corridor, severity)

    def override_weather(self, weather: dict | None):
        """Override weather features (set None to resume simulator weather)."""
        self._manual_weather_override = weather
