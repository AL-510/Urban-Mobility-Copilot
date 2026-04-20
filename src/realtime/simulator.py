"""Simulated real-time disruption engine.

Generates time-evolving incidents, weather changes, and service alerts that
update the network state dynamically.  This powers the "live" dashboard
experience without requiring real external APIs.

Incident lifecycle:
  1. Created (severity escalation possible)
  2. Active (affects nearby nodes)
  3. Resolving (severity decreasing)
  4. Cleared
"""

import logging
import random
import threading
import time
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# Portland-area corridors for realistic incident placement
CORRIDORS = [
    {"name": "I-5 Northbound",     "lat": 45.520, "lon": -122.681, "radius_deg": 0.008},
    {"name": "I-405 Loop",         "lat": 45.517, "lon": -122.685, "radius_deg": 0.005},
    {"name": "Burnside Bridge",    "lat": 45.523, "lon": -122.670, "radius_deg": 0.003},
    {"name": "Morrison Bridge",    "lat": 45.518, "lon": -122.670, "radius_deg": 0.003},
    {"name": "Steel Bridge",       "lat": 45.527, "lon": -122.672, "radius_deg": 0.003},
    {"name": "SW Broadway",        "lat": 45.515, "lon": -122.679, "radius_deg": 0.004},
    {"name": "NW 23rd Ave",        "lat": 45.530, "lon": -122.699, "radius_deg": 0.004},
    {"name": "SE Hawthorne Blvd",  "lat": 45.512, "lon": -122.655, "radius_deg": 0.005},
    {"name": "MLK Jr Blvd",        "lat": 45.525, "lon": -122.661, "radius_deg": 0.005},
    {"name": "Division St",        "lat": 45.505, "lon": -122.660, "radius_deg": 0.005},
]

INCIDENT_TYPES = [
    {"type": "accident",       "weight": 0.30, "duration_range": (15, 90),  "severity_range": (0.4, 0.9)},
    {"type": "construction",   "weight": 0.20, "duration_range": (60, 480), "severity_range": (0.2, 0.6)},
    {"type": "road_closure",   "weight": 0.10, "duration_range": (30, 240), "severity_range": (0.7, 1.0)},
    {"type": "special_event",  "weight": 0.15, "duration_range": (120, 360),"severity_range": (0.3, 0.7)},
    {"type": "weather_hazard", "weight": 0.10, "duration_range": (30, 180), "severity_range": (0.3, 0.8)},
    {"type": "transit_delay",  "weight": 0.15, "duration_range": (10, 60),  "severity_range": (0.2, 0.5)},
]

WEATHER_SCENARIOS = [
    {"name": "clear",          "severity": 0.0,  "precip": 0.0, "wind": 5,  "visibility": 10000},
    {"name": "light_rain",     "severity": 0.15, "precip": 2.0, "wind": 12, "visibility": 8000},
    {"name": "moderate_rain",  "severity": 0.35, "precip": 8.0, "wind": 20, "visibility": 5000},
    {"name": "heavy_rain",     "severity": 0.6,  "precip": 20,  "wind": 35, "visibility": 2000},
    {"name": "fog",            "severity": 0.3,  "precip": 0.0, "wind": 3,  "visibility": 500},
    {"name": "ice_warning",    "severity": 0.7,  "precip": 1.0, "wind": 10, "visibility": 3000},
    {"name": "wind_advisory",  "severity": 0.4,  "precip": 0.0, "wind": 55, "visibility": 7000},
]


class LiveIncident:
    """A single simulated incident with a lifecycle."""

    _id_counter = 0

    def __init__(
        self,
        incident_type: str,
        corridor: str,
        lat: float,
        lon: float,
        severity: float,
        duration_min: int,
        created_at: datetime,
    ):
        LiveIncident._id_counter += 1
        self.id = f"LIVE-{LiveIncident._id_counter:04d}"
        self.incident_type = incident_type
        self.corridor = corridor
        self.lat = lat
        self.lon = lon
        self.severity = severity
        self.peak_severity = severity
        self.duration_min = duration_min
        self.created_at = created_at
        self.expires_at = created_at + timedelta(minutes=duration_min)
        self.status = "active"
        self.delay_factor = 1.0 + severity * 1.5

    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at

    @property
    def remaining_min(self) -> float:
        return max(0, (self.expires_at - datetime.now()).total_seconds() / 60)

    @property
    def age_min(self) -> float:
        return (datetime.now() - self.created_at).total_seconds() / 60

    def tick(self):
        """Update incident state based on elapsed time."""
        progress = self.age_min / self.duration_min
        if progress >= 1.0:
            self.status = "cleared"
            self.severity = 0
        elif progress >= 0.75:
            self.status = "resolving"
            self.severity = self.peak_severity * (1 - progress)
        else:
            self.status = "active"
        self.delay_factor = 1.0 + self.severity * 1.5

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "incident_type": self.incident_type,
            "corridor": self.corridor,
            "lat": self.lat,
            "lon": self.lon,
            "severity": round(self.severity, 3),
            "delay_factor": round(self.delay_factor, 2),
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "remaining_min": round(self.remaining_min, 1),
            "duration_min": self.duration_min,
        }


class DisruptionSimulator:
    """Background engine that generates and manages live disruptions.

    Runs on a configurable tick interval (default 30s).  Each tick:
      1. Expire old incidents
      2. Probabilistically create new incidents
      3. Update weather conditions (slowly drift)
      4. Update the shared network state
    """

    def __init__(
        self,
        tick_interval_s: int = 30,
        incident_rate_per_min: float = 0.15,
        max_concurrent_incidents: int = 8,
    ):
        self.tick_interval = tick_interval_s
        self.incident_rate = incident_rate_per_min
        self.max_incidents = max_concurrent_incidents

        self.incidents: list[LiveIncident] = []
        self.weather: dict[str, Any] = dict(WEATHER_SCENARIOS[0])  # start clear
        self.weather["last_change"] = datetime.now()
        self.service_alerts: list[dict] = []

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._tick_count = 0
        self.last_tick: datetime = datetime.now()

    # ── public API (thread-safe) ────────────────────────────────

    def start(self):
        """Start the background simulation loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"DisruptionSimulator started (tick={self.tick_interval}s)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_state(self) -> dict:
        """Return current network disruption state (thread-safe snapshot)."""
        with self._lock:
            return {
                "active_incidents": [i.to_dict() for i in self.incidents if i.status != "cleared"],
                "weather": dict(self.weather),
                "service_alerts": list(self.service_alerts),
                "last_tick": self.last_tick.isoformat(),
                "tick_count": self._tick_count,
                "incident_count_active": sum(1 for i in self.incidents if i.status == "active"),
                "incident_count_resolving": sum(1 for i in self.incidents if i.status == "resolving"),
            }

    def get_weather_features(self) -> dict:
        """Return weather dict compatible with FeatureEngine."""
        with self._lock:
            return {
                "weather_severity": self.weather.get("severity", 0),
                "precip_mm": self.weather.get("precip", 0),
                "wind_speed_kmh": self.weather.get("wind", 0),
                "visibility_m": self.weather.get("visibility", 10000),
            }

    def get_incidents_df(self):
        """Return active incidents as a DataFrame compatible with FeatureEngine."""
        import pandas as pd
        with self._lock:
            active = [i for i in self.incidents if i.status in ("active", "resolving")]
            if not active:
                return pd.DataFrame(columns=["lat", "lon", "start_time", "end_time",
                                              "severity", "delay_factor", "incident_type"])
            rows = []
            for inc in active:
                rows.append({
                    "lat": inc.lat,
                    "lon": inc.lon,
                    "start_time": inc.created_at.isoformat(),
                    "end_time": inc.expires_at.isoformat(),
                    "severity": "high" if inc.severity > 0.6 else "medium" if inc.severity > 0.3 else "low",
                    "delay_factor": inc.delay_factor,
                    "incident_type": inc.incident_type,
                })
            return pd.DataFrame(rows)

    def force_inject(self, incident_type: str = "accident", corridor_name: str | None = None,
                     severity: float | None = None) -> dict:
        """Manually inject an incident (for demo / testing)."""
        corridor = next((c for c in CORRIDORS if c["name"] == corridor_name), None)
        if corridor is None:
            corridor = random.choice(CORRIDORS)
        inc = self._create_incident(corridor, override_type=incident_type, override_severity=severity)
        with self._lock:
            self.incidents.append(inc)
        logger.info(f"Force-injected: {inc.id} ({inc.incident_type}) on {inc.corridor}")
        return inc.to_dict()

    # ── background loop ─────────────────────────────────────────

    def _run_loop(self):
        # Seed initial state with 1-2 incidents
        self._tick()
        self._tick()
        while self._running:
            time.sleep(self.tick_interval)
            self._tick()

    def _tick(self):
        with self._lock:
            self._tick_count += 1
            self.last_tick = datetime.now()

            # 1. Update existing incidents
            for inc in self.incidents:
                inc.tick()
            # Remove cleared incidents older than 5 minutes
            self.incidents = [
                i for i in self.incidents
                if i.status != "cleared" or i.age_min < i.duration_min + 5
            ]

            # 2. Possibly create new incidents
            active_count = sum(1 for i in self.incidents if i.status in ("active", "resolving"))
            if active_count < self.max_incidents:
                # Poisson-ish: expected incidents per tick
                expected = self.incident_rate * (self.tick_interval / 60)
                if random.random() < expected:
                    corridor = random.choice(CORRIDORS)
                    inc = self._create_incident(corridor)
                    self.incidents.append(inc)
                    logger.info(f"[tick {self._tick_count}] New incident: {inc.id} "
                                f"({inc.incident_type}) on {inc.corridor}, "
                                f"severity={inc.severity:.2f}, duration={inc.duration_min}min")

            # 3. Update weather (drift every ~5 minutes)
            weather_age = (datetime.now() - self.weather["last_change"]).total_seconds()
            if weather_age > 300 and random.random() < 0.3:
                self._drift_weather()

            # 4. Update service alerts
            self._update_service_alerts()

    def _create_incident(self, corridor: dict, override_type: str | None = None,
                         override_severity: float | None = None) -> LiveIncident:
        # Pick type
        if override_type:
            spec = next((s for s in INCIDENT_TYPES if s["type"] == override_type), INCIDENT_TYPES[0])
        else:
            types = INCIDENT_TYPES
            weights = [t["weight"] for t in types]
            spec = random.choices(types, weights=weights, k=1)[0]

        # Jitter location within corridor
        lat = corridor["lat"] + random.gauss(0, corridor["radius_deg"])
        lon = corridor["lon"] + random.gauss(0, corridor["radius_deg"])

        severity = override_severity if override_severity is not None else random.uniform(*spec["severity_range"])
        duration = random.randint(*spec["duration_range"])

        return LiveIncident(
            incident_type=spec["type"],
            corridor=corridor["name"],
            lat=lat, lon=lon,
            severity=severity,
            duration_min=duration,
            created_at=datetime.now(),
        )

    def _drift_weather(self):
        """Gradually shift weather toward a random scenario."""
        target = random.choice(WEATHER_SCENARIOS)
        alpha = 0.4  # blending factor
        self.weather["severity"] = (1 - alpha) * self.weather.get("severity", 0) + alpha * target["severity"]
        self.weather["precip"] = (1 - alpha) * self.weather.get("precip", 0) + alpha * target["precip"]
        self.weather["wind"] = (1 - alpha) * self.weather.get("wind", 5) + alpha * target["wind"]
        self.weather["visibility"] = (1 - alpha) * self.weather.get("visibility", 10000) + alpha * target["visibility"]
        self.weather["name"] = target["name"]
        self.weather["last_change"] = datetime.now()
        logger.info(f"Weather drift → {target['name']} (severity={self.weather['severity']:.2f})")

    def _update_service_alerts(self):
        """Generate service alerts from current state."""
        alerts = []
        active = [i for i in self.incidents if i.status == "active" and i.severity > 0.4]
        for inc in active[:5]:
            severity_label = "HIGH" if inc.severity > 0.6 else "MODERATE"
            alerts.append({
                "id": inc.id,
                "type": inc.incident_type,
                "title": f"{severity_label}: {inc.incident_type.replace('_', ' ').title()} — {inc.corridor}",
                "body": (f"{inc.incident_type.replace('_', ' ').title()} reported on {inc.corridor}. "
                         f"Expected delay factor {inc.delay_factor:.1f}x. "
                         f"Estimated {inc.remaining_min:.0f} min remaining."),
                "severity": severity_label.lower(),
                "lat": inc.lat,
                "lon": inc.lon,
                "created_at": inc.created_at.isoformat(),
                "expires_at": inc.expires_at.isoformat(),
            })
        # Weather alert
        if self.weather.get("severity", 0) > 0.25:
            alerts.append({
                "id": "WEATHER-LIVE",
                "type": "weather",
                "title": f"Weather Advisory: {self.weather.get('name', 'unknown').replace('_', ' ').title()}",
                "body": (f"Current conditions: {self.weather.get('name', '')}. "
                         f"Wind {self.weather.get('wind', 0):.0f} km/h, "
                         f"Precip {self.weather.get('precip', 0):.1f} mm, "
                         f"Visibility {self.weather.get('visibility', 10000):.0f} m."),
                "severity": "high" if self.weather["severity"] > 0.5 else "moderate",
            })
        self.service_alerts = alerts
