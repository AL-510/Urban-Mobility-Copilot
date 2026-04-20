"""Generate synthetic but realistic incidents, roadworks, events, and advisories."""

import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

INCIDENT_TYPES = [
    "accident",
    "breakdown",
    "roadwork",
    "signal_failure",
    "track_maintenance",
    "flooding",
    "power_outage",
    "medical_emergency",
    "police_activity",
    "special_event",
]

SEVERITY_LEVELS = ["low", "medium", "high", "critical"]

TRANSIT_LINES = [
    "Blue Line MAX", "Red Line MAX", "Green Line MAX", "Orange Line MAX", "Yellow Line MAX",
    "Bus Route 4", "Bus Route 6", "Bus Route 8", "Bus Route 12", "Bus Route 14",
    "Bus Route 15", "Bus Route 20", "Bus Route 33", "Bus Route 54", "Bus Route 72",
]

ROAD_CORRIDORS = [
    "I-5 Northbound", "I-5 Southbound", "I-84 Eastbound", "I-84 Westbound",
    "I-405 Loop", "US-26 Westbound", "US-26 Eastbound",
    "Burnside St", "Powell Blvd", "MLK Jr Blvd", "Hawthorne Blvd",
    "Division St", "Sandy Blvd", "Broadway", "Interstate Ave",
    "82nd Ave", "Barbur Blvd", "Macadam Ave", "NW 23rd Ave",
]

LOCATION_CLUSTERS = [
    {"name": "Downtown Portland", "lat": 45.5152, "lon": -122.6784, "radius": 0.015},
    {"name": "Lloyd District", "lat": 45.5311, "lon": -122.6590, "radius": 0.008},
    {"name": "Pearl District", "lat": 45.5299, "lon": -122.6838, "radius": 0.006},
    {"name": "Hawthorne", "lat": 45.5118, "lon": -122.6325, "radius": 0.008},
    {"name": "Alberta Arts", "lat": 45.5590, "lon": -122.6455, "radius": 0.006},
    {"name": "Hollywood", "lat": 45.5350, "lon": -122.6210, "radius": 0.008},
    {"name": "Sellwood", "lat": 45.4625, "lon": -122.6525, "radius": 0.008},
    {"name": "St Johns", "lat": 45.5900, "lon": -122.7520, "radius": 0.008},
    {"name": "Gateway", "lat": 45.5310, "lon": -122.5660, "radius": 0.010},
    {"name": "Beaverton TC", "lat": 45.4920, "lon": -122.8030, "radius": 0.008},
]


def _random_location() -> tuple[float, float, str]:
    cluster = random.choice(LOCATION_CLUSTERS)
    lat = cluster["lat"] + np.random.normal(0, cluster["radius"])
    lon = cluster["lon"] + np.random.normal(0, cluster["radius"])
    return lat, lon, cluster["name"]


def _incident_duration(incident_type: str, severity: str) -> int:
    """Return duration in minutes."""
    base = {
        "accident": 45, "breakdown": 30, "roadwork": 480, "signal_failure": 90,
        "track_maintenance": 360, "flooding": 240, "power_outage": 120,
        "medical_emergency": 25, "police_activity": 35, "special_event": 180,
    }
    severity_mult = {"low": 0.5, "medium": 1.0, "high": 1.5, "critical": 2.5}
    minutes = base.get(incident_type, 60) * severity_mult.get(severity, 1.0)
    return int(minutes * random.uniform(0.6, 1.4))


def _delay_impact(incident_type: str, severity: str) -> float:
    """Return expected delay factor (1.0 = no delay, 2.0 = double travel time)."""
    base = {
        "accident": 1.8, "breakdown": 1.3, "roadwork": 1.5, "signal_failure": 2.0,
        "track_maintenance": 2.5, "flooding": 3.0, "power_outage": 2.0,
        "medical_emergency": 1.4, "police_activity": 1.3, "special_event": 1.6,
    }
    severity_mult = {"low": 0.6, "medium": 1.0, "high": 1.4, "critical": 2.0}
    factor = base.get(incident_type, 1.5) * severity_mult.get(severity, 1.0)
    return round(min(factor * random.uniform(0.8, 1.2), 5.0), 2)


def generate_incidents(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    incidents_per_day: float = 8.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic incident dataset spanning a date range."""
    random.seed(seed)
    np.random.seed(seed)

    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    days = (end - start).days

    records = []
    for day_offset in range(days):
        date = start + timedelta(days=day_offset)
        is_weekday = date.weekday() < 5
        # More incidents during rush hours on weekdays
        n_incidents = np.random.poisson(incidents_per_day * (1.2 if is_weekday else 0.7))

        for _ in range(n_incidents):
            incident_type = random.choice(INCIDENT_TYPES)
            severity = random.choices(
                SEVERITY_LEVELS, weights=[0.35, 0.35, 0.2, 0.1]
            )[0]

            # Time distribution: peaks at 7-9am and 4-7pm on weekdays
            if is_weekday and random.random() < 0.6:
                hour = random.choice(
                    list(range(7, 10)) * 3 + list(range(16, 19)) * 3 + list(range(24))
                )
            else:
                hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            incident_time = date.replace(hour=hour, minute=minute)

            lat, lon, area = _random_location()
            duration = _incident_duration(incident_type, severity)
            delay_factor = _delay_impact(incident_type, severity)

            is_transit = incident_type in [
                "signal_failure", "track_maintenance", "power_outage"
            ]
            affected = random.choice(TRANSIT_LINES) if is_transit else random.choice(ROAD_CORRIDORS)

            records.append({
                "incident_id": f"INC-{date.strftime('%Y%m%d')}-{len(records):04d}",
                "incident_type": incident_type,
                "severity": severity,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "area": area,
                "affected_entity": affected,
                "is_transit": is_transit,
                "start_time": incident_time.isoformat(),
                "duration_minutes": duration,
                "end_time": (incident_time + timedelta(minutes=duration)).isoformat(),
                "delay_factor": delay_factor,
                "description": _generate_description(incident_type, severity, affected, area),
            })

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} synthetic incidents over {days} days")
    return df


def _generate_description(
    incident_type: str, severity: str, affected: str, area: str
) -> str:
    templates = {
        "accident": f"Vehicle collision reported on {affected} near {area}. {severity.title()} severity. Expect delays.",
        "breakdown": f"Disabled vehicle on {affected} near {area}. {severity.title()} impact on traffic flow.",
        "roadwork": f"Scheduled roadwork on {affected} in the {area} area. Lane restrictions in effect.",
        "signal_failure": f"Signal system malfunction affecting {affected} service near {area}. {severity.title()} disruption to schedules.",
        "track_maintenance": f"Track maintenance underway on {affected}. Reduced service frequency near {area}.",
        "flooding": f"Flooding reported on {affected} near {area}. {severity.title()} impact. Possible route closures.",
        "power_outage": f"Power outage affecting {affected} operations near {area}. Service suspended until restoration.",
        "medical_emergency": f"Emergency response on {affected} near {area}. Brief delays expected.",
        "police_activity": f"Police activity near {affected} in {area}. Minor detours in effect.",
        "special_event": f"Large event near {area} causing congestion on {affected}. Plan alternate routes.",
    }
    return templates.get(incident_type, f"Disruption on {affected} near {area}.")


def generate_advisories(incidents: pd.DataFrame) -> list[dict]:
    """Generate RAG-indexable advisory documents from incidents."""
    advisories = []
    for _, row in incidents.iterrows():
        advisory = {
            "doc_id": f"ADV-{row['incident_id']}",
            "title": f"{row['incident_type'].replace('_', ' ').title()} - {row['affected_entity']}",
            "body": row["description"],
            "incident_type": row["incident_type"],
            "severity": row["severity"],
            "area": row["area"],
            "affected_entity": row["affected_entity"],
            "is_transit": row["is_transit"],
            "lat": row["lat"],
            "lon": row["lon"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "source": "Portland Transport Authority (Synthetic)",
        }
        advisories.append(advisory)
    return advisories


def generate_weather_advisories(weather_df: pd.DataFrame) -> list[dict]:
    """Generate weather alert documents from severe weather periods."""
    advisories = []
    if weather_df.empty:
        return advisories

    for _, row in weather_df.iterrows():
        severity_score = row.get("weather_severity", 0)
        if severity_score < 0.4:
            continue

        ts = row["timestamp"]
        if severity_score >= 0.7:
            level = "Warning"
        elif severity_score >= 0.5:
            level = "Watch"
        else:
            level = "Advisory"

        conditions = []
        if row.get("is_rain", 0) > 0:
            conditions.append(f"rain ({row.get('rain_mm', 0):.1f}mm)")
        if row.get("is_snow", 0) > 0:
            conditions.append(f"snow ({row.get('snow_mm', 0):.1f}mm)")
        if row.get("wind_speed_kmh", 0) > 40:
            conditions.append(f"high winds ({row.get('wind_speed_kmh', 0):.0f} km/h)")
        if row.get("is_fog", 0) > 0:
            conditions.append("fog/low visibility")
        if row.get("is_storm", 0) > 0:
            conditions.append("thunderstorm activity")

        condition_text = ", ".join(conditions) if conditions else "adverse weather"

        advisories.append({
            "doc_id": f"WX-{ts.strftime('%Y%m%d%H%M') if hasattr(ts, 'strftime') else str(ts)[:12]}",
            "title": f"Weather {level}: {condition_text.title()}",
            "body": (
                f"Weather {level.lower()} in effect for the Portland metro area. "
                f"Conditions include {condition_text}. "
                f"Commuters should allow extra travel time and exercise caution. "
                f"Severity index: {severity_score:.2f}/1.00."
            ),
            "incident_type": "weather",
            "severity": "high" if severity_score >= 0.7 else "medium",
            "area": "Portland Metro",
            "affected_entity": "All routes",
            "is_transit": False,
            "lat": get_settings().city_center_lat,
            "lon": get_settings().city_center_lon,
            "start_time": str(ts),
            "end_time": str(ts + timedelta(hours=1) if isinstance(ts, datetime) else ts),
            "source": "National Weather Service (Synthetic)",
        })

    logger.info(f"Generated {len(advisories)} weather advisories")
    return advisories


def save_incidents(df: pd.DataFrame, path: Path | None = None) -> Path:
    settings = get_settings()
    out = path or (settings.synthetic_dir / "incidents.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info(f"Saved {len(df)} incidents to {out}")
    return out


def save_advisories(advisories: list[dict], path: Path | None = None) -> Path:
    settings = get_settings()
    out = path or (settings.synthetic_dir / "advisories.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(advisories, f, indent=2, default=str)
    logger.info(f"Saved {len(advisories)} advisories to {out}")
    return out
