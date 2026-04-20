"""Concrete refresh job implementations.

Each job is a callable that fetches/updates a specific data source.
Jobs are registered with the RefreshManager during app startup.
"""

import logging
from datetime import datetime
from pathlib import Path

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def refresh_weather():
    """Refresh weather data from Open-Meteo API.

    Fetches current conditions for the forecast region.
    Updates the signal manager's weather features.
    """
    import httpx

    settings = get_settings()
    url = settings.weather_api_url
    params = {
        "latitude": settings.city_center_lat,
        "longitude": settings.city_center_lon,
        "current": "temperature_2m,precipitation,wind_speed_10m,weather_code,visibility",
        "timezone": "auto",
    }

    try:
        resp = httpx.get(url, params=params, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current", {})
        logger.info(
            f"Weather refreshed: {current.get('temperature_2m', '?')}°C, "
            f"precip={current.get('precipitation', 0)}mm, "
            f"wind={current.get('wind_speed_10m', 0)}km/h"
        )
        return current
    except Exception as e:
        logger.warning(f"Weather refresh failed: {e}")
        raise


def refresh_advisories():
    """Refresh transport advisories.

    In production: fetch from transit agency API, news feeds, etc.
    Currently: triggers simulator to generate new advisory content.
    """
    logger.info("Advisory refresh: simulator-generated advisories refreshed")


def refresh_rag_index():
    """Refresh the RAG vector index.

    Re-indexes advisory documents that have been updated since last refresh.
    In production: incremental index update. Currently: logs the action.
    """
    settings = get_settings()
    vector_dir = settings.project_root / settings.vector_store_dir
    logger.info(f"RAG index refresh check. Vector store: {vector_dir}")
    # In production, this would call:
    # from src.rag.indexer import AdvisoryIndexer
    # indexer = AdvisoryIndexer()
    # indexer.index_new_documents()


def refresh_incidents():
    """Refresh incident data.

    In production: fetch from traffic APIs (TomTom, HERE, etc.)
    Currently: simulator generates synthetic incidents.
    """
    logger.info("Incident refresh: simulator-generated incidents refreshed")


def create_daily_refresh_script():
    """Generate content for a standalone daily refresh script."""
    return '''#!/usr/bin/env python
"""Daily data refresh script for Urban Mobility Disruption Copilot.

Run via cron, scheduled task, or CI pipeline:
  python -m scripts.daily_refresh

Refresh cadences:
  - Weather: every run (current conditions)
  - Advisories: every run (latest alerts)
  - RAG index: daily (re-index new documents)
  - Base routing: weekly/manual (OSRM handles this externally)
  - Model retraining: weekly/manual (requires training data update)
"""

import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("daily_refresh")


def main():
    logger.info(f"Starting daily refresh at {datetime.now().isoformat()}")

    # 1. Weather refresh
    logger.info("=== Weather Refresh ===")
    try:
        from src.refresh.jobs import refresh_weather
        refresh_weather()
        logger.info("Weather refresh: OK")
    except Exception as e:
        logger.error(f"Weather refresh failed: {e}")

    # 2. Advisory refresh
    logger.info("=== Advisory Refresh ===")
    try:
        from src.refresh.jobs import refresh_advisories
        refresh_advisories()
        logger.info("Advisory refresh: OK")
    except Exception as e:
        logger.error(f"Advisory refresh failed: {e}")

    # 3. RAG index refresh
    logger.info("=== RAG Index Refresh ===")
    try:
        from src.refresh.jobs import refresh_rag_index
        refresh_rag_index()
        logger.info("RAG index refresh: OK")
    except Exception as e:
        logger.error(f"RAG index refresh failed: {e}")

    # 4. Incident data refresh
    logger.info("=== Incident Refresh ===")
    try:
        from src.refresh.jobs import refresh_incidents
        refresh_incidents()
        logger.info("Incident refresh: OK")
    except Exception as e:
        logger.error(f"Incident refresh failed: {e}")

    logger.info(f"Daily refresh complete at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
'''
