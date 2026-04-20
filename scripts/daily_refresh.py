#!/usr/bin/env python
"""Daily data refresh script for Urban Mobility Disruption Copilot.

Run via cron, scheduled task, or CI pipeline:
  python -m scripts.daily_refresh

Refresh cadences:
  - Weather: every run (current conditions)
  - Advisories: every run (latest alerts)
  - RAG index: daily (re-index new documents)
  - Incidents: every run (traffic data)
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
