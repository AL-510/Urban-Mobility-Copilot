"""Setup script: download/generate all data needed for the project.

Usage:
    python -m scripts.setup_data [--skip-osm] [--skip-gtfs] [--skip-weather] [--full-year]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import get_settings
from src.data_ingestion.incident_generator import (
    generate_advisories,
    generate_incidents,
    generate_weather_advisories,
    save_advisories,
    save_incidents,
)
from src.data_ingestion.weather_loader import fetch_weather_history, weather_to_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Setup project data")
    parser.add_argument("--skip-osm", action="store_true", help="Skip OSM download")
    parser.add_argument("--skip-gtfs", action="store_true", help="Skip GTFS download")
    parser.add_argument("--skip-weather", action="store_true", help="Skip weather download")
    parser.add_argument("--full-year", action="store_true", default=True,
                        help="Use full year of data (default: True)")
    parser.add_argument("--half-year", action="store_true", help="Use only 6 months of data")
    args = parser.parse_args()

    if args.half_year:
        args.full_year = False

    settings = get_settings()

    # Date range
    start_date = "2024-01-01"
    end_date = "2024-12-31" if args.full_year else "2024-06-30"
    incidents_per_day = 12.0 if args.full_year else 8.0

    logger.info(f"=== Urban Mobility Copilot - Data Setup ===")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"City: Portland, OR ({settings.city_center_lat}, {settings.city_center_lon})")
    logger.info(f"Radius: {settings.city_radius_km} km")

    # Ensure directories exist
    for d in [settings.raw_dir, settings.processed_dir, settings.synthetic_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. OSM Road Network
    # =========================================================================
    road_graph = None
    if not args.skip_osm:
        try:
            from src.data_ingestion.osm_loader import download_road_network, save_graph, load_graph

            # Check if already downloaded
            cached_path = settings.processed_dir / "graphs" / "road_network.graphml"
            if cached_path.exists():
                logger.info(f"Loading cached OSM graph from {cached_path}")
                road_graph = load_graph("road_network")
            else:
                logger.info("Downloading OSM road network (this may take a few minutes)...")
                road_graph = download_road_network()
                save_graph(road_graph, "road_network")

            logger.info(f"Road network: {road_graph.number_of_nodes()} nodes, "
                        f"{road_graph.number_of_edges()} edges")
        except Exception as e:
            logger.warning(f"OSM download failed: {e}")
            logger.info("Will try to proceed without road network.")

    # =========================================================================
    # 2. GTFS Transit Data
    # =========================================================================
    gtfs_data = None
    if not args.skip_gtfs:
        try:
            from src.data_ingestion.gtfs_loader import download_gtfs, GTFSData

            # Try multiple GTFS sources for TriMet
            gtfs_urls = [
                settings.gtfs_url,
                "https://developer.trimet.org/schedule/gtfs.zip",
                "https://transitfeeds.com/p/trimet/43/latest/download",
            ]

            gtfs_dir = None
            for url in gtfs_urls:
                try:
                    logger.info(f"Trying GTFS download from: {url}")
                    gtfs_dir = download_gtfs(url)
                    break
                except Exception as e:
                    logger.warning(f"  Failed: {e}")
                    # Remove partial download
                    zip_path = settings.raw_dir / "gtfs" / "gtfs.zip"
                    if zip_path.exists():
                        zip_path.unlink()
                    continue

            if gtfs_dir:
                gtfs_data = GTFSData(gtfs_dir)
                logger.info(f"GTFS loaded: {len(gtfs_data.stops)} stops, "
                            f"{len(gtfs_data.routes)} routes, "
                            f"{len(gtfs_data.trips)} trips")
            else:
                logger.warning("All GTFS download attempts failed")

        except Exception as e:
            logger.warning(f"GTFS setup failed: {e}")

    # =========================================================================
    # 3. Weather History
    # =========================================================================
    weather_df = None
    if not args.skip_weather:
        try:
            import pandas as pd
            from datetime import datetime, timedelta

            logger.info(f"Fetching historical weather data ({start_date} to {end_date})...")

            # Fetch in chunks to avoid API limits
            chunks = []
            chunk_start = datetime.fromisoformat(start_date)
            chunk_end_final = datetime.fromisoformat(end_date)

            while chunk_start < chunk_end_final:
                chunk_end = min(chunk_start + timedelta(days=90), chunk_end_final)
                logger.info(f"  Fetching weather: {chunk_start.date()} to {chunk_end.date()}")
                try:
                    chunk = fetch_weather_history(
                        start_date=chunk_start.strftime("%Y-%m-%d"),
                        end_date=chunk_end.strftime("%Y-%m-%d"),
                    )
                    chunks.append(chunk)
                except Exception as e:
                    logger.warning(f"  Weather chunk failed: {e}")
                chunk_start = chunk_end + timedelta(days=1)

            if chunks:
                weather_df = pd.concat(chunks, ignore_index=True)
                weather_features = weather_to_features(weather_df)
                out_path = settings.processed_dir / "weather_features.parquet"
                weather_features.to_parquet(out_path, index=False)
                logger.info(f"Saved {len(weather_features)} hourly weather records to {out_path}")
        except Exception as e:
            logger.warning(f"Weather fetch failed: {e}")

    # =========================================================================
    # 4. Synthetic Incidents (large-scale)
    # =========================================================================
    logger.info(f"Generating synthetic incidents ({incidents_per_day}/day)...")
    incidents = generate_incidents(
        start_date=start_date,
        end_date=end_date,
        incidents_per_day=incidents_per_day,
    )
    save_incidents(incidents)
    logger.info(f"Generated {len(incidents)} incidents")

    # =========================================================================
    # 5. Advisory Documents for RAG
    # =========================================================================
    logger.info("Generating advisory documents...")
    advisories = generate_advisories(incidents)

    if weather_df is not None:
        wx_features = weather_to_features(weather_df)
        wx_advisories = generate_weather_advisories(wx_features)
        advisories.extend(wx_advisories)

    save_advisories(advisories)
    logger.info(f"Total advisories: {len(advisories)}")

    # =========================================================================
    # 6. Build Transport Graph
    # =========================================================================
    logger.info("Building multimodal transport graph...")
    try:
        from src.graph.builder import TransportGraph

        graph = TransportGraph()

        # Add road network
        if road_graph is not None:
            graph.add_road_network(road_graph)
        else:
            # Try loading cached
            try:
                from src.data_ingestion.osm_loader import load_graph
                cached = load_graph("road_network")
                graph.add_road_network(cached)
            except FileNotFoundError:
                logger.warning("No road network available")

        # Add transit network
        if gtfs_data is not None and not gtfs_data.stops.empty:
            # Compute bounds from road network or use city settings
            if graph.num_nodes > 0:
                positions = graph.get_node_positions()
                bounds = {
                    "min_lat": positions[:, 0].min() - 0.01,
                    "max_lat": positions[:, 0].max() + 0.01,
                    "min_lon": positions[:, 1].min() - 0.01,
                    "max_lon": positions[:, 1].max() + 0.01,
                }
            else:
                r = settings.city_radius_km / 111.0  # rough deg
                bounds = {
                    "min_lat": settings.city_center_lat - r,
                    "max_lat": settings.city_center_lat + r,
                    "min_lon": settings.city_center_lon - r,
                    "max_lon": settings.city_center_lon + r,
                }
            graph.add_transit_network(gtfs_data, bounds=bounds)
            graph.add_transfer_edges(max_distance_m=300)

        if graph.num_nodes > 0:
            graph.save()
            logger.info(f"Saved transport graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

            # Print breakdown
            transit_count = sum(
                1 for _, d in graph.G.nodes(data=True) if d.get("node_type") == "transit"
            )
            road_count = sum(
                1 for _, d in graph.G.nodes(data=True) if d.get("node_type") == "road"
            )
            logger.info(f"  Road nodes: {road_count}, Transit nodes: {transit_count}")

            edge_types = {}
            for _, _, d in graph.G.edges(data=True):
                et = d.get("edge_type", -1)
                edge_types[et] = edge_types.get(et, 0) + 1
            type_names = {0: "road", 1: "transit", 2: "walk", 3: "transfer"}
            for et, count in sorted(edge_types.items()):
                logger.info(f"  {type_names.get(et, f'type_{et}')} edges: {count}")
        else:
            logger.info("No real data available, building demo grid graph...")
            from src.api.app import _build_demo_graph
            graph = _build_demo_graph(settings)
            logger.info(f"Saved demo graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

    except Exception as e:
        logger.error(f"Graph building failed: {e}", exc_info=True)

    logger.info("=" * 60)
    logger.info("Data setup complete!")
    logger.info(f"  Incidents: {len(incidents)}")
    logger.info(f"  Advisories: {len(advisories)}")
    logger.info(f"  Weather records: {len(weather_df) if weather_df is not None else 0}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
