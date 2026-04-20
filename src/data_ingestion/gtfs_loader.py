"""Load and parse GTFS static transit data."""

import io
import logging
import zipfile
from pathlib import Path

import httpx
import pandas as pd

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

GTFS_FILES = [
    "agency.txt",
    "calendar.txt",
    "calendar_dates.txt",
    "routes.txt",
    "trips.txt",
    "stops.txt",
    "stop_times.txt",
    "shapes.txt",
]


def download_gtfs(url: str | None = None) -> Path:
    """Download GTFS zip and extract to raw data directory."""
    settings = get_settings()
    url = url or settings.gtfs_url
    out_dir = settings.raw_dir / "gtfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "gtfs.zip"

    if zip_path.exists():
        logger.info("GTFS zip already exists, skipping download")
    else:
        logger.info(f"Downloading GTFS from {url}")
        resp = httpx.get(url, follow_redirects=True, timeout=120)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
        logger.info(f"Downloaded {len(resp.content) / 1e6:.1f} MB")

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in GTFS_FILES:
            if name in zf.namelist():
                zf.extract(name, out_dir)
    logger.info(f"Extracted GTFS to {out_dir}")
    return out_dir


class GTFSData:
    """Parsed GTFS static feed."""

    def __init__(self, gtfs_dir: Path | None = None):
        settings = get_settings()
        self.dir = gtfs_dir or (settings.raw_dir / "gtfs")
        self._cache: dict[str, pd.DataFrame] = {}

    def _load(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            path = self.dir / f"{name}.txt"
            if not path.exists():
                logger.warning(f"GTFS file not found: {path}")
                return pd.DataFrame()
            self._cache[name] = pd.read_csv(path, dtype=str)
        return self._cache[name]

    @property
    def stops(self) -> pd.DataFrame:
        df = self._load("stops")
        if not df.empty:
            df["stop_lat"] = df["stop_lat"].astype(float)
            df["stop_lon"] = df["stop_lon"].astype(float)
        return df

    @property
    def routes(self) -> pd.DataFrame:
        return self._load("routes")

    @property
    def trips(self) -> pd.DataFrame:
        return self._load("trips")

    @property
    def stop_times(self) -> pd.DataFrame:
        return self._load("stop_times")

    @property
    def shapes(self) -> pd.DataFrame:
        df = self._load("shapes")
        if not df.empty:
            df["shape_pt_lat"] = df["shape_pt_lat"].astype(float)
            df["shape_pt_lon"] = df["shape_pt_lon"].astype(float)
            df["shape_pt_sequence"] = df["shape_pt_sequence"].astype(int)
        return df

    def get_route_stops(self, route_id: str) -> pd.DataFrame:
        """Get ordered stops for a specific route."""
        trips = self.trips[self.trips["route_id"] == route_id]
        if trips.empty:
            return pd.DataFrame()
        trip_id = trips.iloc[0]["trip_id"]
        st = self.stop_times[self.stop_times["trip_id"] == trip_id].copy()
        st["stop_sequence"] = st["stop_sequence"].astype(int)
        st = st.sort_values("stop_sequence")
        return st.merge(self.stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]], on="stop_id")
