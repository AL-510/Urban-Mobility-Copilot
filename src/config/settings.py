"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    log_level: str = "info"

    # City / ML Region (where ST-GAT forecasting is trained and valid)
    city_name: str = "portland"
    city_center_lat: float = 45.5152
    city_center_lon: float = -122.6784
    city_radius_km: float = 8.0

    # Service Area (broader region for user-facing routing)
    service_area_enabled: bool = True
    service_area_min_lat: float = 45.25
    service_area_max_lat: float = 45.75
    service_area_min_lon: float = -123.1
    service_area_max_lon: float = -122.3

    # OSRM routing (for broader routing beyond the ML graph)
    osrm_base_url: str = "https://router.project-osrm.org"

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    data_synthetic_dir: str = "data/synthetic"
    vector_store_dir: str = "data/vector_store"
    model_checkpoint_dir: str = "checkpoints"

    # Model
    model_name: str = "stgat_v1"
    device: str = "cpu"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "mobility_advisories"

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"

    # Weather
    weather_api_url: str = "https://api.open-meteo.com/v1/forecast"

    # GTFS
    gtfs_url: str = "https://developer.trimet.org/schedule/gtfs.zip"
    gtfs_rt_enabled: bool = False

    # Graph model hyperparameters
    node_feat_dim: int = 24
    edge_feat_dim: int = 12
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    temporal_window: int = 12  # 12 x 5min = 60min lookback
    forecast_horizons: list[int] = [6, 12, 18]  # 30, 60, 90 min ahead
    num_quantiles: int = 3  # 10th, 50th, 90th percentile
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 10
    patience: int = 5

    @property
    def raw_dir(self) -> Path:
        return self.project_root / self.data_raw_dir

    @property
    def processed_dir(self) -> Path:
        return self.project_root / self.data_processed_dir

    @property
    def synthetic_dir(self) -> Path:
        return self.project_root / self.data_synthetic_dir

    @property
    def checkpoint_dir(self) -> Path:
        return self.project_root / self.model_checkpoint_dir

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
