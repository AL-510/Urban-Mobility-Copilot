"""Shared Qdrant client factory — tries remote, falls back to local embedded mode."""

import logging
from pathlib import Path

from qdrant_client import QdrantClient

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Get or create a Qdrant client.

    Strategy:
    1. Try connecting to remote Qdrant (Docker container)
    2. Fall back to local embedded mode (no Docker needed)
    """
    global _client
    if _client is not None:
        return _client

    settings = get_settings()

    # Try remote first
    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=3,
        )
        client.get_collections()
        logger.info(f"Connected to remote Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
        _client = client
        return _client
    except Exception:
        pass

    # Fall back to local embedded mode
    local_path = str(settings.project_root / settings.vector_store_dir / "qdrant_local")
    Path(local_path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Using local Qdrant at {local_path}")
    _client = QdrantClient(path=local_path)
    return _client


def reset_client():
    """Reset the cached client (for testing)."""
    global _client
    _client = None
