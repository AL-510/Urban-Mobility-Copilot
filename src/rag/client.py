"""Shared Qdrant client factory — remote only, no embedded fallback."""

import logging
from pathlib import Path

from qdrant_client import QdrantClient

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None


def is_qdrant_reachable() -> bool:
    """Lightweight check: is the remote Qdrant server reachable?

    Used before loading sentence-transformers to avoid OOM on memory-constrained
    deployments (e.g. Render free tier / 512MB) when Qdrant isn't available.
    """
    settings = get_settings()
    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=3,
        )
        client.get_collections()
        return True
    except Exception:
        return False


def get_qdrant_client() -> QdrantClient:
    """Get or create a Qdrant client (remote only — no embedded fallback).

    Raises RuntimeError if the server is not reachable.
    Call is_qdrant_reachable() first to guard against this.
    """
    global _client
    if _client is not None:
        return _client

    settings = get_settings()
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
    except Exception as e:
        raise RuntimeError(f"Qdrant not reachable at {settings.qdrant_host}:{settings.qdrant_port}: {e}") from e


def reset_client():
    """Reset the cached client (for testing)."""
    global _client
    _client = None
