"""RAG indexer: chunks, embeds, and indexes advisory documents into Qdrant."""

import logging
import uuid
from datetime import datetime

from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from src.config.settings import get_settings
from src.rag.client import get_qdrant_client

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


class AdvisoryIndexer:
    """Indexes transport advisories and alerts into a vector database."""

    def __init__(self):
        self.settings = get_settings()
        self.client = get_qdrant_client()
        self.model = SentenceTransformer(self.settings.embedding_model)
        self.collection = self.settings.qdrant_collection

    def create_collection(self, recreate: bool = False) -> None:
        """Create or recreate the Qdrant collection."""
        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection in collections:
            if recreate:
                self.client.delete_collection(self.collection)
                logger.info(f"Deleted existing collection: {self.collection}")
            else:
                logger.info(f"Collection already exists: {self.collection}")
                return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created collection: {self.collection}")

    def index_advisories(self, advisories: list[dict], batch_size: int = 64) -> int:
        """Index advisory documents.

        Each advisory is chunked and embedded.
        Metadata includes: incident_type, severity, area, lat, lon, time range.
        """
        points = []
        for adv in advisories:
            chunks = self._chunk_advisory(adv)
            for chunk in chunks:
                embedding = self.model.encode(chunk["text"], show_progress_bar=False)
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "doc_id": adv.get("doc_id", ""),
                        "title": adv.get("title", ""),
                        "text": chunk["text"],
                        "chunk_type": chunk["type"],
                        "incident_type": adv.get("incident_type", ""),
                        "severity": adv.get("severity", ""),
                        "area": adv.get("area", ""),
                        "affected_entity": adv.get("affected_entity", ""),
                        "is_transit": adv.get("is_transit", False),
                        "lat": adv.get("lat", 0),
                        "lon": adv.get("lon", 0),
                        "start_time": str(adv.get("start_time", "")),
                        "end_time": str(adv.get("end_time", "")),
                        "source": adv.get("source", ""),
                    },
                )
                points.append(point)

                if len(points) >= batch_size:
                    self.client.upsert(
                        collection_name=self.collection,
                        points=points,
                    )
                    points = []

        if points:
            self.client.upsert(collection_name=self.collection, points=points)

        total = len(advisories)
        logger.info(f"Indexed {total} advisories into {self.collection}")
        return total

    def _chunk_advisory(self, advisory: dict) -> list[dict]:
        """Chunk an advisory into indexable pieces.

        Strategy:
        - Title + body as main chunk
        - Title alone as a short chunk (for title-based retrieval)
        - If body is long (>500 chars), split into paragraphs
        """
        title = advisory.get("title", "")
        body = advisory.get("body", "")
        chunks = []

        # Main chunk: title + body
        main_text = f"{title}. {body}".strip()
        if main_text:
            chunks.append({"text": main_text, "type": "full"})

        # Title-only chunk for concise retrieval
        if title and len(body) > 100:
            chunks.append({"text": title, "type": "title"})

        # Split long bodies
        if len(body) > 500:
            paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
            for i, para in enumerate(paragraphs):
                if len(para) > 50:
                    chunks.append({
                        "text": f"{title}. {para}",
                        "type": f"paragraph_{i}",
                    })

        return chunks if chunks else [{"text": title or "No content", "type": "empty"}]

    def get_collection_info(self) -> dict:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection)
            return {
                "name": self.collection,
                "points_count": info.points_count,
                "vectors_count": getattr(info, "vectors_count", info.points_count),
                "status": info.status.value if hasattr(info.status, "value") else str(info.status),
            }
        except Exception as e:
            return {"name": self.collection, "error": str(e)}
