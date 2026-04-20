"""RAG retriever: queries the vector database for relevant advisories."""

import logging
from datetime import datetime

from qdrant_client.http.models import FieldCondition, Filter, MatchValue, Range
from sentence_transformers import SentenceTransformer

from src.config.settings import get_settings
from src.rag.client import get_qdrant_client

logger = logging.getLogger(__name__)


class AdvisoryRetriever:
    """Retrieves relevant transport advisories from the vector store."""

    def __init__(self):
        self.settings = get_settings()
        self.client = get_qdrant_client()
        self.model = SentenceTransformer(self.settings.embedding_model)
        self.collection = self.settings.qdrant_collection

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        area: str | None = None,
        incident_type: str | None = None,
        severity_min: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
        radius_deg: float = 0.02,
    ) -> list[dict]:
        """Retrieve relevant advisories using semantic search with metadata filters.

        Args:
            query: natural language query (e.g., "delays on Blue Line MAX")
            top_k: number of results
            area: filter by area name
            incident_type: filter by type
            severity_min: minimum severity level
            lat, lon: center of spatial filter
            radius_deg: spatial search radius in degrees (~2km per 0.02)

        Returns:
            List of advisory dicts with relevance scores
        """
        query_embedding = self.model.encode(query, show_progress_bar=False)

        # Build filter conditions
        must_conditions = []
        if area:
            must_conditions.append(
                FieldCondition(key="area", match=MatchValue(value=area))
            )
        if incident_type:
            must_conditions.append(
                FieldCondition(key="incident_type", match=MatchValue(value=incident_type))
            )

        search_filter = Filter(must=must_conditions) if must_conditions else None

        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=query_embedding.tolist(),
                query_filter=search_filter,
                limit=top_k * 2,  # Over-fetch for post-filtering
            )
            results = response.points
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

        # Post-filter by spatial proximity if lat/lon provided
        advisories = []
        for hit in results:
            payload = hit.payload
            score = hit.score

            if lat is not None and lon is not None:
                doc_lat = payload.get("lat", 0)
                doc_lon = payload.get("lon", 0)
                dist = ((doc_lat - lat) ** 2 + (doc_lon - lon) ** 2) ** 0.5
                if dist > radius_deg:
                    # Reduce score but don't exclude
                    score *= 0.5

            advisory = {
                "doc_id": payload.get("doc_id", ""),
                "title": payload.get("title", ""),
                "body": payload.get("text", ""),
                "incident_type": payload.get("incident_type", ""),
                "severity": payload.get("severity", ""),
                "area": payload.get("area", ""),
                "affected_entity": payload.get("affected_entity", ""),
                "is_transit": payload.get("is_transit", False),
                "lat": payload.get("lat", 0),
                "lon": payload.get("lon", 0),
                "start_time": payload.get("start_time", ""),
                "end_time": payload.get("end_time", ""),
                "source": payload.get("source", ""),
                "relevance_score": round(score, 4),
            }
            advisories.append(advisory)

        # Deduplicate by doc_id
        seen = set()
        unique = []
        for adv in advisories:
            if adv["doc_id"] not in seen:
                seen.add(adv["doc_id"])
                unique.append(adv)

        return sorted(unique, key=lambda x: x["relevance_score"], reverse=True)[:top_k]

    def retrieve_for_route(
        self,
        route: dict,
        timestamp: datetime,
        top_k: int = 5,
    ) -> list[dict]:
        """Retrieve advisories relevant to a specific route.

        Builds a query from route characteristics and spatial context.
        """
        coords = route.get("coordinates", [])
        modes = route.get("modes", [])
        name = route.get("name", "route")

        # Build query from route context
        query_parts = [f"disruptions affecting commute via {name}"]
        if "transit" in modes:
            query_parts.append("transit delays service alerts")
        if "drive" in modes:
            query_parts.append("traffic congestion road closures")
        query = " ".join(query_parts)

        # Use route midpoint for spatial filtering
        lat, lon = None, None
        if coords:
            mid = len(coords) // 2
            lat, lon = coords[mid][0], coords[mid][1]

        return self.retrieve(
            query=query,
            top_k=top_k,
            lat=lat,
            lon=lon,
            radius_deg=0.03,
        )

    def retrieve_active_alerts(
        self,
        timestamp: datetime | None = None,
        area: str | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        """Retrieve currently active alerts and advisories."""
        query = "active service alert disruption advisory"
        return self.retrieve(
            query=query,
            top_k=top_k,
            area=area,
        )
