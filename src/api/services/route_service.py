"""Route service: orchestrates route generation, scoring, retrieval, and explanation.

Six-layer architecture:
1. OSRM — real road geometry for ANY origin/destination worldwide
2. RegionRegistry — determines which forecast regions a route passes through
3. ML Predictions — ST-GAT disruption forecasting for covered regions
4. Fallback Scoring — heuristic time/distance scoring outside forecast regions
5. RAG Evidence — advisory retrieval grounded in route context
6. Explanation — natural language reasoning for recommendations
"""

import logging
import uuid
from datetime import datetime

import numpy as np

from src.api.schemas.routes import (
    EvidenceCitation,
    Explanation,
    RiskFactor,
    RouteResponse,
    ScoredRoute,
    Segment,
)
from src.config.settings import get_settings
from src.explanation.engine import ExplanationEngine
from src.graph.builder import TransportGraph
from src.rag.retriever import AdvisoryRetriever
from src.realtime.signal_manager import SignalManager
from src.regions.registry import RegionRegistry
from src.routes.osrm_router import OSRMRouter
from src.routes.scorer import RouteScorer

logger = logging.getLogger(__name__)


class RouteService:
    """Main service that orchestrates the full route recommendation pipeline.

    Key design: ALL routes use OSRM for geometry (real road paths).
    The ML graph is used only for disruption predictions, never for geometry.
    """

    def __init__(
        self,
        route_scorer: RouteScorer,
        retriever: AdvisoryRetriever | None,
        explanation_engine: ExplanationEngine,
        signal_manager: SignalManager | None = None,
        graph: TransportGraph | None = None,
        region_registry: RegionRegistry | None = None,
    ):
        self.scorer = route_scorer
        self.retriever = retriever
        self.explainer = explanation_engine
        self.signal_manager = signal_manager
        self.graph = graph
        self.osrm = OSRMRouter()
        self.region_registry = region_registry or RegionRegistry()

    async def get_routes(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        departure_time: datetime,
        preference: str = "balanced",
        max_routes: int = 5,
        horizon_minutes: int = 30,
        weather_features: dict | None = None,
        incidents=None,
    ) -> RouteResponse:
        """Full pipeline: OSRM geometry -> region analysis -> score -> evidence -> explain."""
        request_id = str(uuid.uuid4())[:8]
        logger.info(
            f"[{request_id}] Route request: ({origin_lat},{origin_lon}) -> "
            f"({dest_lat},{dest_lon}) at {departure_time}, pref={preference}"
        )

        # Pull live signals if not explicitly provided
        signal_source = "static"
        if self.signal_manager:
            if weather_features is None:
                weather_features = self.signal_manager.get_weather_features()
                signal_source = "live"
            if incidents is None:
                incidents = self.signal_manager.get_incidents_df()
                signal_source = "live"

        # Step 1: Get real road geometry from OSRM (works worldwide)
        try:
            candidates = await self._get_osrm_routes(
                origin_lat, origin_lon, dest_lat, dest_lon, max_routes
            )
        except Exception as e:
            logger.error(f"[{request_id}] OSRM routing failed: {e}")
            candidates = []

        if not candidates:
            logger.warning(f"[{request_id}] No routes available")
            return self._empty_response(
                request_id, departure_time, signal_source,
                reason="Could not find routes between these locations. "
                       "The routing service may be temporarily unavailable, or "
                       "the locations may not be routable (e.g., across oceans).",
            )

        logger.info(f"[{request_id}] OSRM returned {len(candidates)} routes")

        # Step 2: Analyze forecast region coverage for the primary route
        primary_coords = candidates[0].get("coordinates", [])
        confidence_tier, coverage_pct, confidence_note, matching_regions = (
            self.region_registry.analyze_route_coverage(primary_coords)
        )
        logger.info(f"[{request_id}] Coverage: {confidence_tier} ({coverage_pct}%)")

        # Step 3: Score routes — ML-enhanced or fallback
        try:
            if confidence_tier in ("full", "partial") and self.graph:
                scored = self._ml_score_routes(
                    candidates, departure_time, preference,
                    weather_features, incidents, horizon_minutes,
                )
            else:
                scored = self._base_score_routes(candidates, preference)
        except Exception as e:
            logger.error(f"[{request_id}] Scoring failed, using base scoring: {e}")
            scored = self._base_score_routes(candidates, preference)
            confidence_tier = "base"
            confidence_note = "Scoring fell back to time/distance estimates due to an internal error."

        # Attach confidence metadata to each route
        for r in scored:
            route_coords = r.get("coordinates", [])
            try:
                r_tier, r_pct, r_note, _ = self.region_registry.analyze_route_coverage(route_coords)
            except Exception:
                r_tier, r_pct, r_note = "base", 0.0, ""
            r["confidence_tier"] = r_tier
            r["confidence_note"] = r_note
            r["forecast_coverage_pct"] = r_pct

        logger.info(f"[{request_id}] Scored {len(scored)} routes")

        # Step 4: Retrieve evidence for recommended route
        evidence = []
        active_alerts = []
        if self.retriever and scored and confidence_tier in ("full", "partial"):
            try:
                evidence = self.retriever.retrieve_for_route(
                    scored[0], departure_time, top_k=5
                )
                active_alerts = self.retriever.retrieve_active_alerts(
                    departure_time, top_k=3
                )
            except Exception as e:
                logger.warning(f"[{request_id}] Evidence retrieval failed: {e}")

        # Merge live service alerts
        if self.signal_manager:
            try:
                live_alerts = self.signal_manager.get_service_alerts()
                for la in live_alerts[:3]:
                    active_alerts.append({
                        "doc_id": la.get("id", ""),
                        "title": la.get("title", ""),
                        "body": la.get("body", ""),
                        "relevance_score": 0.9,
                        "source": "live_simulator",
                    })
            except Exception as e:
                logger.warning(f"[{request_id}] Live alerts retrieval failed: {e}")

        # Step 5: Generate explanation
        explanation_data = {
            "summary": "", "reasoning": "", "factors": [],
            "evidence_citations": [], "comparison": "", "confidence": "medium",
        }
        if scored:
            try:
                recommended = scored[0]
                alternatives = scored[1:]
                explanation_data = self.explainer.explain_recommendation(
                    recommended, alternatives, evidence, weather_features
                )
            except Exception as e:
                logger.warning(f"[{request_id}] Explanation generation failed: {e}")

            # Adjust explanation for confidence tier
            if confidence_tier == "base":
                explanation_data["confidence"] = "low"
                explanation_data["summary"] = (
                    "Outside forecast region \u2014 showing estimated travel times "
                    "based on road distance and typical conditions. "
                    + explanation_data.get("summary", "")
                )
                explanation_data["reasoning"] = (
                    "This route is outside all trained forecast regions. "
                    "Predictions are based on OSRM road routing estimates, not the "
                    "ST-GAT disruption model. For full predictive intelligence, "
                    "route within the Portland metro area. "
                    + explanation_data.get("reasoning", "")
                )
            elif confidence_tier == "partial":
                explanation_data["confidence"] = "medium"
                explanation_data["summary"] = (
                    f"Partial forecast coverage ({coverage_pct:.0f}%) \u2014 "
                    + explanation_data.get("summary", "")
                )

        # Step 6: Build response
        route_models = self._build_route_models(scored)
        explanation = Explanation(
            summary=explanation_data.get("summary", ""),
            reasoning=explanation_data.get("reasoning", ""),
            factors=[RiskFactor(**f) for f in explanation_data.get("factors", [])],
            evidence_citations=[
                EvidenceCitation(**c) for c in explanation_data.get("evidence_citations", [])
            ],
            comparison=explanation_data.get("comparison", ""),
            confidence=explanation_data.get("confidence", "medium"),
        )

        alert_models = [
            EvidenceCitation(
                doc_id=a.get("doc_id", ""),
                title=a.get("title", ""),
                snippet=a.get("body", "")[:200],
                relevance=a.get("relevance_score", 0),
                source=a.get("source", ""),
            )
            for a in active_alerts
        ]

        last_update = None
        if self.signal_manager:
            try:
                last_update = self.signal_manager.simulator.last_tick.isoformat()
            except Exception:
                pass

        return RouteResponse(
            request_id=request_id,
            timestamp=departure_time,
            routes=route_models,
            explanation=explanation,
            active_alerts=alert_models,
            signal_source=signal_source,
            last_signal_update=last_update,
        )

    async def _get_osrm_routes(
        self,
        origin_lat: float, origin_lon: float,
        dest_lat: float, dest_lon: float,
        max_routes: int,
    ) -> list[dict]:
        """Get real road routes from OSRM. Works for any coordinates worldwide."""
        osrm_routes = await self.osrm.get_multi_profile_routes(
            origin_lat, origin_lon, dest_lat, dest_lon
        )

        candidates = []
        for r in osrm_routes[:max_routes]:
            candidates.append({
                "name": r["name"],
                "strategy": r["strategy"],
                "path": [],  # No graph node path — OSRM geometry is authoritative
                "segments": [{
                    "from_node": 0,
                    "to_node": 0,
                    "mode": r["modes"][0] if r["modes"] else "drive",
                    "travel_time_s": r["duration_s"],
                    "length_m": r["distance_m"],
                    "edge_type": 0,
                }],
                "coordinates": r["coordinates"],
                "total_time_s": r["duration_s"],
                "total_distance_m": r["distance_m"],
                "modes": r["modes"],
                "num_transfers": 0,
            })

        return candidates

    def _ml_score_routes(
        self,
        candidates: list[dict],
        departure_time: datetime,
        preference: str,
        weather_features: dict | None,
        incidents,
        horizon_minutes: int,
    ) -> list[dict]:
        """Score routes using ML predictions overlaid on OSRM geometry."""
        scored = self.scorer.score_routes_with_coordinates(
            candidates, departure_time, preference,
            weather_features, incidents, horizon_minutes,
            graph=self.graph,
        )
        return scored

    def _base_score_routes(self, candidates: list[dict], preference: str) -> list[dict]:
        """Score routes using only time/distance when outside all forecast regions."""
        scored = []
        for route in candidates:
            time_s = route.get("total_time_s", 0)
            time_score = min(time_s / 7200, 1.0) if time_s > 0 else 0.5

            scored.append({
                **route,
                "predicted_time_s": round(time_s),
                "predicted_time_q90_s": round(time_s * 1.2),
                "disruption_prob": 0.0,
                "total_delay_median_min": 0.0,
                "total_delay_q90_min": 0.0,
                "reliability_score": 0.8,
                "comfort_score": 0.7,
                "time_score": round(1 - time_score, 3),
                "risk_score": 1.0,
                "composite_score": round(time_score, 4),
                "risk_factors": [],
            })

        scored.sort(key=lambda x: x["composite_score"])
        for i, route in enumerate(scored):
            route["rank"] = i + 1
            route["is_recommended"] = (i == 0)

        return scored

    def _build_route_models(self, scored: list[dict]) -> list[ScoredRoute]:
        """Convert scored dicts to Pydantic models."""
        route_models = []
        for r in scored:
            try:
                route_models.append(ScoredRoute(
                    name=r["name"],
                    strategy=r["strategy"],
                    rank=r.get("rank", 0),
                    is_recommended=r.get("is_recommended", False),
                    coordinates=r.get("coordinates", []),
                    modes=r.get("modes", []),
                    num_transfers=r.get("num_transfers", 0),
                    total_distance_m=r.get("total_distance_m", 0),
                    total_time_s=r.get("total_time_s", 0),
                    predicted_time_s=r.get("predicted_time_s", 0),
                    predicted_time_q90_s=r.get("predicted_time_q90_s", 0),
                    disruption_prob=r.get("disruption_prob", 0),
                    total_delay_median_min=r.get("total_delay_median_min", 0),
                    total_delay_q90_min=r.get("total_delay_q90_min", 0),
                    reliability_score=r.get("reliability_score", 0),
                    comfort_score=r.get("comfort_score", 0),
                    time_score=r.get("time_score", 0),
                    risk_score=r.get("risk_score", 0),
                    composite_score=max(0, min(r.get("composite_score", 0), 1.0)),
                    risk_factors=[RiskFactor(**rf) for rf in r.get("risk_factors", [])],
                    segments=[
                        Segment(
                            from_node=s["from_node"],
                            to_node=s["to_node"],
                            mode=s["mode"],
                            travel_time_s=s["travel_time_s"],
                            length_m=s["length_m"],
                        )
                        for s in r.get("segments", [])
                    ],
                    confidence_tier=r.get("confidence_tier", "full"),
                    confidence_note=r.get("confidence_note", ""),
                    forecast_coverage_pct=r.get("forecast_coverage_pct", 100.0),
                ))
            except Exception as e:
                logger.warning(f"Failed to build route model for {r.get('name', '?')}: {e}")
                continue
        return route_models

    def _empty_response(
        self, request_id: str, departure_time: datetime, signal_source: str,
        reason: str = "Could not find routes between these locations.",
    ) -> RouteResponse:
        """Return an empty response when no routes could be generated."""
        return RouteResponse(
            request_id=request_id,
            timestamp=departure_time,
            routes=[],
            explanation=Explanation(
                summary=reason,
                reasoning="The routing service could not generate any valid routes for "
                         "this origin-destination pair. This can happen if the locations "
                         "are across water bodies, the routing service is temporarily "
                         "unavailable, or the locations are unreachable by road.",
                factors=[],
                evidence_citations=[],
                comparison="",
                confidence="low",
            ),
            active_alerts=[],
            signal_source=signal_source,
            last_signal_update=None,
        )
