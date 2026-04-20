"""Pydantic schemas for API request/response models."""

from datetime import datetime

from pydantic import BaseModel, Field


class RouteRequest(BaseModel):
    origin_lat: float = Field(..., ge=-90, le=90, description="Origin latitude")
    origin_lon: float = Field(..., ge=-180, le=180, description="Origin longitude")
    dest_lat: float = Field(..., ge=-90, le=90, description="Destination latitude")
    dest_lon: float = Field(..., ge=-180, le=180, description="Destination longitude")
    departure_time: datetime = Field(default_factory=datetime.now, description="Departure time (ISO 8601)")
    preference: str = Field(
        default="balanced",
        description="Route preference: fastest, cheapest, least_risky, balanced",
    )
    max_routes: int = Field(default=5, ge=1, le=10, description="Maximum candidate routes")
    horizon_minutes: int = Field(default=30, description="Forecast horizon in minutes")

    model_config = {"json_schema_extra": {
        "example": {
            "origin_lat": 45.5231,
            "origin_lon": -122.6765,
            "dest_lat": 45.5122,
            "dest_lon": -122.6587,
            "departure_time": "2024-06-15T08:30:00",
            "preference": "balanced",
            "max_routes": 5,
            "horizon_minutes": 30,
        }
    }}


class Segment(BaseModel):
    from_node: int
    to_node: int
    mode: str
    travel_time_s: float
    length_m: float


class RiskFactor(BaseModel):
    type: str
    severity: str
    description: str
    source: str = "model_prediction"


class EvidenceCitation(BaseModel):
    doc_id: str
    title: str
    snippet: str
    relevance: float = 0.0
    source: str = ""


class ScoredRoute(BaseModel):
    name: str
    strategy: str
    rank: int = 0
    is_recommended: bool = False
    coordinates: list[list[float]]
    modes: list[str]
    num_transfers: int = 0
    total_distance_m: float
    total_time_s: float
    predicted_time_s: float
    predicted_time_q90_s: float
    disruption_prob: float
    total_delay_median_min: float
    total_delay_q90_min: float
    reliability_score: float
    comfort_score: float
    time_score: float
    risk_score: float
    composite_score: float
    risk_factors: list[RiskFactor] = []
    segments: list[Segment] = []
    # Confidence tier: "full" (inside ML region), "partial" (overlaps), "base" (outside)
    confidence_tier: str = "full"
    confidence_note: str = ""
    forecast_coverage_pct: float = 100.0  # % of route inside forecast region


class Explanation(BaseModel):
    summary: str
    reasoning: str
    factors: list[RiskFactor]
    evidence_citations: list[EvidenceCitation]
    comparison: str
    confidence: str


class RouteResponse(BaseModel):
    request_id: str
    timestamp: datetime
    routes: list[ScoredRoute]
    explanation: Explanation
    active_alerts: list[EvidenceCitation] = []
    signal_source: str | None = None  # "live" or "static"
    last_signal_update: str | None = None  # ISO timestamp of last signal refresh


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    graph_nodes: int
    graph_edges: int
    vector_db_status: str
    realtime_enabled: bool = False


class ErrorResponse(BaseModel):
    detail: str
    code: str = "INTERNAL_ERROR"
