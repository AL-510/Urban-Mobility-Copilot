export interface RouteRequest {
  origin_lat: number;
  origin_lon: number;
  dest_lat: number;
  dest_lon: number;
  departure_time: string;
  preference: "fastest" | "cheapest" | "least_risky" | "balanced";
  max_routes: number;
  horizon_minutes: number;
}

export interface Segment {
  from_node: number;
  to_node: number;
  mode: string;
  travel_time_s: number;
  length_m: number;
}

export interface RiskFactor {
  type: string;
  severity: string;
  description: string;
  source: string;
}

export interface EvidenceCitation {
  doc_id: string;
  title: string;
  snippet: string;
  relevance: number;
  source: string;
}

export interface ScoredRoute {
  name: string;
  strategy: string;
  rank: number;
  is_recommended: boolean;
  coordinates: [number, number][];
  modes: string[];
  num_transfers: number;
  total_distance_m: number;
  total_time_s: number;
  predicted_time_s: number;
  predicted_time_q90_s: number;
  disruption_prob: number;
  total_delay_median_min: number;
  total_delay_q90_min: number;
  reliability_score: number;
  comfort_score: number;
  time_score: number;
  risk_score: number;
  composite_score: number;
  risk_factors: RiskFactor[];
  segments: Segment[];
  confidence_tier: "full" | "partial" | "base";
  confidence_note: string;
  forecast_coverage_pct: number;
}

export interface Explanation {
  summary: string;
  reasoning: string;
  factors: RiskFactor[];
  evidence_citations: EvidenceCitation[];
  comparison: string;
  confidence: string;
}

export interface RouteResponse {
  request_id: string;
  timestamp: string;
  routes: ScoredRoute[];
  explanation: Explanation;
  active_alerts: EvidenceCitation[];
  signal_source?: string;
  last_signal_update?: string;
}

export interface NetworkStatus {
  timestamp: string;
  last_signal_update: string;
  tick_count: number;
  incidents: {
    active: number;
    resolving: number;
    details: IncidentDetail[];
  };
  weather: {
    condition: string;
    severity: number;
    precip_mm: number;
    wind_speed_kmh: number;
    visibility_m: number;
  };
  alerts: ServiceAlert[];
  signal_source: string;
}

export interface IncidentDetail {
  id: string;
  incident_type: string;
  corridor: string;
  lat: number;
  lon: number;
  severity: number;
  delay_factor: number;
  status: string;
  created_at: string;
  expires_at: string;
  remaining_min: number;
  duration_min: number;
}

export interface ServiceAlert {
  id: string;
  type: string;
  title: string;
  body: string;
  severity: string;
  lat?: number;
  lon?: number;
}

export interface GeocodeSuggestion {
  display_name: string;
  short_name?: string;
  lat: number;
  lon: number;
  type: string;
  category: string;
  importance?: number;
  address?: Record<string, string>;
}

export interface LatLng {
  lat: number;
  lon: number;
  label?: string;
}

export type PlanningMode = "pre-trip" | "live";

export interface DataFreshness {
  timestamp: string;
  sources: Record<
    string,
    {
      source: string;
      last_refresh: string | null;
      next_refresh: string | null;
      status: string;
      error_message: string;
      refresh_count: number;
      cadence: string;
      freshness_seconds: number | null;
    }
  >;
}

export interface ForecastRegion {
  name: string;
  center_lat: number;
  center_lon: number;
  radius_km: number;
  model_name: string;
  enabled: boolean;
}
