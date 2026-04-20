import { RouteRequest, RouteResponse, NetworkStatus, GeocodeSuggestion, DataFreshness, ForecastRegion } from "@/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function fetchRoutes(request: RouteRequest): Promise<RouteResponse> {
  let res: Response;
  try {
    // 120s timeout — allows for Render free-tier cold start (model loading takes ~60s)
    res = await fetch(`${API_URL}/api/v1/routes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal: AbortSignal.timeout(120000),
    });
  } catch (e: any) {
    if (e?.name === "TimeoutError" || e?.name === "AbortError") {
      throw new Error("The backend is taking longer than expected to respond. If this is a fresh deployment, it may still be warming up — please try again in 30 seconds.");
    }
    throw new Error("Cannot reach the routing server. Check that the backend is running.");
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

export async function fetchAlerts(area?: string): Promise<any> {
  try {
    const params = new URLSearchParams();
    if (area) params.set("area", area);
    params.set("top_k", "10");

    const res = await fetch(`${API_URL}/api/v1/alerts?${params}`);
    if (!res.ok) return { results: [] };
    return res.json();
  } catch {
    return { results: [] };
  }
}

export async function fetchHealth(): Promise<any> {
  const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(5000) });
  return res.json();
}

export async function fetchNetworkStatus(): Promise<NetworkStatus> {
  const res = await fetch(`${API_URL}/api/v1/network-status`, { signal: AbortSignal.timeout(5000) });
  if (!res.ok) throw new Error("Failed to fetch network status");
  return res.json();
}

export async function injectIncident(
  type: string = "accident",
  corridor?: string,
  severity?: number
): Promise<any> {
  const params = new URLSearchParams({ incident_type: type });
  if (corridor) params.set("corridor", corridor);
  if (severity !== undefined) params.set("severity", severity.toString());

  const res = await fetch(`${API_URL}/api/v1/inject-incident?${params}`, {
    method: "POST",
  });
  return res.json();
}

export async function refreshSignals(): Promise<NetworkStatus> {
  const res = await fetch(`${API_URL}/api/v1/refresh-signals`, {
    method: "POST",
  });
  return res.json();
}

export async function geocodeSearch(query: string): Promise<GeocodeSuggestion[]> {
  if (!query || query.length < 1) return [];
  try {
    const res = await fetch(
      `${API_URL}/api/v1/autocomplete?q=${encodeURIComponent(query)}&limit=6`,
      { signal: AbortSignal.timeout(8000) }
    );
    if (!res.ok) return [];
    return res.json();
  } catch {
    return []; // Graceful degradation for autocomplete
  }
}

export async function geocodeFull(query: string): Promise<GeocodeSuggestion[]> {
  if (!query || query.length < 1) return [];
  try {
    const res = await fetch(
      `${API_URL}/api/v1/geocode?q=${encodeURIComponent(query)}&limit=8`,
      { signal: AbortSignal.timeout(10000) }
    );
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

export async function reverseGeocode(
  lat: number,
  lon: number
): Promise<{ display_name: string; short_name: string; lat: number; lon: number }> {
  try {
    const res = await fetch(
      `${API_URL}/api/v1/reverse-geocode?lat=${lat}&lon=${lon}`,
      { signal: AbortSignal.timeout(8000) }
    );
    if (!res.ok) {
      return { display_name: `${lat.toFixed(4)}, ${lon.toFixed(4)}`, short_name: `${lat.toFixed(4)}, ${lon.toFixed(4)}`, lat, lon };
    }
    return res.json();
  } catch {
    // Return coordinate fallback instead of throwing
    return { display_name: `${lat.toFixed(4)}, ${lon.toFixed(4)}`, short_name: `${lat.toFixed(4)}, ${lon.toFixed(4)}`, lat, lon };
  }
}

export async function fetchDataFreshness(): Promise<DataFreshness> {
  try {
    const res = await fetch(`${API_URL}/api/v1/data-freshness`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return { timestamp: "", sources: {} };
    return res.json();
  } catch {
    return { timestamp: "", sources: {} };
  }
}

export async function fetchRegions(): Promise<ForecastRegion[]> {
  try {
    const res = await fetch(`${API_URL}/api/v1/regions`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return [];
    const data = await res.json();
    return data.regions || [];
  } catch {
    return [];
  }
}

export async function triggerRefresh(source: string): Promise<void> {
  await fetch(`${API_URL}/api/v1/data-freshness/${source}/refresh`, { method: "POST" }).catch(() => {});
}
