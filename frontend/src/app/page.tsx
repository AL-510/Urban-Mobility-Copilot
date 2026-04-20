"use client";

import { useState, useCallback, useRef } from "react";
import dynamic from "next/dynamic";
import { SearchPanel } from "@/components/ui/SearchPanel";
import { RouteCards } from "@/components/routes/RouteCards";
import { ExplanationPanel } from "@/components/explanation/ExplanationPanel";
import { AlertsBanner } from "@/components/ui/AlertsBanner";
import { Header } from "@/components/ui/Header";
import { NetworkStatusBar } from "@/components/ui/NetworkStatusBar";
import { fetchRoutes, reverseGeocode } from "@/lib/api";
import type { RouteResponse, RouteRequest, ScoredRoute, NetworkStatus, LatLng, PlanningMode } from "@/types";

const RouteMap = dynamic(() => import("@/components/map/RouteMap"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full bg-slate-200 animate-pulse flex items-center justify-center rounded-lg">
      <span className="text-slate-500">Loading map...</span>
    </div>
  ),
});

export default function HomePage() {
  const [response, setResponse] = useState<RouteResponse | null>(null);
  const [selectedRoute, setSelectedRoute] = useState<ScoredRoute | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [networkStatus, setNetworkStatus] = useState<NetworkStatus | null>(null);

  // Location state
  const [origin, setOrigin] = useState<LatLng | null>(null);
  const [destination, setDestination] = useState<LatLng | null>(null);
  const [currentLocation, setCurrentLocation] = useState<LatLng | null>(null);
  const [geolocating, setGeolocating] = useState(false);
  const [mapClickTarget, setMapClickTarget] = useState<"origin" | "destination">("origin");
  const [geoError, setGeoError] = useState<string | null>(null);

  // Planning mode
  const [planningMode, setPlanningMode] = useState<PlanningMode>("pre-trip");

  // Map control ref for recentering
  const mapRecenterRef = useRef<{ lat: number; lon: number; zoom?: number } | null>(null);
  const [mapRecenterTrigger, setMapRecenterTrigger] = useState(0);

  const recenterMap = useCallback((lat: number, lon: number, zoom?: number) => {
    mapRecenterRef.current = { lat, lon, zoom };
    setMapRecenterTrigger((n) => n + 1);
  }, []);

  const handleSearch = useCallback(async (request: RouteRequest) => {
    setLoading(true);
    setError(null);
    setSelectedRoute(null);
    try {
      const data = await fetchRoutes(request);
      setResponse(data);
      if (data.routes.length > 0) {
        setSelectedRoute(data.routes[0]);
      } else {
        // No routes returned — show explanation
        setError(
          data.explanation?.summary ||
          "No routes found between these locations. Try adjusting your origin or destination."
        );
      }
    } catch (err: any) {
      const message = err.message || "Failed to fetch routes";
      if (message.includes("longer than expected") || message.includes("warming up")) {
        setError(message);
      } else if (message.includes("fetch") || message.includes("network") || message.includes("Failed")) {
        setError("Cannot reach the routing server. The backend may still be starting up — please wait 30 seconds and try again.");
      } else {
        setError(message);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  const handleNetworkStatusUpdate = useCallback(
    (status: NetworkStatus) => {
      setNetworkStatus(status);
    },
    []
  );

  const handleRequestGeolocation = useCallback(() => {
    setGeoError(null);

    // Check secure context
    if (typeof window !== "undefined" && !window.isSecureContext) {
      setGeoError(
        "Geolocation requires HTTPS. On localhost, use http://localhost:3000 (not an IP address)."
      );
      return;
    }

    if (!navigator.geolocation) {
      setGeoError("Geolocation is not supported by your browser.");
      return;
    }

    setGeolocating(true);
    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude } = position.coords;
        let label = `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`;
        try {
          const result = await reverseGeocode(latitude, longitude);
          label = result.short_name || result.display_name.split(",").slice(0, 2).join(",");
        } catch {
          // Use coordinate label as fallback
        }

        const loc: LatLng = { lat: latitude, lon: longitude, label };
        setCurrentLocation(loc);
        setOrigin(loc);
        setGeolocating(false);
        setGeoError(null);
        recenterMap(latitude, longitude, 14);
      },
      (err) => {
        setGeolocating(false);
        switch (err.code) {
          case err.PERMISSION_DENIED:
            setGeoError(
              "Location permission denied. Grant access in your browser's address bar, or search by address."
            );
            break;
          case err.POSITION_UNAVAILABLE:
            setGeoError(
              "Location unavailable. Check that location services are enabled in your device settings."
            );
            break;
          case err.TIMEOUT:
            setGeoError(
              "Location request timed out. Try again or search by address."
            );
            break;
          default:
            setGeoError("Unable to get your location. Try searching by address.");
        }
      },
      { enableHighAccuracy: false, timeout: 15000, maximumAge: 60000 }
    );
  }, [recenterMap]);

  const handleMapClick = useCallback(async (lat: number, lon: number) => {
    let label = `${lat.toFixed(4)}, ${lon.toFixed(4)}`;
    try {
      const result = await reverseGeocode(lat, lon);
      label = result.short_name || result.display_name.split(",").slice(0, 2).join(",");
    } catch {
      // Use coordinate label — reverseGeocode already handles errors gracefully
    }

    const loc: LatLng = { lat, lon, label };
    if (!origin) {
      setOrigin(loc);
      setMapClickTarget("destination");
    } else if (!destination) {
      setDestination(loc);
      setMapClickTarget("origin");
    } else {
      if (mapClickTarget === "origin") {
        setOrigin(loc);
        setMapClickTarget("destination");
      } else {
        setDestination(loc);
        setMapClickTarget("origin");
      }
    }
  }, [origin, destination, mapClickTarget]);

  const handleOriginChange = useCallback((loc: LatLng | null) => {
    setOrigin(loc);
    if (loc) {
      recenterMap(loc.lat, loc.lon, 14);
    }
  }, [recenterMap]);

  const handleDestinationChange = useCallback((loc: LatLng | null) => {
    setDestination(loc);
    if (loc) {
      recenterMap(loc.lat, loc.lon, 14);
    }
  }, [recenterMap]);

  // Determine effective confidence tier for display
  const primaryTier = response?.routes?.[0]?.confidence_tier;
  const coveragePct = response?.routes?.[0]?.forecast_coverage_pct ?? 0;

  return (
    <div className="h-screen flex flex-col">
      <Header />
      <NetworkStatusBar
        autoRefreshInterval={30000}
        onStatusUpdate={handleNetworkStatusUpdate}
      />

      {response?.active_alerts && response.active_alerts.length > 0 && (
        <AlertsBanner alerts={response.active_alerts} />
      )}

      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel */}
        <div className="w-[420px] min-w-[380px] border-r border-slate-200 flex flex-col bg-white overflow-y-auto">
          <SearchPanel
            onSearch={handleSearch}
            loading={loading}
            origin={origin}
            destination={destination}
            onOriginChange={handleOriginChange}
            onDestinationChange={handleDestinationChange}
            planningMode={planningMode}
            onPlanningModeChange={setPlanningMode}
            onRequestGeolocation={handleRequestGeolocation}
            geolocating={geolocating}
          />

          {/* Geolocation error */}
          {geoError && (
            <div className="mx-4 mb-2 p-2.5 bg-amber-50 border border-amber-200 rounded-lg text-amber-800 text-xs">
              <div className="flex items-start gap-2">
                <svg className="w-4 h-4 mt-0.5 shrink-0 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div>
                  <span>{geoError}</span>
                  <button
                    onClick={() => setGeoError(null)}
                    className="ml-2 text-amber-600 underline hover:text-amber-800"
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* API error */}
          {error && (
            <div className="mx-4 mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
              <div className="flex items-start gap-2">
                <svg className="w-4 h-4 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div>
                  <p>{error}</p>
                  <button
                    onClick={() => setError(null)}
                    className="mt-1 text-red-500 underline hover:text-red-700 text-xs"
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            </div>
          )}

          {loading && (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                <p className="text-slate-500 text-sm">
                  {planningMode === "live" ? "Analyzing live conditions..." : "Forecasting disruption risk..."}
                </p>
                <p className="text-slate-400 text-xs mt-2">
                  First request may take up to 60s while the backend warms up.
                </p>
              </div>
            </div>
          )}

          {response && !loading && (
            <>
              {/* Signal freshness + planning mode */}
              <div className="mx-4 mb-2 flex items-center gap-2 text-xs text-slate-500">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    response.signal_source === "live" ? "bg-emerald-500" : "bg-slate-400"
                  }`}
                />
                <span>
                  {planningMode === "live" ? "Live routing" : "Pre-trip forecast"}
                  {response.signal_source === "live" ? " with live signals" : " with static data"}
                  {response.last_signal_update && (
                    <> &middot; Updated {new Date(response.last_signal_update).toLocaleTimeString()}</>
                  )}
                </span>
              </div>

              {/* Confidence tier banner */}
              {response.routes.length > 0 && primaryTier !== "full" && (
                <div className={`mx-4 mb-2 px-3 py-2 rounded-lg text-xs border ${
                  primaryTier === "partial"
                    ? "bg-yellow-50 border-yellow-200 text-yellow-800"
                    : "bg-slate-50 border-slate-200 text-slate-600"
                }`}>
                  <div className="flex items-center gap-2 mb-1">
                    <svg className="w-3.5 h-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span className="font-medium">
                      {primaryTier === "partial"
                        ? `Partial Forecast Coverage (${coveragePct.toFixed(0)}%)`
                        : "Outside Forecast Region"}
                    </span>
                  </div>
                  {primaryTier === "partial" ? (
                    <p>{response.routes[0].confidence_note || `Disruption predictions have reduced confidence for portions of this route.`}</p>
                  ) : (
                    <p>
                      {response.routes[0].confidence_note ||
                        "Showing estimated travel times based on road routing. For full disruption intelligence, route within the Portland metro area."}
                    </p>
                  )}
                </div>
              )}

              {/* Empty routes message */}
              {response.routes.length === 0 && !error && (
                <div className="mx-4 mb-4 p-4 bg-slate-50 border border-slate-200 rounded-lg text-center">
                  <svg className="w-8 h-8 mx-auto mb-2 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                  </svg>
                  <p className="text-sm text-slate-600 font-medium">No routes found</p>
                  <p className="text-xs text-slate-400 mt-1">{response.explanation?.summary || "Try different locations."}</p>
                </div>
              )}

              <RouteCards
                routes={response.routes}
                selectedRoute={selectedRoute}
                onSelectRoute={setSelectedRoute}
              />
            </>
          )}
        </div>

        {/* Map Area */}
        <div className="flex-1 flex flex-col">
          <div className="flex-1 relative">
            <RouteMap
              routes={response?.routes || []}
              selectedRoute={selectedRoute}
              onSelectRoute={setSelectedRoute}
              origin={origin}
              destination={destination}
              currentLocation={currentLocation}
              onMapClick={handleMapClick}
              mapClickTarget={mapClickTarget}
              recenterTarget={mapRecenterRef.current}
              recenterTrigger={mapRecenterTrigger}
            />
            {/* Map click hint */}
            {(!origin || !destination) && !loading && (
              <div className="absolute top-3 left-1/2 -translate-x-1/2 bg-white/90 backdrop-blur-sm px-4 py-2 rounded-full shadow-md text-sm text-slate-700 border border-slate-200 z-[1000]">
                Click map to set {!origin ? "origin" : "destination"}
              </div>
            )}
          </div>

          {/* Explanation Panel (bottom drawer) */}
          {response && selectedRoute && !loading && (
            <ExplanationPanel
              explanation={response.explanation}
              route={selectedRoute}
            />
          )}
        </div>
      </div>
    </div>
  );
}
