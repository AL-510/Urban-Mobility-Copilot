"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { geocodeSearch } from "@/lib/api";
import type { RouteRequest, GeocodeSuggestion, LatLng, PlanningMode } from "@/types";

const PRESETS = [
  { label: "Downtown to Lloyd", olat: 45.5189, olon: -122.6795, dlat: 45.5311, dlon: -122.6590, oname: "Downtown Portland", dname: "Lloyd District" },
  { label: "Pearl to Hawthorne", olat: 45.5299, olon: -122.6838, dlat: 45.5118, dlon: -122.6325, oname: "Pearl District", dname: "Hawthorne" },
  { label: "Alberta to PSU", olat: 45.5590, olon: -122.6455, dlat: 45.5116, dlon: -122.6838, oname: "Alberta Arts", dname: "Portland State University" },
  { label: "Gateway to Downtown", olat: 45.5310, olon: -122.5660, dlat: 45.5152, dlon: -122.6784, oname: "Gateway Transit", dname: "Pioneer Square" },
];

interface Props {
  onSearch: (req: RouteRequest) => void;
  loading: boolean;
  origin: LatLng | null;
  destination: LatLng | null;
  onOriginChange: (loc: LatLng | null) => void;
  onDestinationChange: (loc: LatLng | null) => void;
  planningMode: PlanningMode;
  onPlanningModeChange: (mode: PlanningMode) => void;
  onRequestGeolocation: () => void;
  geolocating: boolean;
}

export function SearchPanel({
  onSearch,
  loading,
  origin,
  destination,
  onOriginChange,
  onDestinationChange,
  planningMode,
  onPlanningModeChange,
  onRequestGeolocation,
  geolocating,
}: Props) {
  const [originText, setOriginText] = useState(origin?.label || "");
  const [destText, setDestText] = useState(destination?.label || "");
  const [originSuggestions, setOriginSuggestions] = useState<GeocodeSuggestion[]>([]);
  const [destSuggestions, setDestSuggestions] = useState<GeocodeSuggestion[]>([]);
  const [showOriginDropdown, setShowOriginDropdown] = useState(false);
  const [showDestDropdown, setShowDestDropdown] = useState(false);
  const [originSearching, setOriginSearching] = useState(false);
  const [destSearching, setDestSearching] = useState(false);
  const [originNoResults, setOriginNoResults] = useState(false);
  const [destNoResults, setDestNoResults] = useState(false);
  const [preference, setPreference] = useState<RouteRequest["preference"]>("balanced");
  const [departureTime, setDepartureTime] = useState(new Date().toISOString().slice(0, 16));
  const [horizon, setHorizon] = useState(30);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const originRef = useRef<HTMLDivElement>(null);
  const destRef = useRef<HTMLDivElement>(null);
  const originTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const destTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Sync external origin/dest changes to text fields
  useEffect(() => {
    if (origin?.label) setOriginText(origin.label);
  }, [origin?.label]);

  useEffect(() => {
    if (destination?.label) setDestText(destination.label);
  }, [destination?.label]);

  // Auto-detect planning mode from departure time
  useEffect(() => {
    const depDate = new Date(departureTime);
    const now = new Date();
    const diffMin = (depDate.getTime() - now.getTime()) / 60000;
    if (diffMin < 5) {
      onPlanningModeChange("live");
    } else {
      onPlanningModeChange("pre-trip");
    }
  }, [departureTime, onPlanningModeChange]);

  // Close dropdowns on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (originRef.current && !originRef.current.contains(e.target as Node)) {
        setShowOriginDropdown(false);
      }
      if (destRef.current && !destRef.current.contains(e.target as Node)) {
        setShowDestDropdown(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const doSearch = useCallback(
    (
      query: string,
      setSuggestions: (s: GeocodeSuggestion[]) => void,
      setSearching: (b: boolean) => void,
      setNoResults: (b: boolean) => void,
      timeoutRef: React.MutableRefObject<NodeJS.Timeout | null>,
    ) => {
      if (query.length < 2) {
        setSuggestions([]);
        setNoResults(false);
        return;
      }
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      setSearching(true);
      setNoResults(false);
      timeoutRef.current = setTimeout(async () => {
        const results = await geocodeSearch(query);
        setSuggestions(results);
        setSearching(false);
        setNoResults(results.length === 0);
      }, 250);
    },
    []
  );

  const handleOriginInput = (val: string) => {
    setOriginText(val);
    setShowOriginDropdown(true);
    // Clear origin coordinates when user types (they haven't selected yet)
    if (origin && val !== origin.label) {
      onOriginChange(null);
    }
    doSearch(val, setOriginSuggestions, setOriginSearching, setOriginNoResults, originTimeoutRef);
  };

  const handleDestInput = (val: string) => {
    setDestText(val);
    setShowDestDropdown(true);
    if (destination && val !== destination.label) {
      onDestinationChange(null);
    }
    doSearch(val, setDestSuggestions, setDestSearching, setDestNoResults, destTimeoutRef);
  };

  const selectOrigin = (s: GeocodeSuggestion) => {
    const label = s.short_name || s.display_name.split(",").slice(0, 2).join(", ");
    setOriginText(label);
    onOriginChange({ lat: s.lat, lon: s.lon, label });
    setShowOriginDropdown(false);
    setOriginSuggestions([]);
    setOriginNoResults(false);
  };

  const selectDest = (s: GeocodeSuggestion) => {
    const label = s.short_name || s.display_name.split(",").slice(0, 2).join(", ");
    setDestText(label);
    onDestinationChange({ lat: s.lat, lon: s.lon, label });
    setShowDestDropdown(false);
    setDestSuggestions([]);
    setDestNoResults(false);
  };

  const handlePreset = (p: typeof PRESETS[0]) => {
    setOriginText(p.oname);
    setDestText(p.dname);
    onOriginChange({ lat: p.olat, lon: p.olon, label: p.oname });
    onDestinationChange({ lat: p.dlat, lon: p.dlon, label: p.dname });
  };

  const handleUseMyLocation = () => {
    onRequestGeolocation();
  };

  const handleSetLiveNow = () => {
    setDepartureTime(new Date().toISOString().slice(0, 16));
    onPlanningModeChange("live");
  };

  const handleClearOrigin = () => {
    setOriginText("");
    onOriginChange(null);
    setOriginSuggestions([]);
  };

  const handleClearDest = () => {
    setDestText("");
    onDestinationChange(null);
    setDestSuggestions([]);
  };

  const handleSubmit = () => {
    if (!origin || !destination) return;
    onSearch({
      origin_lat: origin.lat,
      origin_lon: origin.lon,
      dest_lat: destination.lat,
      dest_lon: destination.lon,
      departure_time: new Date(departureTime).toISOString(),
      preference,
      max_routes: 5,
      horizon_minutes: horizon,
    });
  };

  const canSubmit = origin && destination && !loading;

  return (
    <div className="p-4 border-b border-slate-200 space-y-3">
      {/* Planning mode indicator */}
      <div className="flex items-center gap-2">
        <button
          onClick={handleSetLiveNow}
          className={`flex-1 py-1.5 text-xs font-medium rounded-l-lg border transition-all ${
            planningMode === "live"
              ? "bg-emerald-600 text-white border-emerald-600"
              : "bg-white text-slate-500 border-slate-300 hover:border-emerald-300"
          }`}
        >
          {planningMode === "live" && <span className="inline-block w-1.5 h-1.5 bg-white rounded-full mr-1 animate-pulse" />}
          Live — Leave Now
        </button>
        <button
          onClick={() => onPlanningModeChange("pre-trip")}
          className={`flex-1 py-1.5 text-xs font-medium rounded-r-lg border transition-all ${
            planningMode === "pre-trip"
              ? "bg-blue-600 text-white border-blue-600"
              : "bg-white text-slate-500 border-slate-300 hover:border-blue-300"
          }`}
        >
          Pre-Trip — Plan Ahead
        </button>
      </div>

      {/* Origin input */}
      <div ref={originRef} className="relative">
        <label className="text-xs font-medium text-slate-600">From</label>
        <div className="flex gap-1.5 mt-1">
          <div className="relative flex-1">
            <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-emerald-500 text-sm font-bold">A</span>
            <input
              type="text"
              value={originText}
              onChange={(e) => handleOriginInput(e.target.value)}
              onFocus={() => {
                if (originSuggestions.length > 0) setShowOriginDropdown(true);
              }}
              placeholder="Search address, place, or landmark..."
              className="w-full pl-7 pr-8 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
            {originText && (
              <button
                onClick={handleClearOrigin}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
          <button
            onClick={handleUseMyLocation}
            disabled={geolocating}
            title="Use my location"
            className="px-2.5 py-2 border border-slate-300 rounded-lg hover:bg-blue-50 hover:border-blue-300 transition-colors disabled:opacity-50"
          >
            {geolocating ? (
              <span className="w-4 h-4 block border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            ) : (
              <svg className="w-4 h-4 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <circle cx="12" cy="12" r="3" />
                <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
              </svg>
            )}
          </button>
        </div>
        {showOriginDropdown && (
          <SuggestionDropdown
            suggestions={originSuggestions}
            onSelect={selectOrigin}
            searching={originSearching}
            noResults={originNoResults}
            query={originText}
          />
        )}
      </div>

      {/* Destination input */}
      <div ref={destRef} className="relative">
        <label className="text-xs font-medium text-slate-600">To</label>
        <div className="flex gap-1.5 mt-1">
          <div className="relative flex-1">
            <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-red-500 text-sm font-bold">B</span>
            <input
              type="text"
              value={destText}
              onChange={(e) => handleDestInput(e.target.value)}
              onFocus={() => {
                if (destSuggestions.length > 0) setShowDestDropdown(true);
              }}
              placeholder="Search address, place, or landmark..."
              className="w-full pl-7 pr-8 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
            {destText && (
              <button
                onClick={handleClearDest}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        </div>
        {showDestDropdown && (
          <SuggestionDropdown
            suggestions={destSuggestions}
            onSelect={selectDest}
            searching={destSearching}
            noResults={destNoResults}
            query={destText}
          />
        )}
      </div>

      {/* Departure time */}
      {planningMode === "pre-trip" ? (
        <div>
          <label className="text-xs font-medium text-slate-600">Departure Time</label>
          <input
            type="datetime-local"
            value={departureTime}
            onChange={(e) => setDepartureTime(e.target.value)}
            className="w-full mt-1 px-2.5 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
      ) : (
        <div className="flex items-center gap-2 text-xs text-emerald-600 bg-emerald-50 rounded-lg px-3 py-1.5 border border-emerald-200">
          <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
          Departing now — using live network conditions
        </div>
      )}

      {/* Preference selector */}
      <div>
        <label className="text-xs font-medium text-slate-600">Optimize For</label>
        <div className="grid grid-cols-4 gap-1.5 mt-1.5">
          {(["balanced", "fastest", "least_risky", "cheapest"] as const).map((p) => (
            <button
              key={p}
              onClick={() => setPreference(p)}
              className={`px-2 py-1.5 text-xs rounded-md border transition-all ${
                preference === p
                  ? "bg-blue-600 text-white border-blue-600 shadow-sm"
                  : "bg-white text-slate-600 border-slate-300 hover:border-blue-300"
              }`}
            >
              {p === "least_risky" ? "Safest" : p === "balanced" ? "Balanced" : p === "fastest" ? "Fastest" : "Cheapest"}
            </button>
          ))}
        </div>
      </div>

      {/* Advanced options toggle */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="text-xs text-slate-500 hover:text-blue-600 transition-colors"
      >
        {showAdvanced ? "Hide" : "Show"} advanced options
      </button>

      {showAdvanced && (
        <div className="space-y-3 pt-1">
          <div>
            <label className="text-xs font-medium text-slate-600">
              Forecast Horizon: <span className="text-blue-600">{horizon} min</span>
            </label>
            <input
              type="range" min={30} max={90} step={30} value={horizon}
              onChange={(e) => setHorizon(parseInt(e.target.value))}
              className="w-full mt-1"
            />
          </div>

          {/* Quick presets */}
          <div>
            <label className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Quick Routes (Portland)</label>
            <div className="flex flex-wrap gap-1.5 mt-1.5">
              {PRESETS.map((p) => (
                <button
                  key={p.label}
                  onClick={() => handlePreset(p)}
                  className="px-2.5 py-1 text-xs bg-slate-100 hover:bg-blue-50 hover:text-blue-600 rounded-full transition-colors border border-slate-200"
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Search button */}
      <button
        onClick={handleSubmit}
        disabled={!canSubmit}
        className="w-full py-2.5 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed shadow-md"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            Analyzing routes...
          </span>
        ) : (
          planningMode === "live" ? "Find Best Routes Now" : "Plan Best Routes"
        )}
      </button>

      {!origin && !destination && (
        <p className="text-[11px] text-slate-400 text-center">
          Search for places, use your location, or click the map to set origin & destination
        </p>
      )}
    </div>
  );
}

function SuggestionDropdown({
  suggestions,
  onSelect,
  searching,
  noResults,
  query,
}: {
  suggestions: GeocodeSuggestion[];
  onSelect: (s: GeocodeSuggestion) => void;
  searching: boolean;
  noResults: boolean;
  query: string;
}) {
  if (!searching && suggestions.length === 0 && !noResults) return null;

  return (
    <div className="absolute z-50 top-full left-0 right-0 mt-1 bg-white border border-slate-200 rounded-lg shadow-lg max-h-56 overflow-y-auto">
      {searching && (
        <div className="px-3 py-2 text-xs text-slate-500 flex items-center gap-2">
          <span className="w-3 h-3 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
          Searching...
        </div>
      )}

      {!searching && noResults && query.length >= 2 && (
        <div className="px-3 py-2.5 text-xs text-slate-500">
          No results for &ldquo;{query}&rdquo;. Try a different search or click the map.
        </div>
      )}

      {suggestions.map((s, i) => {
        const main = s.short_name || s.display_name.split(",").slice(0, 2).join(", ");
        const secondary = s.display_name.split(",").slice(2, 4).join(", ").trim();
        const typeLabel = s.type && s.type !== "yes" ? s.type.replace(/_/g, " ") : "";

        return (
          <button
            key={`${s.lat}-${s.lon}-${i}`}
            onClick={() => onSelect(s)}
            className="w-full text-left px-3 py-2 hover:bg-blue-50 transition-colors border-b border-slate-100 last:border-0"
          >
            <div className="flex items-center gap-2">
              <svg className="w-3.5 h-3.5 text-slate-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              <div className="min-w-0 flex-1">
                <div className="text-sm text-slate-800 truncate">{main}</div>
                {secondary && (
                  <div className="text-[11px] text-slate-400 truncate">{secondary}</div>
                )}
              </div>
              {typeLabel && (
                <span className="text-[10px] text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded shrink-0">
                  {typeLabel}
                </span>
              )}
            </div>
          </button>
        );
      })}
    </div>
  );
}
