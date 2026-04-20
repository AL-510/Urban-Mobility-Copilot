"use client";

import { useEffect, useState, useCallback } from "react";
import { fetchNetworkStatus, refreshSignals } from "@/lib/api";
import type { NetworkStatus } from "@/types";

interface Props {
  autoRefreshInterval?: number; // ms, 0 = disabled
  onStatusUpdate?: (status: NetworkStatus) => void;
}

export function NetworkStatusBar({ autoRefreshInterval = 30000, onStatusUpdate }: Props) {
  const [status, setStatus] = useState<NetworkStatus | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(autoRefreshInterval > 0);
  const [refreshing, setRefreshing] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const refresh = useCallback(async () => {
    try {
      setRefreshing(true);
      const data = await fetchNetworkStatus();
      setStatus(data);
      setLastRefresh(new Date());
      onStatusUpdate?.(data);
    } catch {
      // Silently fail — status bar is non-critical
    } finally {
      setRefreshing(false);
    }
  }, [onStatusUpdate]);

  // Initial load
  useEffect(() => {
    refresh();
  }, [refresh]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh || autoRefreshInterval <= 0) return;
    const interval = setInterval(refresh, autoRefreshInterval);
    return () => clearInterval(interval);
  }, [autoRefresh, autoRefreshInterval, refresh]);

  if (!status) return null;

  const weatherSeverity = status.weather?.severity ?? 0;
  const activeIncidents = status.incidents?.active ?? 0;
  const resolvingIncidents = status.incidents?.resolving ?? 0;
  const weatherCondition = status.weather?.condition?.replace(/_/g, " ") ?? "unknown";

  const severityColor =
    activeIncidents >= 3
      ? "bg-red-500"
      : activeIncidents >= 1
        ? "bg-amber-500"
        : "bg-emerald-500";

  const weatherColor =
    weatherSeverity > 0.5
      ? "text-red-400"
      : weatherSeverity > 0.2
        ? "text-amber-400"
        : "text-emerald-400";

  const timeSince = lastRefresh
    ? `${Math.round((Date.now() - lastRefresh.getTime()) / 1000)}s ago`
    : "—";

  return (
    <div className="bg-slate-900 text-white text-xs px-4 py-1.5 flex items-center gap-4 justify-between">
      <div className="flex items-center gap-4">
        {/* Live indicator */}
        <div className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full ${severityColor} ${refreshing ? "animate-pulse" : ""}`} />
          <span className="font-medium">LIVE</span>
        </div>

        {/* Incidents */}
        <div className="flex items-center gap-1">
          <span className="text-slate-400">Incidents:</span>
          <span className={activeIncidents > 0 ? "text-amber-400 font-medium" : "text-emerald-400"}>
            {activeIncidents} active
          </span>
          {resolvingIncidents > 0 && (
            <span className="text-slate-400">· {resolvingIncidents} resolving</span>
          )}
        </div>

        {/* Weather */}
        <div className="flex items-center gap-1">
          <span className="text-slate-400">Weather:</span>
          <span className={weatherColor}>{weatherCondition}</span>
          {weatherSeverity > 0 && (
            <span className="text-slate-500">
              ({(weatherSeverity * 100).toFixed(0)}%)
            </span>
          )}
        </div>

        {/* Signal source */}
        <span className="text-slate-500">
          [{status.signal_source}]
        </span>
      </div>

      <div className="flex items-center gap-3">
        {/* Last updated */}
        <span className="text-slate-500">Updated {timeSince}</span>

        {/* Auto-refresh toggle */}
        <button
          onClick={() => setAutoRefresh(!autoRefresh)}
          className={`px-2 py-0.5 rounded text-xs ${
            autoRefresh
              ? "bg-emerald-800 text-emerald-300"
              : "bg-slate-700 text-slate-400"
          }`}
        >
          {autoRefresh ? "Auto ●" : "Auto ○"}
        </button>

        {/* Manual refresh */}
        <button
          onClick={refresh}
          disabled={refreshing}
          className="px-2 py-0.5 rounded bg-slate-700 hover:bg-slate-600 text-slate-300 disabled:opacity-50"
        >
          {refreshing ? "..." : "Refresh"}
        </button>
      </div>
    </div>
  );
}
