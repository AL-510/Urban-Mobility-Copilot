"use client";

import type { ScoredRoute } from "@/types";
import { clsx } from "clsx";

interface Props {
  routes: ScoredRoute[];
  selectedRoute: ScoredRoute | null;
  onSelectRoute: (route: ScoredRoute) => void;
}

const ROUTE_COLORS = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626"];

function riskColor(prob: number): string {
  if (prob < 0.2) return "text-green-600 bg-green-50";
  if (prob < 0.5) return "text-amber-600 bg-amber-50";
  return "text-red-600 bg-red-50";
}

function riskLabel(prob: number): string {
  if (prob < 0.2) return "Low Risk";
  if (prob < 0.5) return "Moderate";
  return "High Risk";
}

function modeIcon(mode: string): string {
  switch (mode) {
    case "transit": return "\u{1F68C}";
    case "drive": return "\u{1F697}";
    case "walk": return "\u{1F6B6}";
    default: return "\u{1F4CD}";
  }
}

export function RouteCards({ routes, selectedRoute, onSelectRoute }: Props) {
  if (routes.length === 0) return null;

  return (
    <div className="flex-1 overflow-y-auto px-4 pb-4 space-y-3">
      <div className="flex items-center justify-between py-2">
        <h2 className="text-sm font-semibold text-slate-700">
          {routes.length} Routes Found
        </h2>
        <span className="text-xs text-slate-400">Ranked by preference</span>
      </div>

      {routes.map((route, idx) => {
        const isSelected = selectedRoute?.name === route.name;
        const color = ROUTE_COLORS[idx % ROUTE_COLORS.length];
        const timeMin = Math.round(route.predicted_time_s / 60);
        const worstMin = Math.round(route.predicted_time_q90_s / 60);
        const distKm = (route.total_distance_m / 1000).toFixed(1);

        return (
          <button
            key={route.name}
            onClick={() => onSelectRoute(route)}
            className={clsx(
              "w-full text-left rounded-xl border-2 p-3.5 transition-all",
              isSelected
                ? "border-blue-500 bg-blue-50/50 shadow-md"
                : "border-slate-200 bg-white hover:border-slate-300 hover:shadow-sm"
            )}
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                {route.is_recommended && (
                  <span className="px-2 py-0.5 bg-blue-600 text-white text-[10px] font-bold rounded-full uppercase">
                    Best
                  </span>
                )}
                <span className="font-semibold text-sm text-slate-800">{route.name}</span>
                {route.confidence_tier && route.confidence_tier !== "full" && (
                  <span className={clsx(
                    "px-1.5 py-0.5 text-[10px] font-medium rounded-full",
                    route.confidence_tier === "partial"
                      ? "bg-yellow-100 text-yellow-700"
                      : "bg-slate-100 text-slate-500"
                  )}>
                    {route.confidence_tier === "partial" ? "Partial Forecast" : "No Forecast"}
                  </span>
                )}
              </div>
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: color }}
              />
            </div>

            {/* Metrics row */}
            <div className="flex items-center gap-3 text-sm mb-2">
              <div className="flex items-center gap-1">
                <span className="text-lg font-bold text-slate-900">{timeMin}</span>
                <span className="text-slate-500 text-xs">min</span>
              </div>
              <span className="text-slate-300">|</span>
              <span className="text-xs text-slate-500">{distKm} km</span>
              <span className="text-slate-300">|</span>
              <span className={clsx("text-xs px-2 py-0.5 rounded-full font-medium", riskColor(route.disruption_prob))}>
                {riskLabel(route.disruption_prob)}
              </span>
            </div>

            {/* Modes */}
            <div className="flex items-center gap-1 mb-2">
              {route.modes.map((mode) => (
                <span
                  key={mode}
                  className="px-2 py-0.5 bg-slate-100 rounded text-xs text-slate-600"
                >
                  {modeIcon(mode)} {mode}
                </span>
              ))}
              {route.num_transfers > 0 && (
                <span className="text-xs text-slate-400">
                  {route.num_transfers} transfer{route.num_transfers > 1 ? "s" : ""}
                </span>
              )}
            </div>

            {/* Score bars */}
            <div className="grid grid-cols-4 gap-2">
              <ScoreBar label="Time" value={route.time_score} />
              <ScoreBar label="Risk" value={route.risk_score} />
              <ScoreBar label="Reliable" value={route.reliability_score} />
              <ScoreBar label="Comfort" value={route.comfort_score} />
            </div>

            {/* Uncertainty band with visual bar */}
            <div className="mt-2 space-y-1">
              <div className="flex items-center gap-2">
                <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden relative">
                  <div
                    className="h-full bg-blue-400 rounded-l-full"
                    style={{ width: `${Math.min((timeMin / Math.max(worstMin, 1)) * 100, 100)}%` }}
                  />
                  <div
                    className="absolute top-0 right-0 h-full bg-red-200 rounded-r-full"
                    style={{ width: `${Math.max(100 - (timeMin / Math.max(worstMin, 1)) * 100, 0)}%` }}
                  />
                </div>
              </div>
              <div className="flex justify-between text-[10px] text-slate-500">
                <span>{timeMin}min expected</span>
                {route.total_delay_median_min > 1 && (
                  <span className="text-amber-600">+{route.total_delay_median_min.toFixed(0)}m delay</span>
                )}
                <span className="text-slate-400">{worstMin}min worst</span>
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}

function ScoreBar({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100);
  const barColor =
    pct >= 70 ? "bg-green-500" : pct >= 40 ? "bg-amber-500" : "bg-red-500";

  return (
    <div>
      <div className="flex items-center justify-between mb-0.5">
        <span className="text-[10px] text-slate-500">{label}</span>
        <span className="text-[10px] font-medium text-slate-700">{pct}%</span>
      </div>
      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div
          className={clsx("h-full rounded-full transition-all", barColor)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
