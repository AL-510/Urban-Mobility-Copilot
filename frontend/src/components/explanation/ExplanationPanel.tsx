"use client";

import { useState } from "react";
import type { Explanation, ScoredRoute } from "@/types";
import { clsx } from "clsx";

interface Props {
  explanation: Explanation;
  route: ScoredRoute;
}

export function ExplanationPanel({ explanation, route }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [activeTab, setActiveTab] = useState<"analysis" | "model" | "evidence">("analysis");

  const confidenceColor = {
    high: "text-emerald-600 bg-emerald-50 border-emerald-200",
    medium: "text-amber-600 bg-amber-50 border-amber-200",
    low: "text-red-600 bg-red-50 border-red-200",
  }[explanation.confidence] || "text-slate-600 bg-slate-50 border-slate-200";

  const disruptionPct = (route.disruption_prob * 100).toFixed(0);
  const predictedMin = Math.round(route.predicted_time_s / 60);
  const worstCaseMin = Math.round(route.predicted_time_q90_s / 60);
  const delayMedian = route.total_delay_median_min;
  const delayQ90 = route.total_delay_q90_min;

  const modelFactors = explanation.factors.filter(f => f.source !== "retrieved_advisory");
  const evidenceFactors = explanation.factors.filter(f => f.source === "retrieved_advisory");

  return (
    <div className="bg-white border-t border-slate-200 shadow-lg">
      {/* Collapsed summary bar */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-2.5 flex items-center justify-between hover:bg-slate-50 transition-colors"
      >
        <div className="flex items-center gap-3 min-w-0">
          <svg
            className={clsx("w-4 h-4 text-slate-400 transition-transform", expanded && "rotate-180")}
            fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 15l7-7 7 7" />
          </svg>
          <span className="text-sm font-medium text-slate-700 truncate">
            {explanation.summary || `Recommended: ${route.name}`}
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0 ml-2">
          {/* Quick stats in collapsed bar */}
          <span className="text-xs text-slate-500 hidden sm:inline">
            {disruptionPct}% risk
          </span>
          <span className={clsx("text-xs px-2 py-0.5 rounded-full font-medium border", confidenceColor)}>
            {explanation.confidence} confidence
          </span>
        </div>
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="border-t border-slate-100">
          {/* Tab navigation */}
          <div className="flex border-b border-slate-100 px-4">
            {(["analysis", "model", "evidence"] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={clsx(
                  "px-3 py-2 text-xs font-medium border-b-2 transition-colors capitalize",
                  activeTab === tab
                    ? "border-blue-500 text-blue-600"
                    : "border-transparent text-slate-500 hover:text-slate-700"
                )}
              >
                {tab === "model" ? "Model Predictions" : tab === "evidence" ? "Evidence" : "Analysis"}
              </button>
            ))}
          </div>

          <div className="px-4 pb-4 pt-3">
            {/* Analysis Tab */}
            {activeTab === "analysis" && (
              <div className="space-y-3">
                <p className="text-sm text-slate-700 leading-relaxed">
                  {explanation.reasoning}
                </p>

                {explanation.comparison && (
                  <div className="bg-slate-50 rounded-lg p-3 border border-slate-100">
                    <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">
                      vs. Alternatives
                    </h4>
                    <p className="text-sm text-slate-600">{explanation.comparison}</p>
                  </div>
                )}

                {/* Contributing factors (all) */}
                {explanation.factors.length > 0 && (
                  <div>
                    <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
                      Contributing Factors
                    </h4>
                    <div className="space-y-1.5">
                      {explanation.factors.map((f, i) => (
                        <div key={i} className="flex items-start gap-2 text-sm">
                          <span className={clsx(
                            "shrink-0 w-2 h-2 rounded-full mt-1.5",
                            f.severity === "high" ? "bg-red-500" :
                            f.severity === "medium" ? "bg-amber-500" :
                            "bg-blue-500"
                          )} />
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span className="text-slate-700">{f.description}</span>
                            <span className={clsx(
                              "text-[10px] px-1.5 py-0.5 rounded-full font-medium",
                              f.source === "retrieved_advisory"
                                ? "bg-purple-50 text-purple-600"
                                : "bg-blue-50 text-blue-600"
                            )}>
                              {f.source === "retrieved_advisory" ? "evidence" : "model"}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Model Predictions Tab */}
            {activeTab === "model" && (
              <div className="space-y-4">
                {/* Prediction summary cards */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <MetricCard
                    label="Disruption Risk"
                    value={`${disruptionPct}%`}
                    color={route.disruption_prob < 0.2 ? "emerald" : route.disruption_prob < 0.5 ? "amber" : "red"}
                    sublabel="probability"
                  />
                  <MetricCard
                    label="Predicted Time"
                    value={`${predictedMin}m`}
                    color="blue"
                    sublabel="median estimate"
                  />
                  <MetricCard
                    label="Worst Case"
                    value={`${worstCaseMin}m`}
                    color="slate"
                    sublabel="90th percentile"
                  />
                  <MetricCard
                    label="Reliability"
                    value={`${(route.reliability_score * 100).toFixed(0)}%`}
                    color={route.reliability_score > 0.7 ? "emerald" : route.reliability_score > 0.4 ? "amber" : "red"}
                    sublabel="consistency"
                  />
                </div>

                {/* Uncertainty visualization */}
                <div>
                  <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
                    Travel Time Uncertainty
                  </h4>
                  <UncertaintyBar
                    median={predictedMin}
                    q90={worstCaseMin}
                    delayMedian={delayMedian}
                    delayQ90={delayQ90}
                  />
                </div>

                {/* Model-derived risk factors */}
                {modelFactors.length > 0 && (
                  <div>
                    <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
                      Model-Identified Risks
                    </h4>
                    <div className="space-y-1">
                      {modelFactors.map((f, i) => (
                        <div key={i} className="flex items-center gap-2 text-sm py-1 px-2 rounded bg-slate-50">
                          <SeverityBadge severity={f.severity} />
                          <span className="text-slate-700">{f.description}</span>
                          <span className="text-[10px] text-slate-400 ml-auto">{f.type}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Route scores breakdown */}
                <div>
                  <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
                    Composite Score Breakdown
                  </h4>
                  <div className="space-y-1.5">
                    <ScoreRow label="Time Score" value={route.time_score} />
                    <ScoreRow label="Risk Score" value={route.risk_score} />
                    <ScoreRow label="Reliability" value={route.reliability_score} />
                    <ScoreRow label="Comfort" value={route.comfort_score} />
                    <div className="border-t border-slate-200 pt-1.5 mt-1.5">
                      <ScoreRow label="Composite" value={route.composite_score} bold />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Evidence Tab */}
            {activeTab === "evidence" && (
              <div className="space-y-3">
                {explanation.evidence_citations.length === 0 && evidenceFactors.length === 0 ? (
                  <p className="text-sm text-slate-500 italic">
                    No retrieved evidence for this route. Predictions are based solely on model inference.
                  </p>
                ) : (
                  <>
                    {explanation.evidence_citations.length > 0 && (
                      <div>
                        <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
                          Retrieved Advisories
                        </h4>
                        <div className="space-y-2">
                          {explanation.evidence_citations.map((cit, i) => (
                            <div key={i} className="bg-slate-50 rounded-lg p-3 border border-slate-100">
                              <div className="flex items-start justify-between gap-2 mb-1.5">
                                <span className="text-xs font-semibold text-slate-700">{cit.title}</span>
                                <RelevanceBadge relevance={cit.relevance} />
                              </div>
                              <p className="text-xs text-slate-600 leading-relaxed">{cit.snippet}</p>
                              <div className="flex items-center gap-2 mt-2">
                                <span className={clsx(
                                  "text-[10px] px-1.5 py-0.5 rounded font-medium",
                                  cit.source === "live_simulator"
                                    ? "bg-emerald-50 text-emerald-600"
                                    : "bg-slate-200 text-slate-600"
                                )}>
                                  {cit.source === "live_simulator" ? "LIVE" : cit.source}
                                </span>
                                <span className="text-[10px] text-slate-400">ID: {cit.doc_id}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {evidenceFactors.length > 0 && (
                      <div>
                        <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
                          Evidence-Based Factors
                        </h4>
                        <div className="space-y-1">
                          {evidenceFactors.map((f, i) => (
                            <div key={i} className="flex items-center gap-2 text-sm py-1 px-2 rounded bg-purple-50/50">
                              <span className="w-1.5 h-1.5 rounded-full bg-purple-400 shrink-0" />
                              <span className="text-slate-700">{f.description}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function MetricCard({ label, value, color, sublabel }: {
  label: string; value: string; color: string; sublabel: string;
}) {
  const colorMap: Record<string, string> = {
    emerald: "text-emerald-700 bg-emerald-50 border-emerald-200",
    amber: "text-amber-700 bg-amber-50 border-amber-200",
    red: "text-red-700 bg-red-50 border-red-200",
    blue: "text-blue-700 bg-blue-50 border-blue-200",
    slate: "text-slate-700 bg-slate-50 border-slate-200",
  };

  return (
    <div className={clsx("rounded-lg p-2.5 border text-center", colorMap[color] || colorMap.slate)}>
      <div className="text-lg font-bold">{value}</div>
      <div className="text-[10px] font-medium uppercase tracking-wide opacity-80">{label}</div>
      <div className="text-[10px] opacity-60">{sublabel}</div>
    </div>
  );
}

function UncertaintyBar({ median, q90, delayMedian, delayQ90 }: {
  median: number; q90: number; delayMedian: number; delayQ90: number;
}) {
  const maxVal = q90 * 1.1;
  const medianPct = (median / maxVal) * 100;
  const q90Pct = (q90 / maxVal) * 100;
  const basePct = Math.max(((median - delayMedian) / maxVal) * 100, 5);

  return (
    <div className="space-y-1">
      <div className="relative h-6 bg-slate-100 rounded-full overflow-hidden">
        {/* Base travel time */}
        <div
          className="absolute left-0 top-0 h-full bg-blue-200 rounded-l-full"
          style={{ width: `${basePct}%` }}
        />
        {/* Median delay zone */}
        <div
          className="absolute top-0 h-full bg-amber-200"
          style={{ left: `${basePct}%`, width: `${medianPct - basePct}%` }}
        />
        {/* Q90 tail */}
        <div
          className="absolute top-0 h-full bg-red-200 rounded-r-full"
          style={{ left: `${medianPct}%`, width: `${q90Pct - medianPct}%` }}
        />
        {/* Median marker */}
        <div
          className="absolute top-0 h-full w-0.5 bg-blue-600"
          style={{ left: `${medianPct}%` }}
        />
      </div>
      <div className="flex justify-between text-[10px] text-slate-500">
        <span>Base: {Math.round(median - delayMedian)}m</span>
        <span>Median: {median}m (+{delayMedian.toFixed(0)}m delay)</span>
        <span>Worst: {q90}m</span>
      </div>
    </div>
  );
}

function SeverityBadge({ severity }: { severity: string }) {
  const colors: Record<string, string> = {
    high: "bg-red-100 text-red-700",
    medium: "bg-amber-100 text-amber-700",
    low: "bg-blue-100 text-blue-700",
    info: "bg-slate-100 text-slate-600",
  };

  return (
    <span className={clsx("text-[10px] px-1.5 py-0.5 rounded font-medium shrink-0", colors[severity] || colors.info)}>
      {severity}
    </span>
  );
}

function RelevanceBadge({ relevance }: { relevance: number }) {
  const pct = Math.round(relevance * 100);
  const color = pct >= 70 ? "text-emerald-600" : pct >= 40 ? "text-amber-600" : "text-slate-500";

  return (
    <span className={clsx("text-[10px] font-medium shrink-0", color)}>
      {pct}% relevant
    </span>
  );
}

function ScoreRow({ label, value, bold }: { label: string; value: number; bold?: boolean }) {
  const pct = Math.round(value * 100);
  const color = pct >= 70 ? "bg-emerald-500" : pct >= 40 ? "bg-amber-500" : "bg-red-500";

  return (
    <div className="flex items-center gap-3">
      <span className={clsx("text-xs w-20 shrink-0", bold ? "font-semibold text-slate-800" : "text-slate-600")}>
        {label}
      </span>
      <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
        <div className={clsx("h-full rounded-full", color)} style={{ width: `${pct}%` }} />
      </div>
      <span className={clsx("text-xs w-8 text-right", bold ? "font-bold text-slate-800" : "text-slate-600")}>
        {pct}%
      </span>
    </div>
  );
}
