"use client";

import type { EvidenceCitation } from "@/types";

interface Props {
  alerts: EvidenceCitation[];
}

export function AlertsBanner({ alerts }: Props) {
  if (alerts.length === 0) return null;

  return (
    <div className="bg-amber-50 border-b border-amber-200 px-4 py-2">
      <div className="flex items-center gap-2 overflow-x-auto">
        <span className="shrink-0 text-amber-600 text-xs font-semibold uppercase">Active Alerts:</span>
        {alerts.map((alert, i) => (
          <span
            key={i}
            className="shrink-0 text-xs bg-amber-100 text-amber-800 px-2.5 py-1 rounded-full"
          >
            {alert.title}
          </span>
        ))}
      </div>
    </div>
  );
}
