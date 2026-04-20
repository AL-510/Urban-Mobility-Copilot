"use client";

import { useEffect, useState } from "react";
import { fetchHealth } from "@/lib/api";

type BackendStatus = "checking" | "live" | "starting" | "offline";

export function Header() {
  const [version, setVersion] = useState("0.5.0");
  const [status, setStatus] = useState<BackendStatus>("checking");
  const [failCount, setFailCount] = useState(0);

  useEffect(() => {
    let mounted = true;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    const check = (currentFails: number) => {
      fetchHealth()
        .then((h) => {
          if (!mounted) return;
          setVersion(h.version || "0.5.0");
          setStatus("live");
          setFailCount(0);
          // Re-check every 30s once live
          retryTimer = setTimeout(() => check(0), 30000);
        })
        .catch(() => {
          if (!mounted) return;
          const newFails = currentFails + 1;
          setFailCount(newFails);
          // First 12 failures (~60s) = "Starting", after that = "Offline"
          setStatus(newFails <= 12 ? "starting" : "offline");
          // Retry every 5s while not live
          retryTimer = setTimeout(() => check(newFails), 5000);
        });
    };

    check(0);
    return () => {
      mounted = false;
      if (retryTimer) clearTimeout(retryTimer);
    };
  }, []);

  const statusConfig = {
    checking: { dot: "bg-yellow-400 animate-pulse", label: "Checking..." },
    live:     { dot: "bg-green-400 animate-pulse",  label: "Live" },
    starting: { dot: "bg-yellow-400 animate-pulse", label: "Starting…" },
    offline:  { dot: "bg-red-400",                  label: "Offline" },
  };
  const { dot, label } = statusConfig[status];

  return (
    <header className="gradient-primary text-white px-6 py-3 flex items-center justify-between shadow-lg">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 bg-white/20 rounded-lg flex items-center justify-center">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 6v6l4 2" />
          </svg>
        </div>
        <div>
          <h1 className="text-lg font-bold tracking-tight">Urban Mobility Copilot</h1>
          <p className="text-xs text-blue-200">Disruption-aware route intelligence</p>
        </div>
      </div>
      <div className="flex items-center gap-4 text-sm">
        <span className="text-blue-200">Portland, OR</span>
        <span className="text-blue-300/70 text-xs">+ Global Routing</span>
        <div className="flex items-center gap-1.5" title={status === "starting" ? "Backend is loading model & graph (~60s on first start)" : undefined}>
          <span className={`w-2 h-2 rounded-full ${dot}`} />
          <span className="text-blue-200 text-xs">{label}</span>
        </div>
      </div>
    </header>
  );
}
