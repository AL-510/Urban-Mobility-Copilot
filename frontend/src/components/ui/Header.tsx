"use client";

import { useEffect, useState } from "react";
import { fetchHealth } from "@/lib/api";

export function Header() {
  const [version, setVersion] = useState("0.5.0");
  const [backendUp, setBackendUp] = useState(true);

  useEffect(() => {
    let mounted = true;
    const check = () => {
      fetchHealth()
        .then((h) => {
          if (!mounted) return;
          setVersion(h.version || "0.5.0");
          setBackendUp(true);
        })
        .catch(() => {
          if (!mounted) return;
          setBackendUp(false);
          // Retry after 5s if initial check fails (backend may still be starting)
          setTimeout(check, 5000);
        });
    };
    check();
    // Also re-check periodically
    const interval = setInterval(check, 30000);
    return () => { mounted = false; clearInterval(interval); };
  }, []);

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
        <div className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full ${backendUp ? "bg-green-400 animate-pulse" : "bg-red-400"}`} />
          <span className="text-blue-200 text-xs">{backendUp ? "Live" : "Offline"}</span>
        </div>
      </div>
    </header>
  );
}
