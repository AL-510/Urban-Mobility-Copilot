"use client";

import { useEffect, useMemo } from "react";
import { MapContainer, TileLayer, Polyline, Marker, Popup, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";
import type { ScoredRoute, LatLng } from "@/types";

// Default: world view if no location known; overridden dynamically
const DEFAULT_LAT = 45.5152;
const DEFAULT_LON = -122.6784;
const DEFAULT_ZOOM = 12;

const ROUTE_COLORS = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626"];

const originIcon = new L.DivIcon({
  className: "",
  html: `<div style="width:16px;height:16px;background:#22c55e;border:3px solid white;border-radius:50%;box-shadow:0 2px 6px rgba(0,0,0,0.3)"></div>`,
  iconSize: [16, 16],
  iconAnchor: [8, 8],
});

const destIcon = new L.DivIcon({
  className: "",
  html: `<div style="width:16px;height:16px;background:#ef4444;border:3px solid white;border-radius:50%;box-shadow:0 2px 6px rgba(0,0,0,0.3)"></div>`,
  iconSize: [16, 16],
  iconAnchor: [8, 8],
});

const currentLocIcon = new L.DivIcon({
  className: "",
  html: `<div style="width:18px;height:18px;background:#3b82f6;border:3px solid white;border-radius:50%;box-shadow:0 0 0 6px rgba(59,130,246,0.2), 0 2px 6px rgba(0,0,0,0.3)"></div>`,
  iconSize: [18, 18],
  iconAnchor: [9, 9],
});

interface Props {
  routes: ScoredRoute[];
  selectedRoute: ScoredRoute | null;
  onSelectRoute: (route: ScoredRoute) => void;
  origin?: LatLng | null;
  destination?: LatLng | null;
  currentLocation?: LatLng | null;
  onMapClick?: (lat: number, lon: number) => void;
  mapClickTarget?: "origin" | "destination";
  recenterTarget?: { lat: number; lon: number; zoom?: number } | null;
  recenterTrigger?: number;
}

function FitBounds({ routes, origin, destination }: { routes: ScoredRoute[]; origin?: LatLng | null; destination?: LatLng | null }) {
  const map = useMap();
  useEffect(() => {
    const allCoords: [number, number][] = [];

    routes.forEach((r) => {
      r.coordinates.forEach((c) => allCoords.push([c[0], c[1]]));
    });

    if (origin) allCoords.push([origin.lat, origin.lon]);
    if (destination) allCoords.push([destination.lat, destination.lon]);

    if (allCoords.length < 2) return;
    const bounds = L.latLngBounds(allCoords);
    map.fitBounds(bounds, { padding: [50, 50] });
  }, [routes, origin, destination, map]);
  return null;
}

function Recenterer({ target, trigger }: { target?: { lat: number; lon: number; zoom?: number } | null; trigger?: number }) {
  const map = useMap();
  useEffect(() => {
    if (target) {
      map.setView([target.lat, target.lon], target.zoom || map.getZoom(), { animate: true });
    }
  }, [trigger, target, map]);
  return null;
}

function MapClickHandler({ onClick }: { onClick: (lat: number, lon: number) => void }) {
  useMapEvents({
    click: (e) => {
      onClick(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

export default function RouteMap({
  routes,
  selectedRoute,
  onSelectRoute,
  origin,
  destination,
  currentLocation,
  onMapClick,
  mapClickTarget,
  recenterTarget,
  recenterTrigger,
}: Props) {
  // Use first known location as initial center
  const initialCenter = useMemo(() => {
    if (currentLocation) return [currentLocation.lat, currentLocation.lon] as [number, number];
    if (origin) return [origin.lat, origin.lon] as [number, number];
    return [DEFAULT_LAT, DEFAULT_LON] as [number, number];
  }, []);

  // Derive origin/dest from routes if not set
  const routeOrigin = useMemo(() => {
    if (origin) return origin;
    if (routes.length > 0 && routes[0].coordinates.length > 0) {
      const c = routes[0].coordinates[0];
      return { lat: c[0], lon: c[1] };
    }
    return null;
  }, [origin, routes]);

  const routeDest = useMemo(() => {
    if (destination) return destination;
    if (routes.length > 0 && routes[0].coordinates.length > 0) {
      const c = routes[0].coordinates[routes[0].coordinates.length - 1];
      return { lat: c[0], lon: c[1] };
    }
    return null;
  }, [destination, routes]);

  return (
    <MapContainer
      center={initialCenter}
      zoom={DEFAULT_ZOOM}
      className="w-full h-full z-0"
      zoomControl={false}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
      />

      <FitBounds routes={routes} origin={origin} destination={destination} />
      <Recenterer target={recenterTarget} trigger={recenterTrigger} />

      {onMapClick && <MapClickHandler onClick={onMapClick} />}

      {/* Render all routes */}
      {routes.map((route, idx) => {
        const isSelected = selectedRoute?.name === route.name;
        const color = ROUTE_COLORS[idx % ROUTE_COLORS.length];
        return (
          <Polyline
            key={route.name}
            positions={route.coordinates.map((c) => [c[0], c[1]] as [number, number])}
            pathOptions={{
              color: isSelected ? color : `${color}66`,
              weight: isSelected ? 6 : 3,
              opacity: isSelected ? 1 : 0.5,
              dashArray: isSelected ? undefined : "8 6",
            }}
            eventHandlers={{
              click: () => onSelectRoute(route),
            }}
          >
            <Popup>
              <div className="text-sm">
                <strong>{route.name}</strong>
                <br />
                {Math.round(route.predicted_time_s / 60)} min
                {" | "}
                {(route.disruption_prob * 100).toFixed(0)}% risk
                {" | "}
                {(route.reliability_score * 100).toFixed(0)}% reliable
                {route.confidence_tier !== "full" && (
                  <>
                    <br />
                    <span className="text-amber-600 text-xs">
                      {route.confidence_tier === "partial" ? "Partial forecast" : "No forecast data"}
                    </span>
                  </>
                )}
              </div>
            </Popup>
          </Polyline>
        );
      })}

      {/* Current location marker (blue pulsing dot) — only if different from origin */}
      {currentLocation && origin && (
        Math.abs(currentLocation.lat - origin.lat) > 0.0001 ||
        Math.abs(currentLocation.lon - origin.lon) > 0.0001
      ) && (
        <Marker position={[currentLocation.lat, currentLocation.lon]} icon={currentLocIcon}>
          <Popup>Your location</Popup>
        </Marker>
      )}

      {/* Origin marker */}
      {routeOrigin && (
        <Marker position={[routeOrigin.lat, routeOrigin.lon]} icon={originIcon}>
          <Popup>
            <strong>Origin</strong>
            {routeOrigin.label && <><br />{routeOrigin.label}</>}
          </Popup>
        </Marker>
      )}

      {/* Destination marker */}
      {routeDest && (
        <Marker position={[routeDest.lat, routeDest.lon]} icon={destIcon}>
          <Popup>
            <strong>Destination</strong>
            {routeDest.label && <><br />{routeDest.label}</>}
          </Popup>
        </Marker>
      )}
    </MapContainer>
  );
}
