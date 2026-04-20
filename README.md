# Urban Mobility Disruption Copilot

**Disruption-aware route intelligence for urban commuters.**

A full-stack ML-powered routing application that predicts transportation disruptions using a custom spatio-temporal graph neural network, then recommends optimal routes grounded in real evidence. Routes work globally via OSRM; disruption forecasting is active inside trained regions (Portland, OR as the flagship).

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![Next.js](https://img.shields.io/badge/Next.js-14-black) ![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red) ![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

<!-- Uncomment after deploying and replace URL:
![App Screenshot](docs/screenshots/app-overview.png)
-->

---

## Why This Exists (and How It Differs from Google Maps)

Google Maps optimizes for **current conditions** — it tells you what traffic looks like *right now*.

This system optimizes for **predicted future conditions** — it tells you what disruptions are *likely to happen* on your route in the next 30–90 minutes, using a trained graph neural network, and explains *why* it recommends a specific route with supporting evidence from a retrieval-augmented generation (RAG) pipeline.

| Capability | Google Maps | This Project |
|---|---|---|
| Current traffic | Yes | Yes (via OSRM) |
| Future disruption prediction | No | Yes (ST-GAT, 30/60/90 min) |
| Quantile delay estimates | No | Yes (10th/50th/90th percentile) |
| Confidence-tiered routing | No | Yes (full / partial / base) |
| Evidence-grounded explanations | No | Yes (RAG with Qdrant) |
| Live disruption simulation | No | Yes (incidents, weather, alerts) |
| Multi-criteria scoring | Limited | Yes (time, risk, reliability, comfort) |

---

## Core Features

- **ST-GAT Forecasting** — Custom Spatio-Temporal Graph Attention Network trained on Portland's road/transit network. Predicts disruption probability and delay distributions at 30/60/90-minute horizons with **0.937 mean AUROC** across all forecast horizons.
- **Global Road Routing** — OSRM provides real road geometry for any origin/destination worldwide. Routes draw along actual roads, not straight lines.
- **Confidence Tiers** — Inside the trained region, routes get full ML predictions. Outside, they get time/distance heuristic scoring with clear "outside forecast region" labeling. The system never silently lies about its confidence.
- **Evidence Grounding** — Route explanations cite retrieved advisories from a Qdrant vector database (23,043 indexed embeddings). Each recommendation includes supporting evidence.
- **Live Disruption Simulation** — A background simulator generates synthetic incidents, weather events, and service alerts every 30 seconds, driving the live risk signal.
- **Multi-Criteria Scoring** — Routes are ranked by a composite of time, disruption risk, reliability, and comfort, with user-selectable preference weighting.
- **Background Data Refresh** — Four scheduled jobs continuously refresh weather (every 30 min), advisories (every 15 min), incidents (every 15 min), and the RAG index (daily).

---

## Architecture

```
+---------------------------------------------------------------+
|  Frontend  (Next.js 14 + Leaflet + Tailwind CSS)              |
|  Search/Autocomplete | Route Cards | Explanation Panel | Map  |
+-----------------------------+---------------------------------+
                              | REST API (JSON)
+-----------------------------v---------------------------------+
|  Backend  (FastAPI 0.5.0 + PyTorch 2.1 + Qdrant)             |
|                                                               |
|  1. OSRM Router          real road geometry (global)          |
|  2. Region Registry      coverage analysis                    |
|  3. ST-GAT Predictor     disruption forecasts (Portland)      |
|  4. Route Scorer         multi-criteria ranking               |
|  5. RAG Retriever        advisory evidence (Qdrant)           |
|  6. Explanation Engine   natural language reasoning           |
|                                                               |
|  Background:  RefreshManager (4 jobs) | DisruptionSimulator   |
+---------------------------------------------------------------+
```

### Six-Layer Pipeline

| Layer | Component | Scope |
|---|---|---|
| 1. Geometry | `OSRMRouter` | **Global** — real road paths for any coordinates |
| 2. Coverage | `RegionRegistry` | Samples 20 points per route; determines forecast region overlap |
| 3. Prediction | `ST-GAT` + `RouteScorer` | **Portland metro** — ML disruption forecasts |
| 4. Fallback | `BaseScorer` | **Global** — time/distance heuristic scoring when outside all regions |
| 5. Evidence | `RAGRetriever` (Qdrant) | **Portland metro** — advisory grounding for explanations |
| 6. Explanation | `ExplanationEngine` | **All routes** — natural language reasoning with citations |

### Confidence Tiers

The system is transparent about where it has predictive intelligence:

| Tier | Condition | User Experience |
|---|---|---|
| **Full** | ≥80% of route inside forecast region | Full disruption predictions + evidence citations + high-confidence badge |
| **Partial** | 1–79% of route inside region | Mixed predictions + coverage % displayed + medium-confidence badge |
| **Base** | 0% inside any region | Time/distance estimates only + clear "outside forecast region" banner |

---

## ML Model Details

### ST-GAT (Spatio-Temporal Graph Attention Network)

The core forecasting model is a custom architecture combining spatial graph attention over the transport network with temporal self-attention over the recent history window.

#### Architecture

| Property | Value |
|---|---|
| **Parameters** | 101,973 (all trainable) |
| **Graph layers** | 2-layer GAT with 4 attention heads each |
| **Hidden dimension** | 64 |
| **Node feature dim** | 24 (road/transit attributes, historical delay, congestion signal) |
| **Edge feature dim** | 12 (distance, speed limit, road class, transit route type) |
| **Temporal window** | 12 steps × 5 min = 60-minute lookback |
| **Forecast horizons** | 30 min, 60 min, 90 min |
| **Output heads** | Disruption probability + delay quantiles (10th, 50th, 90th pct.) |
| **Transport graph** | 639 nodes, 1,292 edges (Portland road + transit + transfer) |

#### Training

| Property | Value |
|---|---|
| **Training data** | Portland transport graph + synthetic disruption scenarios |
| **Epochs trained** | 9 |
| **Final validation loss** | 0.5487 |
| **Loss function** | Quantile regression loss (pinball) + BCE for disruption probability |
| **Optimizer** | Adam |
| **Inference latency (CPU)** | p50 = 38.2 ms, p95 = 45.2 ms, p99 = 47.9 ms |

#### Evaluation Results (Held-Out Test Set)

| Metric | H+30 min | H+60 min | H+90 min | **Mean** |
|---|---|---|---|---|
| **AUROC** | 0.9593 | 0.9516 | 0.9002 | **0.9370** |
| **Avg. Precision** | 0.2810 | 0.2749 | 0.1538 | 0.2366 |
| **Brier Score** | 0.0025 | 0.0022 | 0.0028 | 0.0025 |
| **Delay MAE (min)** | 0.434 | 0.435 | 0.441 | 0.436 |
| **Speed Ratio MAE** | 0.0435 | 0.0435 | 0.0441 | 0.0437 |
| **Q90 Coverage** | 0.904 | 0.903 | 0.903 | 0.903 |

*AUROC degrades gracefully at the 90-minute horizon (0.9002) as expected for longer-range forecasts. Quantile coverage at the 90th percentile is well-calibrated (~0.903 vs target 0.90).*

#### Comparison with LSTM Baseline

An LSTM-based baseline (57,100 parameters) is included in `checkpoints/` for comparison. The ST-GAT outperforms it by **+3.1 AUROC points** on average, demonstrating the value of graph-structured message passing over sequence-only modeling.

| Model | Params | Mean AUROC | H+30 AUROC | H+60 AUROC | H+90 AUROC | Inference |
|---|---|---|---|---|---|---|
| **ST-GAT (this work)** | 101,973 | **0.9370** | 0.9593 | 0.9516 | 0.9002 | 39.3 ms |
| LSTM Baseline | 57,100 | 0.9061 | 0.9317 | 0.9078 | 0.8788 | 4.6 ms |

*The LSTM is significantly faster (~4.6 ms vs 39.3 ms) but the ST-GAT's explicit modeling of graph topology yields meaningfully better disruption detection, especially at the 30-min and 60-min horizons most relevant to pre-trip planning.*

#### How the Model Is Used at Inference

1. `OSRMRouter` fetches real road geometry for each candidate route
2. `RouteScorer.score_routes_with_coordinates()` samples up to 20 evenly-spaced points along the route
3. Each sample point is matched to the nearest graph node within 2 km via `graph.nearest_node_within()`
4. `DisruptionPredictor.predict_for_nodes()` runs the ST-GAT forward pass on matched nodes
5. Node-level predictions are aggregated (weighted by proximity) into route-level scores
6. Routes with no graph node matches fall back to time/distance heuristics

---

## Live System State (Verified)

The following was confirmed against a running local instance:

| Check | Result |
|---|---|
| Backend version | v0.5.0 |
| Model loaded | `stgat_v1` (101,973 params) |
| Graph loaded | 639 nodes, 1,292 edges |
| Vector DB | Connected (Qdrant, 23,043 vectors) |
| Real-time simulator | Running (incidents + weather every 30s) |
| Portland route (Hawthorne → Pearl) | 3 routes, 135+ coordinates, **Full tier**, disruption_prob ≈ 0.017 |
| NYC route (Times Square → Grand Central) | 2 routes, correct **Base tier** with "outside forecast region" label |
| London route | 3 routes, correct **Base tier** |
| Global autocomplete | Returns results worldwide (no Portland bias) |
| Data freshness | All 4 sources (weather, advisories, incidents, rag_index) reporting success |

---

## Demo Usage Flow

1. **Open** `http://localhost:3000`
2. **Set origin** — click the map, use "My Location", or type (e.g., *"Pioneer Square Portland"*)
3. **Set destination** — type or click (e.g., *"Lloyd Center"*, *"Times Square NYC"*, *"Eiffel Tower Paris"*)
4. **Choose mode** — "Live" for current conditions or "Pre-Trip" to set a future departure time
5. **Select preference** — Balanced, Fastest, Safest, or Cheapest
6. **Click "Find Best Routes"** — the system fetches OSRM geometry, runs ML scoring (if Portland) or fallback heuristics, retrieves evidence, and returns ranked results
7. **Review routes** — route cards show predicted time, disruption risk, reliability, and a confidence tier badge
8. **Read the explanation** — expand the bottom panel for Analysis, Model Predictions, and Evidence tabs

**Try outside Portland**: NYC, London, Mumbai all work with correct Base-tier labeling.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker (optional, for Qdrant)

### 1. Clone and Configure

```bash
git clone https://github.com/YOUR_USERNAME/urban-mobility-copilot.git
cd urban-mobility-copilot

cp .env.example .env
# Defaults work for local development — no edits required
```

### 2. Backend

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
uvicorn src.api.app:app --reload --port 8000
```

First startup takes 30–60 seconds (loads graph, model weights, embedding model, and initializes the disruption simulator).

Verify: `curl http://localhost:8000/health`

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### 4. Qdrant (Optional)

```bash
docker run -d -p 6333:6333 qdrant/qdrant:latest
```

Without Qdrant, the RAG evidence panel shows "No retrieved evidence" but all routing and scoring function normally.

### Docker Compose (Full Stack)

```bash
docker compose up --build
```

Starts backend (:8000), frontend (:3000), and Qdrant (:6333) together with health-check dependencies.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/routes` | Generate scored route recommendations |
| `GET` | `/api/v1/autocomplete?q=...` | Place search autocomplete (global, no regional bias) |
| `GET` | `/api/v1/geocode?q=...` | Forward geocode (Nominatim) |
| `GET` | `/api/v1/reverse-geocode?lat=&lon=` | Coordinates → address |
| `GET` | `/api/v1/network-status` | Live disruption state from simulator |
| `GET` | `/api/v1/data-freshness` | Data source freshness (4 sources) |
| `GET` | `/api/v1/regions` | Forecast region metadata |
| `GET` | `/api/v1/regions/coverage?lat=&lon=` | Is this point inside a forecast region? |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe (model + graph + dependencies) |
| `POST` | `/api/v1/inject-incident` | Inject a test disruption event |
| `POST` | `/api/v1/refresh-signals` | Force a signal refresh tick |

Interactive docs: `http://localhost:8000/docs`

### Route Request Schema

```json
POST /api/v1/routes
{
  "origin": {"lat": 45.5122, "lon": -122.6587, "name": "Hawthorne Bridge"},
  "destination": {"lat": 45.5231, "lon": -122.6765, "name": "Pearl District"},
  "mode": "live",
  "preference": "balanced",
  "departure_time": null
}
```

### Route Response Fields

| Field | Description |
|---|---|
| `routes[].geometry` | GeoJSON LineString (real road coordinates from OSRM) |
| `routes[].duration_min` | Estimated travel time |
| `routes[].disruption_prob` | Predicted disruption probability (0–1) |
| `routes[].reliability_score` | Route reliability (0–1, higher is better) |
| `routes[].composite_score` | Multi-criteria ranking score |
| `routes[].confidence_tier` | `"full"`, `"partial"`, or `"base"` |
| `routes[].coverage_pct` | % of route inside trained forecast region |
| `explanation` | Natural language reasoning with evidence citations |
| `metadata.model_version` | Which model checkpoint scored this request |

---

## Environment Variables

See `.env.example` for the complete list. Key variables:

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL used by the frontend |
| `OSRM_BASE_URL` | `https://router.project-osrm.org` | OSRM routing engine |
| `QDRANT_HOST` | `localhost` | Qdrant vector DB host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `CITY_NAME` | `portland` | Forecast region identifier |
| `CITY_CENTER_LAT` | `45.5152` | Forecast region center latitude |
| `CITY_CENTER_LON` | `-122.6784` | Forecast region center longitude |
| `CITY_RADIUS_KM` | `8` | Forecast region radius |
| `MODEL_NAME` | `stgat_v1` | Model checkpoint to load |
| `DEVICE` | `cpu` | PyTorch device (`cpu` or `cuda`) |
| `LOG_LEVEL` | `info` | Logging verbosity |

For cloud deployment, the critical override is `NEXT_PUBLIC_API_URL` — set it to your deployed backend URL.

---

## Project Structure

```
urban-mobility-copilot/
  src/
    api/
      app.py               # FastAPI entrypoint, startup lifecycle
      services/            # RouteService — orchestrates the six-layer pipeline
      routers/             # Endpoint handlers (routes, geocode, health, freshness...)
      schemas/             # Pydantic request/response models
    config/                # Settings via pydantic-settings + .env
    models/
      stgat.py             # ST-GAT architecture (GAT layers + temporal attention)
      lstm_baseline.py     # LSTM comparison model
    graph/
      transport_graph.py   # NetworkX graph builder + nearest-node lookup
    inference/
      disruption_predictor.py  # Model inference wrapper
    routes/
      osrm_router.py       # OSRM HTTP client (global road geometry)
      scorer.py            # RouteScorer (ML + fallback scoring)
    regions/
      registry.py          # RegionRegistry (coverage analysis, tier assignment)
    rag/
      retriever.py         # Qdrant RAG retriever
      indexer.py           # RAG index builder
    explanation/
      engine.py            # Natural language explanation generator
    realtime/
      simulator.py         # DisruptionSimulator (synthetic incidents/weather)
      signal_manager.py    # Live signal aggregation
    refresh/
      manager.py           # RefreshManager — 4 background data jobs
    training/              # Training pipeline (data prep + training loop)
    preprocessing/         # Feature engineering, graph construction
    data_ingestion/        # GTFS + OSM data loading utilities
  frontend/
    src/app/page.tsx       # Main page (map + search + route cards)
    src/components/        # Map, SearchPanel, RouteList, ExplanationPanel, Header
    src/lib/api.ts         # Typed API client with timeout + error handling
    src/types/             # TypeScript types matching backend schemas
  docker/
    Dockerfile.backend     # Multi-stage Python build with HEALTHCHECK
    Dockerfile.frontend    # Node build + standalone Next.js output
  docker-compose.yml       # Full stack with service_healthy dependencies
  checkpoints/
    stgat_v1_best.pt       # ST-GAT weights (best validation checkpoint)
    lstm_baseline_best.pt  # LSTM baseline weights
    training_history.json  # Loss curves + epoch metadata
  data/processed/
    transport_graph.pkl         # Full Portland transport graph (NetworkX)
    transport_graph_train.pkl   # Training-split subgraph
    transport_graph.meta.json   # Graph metadata (639 nodes, 1292 edges)
    graphs/road_network.graphml # OSMnx road network (raw)
    weather_features.parquet    # Historical weather feature data
  docs/
    evaluation_results.json     # ST-GAT vs LSTM benchmark results
  scripts/                      # run_backend.sh, daily_refresh.py
  tests/                        # Pytest test suite
  render.yaml                   # Render.com Blueprint spec
  pyproject.toml                # Project metadata and dev dependencies
  requirements.txt              # Pinned production dependencies
```

---

## Deployment

### Vercel (Frontend) + Render (Backend)

Recommended for portfolio demo.

#### Backend on Render

1. Push repo to GitHub
2. New Web Service → connect repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn src.api.app:app --host 0.0.0.0 --port $PORT`
5. Python version: 3.11.0
6. Health check path: `/health`
7. Environment variables:
   ```
   DEVICE=cpu
   OSRM_BASE_URL=https://router.project-osrm.org
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   LOG_LEVEL=info
   ```
8. Note your backend URL (e.g., `https://urban-mobility-backend.onrender.com`)

The `render.yaml` file provides a Blueprint spec for one-click deployment.

#### Frontend on Vercel

1. Import GitHub repo on [vercel.com](https://vercel.com)
2. Root directory: `frontend`
3. Framework: Next.js (auto-detected)
4. Environment variable: `NEXT_PUBLIC_API_URL=https://your-backend.onrender.com`
5. Deploy

The `frontend/vercel.json` is pre-configured.

#### Qdrant (Optional)

- **Without Qdrant**: Routing and scoring work fully; RAG evidence panel shows "No retrieved evidence"
- **Qdrant Cloud**: Free cluster at [cloud.qdrant.io](https://cloud.qdrant.io) — set `QDRANT_HOST` and `QDRANT_PORT` in Render env vars
- **Self-hosted**: `docker run -d -p 6333:6333 qdrant/qdrant:latest`

### Docker Compose (Self-Hosted)

```bash
export NEXT_PUBLIC_API_URL=http://your-server:8000
docker compose up --build -d
```

### Health Checks

| Endpoint | Returns |
|---|---|
| `GET /health` | `{"status": "healthy", "version": "0.5.0"}` |
| `GET /ready` | `{"ready": true, "checks": {"model_loaded": true, "graph_available": true, ...}}` |

---

## Known Limitations

1. **Single forecast region** — Only Portland, OR has a trained ST-GAT model. All other cities use time/distance fallback (clearly labeled as Base tier in the UI). Adding a new city requires training a regional model and registering it in `src/regions/registry.py`.
2. **Public OSRM server** — Uses the public demo OSRM server (rate limits apply). For production traffic, self-host OSRM.
3. **Synthetic disruption data** — The simulator generates synthetic incidents. Real-world deployment would require transit agency GTFS-RT feeds.
4. **Nominatim rate limits** — OSM's geocoding service has a 1 req/s fair-use policy. Handled gracefully with timeouts and fallbacks.
5. **No authentication** — Portfolio demo, not a multi-tenant service.
6. **Render free plan cold starts** — Backend sleeps after 15 min of inactivity; first request after sleep takes 30–60s to load the model and graph.

---

## Future Work

- Additional trained forecast regions (Seattle, San Francisco, Chicago)
- GTFS-RT integration for live transit disruption data
- Historical disruption data pipeline for automated model retraining
- Multi-modal trip planning (transit + drive + bike in one itinerary)
- User accounts with saved routes and push notification preferences

---

## Resume Bullets

> - Built a **full-stack disruption-aware routing platform** using a custom ST-GAT graph neural network (0.937 AUROC), OSRM global routing, and Qdrant RAG — producing explained, confidence-tiered route recommendations that distinguish model-backed forecasts from fallback estimates
> - Designed a **six-layer intelligence pipeline** (geometry → coverage → ML prediction → fallback → RAG evidence → explanation) with a pluggable region registry — globally usable while preserving regional ML intelligence with transparent confidence tiering
> - Implemented **production-hardened architecture**: graceful degradation on all dependency failures, background data refresh across 4 sources, Docker-based deployment with health checks, and a readiness probe validating model + graph + vector DB state before serving traffic

---

## License

MIT

---

*Built as a portfolio project demonstrating applied ML engineering, system design, and full-stack development.*
