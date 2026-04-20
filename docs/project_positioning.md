# Urban Mobility Disruption Copilot — Project Positioning

## GitHub Summary (for repo description)

City-scale multimodal commuter intelligence platform: ST-GAT disruption forecasting (0.937 AUROC), six-layer architecture (global OSRM routing → region registry → ML predictions → fallback scoring → RAG evidence → explanation), confidence-tiered routing, daily data refresh, Nominatim geocoding, browser geolocation, pre-trip and live planning modes. Built with PyTorch, FastAPI, Next.js.

## Resume Bullets (pick 3-5)

1. **Designed and built a city-scale multimodal commuter intelligence platform** combining spatio-temporal graph neural networks, retrieval-augmented generation, and uncertainty-aware forecasting to predict route-level disruption risk 30-90 minutes ahead across road and transit networks

2. **Implemented a custom ST-GAT achieving 0.937 AUROC** on Portland disruption forecasting (+3.4% over trained LSTM baseline), with near-perfect quantile calibration (90.3% Q90 coverage) and sub-40ms inference latency on 639-node transport graphs

3. **Engineered a six-layer geography architecture** — global OSRM routing → pluggable region registry → coordinate-sampled ML predictions → fallback scoring → RAG evidence → explanation — with confidence-tiered transparency and automatic degradation outside trained regions

3b. **Built a daily data refresh architecture** with background-threaded RefreshManager, per-source freshness tracking, configurable cadences, and manual trigger API for production-ready observability

4. **Built a simulated real-time disruption engine** with incident lifecycle management, weather drift, and dynamic route rescoring — integrated via thread-safe signal manager bridging live conditions to the ML inference pipeline

5. **Developed a RAG subsystem** using Qdrant vector search and sentence-transformers to retrieve and ground route explanations in transport advisories, weather alerts, and maintenance notices (23K vectors indexed)

6. **Engineered a full-stack deployment** with FastAPI backend, Next.js frontend with interactive maps, Nominatim geocoding, browser geolocation, uncertainty visualization, Docker Compose orchestration, and comprehensive evaluation framework

## Interview Explanation (2-minute version)

"This project is a city-scale commuter intelligence system that predicts disruption risk on multimodal transport routes and recommends alternatives with grounded explanations.

The core ML component is a Spatio-Temporal Graph Attention Network — it takes the city's road and transit network as a graph, overlays temporal features like weather, incident history, and time-of-day patterns, and predicts disruption probability and delay distributions at 30, 60, and 90-minute horizons. I used quantile regression heads so the model outputs calibrated uncertainty — the 90th percentile predictions cover exactly 90.3% of actual outcomes, which is critical for real-world reliability.

The system has a two-layer geography architecture. Layer one is OSRM-based routing that works for arbitrary locations — users can search any address, use their GPS location, or click the map. Layer two is the ML forecast region where the trained ST-GAT model is valid. Every route gets a confidence tier — full, partial, or base — telling the user exactly what level of intelligence is being applied. This transparency is something most routing products don't offer.

The system runs on real Portland, OR data — 17,000 OSM nodes, TriMet GTFS schedules, Open-Meteo weather, and synthetic incidents modeled after real-world patterns. I built a simulated real-time layer that generates incident lifecycles and weather drift, so routes are rescored dynamically as conditions change.

For explainability, I built a RAG pipeline that retrieves relevant transport advisories from a Qdrant vector database and grounds the route explanations in actual evidence — separating what the model predicts from what supporting documentation says.

The trained ST-GAT achieves 0.937 AUROC, outperforming a trained LSTM baseline by 3.4% on AUROC and 34% on average precision — demonstrating that the graph structure captures spatial disruption propagation that pure temporal models miss.

The whole system is deployable — FastAPI backend, Next.js frontend with Leaflet maps, Nominatim geocoding, browser geolocation, uncertainty visualizations, Docker Compose for one-command deployment."

## Architecture Paragraph (for written applications)

The Urban Mobility Disruption Copilot is an end-to-end applied AI system for city-scale commuter intelligence. It ingests real multimodal transport data — OpenStreetMap road networks (17K nodes), GTFS transit schedules, hourly weather observations, and incident records — into a unified spatio-temporal graph representation. A custom Graph Attention Network with temporal self-attention and multi-horizon quantile regression heads forecasts disruption probability and delay distributions at 30/60/90-minute horizons (0.937 AUROC, 90.3% calibrated Q90 coverage). A two-layer geography architecture separates OSRM-based broad routing from ML-powered forecasting, with confidence-tiered scoring that transparently labels prediction reliability. A retrieval-augmented generation subsystem indexes 23K transport advisory vectors in Qdrant and grounds route explanations in retrieved evidence. A simulated real-time layer with background-thread incident lifecycle management and weather drift enables dynamic route rescoring. The system serves predictions through a FastAPI backend with sub-40ms model inference, displayed in a Next.js frontend featuring interactive maps, Nominatim geocoding, browser geolocation, uncertainty bands, and tabbed model/evidence explanation panels.

## Key Technical Differentiators

1. **Not a toy model**: Trained on real Portland data, evaluated rigorously against a trained baseline, with quantile calibration that matches theoretical coverage
2. **Full pipeline, not just a model**: Data ingestion → feature engineering → training → inference → scoring → retrieval → explanation → visualization
3. **Six-layer architecture**: Global OSRM routing → region registry → ML overlay → fallback scoring → RAG evidence → explanation — routes work anywhere, forecasts apply where valid
4. **Confidence transparency**: Every route shows whether it has full, partial, or no forecast coverage — a design pattern missing from production routing products
5. **Uncertainty-aware**: Quantile regression provides calibrated confidence intervals, not just point predictions
6. **Real-time capable**: Signal manager architecture cleanly separates simulation from inference, making it straightforward to plug in real feeds
7. **Evidence-grounded explanations**: RAG retrieval ensures explanations reference actual advisories, not hallucinated text
8. **Graph structure matters**: Demonstrated measurable gain from spatial graph attention over temporal-only baseline

## How This Differs from Google Maps (interview answer)

"Google Maps tells you the fastest route right now. This system tells you which route is most likely to *stay* reliable over the next 30-90 minutes. It forecasts disruption risk using a graph neural network that models how incidents propagate spatially through the transport network — something a pure ETA model can't capture. It gives you calibrated uncertainty bands, not just a point estimate. And it grounds every recommendation in retrieved evidence from transport advisories, so you can see *why* the system recommends a slightly slower but more reliable route. It also supports both pre-trip planning (which route should I take for my 8am commute?) and live adaptive routing (conditions just changed, should I switch?).

Unlike Google Maps, this system has a two-layer architecture that transparently tells you when you're inside the forecasting region with full intelligence versus when you're outside it with reduced confidence. That kind of model transparency is something no consumer routing product offers."

## Skills Demonstrated

- **ML Engineering**: PyTorch model architecture, training pipeline optimization (22x speedup), gradient accumulation, quantile loss, evaluation metrics
- **Graph ML**: Graph attention networks, edge features, spatial propagation modeling
- **Systems Design**: Thread-safe real-time engine, signal manager pattern, two-layer geography, confidence-tiered scoring, clean service architecture
- **Data Engineering**: Multi-source data fusion (OSM + GTFS + weather + incidents), feature engineering with O(1) caching
- **RAG/NLP**: Vector indexing, semantic retrieval, evidence-grounded generation
- **Geospatial**: OSRM routing integration, Nominatim geocoding, browser geolocation, coordinate snapping
- **Full-Stack**: FastAPI backend, Next.js frontend, Leaflet maps, geocoding, browser geolocation, Docker Compose
- **Product Design**: Pre-trip vs live planning modes, address search UX, map-click interaction, uncertainty visualization, confidence transparency
- **Evaluation**: AUROC, Brier score, calibration analysis, latency benchmarking, baseline comparison
