"""FastAPI application entrypoint with startup initialization."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import (
    set_app_state, set_refresh_manager, set_region_registry,
    set_retriever, set_route_service, set_signal_manager,
)
from src.api.routers import evidence, geocode, health, network, routes
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def _init_services():
    """Initialize all services on startup."""
    settings = get_settings()
    state = {
        "model_loaded": False,
        "graph_nodes": 0,
        "graph_edges": 0,
        "vector_db_status": "disconnected",
        "realtime_enabled": False,
    }

    # 1. Load or build transport graph
    from src.graph.builder import TransportGraph

    train_graph_path = settings.processed_dir / "transport_graph_train.pkl"
    graph_path = settings.processed_dir / "transport_graph.pkl"

    if train_graph_path.exists():
        logger.info("Loading training subgraph (matches trained model)...")
        graph = TransportGraph.load(train_graph_path)
    elif graph_path.exists():
        logger.info("Loading existing transport graph...")
        graph = TransportGraph.load(graph_path)
    else:
        logger.info("Building demo transport graph...")
        graph = _build_demo_graph(settings)

    state["graph_nodes"] = graph.num_nodes
    state["graph_edges"] = graph.num_edges
    logger.info(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

    # 2. Load or create predictor
    from src.inference.predictor import DisruptionPredictor, create_demo_predictor
    checkpoint = settings.checkpoint_dir / f"{settings.model_name}_best.pt"
    if checkpoint.exists():
        predictor = DisruptionPredictor(graph=graph, checkpoint_path=checkpoint)
        state["model_loaded"] = True
    else:
        logger.warning("No trained model found, using demo predictor")
        predictor = create_demo_predictor(graph)
        state["model_loaded"] = True

    # 3. Start real-time disruption simulator
    from src.realtime.simulator import DisruptionSimulator
    from src.realtime.signal_manager import SignalManager

    simulator = DisruptionSimulator(tick_interval_s=30, incident_rate_per_min=0.15)
    signal_manager = SignalManager(simulator)
    signal_manager.start()
    set_signal_manager(signal_manager)
    state["realtime_enabled"] = True
    logger.info("Real-time disruption simulator started")

    # 4. Initialize region registry
    from src.regions.registry import RegionRegistry
    region_registry = RegionRegistry()
    set_region_registry(region_registry)
    logger.info(f"Region registry: {len(region_registry.regions)} forecast region(s)")

    # 5. Initialize route scorer and explainer
    from src.explanation.engine import ExplanationEngine
    from src.routes.scorer import RouteScorer

    scorer = RouteScorer(predictor)
    explainer = ExplanationEngine()

    # 6. Initialize RAG retriever
    retriever = None
    try:
        from src.rag.retriever import AdvisoryRetriever
        retriever = AdvisoryRetriever()
        collections = retriever.client.get_collections()
        has_data = any(c.name == settings.qdrant_collection for c in collections.collections)
        state["vector_db_status"] = "connected" if has_data else "connected (empty)"
        set_retriever(retriever)
        logger.info(f"Vector DB connected, collection exists: {has_data}")
    except Exception as e:
        logger.warning(f"Vector DB not available: {e}")
        state["vector_db_status"] = "disconnected"

    # 7. Create route service (OSRM-first, ML overlay via region registry)
    from src.api.services.route_service import RouteService
    route_service = RouteService(
        route_scorer=scorer,
        retriever=retriever,
        explanation_engine=explainer,
        signal_manager=signal_manager,
        graph=graph,
        region_registry=region_registry,
    )
    set_route_service(route_service)

    # 8. Start data refresh manager
    from src.refresh.manager import RefreshManager
    from src.refresh.jobs import (
        refresh_weather, refresh_advisories,
        refresh_rag_index, refresh_incidents,
    )

    refresh_mgr = RefreshManager()
    refresh_mgr.register_job("weather", refresh_weather, 1800, "every 30 min")
    refresh_mgr.register_job("advisories", refresh_advisories, 900, "every 15 min")
    refresh_mgr.register_job("incidents", refresh_incidents, 900, "every 15 min")
    refresh_mgr.register_job("rag_index", refresh_rag_index, 86400, "daily")
    refresh_mgr.start()
    set_refresh_manager(refresh_mgr)
    logger.info("Refresh manager started with 4 jobs")

    set_app_state(state)
    logger.info("All services initialized (v0.5.0 — global routing)")


def _build_demo_graph(settings) -> "TransportGraph":
    """Build a simplified demo graph for local development."""
    from src.graph.builder import TransportGraph, EDGE_ROAD, EDGE_TRANSIT, EDGE_TRANSFER
    import numpy as np

    graph = TransportGraph()

    center_lat = settings.city_center_lat
    center_lon = settings.city_center_lon
    grid_size = 15
    spacing = 0.003

    for i in range(grid_size):
        for j in range(grid_size):
            lat = center_lat + (i - grid_size // 2) * spacing
            lon = center_lon + (j - grid_size // 2) * spacing
            graph._get_or_create_node(f"road_{i}_{j}", lat, lon, "road")

    for i in range(grid_size):
        for j in range(grid_size):
            nid = graph.node_id_map[f"road_{i}_{j}"]
            if j + 1 < grid_size:
                nid2 = graph.node_id_map[f"road_{i}_{j+1}"]
                dist = spacing * 111000 * np.cos(np.radians(center_lat))
                for src, dst in [(nid, nid2), (nid2, nid)]:
                    graph.G.add_edge(src, dst, edge_type=EDGE_ROAD,
                                     length_m=dist, speed_kph=40,
                                     base_travel_time_s=dist / (40 / 3.6),
                                     lanes=2, highway_class="secondary", capacity=900)
            if i + 1 < grid_size:
                nid2 = graph.node_id_map[f"road_{i+1}_{j}"]
                dist = spacing * 111000
                for src, dst in [(nid, nid2), (nid2, nid)]:
                    graph.G.add_edge(src, dst, edge_type=EDGE_ROAD,
                                     length_m=dist, speed_kph=40,
                                     base_travel_time_s=dist / (40 / 3.6),
                                     lanes=2, highway_class="secondary", capacity=900)

    transit_stops = []
    for k in range(grid_size):
        lat = center_lat + (k - grid_size // 2) * spacing
        lon = center_lon
        nid = graph._get_or_create_node(
            f"transit_stop_{k}", lat, lon, "transit", stop_name=f"Stop {k}"
        )
        transit_stops.append(nid)

    for k in range(len(transit_stops) - 1):
        src, dst = transit_stops[k], transit_stops[k + 1]
        dist = spacing * 111000
        for s, d in [(src, dst), (dst, src)]:
            graph.G.add_edge(s, d, edge_type=EDGE_TRANSIT,
                             base_travel_time_s=90, length_m=dist,
                             route_id="DEMO_LINE", frequency_min=10)

    for k in range(grid_size):
        transit_nid = transit_stops[k]
        road_nid = graph.node_id_map.get(f"road_{k}_{grid_size // 2}")
        if road_nid is not None:
            for s, d in [(transit_nid, road_nid), (road_nid, transit_nid)]:
                graph.G.add_edge(s, d, edge_type=EDGE_TRANSFER,
                                 base_travel_time_s=60, length_m=50)

    graph.save()
    return graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger.info("Starting Urban Mobility Disruption Copilot...")
    _init_services()
    yield
    # Shutdown
    from src.api.dependencies import get_signal_manager, get_refresh_manager
    sm = get_signal_manager()
    if sm:
        sm.stop()
    rm = get_refresh_manager()
    if rm:
        rm.stop()
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Urban Mobility Disruption Copilot",
        description="City-scale multimodal commuter intelligence with spatio-temporal graph deep learning",
        version="0.5.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import freshness/regions routers
    from src.api.routers import freshness, regions

    app.include_router(health.router)
    app.include_router(routes.router)
    app.include_router(evidence.router)
    app.include_router(network.router)
    app.include_router(geocode.router)
    app.include_router(freshness.router)
    app.include_router(regions.router)

    return app


app = create_app()
