"""FastAPI dependency injection: shared application state and service instances."""

import logging

logger = logging.getLogger(__name__)

# Global application state (initialized in app startup)
_app_state: dict = {}
_route_service = None
_retriever = None
_signal_manager = None
_refresh_manager = None
_region_registry = None


def set_app_state(state: dict):
    global _app_state
    _app_state = state


def get_app_state() -> dict:
    return _app_state


def set_route_service(service):
    global _route_service
    _route_service = service


def get_route_service():
    if _route_service is None:
        raise RuntimeError("Route service not initialized")
    return _route_service


def set_retriever(retriever):
    global _retriever
    _retriever = retriever


def get_retriever():
    return _retriever


def set_signal_manager(manager):
    global _signal_manager
    _signal_manager = manager


def get_signal_manager():
    return _signal_manager


def set_refresh_manager(manager):
    global _refresh_manager
    _refresh_manager = manager


def get_refresh_manager():
    return _refresh_manager


def set_region_registry(registry):
    global _region_registry
    _region_registry = registry


def get_region_registry():
    return _region_registry
