"""FastAPI router for route endpoints."""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from src.api.schemas.routes import ErrorResponse, RouteRequest, RouteResponse
from src.api.dependencies import get_route_service

router = APIRouter(prefix="/api/v1", tags=["routes"])
logger = logging.getLogger(__name__)


@router.post(
    "/routes",
    response_model=RouteResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Get scored route recommendations",
    description="Generate, score, and explain candidate commute routes between origin and destination.",
)
async def get_routes(
    request: RouteRequest,
    route_service=Depends(get_route_service),
) -> RouteResponse:
    try:
        response = await route_service.get_routes(
            origin_lat=request.origin_lat,
            origin_lon=request.origin_lon,
            dest_lat=request.dest_lat,
            dest_lon=request.dest_lon,
            departure_time=request.departure_time,
            preference=request.preference,
            max_routes=request.max_routes,
            horizon_minutes=request.horizon_minutes,
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Route generation failed")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
