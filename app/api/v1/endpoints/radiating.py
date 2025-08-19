"""
Radiating API Endpoints

FastAPI endpoints for the Universal Radiating Coverage System.
Provides REST API access to radiating functionality.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, Any
import json
import logging
import asyncio
from datetime import datetime

from app.services.radiating.radiating_service import get_radiating_service
from app.services.radiating.models.responses import (
    RadiatingQueryRequest,
    RadiatingQueryResponse,
    RadiatingSettings,
    RadiatingSystemStatus,
    RadiatingCoverage,
    RadiatingPreviewRequest,
    RadiatingPreviewResponse,
    RadiatingToggleRequest,
    RadiatingToggleResponse
)
from app.langchain.radiating_agent_system import get_radiating_agent_pool

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=RadiatingQueryResponse)
async def execute_radiating_query(
    request: RadiatingQueryRequest,
    background_tasks: BackgroundTasks
):
    """
    Execute a query with radiating coverage.
    
    This endpoint processes queries using the radiating coverage system,
    which explores knowledge graphs in a radiating pattern to discover
    related entities and relationships.
    
    Args:
        request: Query request with parameters
        background_tasks: FastAPI background tasks
        
    Returns:
        RadiatingQueryResponse with results and optional coverage data
    """
    try:
        logger.info(f"Executing radiating query: {request.query[:100]}...")
        
        service = get_radiating_service()
        
        if request.stream:
            # For streaming, we'll use the agent system
            return await _handle_streaming_query(request)
        
        # Add force_web_search to filters if specified in request
        filters = request.filters.copy() if request.filters else {}
        if request.force_web_search is not None:
            filters['force_web_search'] = request.force_web_search
        
        # Execute non-streaming query
        result = await service.execute_radiating_query(
            query=request.query,
            max_depth=request.max_depth,
            strategy=request.strategy,
            filters=filters,
            include_coverage=request.include_coverage_data
        )
        
        # Create response
        response = RadiatingQueryResponse(
            query_id=result['query_id'],
            status=result['status'],
            response=result.get('response'),
            coverage=RadiatingCoverage(**result['coverage']) if 'coverage' in result else None,
            entities=result.get('entities', []),
            relationships=result.get('relationships', []),
            processing_time_ms=result.get('processing_time_ms'),
            timestamp=datetime.now()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error executing radiating query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _handle_streaming_query(request: RadiatingQueryRequest):
    """Handle streaming query using the agent system"""
    
    async def generate():
        """Generate streaming response"""
        try:
            # Get agent from pool with radiating config
            agent_pool = get_radiating_agent_pool()
            
            # Build radiating config from request
            filters = request.filters.copy() if request.filters else {}
            if request.force_web_search is not None:
                filters['force_web_search'] = request.force_web_search
            
            radiating_config = {
                'max_depth': request.max_depth,
                'strategy': request.strategy,
                'filters': filters
            }
            
            # Get agent with config
            agent = await agent_pool.get_agent(radiating_config=radiating_config)
            
            # Process with radiation
            context = {
                'max_depth': request.max_depth,
                'strategy': request.strategy,
                'filters': filters
            }
            
            async for chunk in agent.process_with_radiation(
                query=request.query,
                context=context,
                stream=True
            ):
                # Convert chunk to JSON and yield
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Release agent back to pool
            await agent_pool.release_agent(agent.agent_id)
            
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            error_chunk = {
                'type': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/settings", response_model=RadiatingSettings)
async def get_radiating_settings():
    """
    Get current radiating system settings.
    
    Returns the current configuration of the radiating coverage system,
    including traversal parameters, caching settings, and processing options.
    
    Returns:
        Current RadiatingSettings
    """
    try:
        service = get_radiating_service()
        return service.settings
        
    except Exception as e:
        logger.error(f"Error getting radiating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/settings", response_model=RadiatingSettings)
async def update_radiating_settings(
    settings: RadiatingSettings
):
    """
    Update radiating system settings.
    
    Updates the configuration of the radiating coverage system.
    Changes take effect immediately for new queries.
    
    Args:
        settings: New settings to apply
        
    Returns:
        Updated RadiatingSettings
    """
    try:
        logger.info("Updating radiating settings")
        
        service = get_radiating_service()
        updated_settings = await service.update_settings(settings)
        
        return updated_settings
        
    except Exception as e:
        logger.error(f"Error updating radiating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/toggle", response_model=RadiatingToggleResponse)
async def toggle_radiating_system(
    request: RadiatingToggleRequest
):
    """
    Enable or disable the radiating system.
    
    Toggles the radiating coverage system on or off. When disabled,
    queries will be processed without radiating exploration.
    
    Args:
        request: Toggle request with enabled state and optional reason
        
    Returns:
        RadiatingToggleResponse with success status
    """
    try:
        logger.info(f"Toggling radiating system: enabled={request.enabled}")
        
        service = get_radiating_service()
        success = await service.toggle_radiating(request.enabled)
        
        message = (
            f"Radiating system {'enabled' if request.enabled else 'disabled'} successfully"
            if success else
            f"Failed to {'enable' if request.enabled else 'disable'} radiating system"
        )
        
        if request.reason:
            logger.info(f"Toggle reason: {request.reason}")
        
        return RadiatingToggleResponse(
            success=success,
            enabled=service.settings.enabled,
            message=message,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error toggling radiating system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=RadiatingSystemStatus)
async def get_radiating_status():
    """
    Get radiating system status and health information.
    
    Returns comprehensive status information about the radiating system,
    including health checks, performance metrics, and active query counts.
    
    Returns:
        RadiatingSystemStatus with system health and metrics
    """
    try:
        service = get_radiating_service()
        status = await service.get_system_status()
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting radiating status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preview", response_model=RadiatingPreviewResponse)
async def preview_radiating_expansion(
    request: RadiatingPreviewRequest
):
    """
    Preview what a radiating expansion would discover.
    
    Provides a preview of entities and relationships that would be
    discovered by a radiating query, without executing the full traversal.
    Useful for understanding the potential scope of a query.
    
    Args:
        request: Preview request with query and parameters
        
    Returns:
        RadiatingPreviewResponse with preview information
    """
    try:
        logger.info(f"Previewing radiating expansion for: {request.query[:100]}...")
        
        service = get_radiating_service()
        preview_data = await service.preview_expansion(
            query=request.query,
            max_depth=request.max_depth,
            max_entities=request.max_entities
        )
        
        response = RadiatingPreviewResponse(
            query=preview_data['query'],
            expanded_queries=preview_data.get('expanded_queries', []),
            potential_entities=preview_data.get('potential_entities', []),
            estimated_coverage=RadiatingCoverage(**preview_data['estimated_coverage'])
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error previewing radiating expansion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/coverage")
async def get_entity_coverage(
    entity_ids: str = Query(..., description="Comma-separated entity IDs")
):
    """
    Get radiating coverage for specific entities.
    
    Analyzes the coverage around specified entities, showing what
    relationships and connected entities exist in the knowledge graph.
    
    Args:
        entity_ids: Comma-separated list of entity IDs
        
    Returns:
        RadiatingCoverage for the specified entities
    """
    try:
        # Parse entity IDs
        ids = [id.strip() for id in entity_ids.split(',') if id.strip()]
        
        if not ids:
            raise HTTPException(status_code=400, detail="No valid entity IDs provided")
        
        logger.info(f"Getting coverage for {len(ids)} entities")
        
        service = get_radiating_service()
        coverage = await service.get_radiating_coverage(ids)
        
        return coverage
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting entity coverage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_radiating_metrics():
    """
    Get detailed metrics for the radiating system.
    
    Returns performance metrics, cache statistics, and usage information
    for monitoring and optimization purposes.
    
    Returns:
        Dict with detailed metrics
    """
    try:
        service = get_radiating_service()
        
        # Get metrics from service
        metrics = {
            'service_metrics': service.metrics,
            'traverser_metrics': service.traverser.metrics if hasattr(service.traverser, 'metrics') else {},
            'cache_metrics': {
                'cache_hits': service.metrics.get('cache_hits', 0),
                'cache_misses': service.metrics.get('cache_misses', 0),
                'hit_rate': (
                    service.metrics.get('cache_hits', 0) / 
                    (service.metrics.get('cache_hits', 0) + service.metrics.get('cache_misses', 0)) * 100
                    if (service.metrics.get('cache_hits', 0) + service.metrics.get('cache_misses', 0)) > 0
                    else 0.0
                )
            },
            'active_queries': len(service.active_queries),
            'settings': service.settings.dict()
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting radiating metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache")
async def clear_radiating_cache():
    """
    Clear the radiating system cache.
    
    Clears all cached data in the radiating system. This can help
    resolve issues with stale data but may temporarily impact performance.
    
    Returns:
        Success message
    """
    try:
        logger.info("Clearing radiating cache")
        
        service = get_radiating_service()
        
        # Clear cache through cache manager
        if hasattr(service.cache_manager, 'clear_all'):
            await service.cache_manager.clear_all()
        
        return {
            "success": True,
            "message": "Radiating cache cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing radiating cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def radiating_health_check():
    """
    Health check endpoint for the radiating system.
    
    Quick health check that verifies the radiating system components
    are responsive and functioning correctly.
    
    Returns:
        Health status dict
    """
    try:
        service = get_radiating_service()
        status = await service.get_system_status()
        
        return {
            "healthy": status.is_healthy,
            "status": status.status,
            "components": {
                "neo4j": status.neo4j_connected,
                "redis": status.redis_connected,
                "service": status.status != "error"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "healthy": False,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }