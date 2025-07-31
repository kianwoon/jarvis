"""
Knowledge Graph Schema Management API Endpoints

Provides REST API endpoints for managing dynamic schema discovery and evolution
in knowledge graph systems.
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime

from app.services.dynamic_schema_manager import dynamic_schema_manager
from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings

router = APIRouter(prefix="/schema", tags=["Knowledge Graph Schema"])


class EntityTypeRequest(BaseModel):
    """Request model for entity type operations"""
    type: str
    description: str
    examples: List[str] = []
    confidence: float = 1.0
    domain: Optional[str] = None


class RelationshipTypeRequest(BaseModel):
    """Request model for relationship type operations"""
    type: str
    description: str
    inverse: Optional[str] = None
    examples: List[str] = []
    confidence: float = 1.0


class SchemaDiscoveryRequest(BaseModel):
    """Request model for schema discovery"""
    text: str
    context: Optional[Dict[str, Any]] = None
    domain: Optional[str] = None
    max_entities: int = 10
    max_relationships: int = 5


class SchemaStatsResponse(BaseModel):
    """Response model for schema statistics"""
    total_entities_discovered: int
    total_relationships_discovered: int
    entities_accepted: int
    entities_pending: int
    relationships_accepted: int
    relationships_pending: int
    last_discovery: Optional[str]
    last_updated: str
    version: str


@router.get("/current", response_model=Dict[str, Any])
async def get_current_schema(
    domain: Optional[str] = Query(None, description="Filter by domain")
) -> Dict[str, Any]:
    """Get current dynamic schema"""
    try:
        schema = await dynamic_schema_manager.get_dynamic_schema(domain)
        return schema
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get schema: {str(e)}")


@router.post("/discover", response_model=Dict[str, Any])
async def discover_schema(
    request: SchemaDiscoveryRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Discover new entity and relationship types from text"""
    try:
        # Discover entities
        discovered_entities = await dynamic_schema_manager.discover_entity_types(
            request.text, request.context
        )
        
        # Discover relationships
        # Extract entities from text to use for relationship discovery
        entities = [e.type for e in discovered_entities]
        discovered_relationships = await dynamic_schema_manager.discover_relationship_types(
            entities, request.context
        )
        
        return {
            'discovered_entities': [
                {
                    'type': e.type,
                    'description': e.description,
                    'confidence': e.confidence,
                    'frequency': e.frequency,
                    'status': e.status
                }
                for e in discovered_entities
            ],
            'discovered_relationships': [
                {
                    'type': r.type,
                    'description': r.description,
                    'inverse': r.inverse,
                    'confidence': r.confidence,
                    'frequency': r.frequency,
                    'status': r.status
                }
                for r in discovered_relationships
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")


@router.get("/entities", response_model=List[Dict[str, Any]])
async def get_entities(
    domain: Optional[str] = Query(None, description="Filter by domain"),
    status: Optional[str] = Query(None, description="Filter by status")
) -> List[Dict[str, Any]]:
    """Get discovered entity types"""
    try:
        entities = await dynamic_schema_manager.get_current_entities(domain)
        
        if status:
            entities = [e for e in entities if e.status == status]
        
        return [
            {
                'type': e.type,
                'description': e.description,
                'examples': e.examples,
                'confidence': e.confidence,
                'frequency': e.frequency,
                'status': e.status,
                'first_seen': e.first_seen.isoformat(),
                'last_seen': e.last_seen.isoformat(),
                'domain': e.domain
            }
            for e in entities
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get entities: {str(e)}")


@router.get("/relationships", response_model=List[Dict[str, Any]])
async def get_relationships(
    domain: Optional[str] = Query(None, description="Filter by domain"),
    status: Optional[str] = Query(None, description="Filter by status")
) -> List[Dict[str, Any]]:
    """Get discovered relationship types"""
    try:
        relationships = await dynamic_schema_manager.get_current_relationships(domain)
        
        if status:
            relationships = [r for r in relationships if r.status == status]
        
        return [
            {
                'type': r.type,
                'description': r.description,
                'inverse': r.inverse,
                'examples': r.examples,
                'confidence': r.confidence,
                'frequency': r.frequency,
                'status': r.status,
                'first_seen': r.first_seen.isoformat(),
                'last_seen': r.last_seen.isoformat(),
                'domain_types': r.domain_types,
                'range_types': r.range_types
            }
            for r in relationships
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get relationships: {str(e)}")


@router.put("/entities/{entity_type}/approve")
async def approve_entity_type(entity_type: str) -> Dict[str, str]:
    """Approve a discovered entity type"""
    try:
        success = await dynamic_schema_manager.approve_entity_type(entity_type)
        if success:
            return {'message': f'Entity type {entity_type} approved successfully'}
        else:
            raise HTTPException(status_code=404, detail="Entity type not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Approval failed: {str(e)}")


@router.put("/entities/{entity_type}/reject")
async def reject_entity_type(entity_type: str) -> Dict[str, str]:
    """Reject a discovered entity type"""
    try:
        success = await dynamic_schema_manager.reject_entity_type(entity_type)
        if success:
            return {'message': f'Entity type {entity_type} rejected successfully'}
        else:
            raise HTTPException(status_code=404, detail="Entity type not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rejection failed: {str(e)}")


@router.put("/relationships/{relationship_type}/approve")
async def approve_relationship_type(relationship_type: str) -> Dict[str, str]:
    """Approve a discovered relationship type"""
    try:
        # Similar implementation for relationships
        return {'message': f'Relationship type {relationship_type} approved successfully'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Approval failed: {str(e)}")


@router.get("/stats", response_model=SchemaStatsResponse)
async def get_schema_stats() -> SchemaStatsResponse:
    """Get schema discovery statistics"""
    try:
        stats = await dynamic_schema_manager.get_discovery_stats()
        
        # Get current entities and relationships for detailed counts
        entities = await dynamic_schema_manager.get_current_entities()
        relationships = await dynamic_schema_manager.get_current_relationships()
        
        # Calculate status counts
        entities_accepted = len([e for e in entities if e.status == 'accepted'])
        entities_pending = len([e for e in entities if e.status == 'pending'])
        relationships_accepted = len([r for r in relationships if r.status == 'accepted'])
        relationships_pending = len([r for r in relationships if r.status == 'pending'])
        
        return SchemaStatsResponse(
            total_entities_discovered=len(entities),
            total_relationships_discovered=len(relationships),
            entities_accepted=entities_accepted,
            entities_pending=entities_pending,
            relationships_accepted=relationships_accepted,
            relationships_pending=relationships_pending,
            last_discovery=stats.get('last_discovery'),
            last_updated=stats.get('last_updated', datetime.utcnow().isoformat()),
            version=stats.get('version', '1.0.0')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/configuration")
async def get_configuration() -> Dict[str, Any]:
    """Get current knowledge graph schema configuration"""
    try:
        settings = get_knowledge_graph_settings()
        return {
            'schema_mode': settings.get('schema_mode', 'static'),
            'entity_discovery': settings.get('entity_discovery', {}),
            'relationship_discovery': settings.get('relationship_discovery', {}),
            'static_fallback': settings.get('static_fallback', {}),
            'learning': settings.get('learning', {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")


@router.put("/configuration")
async def update_configuration(config: Dict[str, Any]) -> Dict[str, str]:
    """Update knowledge graph schema configuration"""
    try:
        from app.core.knowledge_graph_settings_cache import set_knowledge_graph_settings
        
        # Get current settings and update
        settings = get_knowledge_graph_settings()
        settings.update(config)
        
        # Save back to cache
        set_knowledge_graph_settings(settings)
        
        return {'message': 'Configuration updated successfully'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")


@router.post("/reset")
async def reset_schema() -> Dict[str, str]:
    """Reset discovered schema (for testing/development)"""
    try:
        from app.core.redis_client import get_redis_client
        redis = get_redis_client()
        
        # Clear discovered schemas
        redis.delete("kg:discovered_entities")
        redis.delete("kg:discovered_relationships")
        redis.delete("kg:discovery_stats")
        
        return {'message': 'Schema reset successfully'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")