"""
Response Models for Radiating System

Pydantic models for API responses and data structures.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class RadiatingStatus(str, Enum):
    """Status of the radiating system"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PROCESSING = "processing"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ExpandedEntity(BaseModel):
    """Represents an entity discovered through radiating expansion"""
    id: str = Field(..., description="Unique identifier for the entity")
    name: str = Field(..., description="Name of the entity")
    type: str = Field(..., description="Type/category of the entity")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Entity properties")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1)")
    depth: int = Field(..., ge=0, description="Depth level in the radiating traversal")
    source: Optional[str] = Field(None, description="Source of the entity")
    discovered_at: datetime = Field(default_factory=datetime.now, description="Discovery timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DiscoveredRelationship(BaseModel):
    """Represents a relationship discovered through radiating expansion"""
    id: str = Field(..., description="Unique identifier for the relationship")
    source_entity_id: str = Field(..., description="Source entity ID")
    target_entity_id: str = Field(..., description="Target entity ID")
    relationship_type: str = Field(..., description="Type of relationship")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship properties")
    strength: float = Field(..., ge=0.0, le=1.0, description="Relationship strength (0-1)")
    bidirectional: bool = Field(False, description="Whether the relationship is bidirectional")
    discovered_at: datetime = Field(default_factory=datetime.now, description="Discovery timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RadiatingCoverage(BaseModel):
    """Coverage information for radiating exploration"""
    total_entities: int = Field(..., ge=0, description="Total number of entities discovered")
    total_relationships: int = Field(..., ge=0, description="Total number of relationships found")
    max_depth_reached: int = Field(..., ge=0, description="Maximum depth reached in traversal")
    coverage_percentage: float = Field(..., ge=0.0, le=100.0, description="Estimated coverage percentage")
    explored_paths: int = Field(..., ge=0, description="Number of paths explored")
    pruned_paths: int = Field(0, ge=0, description="Number of paths pruned for optimization")
    entity_types: Dict[str, int] = Field(default_factory=dict, description="Count by entity type")
    relationship_types: Dict[str, int] = Field(default_factory=dict, description="Count by relationship type")
    
    class Config:
        schema_extra = {
            "example": {
                "total_entities": 150,
                "total_relationships": 320,
                "max_depth_reached": 3,
                "coverage_percentage": 85.5,
                "explored_paths": 450,
                "pruned_paths": 120,
                "entity_types": {"Person": 50, "Organization": 30, "Location": 20},
                "relationship_types": {"WORKS_AT": 80, "KNOWS": 120, "LOCATED_IN": 40}
            }
        }


class RadiatingSettings(BaseModel):
    """Settings for the radiating system"""
    enabled: bool = Field(True, description="Whether radiating is enabled")
    max_depth: int = Field(3, ge=1, le=10, description="Maximum traversal depth")
    max_entities_per_level: int = Field(50, ge=1, le=500, description="Max entities per depth level")
    relevance_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Minimum relevance score")
    traversal_strategy: str = Field("ADAPTIVE", description="Traversal strategy")
    enable_caching: bool = Field(True, description="Whether to use caching")
    cache_ttl_seconds: int = Field(3600, ge=0, description="Cache TTL in seconds")
    enable_parallel_processing: bool = Field(True, description="Enable parallel processing")
    max_parallel_workers: int = Field(4, ge=1, le=16, description="Maximum parallel workers")
    
    class Config:
        schema_extra = {
            "example": {
                "enabled": True,
                "max_depth": 3,
                "max_entities_per_level": 50,
                "relevance_threshold": 0.3,
                "traversal_strategy": "ADAPTIVE",
                "enable_caching": True,
                "cache_ttl_seconds": 3600,
                "enable_parallel_processing": True,
                "max_parallel_workers": 4
            }
        }


class RadiatingQueryRequest(BaseModel):
    """Request model for radiating query"""
    query: str = Field(..., min_length=1, description="The query to process")
    enable_radiating: bool = Field(True, description="Whether to use radiating coverage")
    max_depth: Optional[int] = Field(None, ge=1, le=10, description="Override max depth")
    strategy: Optional[str] = Field(None, description="Override traversal strategy")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filters to apply")
    stream: bool = Field(True, description="Whether to stream the response")
    include_coverage_data: bool = Field(False, description="Include coverage metadata in response")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What are the connections between AI research and healthcare?",
                "enable_radiating": True,
                "max_depth": 3,
                "strategy": "ADAPTIVE",
                "filters": {"entity_types": ["Organization", "Technology"]},
                "stream": True,
                "include_coverage_data": True
            }
        }


class RadiatingQueryResponse(BaseModel):
    """Response model for radiating query"""
    query_id: str = Field(..., description="Unique query identifier")
    status: str = Field(..., description="Query processing status")
    response: Optional[str] = Field(None, description="Generated response")
    coverage: Optional[RadiatingCoverage] = Field(None, description="Coverage information")
    entities: List[ExpandedEntity] = Field(default_factory=list, description="Discovered entities")
    relationships: List[DiscoveredRelationship] = Field(default_factory=list, description="Discovered relationships")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "query_id": "rq_123456",
                "status": "completed",
                "response": "Based on the radiating analysis, AI research and healthcare...",
                "coverage": {
                    "total_entities": 150,
                    "total_relationships": 320,
                    "max_depth_reached": 3,
                    "coverage_percentage": 85.5
                },
                "entities": [],
                "relationships": [],
                "processing_time_ms": 2350.5,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class RadiatingSystemStatus(BaseModel):
    """Status information for the radiating system"""
    status: RadiatingStatus = Field(..., description="Current system status")
    is_healthy: bool = Field(..., description="Whether the system is healthy")
    active_queries: int = Field(0, ge=0, description="Number of active queries")
    total_queries_processed: int = Field(0, ge=0, description="Total queries processed")
    average_processing_time_ms: float = Field(0.0, ge=0.0, description="Average processing time")
    cache_hit_rate: float = Field(0.0, ge=0.0, le=100.0, description="Cache hit rate percentage")
    neo4j_connected: bool = Field(False, description="Neo4j connection status")
    redis_connected: bool = Field(False, description="Redis connection status")
    last_query_timestamp: Optional[datetime] = Field(None, description="Last query timestamp")
    uptime_seconds: float = Field(0.0, ge=0.0, description="System uptime in seconds")
    errors_last_hour: int = Field(0, ge=0, description="Errors in the last hour")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "status": "active",
                "is_healthy": True,
                "active_queries": 2,
                "total_queries_processed": 1543,
                "average_processing_time_ms": 1850.3,
                "cache_hit_rate": 65.4,
                "neo4j_connected": True,
                "redis_connected": True,
                "last_query_timestamp": "2024-01-15T10:25:00Z",
                "uptime_seconds": 86400.0,
                "errors_last_hour": 0
            }
        }


class RadiatingPreviewRequest(BaseModel):
    """Request model for previewing radiating expansion"""
    query: str = Field(..., min_length=1, description="The query to preview")
    max_depth: int = Field(2, ge=1, le=5, description="Preview depth")
    max_entities: int = Field(20, ge=1, le=100, description="Maximum entities to preview")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Artificial Intelligence applications",
                "max_depth": 2,
                "max_entities": 20
            }
        }


class RadiatingPreviewResponse(BaseModel):
    """Response model for radiating preview"""
    query: str = Field(..., description="Original query")
    expanded_queries: List[str] = Field(default_factory=list, description="Expanded query variations")
    potential_entities: List[ExpandedEntity] = Field(default_factory=list, description="Potential entities")
    estimated_coverage: RadiatingCoverage = Field(..., description="Estimated coverage")
    preview_graph_url: Optional[str] = Field(None, description="URL to preview visualization")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Artificial Intelligence applications",
                "expanded_queries": [
                    "AI applications in industry",
                    "Machine learning use cases",
                    "Deep learning implementations"
                ],
                "potential_entities": [],
                "estimated_coverage": {
                    "total_entities": 50,
                    "total_relationships": 120,
                    "max_depth_reached": 2,
                    "coverage_percentage": 60.0
                }
            }
        }


class RadiatingToggleRequest(BaseModel):
    """Request model for toggling radiating system"""
    enabled: bool = Field(..., description="Whether to enable or disable radiating")
    reason: Optional[str] = Field(None, description="Reason for the toggle")
    
    class Config:
        schema_extra = {
            "example": {
                "enabled": True,
                "reason": "Enabling for enhanced knowledge discovery"
            }
        }


class RadiatingToggleResponse(BaseModel):
    """Response model for toggling radiating system"""
    success: bool = Field(..., description="Whether the toggle was successful")
    enabled: bool = Field(..., description="Current enabled state")
    message: str = Field(..., description="Status message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Toggle timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "success": True,
                "enabled": True,
                "message": "Radiating system enabled successfully",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }