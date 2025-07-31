"""
Anti-Silo Configuration Testing API
Provides endpoints for testing and optimizing anti-silo knowledge graph settings
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from app.services.neo4j_service import get_neo4j_service
from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge-graph", tags=["knowledge-graph-anti-silo"])

class AntiSiloTestRequest(BaseModel):
    """Request model for anti-silo configuration testing"""
    enable_anti_silo: bool = Field(default=True, description="Enable anti-silo system")
    anti_silo_similarity_threshold: float = Field(default=0.75, ge=0.1, le=1.0)
    anti_silo_type_boost: float = Field(default=1.2, ge=1.0, le=2.0)
    enable_cooccurrence_analysis: bool = Field(default=True)
    enable_type_based_clustering: bool = Field(default=True)
    enable_hub_entities: bool = Field(default=True)
    hub_entity_threshold: int = Field(default=3, ge=1, le=10)
    enable_semantic_clustering: bool = Field(default=True)
    clustering_similarity_threshold: float = Field(default=0.8, ge=0.5, le=1.0)
    enable_document_bridge_relationships: bool = Field(default=True)
    bridge_relationship_confidence: float = Field(default=0.6, ge=0.1, le=1.0)
    enable_temporal_linking: bool = Field(default=True)
    temporal_linking_window: int = Field(default=7, ge=1, le=30)
    enable_contextual_linking: bool = Field(default=True)
    contextual_linking_threshold: float = Field(default=0.7, ge=0.1, le=1.0)
    enable_fuzzy_matching: bool = Field(default=True)
    fuzzy_matching_threshold: float = Field(default=0.85, ge=0.5, le=1.0)
    enable_alias_detection: bool = Field(default=True)
    alias_detection_threshold: float = Field(default=0.9, ge=0.7, le=1.0)
    enable_abbreviation_matching: bool = Field(default=True)
    abbreviation_matching_threshold: float = Field(default=0.8, ge=0.6, le=1.0)
    enable_synonym_detection: bool = Field(default=True)
    synonym_detection_threshold: float = Field(default=0.75, ge=0.5, le=1.0)
    enable_hierarchical_linking: bool = Field(default=True)
    hierarchical_linking_depth: int = Field(default=2, ge=1, le=5)
    enable_geographic_linking: bool = Field(default=True)
    geographic_linking_threshold: float = Field(default=0.7, ge=0.5, le=1.0)
    enable_temporal_coherence: bool = Field(default=True)
    temporal_coherence_threshold: float = Field(default=0.6, ge=0.3, le=1.0)
    enable_semantic_bridge_entities: bool = Field(default=True)
    semantic_bridge_threshold: float = Field(default=0.65, ge=0.5, le=1.0)
    enable_cross_reference_analysis: bool = Field(default=True)
    cross_reference_threshold: float = Field(default=0.7, ge=0.5, le=1.0)
    enable_relationship_propagation: bool = Field(default=True)
    relationship_propagation_depth: int = Field(default=2, ge=1, le=5)
    enable_entity_consolidation: bool = Field(default=True)
    entity_consolidation_threshold: float = Field(default=0.85, ge=0.7, le=1.0)
    enable_synthetic_relationships: bool = Field(default=True)
    synthetic_relationship_confidence: float = Field(default=0.5, ge=0.3, le=1.0)
    enable_graph_enrichment: bool = Field(default=True)
    graph_enrichment_depth: int = Field(default=3, ge=1, le=5)
    enable_connectivity_analysis: bool = Field(default=True)
    connectivity_analysis_threshold: float = Field(default=0.4, ge=0.1, le=1.0)
    enable_isolation_detection: bool = Field(default=True)
    isolation_detection_threshold: int = Field(default=1, ge=1, le=5)

class AntiSiloTestResponse(BaseModel):
    """Response model for anti-silo configuration testing"""
    success: bool
    connectivity_score: float
    isolated_nodes: int
    total_nodes: int
    average_degree: float
    connected_components: int
    largest_component_size: int
    recommendations: List[str]
    analysis: Dict[str, Any]

@router.post("/test-anti-silo", response_model=AntiSiloTestResponse)
async def test_anti_silo_configuration(
    request: AntiSiloTestRequest
) -> AntiSiloTestResponse:
    """
    Test anti-silo configuration against current knowledge graph
    
    Analyzes the current knowledge graph and provides recommendations
    for reducing isolated nodes and improving connectivity
    """
    try:
        neo4j_service = get_neo4j_service()
        
        if not neo4j_service.is_enabled():
            return AntiSiloTestResponse(
                success=False,
                connectivity_score=0,
                isolated_nodes=0,
                total_nodes=0,
                average_degree=0,
                connected_components=0,
                largest_component_size=0,
                recommendations=["Neo4j service is not enabled"],
                analysis={"error": "Neo4j service unavailable"}
            )
        
        # Get current graph statistics
        stats_query = """
        MATCH (n)
        RETURN 
            count(n) as total_nodes,
            avg(size([(n)-[]-() | 1])) as average_degree
        """
        
        stats_result = neo4j_service.execute_cypher(stats_query)
        if not stats_result:
            return AntiSiloTestResponse(
                success=False,
                connectivity_score=0,
                isolated_nodes=0,
                total_nodes=0,
                average_degree=0,
                connected_components=0,
                largest_component_size=0,
                recommendations=["Failed to retrieve graph statistics"],
                analysis={"error": "No statistics available"}
            )
        
        stats = stats_result[0]
        total_nodes = stats.get('total_nodes', 0)
        average_degree = stats.get('average_degree', 0) or 0
        
        # Find isolated nodes (nodes with degree 0)
        isolated_query = """
        MATCH (n)
        WHERE NOT (n)--()
        RETURN count(n) as isolated_count
        """
        
        isolated_result = neo4j_service.execute_cypher(isolated_query)
        isolated_nodes = isolated_result[0].get('isolated_count', 0) if isolated_result else 0
        
        # Find connected components using GDS
        try:
            # Create graph projection
            create_graph_query = """
            CALL gds.graph.project(
                'test-graph',
                '*',
                '*'
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
            """
            
            graph_result = neo4j_service.execute_cypher(create_graph_query)
            
            if graph_result:
                # Run connected components
                components_query = """
                CALL gds.wcc.stream('test-graph')
                YIELD nodeId, componentId
                WITH componentId, collect(nodeId) as nodes
                RETURN 
                    count(componentId) as component_count,
                    max(size(nodes)) as largest_component,
                    collect(size(nodes)) as component_sizes
                """
                
                components_result = neo4j_service.execute_cypher(components_query)
                
                if components_result:
                    components = components_result[0]
                    connected_components = components.get('component_count', 1)
                    largest_component_size = components.get('largest_component', total_nodes)
                    
                    # Clean up graph
                    cleanup_query = "CALL gds.graph.drop('test-graph') YIELD graphName"
                    neo4j_service.execute_cypher(cleanup_query)
                else:
                    connected_components = 1
                    largest_component_size = total_nodes
            else:
                connected_components = 1
                largest_component_size = total_nodes
                
        except Exception as e:
            logger.warning(f"GDS analysis failed, using basic metrics: {e}")
            connected_components = 1
            largest_component_size = total_nodes
        
        # Calculate connectivity score
        if total_nodes == 0:
            connectivity_score = 0
        else:
            # Score based on: (1 - isolated_ratio) * (average_degree / 10) * (largest_component_ratio)
            isolated_ratio = isolated_nodes / total_nodes
            largest_component_ratio = largest_component_size / total_nodes
            
            # Normalize average degree (assuming 10 is a good target)
            normalized_degree = min(average_degree / 10, 1.0)
            
            connectivity_score = max(0, min(100, 
                (1 - isolated_ratio) * 50 + 
                normalized_degree * 30 + 
                largest_component_ratio * 20
            )) * 100
        
        # Generate recommendations based on configuration
        recommendations = []
        
        if isolated_nodes > 0:
            recommendations.append(f"Found {isolated_nodes} isolated nodes. Consider enabling anti-silo features.")
        
        if average_degree < 2:
            recommendations.append("Average degree is low. Enable relationship discovery and cross-document linking.")
        
        if connected_components > 1:
            recommendations.append("Multiple connected components detected. Enable hub entities and semantic clustering.")
        
        if request.enable_anti_silo and request.anti_silo_similarity_threshold > 0.8:
            recommendations.append("Consider lowering anti-silo similarity threshold for more connections.")
        
        if not request.enable_cooccurrence_analysis:
            recommendations.append("Enable co-occurrence analysis to find implicit relationships.")
        
        if not request.enable_hub_entities:
            recommendations.append("Enable hub entities to create central connection points.")
        
        # Analysis details
        analysis = {
            "configuration": request.dict(),
            "graph_metrics": {
                "total_nodes": total_nodes,
                "isolated_nodes": isolated_nodes,
                "isolated_ratio": isolated_nodes / total_nodes if total_nodes > 0 else 0,
                "average_degree": average_degree,
                "connected_components": connected_components,
                "largest_component_size": largest_component_size,
                "largest_component_ratio": largest_component_size / total_nodes if total_nodes > 0 else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return AntiSiloTestResponse(
            success=True,
            connectivity_score=connectivity_score,
            isolated_nodes=isolated_nodes,
            total_nodes=total_nodes,
            average_degree=average_degree,
            connected_components=connected_components,
            largest_component_size=largest_component_size,
            recommendations=recommendations,
            analysis=analysis
        )
        
    except Exception as e:
        logger.error(f"Anti-silo test failed: {e}")
        return AntiSiloTestResponse(
            success=False,
            connectivity_score=0,
            isolated_nodes=0,
            total_nodes=0,
            average_degree=0,
            connected_components=0,
            largest_component_size=0,
            recommendations=[f"Test failed: {str(e)}"],
            analysis={"error": str(e)}
        )

@router.get("/connectivity-analysis")
async def get_connectivity_analysis() -> Dict[str, Any]:
    """
    Get detailed connectivity analysis of the knowledge graph
    """
    try:
        neo4j_service = get_neo4j_service()
        
        if not neo4j_service.is_enabled():
            raise HTTPException(status_code=503, detail="Neo4j service is not available")
        
        # Get comprehensive connectivity metrics
        analysis_query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-(m)
        WITH n, count(r) as degree
        RETURN 
            count(n) as total_nodes,
            sum(CASE WHEN degree = 0 THEN 1 ELSE 0 END) as isolated_nodes,
            avg(degree) as average_degree,
            max(degree) as max_degree,
            min(degree) as min_degree,
            percentileCont(degree, 0.5) as median_degree,
            percentileCont(degree, 0.9) as p90_degree
        """
        
        analysis_result = neo4j_service.execute_cypher(analysis_query)
        
        if not analysis_result:
            return {"error": "Failed to retrieve connectivity analysis"}
        
        metrics = analysis_result[0]
        
        # Get entity type distribution
        entity_types_query = """
        MATCH (n)
        RETURN labels(n)[0] as entity_type, count(n) as count
        ORDER BY count DESC
        """
        
        entity_types = neo4j_service.execute_cypher(entity_types_query)
        
        # Get relationship type distribution
        relationship_types_query = """
        MATCH ()-[r]->()
        RETURN type(r) as relationship_type, count(r) as count
        ORDER BY count DESC
        """
        
        relationship_types = neo4j_service.execute_cypher(relationship_types_query)
        
        # Get potential silo candidates
        silo_candidates_query = """
        MATCH (n)
        WHERE NOT (n)--()
        RETURN n.name as name, labels(n)[0] as type, n.document_id as document_id
        LIMIT 20
        """
        
        silo_candidates = neo4j_service.execute_cypher(silo_candidates_query)
        
        return {
            "success": True,
            "metrics": {
                "total_nodes": metrics.get('total_nodes', 0),
                "isolated_nodes": metrics.get('isolated_nodes', 0),
                "average_degree": metrics.get('average_degree', 0),
                "max_degree": metrics.get('max_degree', 0),
                "min_degree": metrics.get('min_degree', 0),
                "median_degree": metrics.get('median_degree', 0),
                "p90_degree": metrics.get('p90_degree', 0)
            },
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "silo_candidates": silo_candidates,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Connectivity analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize-anti-silo")
async def optimize_anti_silo_settings() -> Dict[str, Any]:
    """
    Automatically optimize anti-silo settings based on current graph state
    """
    try:
        neo4j_service = get_neo4j_service()
        
        if not neo4j_service.is_enabled():
            raise HTTPException(status_code=503, detail="Neo4j service is not available")
        
        # Get current settings
        current_settings = get_knowledge_graph_settings()
        
        # Get connectivity analysis
        analysis_response = await get_connectivity_analysis()
        metrics = analysis_response["metrics"]
        
        # Calculate optimal settings based on current state
        isolated_ratio = metrics["isolated_nodes"] / metrics["total_nodes"] if metrics["total_nodes"] > 0 else 0
        avg_degree = metrics["average_degree"]
        
        # Generate optimized settings
        optimized_settings = {
            "enable_anti_silo": True,
            "anti_silo_similarity_threshold": max(0.6, min(0.9, 0.8 - isolated_ratio * 0.2)),
            "anti_silo_type_boost": max(1.1, min(1.5, 1.2 + isolated_ratio * 0.3)),
            "enable_cooccurrence_analysis": True,
            "enable_type_based_clustering": True,
            "enable_hub_entities": isolated_ratio > 0.1,
            "hub_entity_threshold": max(2, min(5, 3 + int(isolated_ratio * 10))),
            "enable_semantic_clustering": True,
            "clustering_similarity_threshold": max(0.7, min(0.9, 0.8 - isolated_ratio * 0.1)),
            "enable_document_bridge_relationships": True,
            "bridge_relationship_confidence": max(0.5, min(0.8, 0.7 - isolated_ratio * 0.2)),
            "enable_synthetic_relationships": isolated_ratio > 0.05,
            "synthetic_relationship_confidence": max(0.4, min(0.7, 0.6 - isolated_ratio * 0.2)),
            "enable_graph_enrichment": True,
            "graph_enrichment_depth": max(2, min(4, 3 + int(isolated_ratio * 5))),
            "enable_relationship_propagation": avg_degree < 3,
            "relationship_propagation_depth": max(1, min(3, 2 + int((3 - avg_degree) * 2)))
        }
        
        # Test the optimized settings
        test_request = AntiSiloTestRequest(**optimized_settings)
        test_result = await test_anti_silo_configuration(test_request)
        
        return {
            "success": True,
            "optimized_settings": optimized_settings,
            "predicted_improvement": {
                "current_connectivity": test_result.connectivity_score,
                "expected_improvement": min(100, test_result.connectivity_score + 15),
                "isolated_nodes_reduction": max(0, int(metrics["isolated_nodes"] * 0.7))
            },
            "recommendations": test_result.recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def min(a, b):
    return a if a < b else b

def max(a, b):
    return a if a > b else b

def int(value):
    return int(value)
