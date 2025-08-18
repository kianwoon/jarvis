"""
Radiating Neo4j Service

Extended Neo4j service for radiating traversal operations.
Provides specialized queries and operations for the Universal Radiating Coverage System.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
import json

from app.services.neo4j_service import Neo4jService, get_neo4j_service
from app.services.radiating.models.radiating_entity import RadiatingEntity
from app.services.radiating.models.radiating_relationship import RadiatingRelationship
from app.services.radiating.models.radiating_graph import RadiatingGraph
from app.services.knowledge_graph_types import ExtractedEntity, ExtractedRelationship

logger = logging.getLogger(__name__)


class RadiatingNeo4jService(Neo4jService):
    """
    Extended Neo4j service for radiating traversal operations.
    Inherits from base Neo4jService and adds radiating-specific functionality.
    """
    
    def __init__(self):
        """Initialize RadiatingNeo4jService"""
        super().__init__()
        self._ensure_radiating_indexes()
    
    def _ensure_radiating_indexes(self):
        """Create indexes specific to radiating traversal"""
        if not self.is_enabled():
            return
        
        try:
            with self.driver.session() as session:
                # Index for traversal depth
                session.run("""
                    CREATE INDEX idx_traversal_depth IF NOT EXISTS 
                    FOR (n:RadiatingEntity)
                    ON (n.traversal_depth)
                """)
                
                # Index for relevance score
                session.run("""
                    CREATE INDEX idx_relevance_score IF NOT EXISTS
                    FOR (n:RadiatingEntity)
                    ON (n.relevance_score)
                """)
                
                # Index for discovery source
                session.run("""
                    CREATE INDEX idx_discovery_source IF NOT EXISTS
                    FOR (n:RadiatingEntity)
                    ON (n.discovery_source)
                """)
                
                # Composite index for efficient radiating queries
                session.run("""
                    CREATE INDEX idx_radiating_composite IF NOT EXISTS
                    FOR (n:RadiatingEntity)
                    ON (n.traversal_depth, n.relevance_score)
                """)
                
                logger.info("Radiating-specific indexes created successfully")
                
        except Exception as e:
            logger.error(f"Failed to create radiating indexes: {e}")
    
    async def get_radiating_neighbors(self, entity_id: str, 
                                     depth: int = 1,
                                     min_relevance: float = 0.1,
                                     max_neighbors: int = 50,
                                     relationship_types: Optional[List[str]] = None) -> List[RadiatingEntity]:
        """
        Get neighboring entities for radiating traversal
        
        Args:
            entity_id: Source entity ID
            depth: Current traversal depth
            min_relevance: Minimum relevance score
            max_neighbors: Maximum number of neighbors to return
            relationship_types: Optional filter for relationship types
            
        Returns:
            List of RadiatingEntity objects
        """
        if not self.is_enabled():
            return []
        
        try:
            # Build relationship type filter
            rel_filter = ""
            if relationship_types:
                rel_types_str = "|".join(relationship_types)
                rel_filter = f"[r:{rel_types_str}]"
            else:
                rel_filter = "[r]"
            
            query = f"""
            MATCH (source {{id: $entity_id}})-{rel_filter}-(neighbor)
            WHERE NOT neighbor.id = $entity_id
            WITH neighbor, r, 
                 CASE 
                    WHEN type(r) IN $priority_types THEN 1.5
                    ELSE 1.0
                 END as type_weight
            RETURN DISTINCT
                neighbor.id as id,
                neighbor.name as name,
                labels(neighbor)[0] as label,
                neighbor.confidence as confidence,
                properties(neighbor) as properties,
                type_weight,
                count(r) as connection_count
            ORDER BY type_weight DESC, connection_count DESC, neighbor.confidence DESC
            LIMIT $max_neighbors
            """
            
            priority_types = relationship_types[:3] if relationship_types else []
            
            with self.driver.session() as session:
                result = session.run(query, {
                    'entity_id': entity_id,
                    'priority_types': priority_types,
                    'max_neighbors': max_neighbors
                })
                
                neighbors = []
                for record in result:
                    # Create RadiatingEntity from result
                    entity = RadiatingEntity(
                        text=record.get('name', ''),
                        label=record.get('label', 'UNKNOWN'),
                        start_char=0,
                        end_char=len(record.get('name', '')),
                        confidence=record.get('confidence', 0.5),
                        canonical_form=record.get('name', ''),
                        properties=record.get('properties', {}),
                        traversal_depth=depth,
                        discovery_source=f"radiating_from_{entity_id}",
                        relevance_score=min(1.0, record.get('type_weight', 1.0) * record.get('confidence', 0.5))
                    )
                    
                    # Set entity ID in properties
                    entity.properties['id'] = record.get('id')
                    
                    # Add connection count to domain metadata
                    entity.add_domain_metadata('connection_count', record.get('connection_count', 1))
                    
                    neighbors.append(entity)
                
                return neighbors
                
        except Exception as e:
            logger.error(f"Failed to get radiating neighbors for {entity_id}: {e}")
            return []
    
    async def get_entity_relationships(self, entity_id: str,
                                      direction: str = "both",
                                      relationship_types: Optional[List[str]] = None,
                                      min_confidence: float = 0.0) -> List[RadiatingRelationship]:
        """
        Get relationships for an entity
        
        Args:
            entity_id: Entity ID
            direction: Relationship direction ('outgoing', 'incoming', 'both')
            relationship_types: Optional filter for relationship types
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of RadiatingRelationship objects
        """
        if not self.is_enabled():
            return []
        
        try:
            relationships = []
            
            # Build direction-specific queries
            queries = []
            if direction in ["outgoing", "both"]:
                queries.append(("outgoing", """
                    MATCH (source {id: $entity_id})-[r]->(target)
                    WHERE r.confidence >= $min_confidence
                    """))
            
            if direction in ["incoming", "both"]:
                queries.append(("incoming", """
                    MATCH (source)-[r]->(target {id: $entity_id})
                    WHERE r.confidence >= $min_confidence
                    """))
            
            with self.driver.session() as session:
                for dir_type, match_clause in queries:
                    # Add relationship type filter if specified
                    if relationship_types:
                        type_filter = f"AND type(r) IN {relationship_types}"
                    else:
                        type_filter = ""
                    
                    query = f"""
                    {match_clause}
                    {type_filter}
                    RETURN 
                        source.id as source_id,
                        source.name as source_name,
                        target.id as target_id,
                        target.name as target_name,
                        type(r) as rel_type,
                        r.confidence as confidence,
                        r.context as context,
                        properties(r) as properties,
                        r.confidence as strength,
                        false as bidirectional
                    """
                    
                    result = session.run(query, {
                        'entity_id': entity_id,
                        'min_confidence': min_confidence
                    })
                    
                    for record in result:
                        rel = RadiatingRelationship(
                            source_entity=record.get('source_id', ''),
                            target_entity=record.get('target_id', ''),
                            relationship_type=record.get('rel_type', 'RELATED'),
                            confidence=record.get('confidence', 0.5),
                            context=record.get('context', ''),
                            properties=record.get('properties', {}),
                            bidirectional=record.get('bidirectional', False),
                            strength_score=record.get('strength', 1.0)
                        )
                        
                        # Set traversal direction based on query
                        rel.traversal_direction = dir_type
                        
                        relationships.append(rel)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to get entity relationships for {entity_id}: {e}")
            return []
    
    async def expand_from_entity(self, entity_id: str,
                                max_depth: int = 3,
                                max_entities_per_level: int = 20,
                                min_relevance: float = 0.1) -> RadiatingGraph:
        """
        Expand graph from a single entity using radiating pattern
        
        Args:
            entity_id: Starting entity ID
            max_depth: Maximum traversal depth
            max_entities_per_level: Maximum entities to explore per depth level
            min_relevance: Minimum relevance score
            
        Returns:
            RadiatingGraph containing discovered entities and relationships
        """
        if not self.is_enabled():
            return RadiatingGraph()
        
        graph = RadiatingGraph()
        visited = set()
        current_level = [entity_id]
        
        try:
            for depth in range(max_depth + 1):
                if not current_level:
                    break
                
                next_level = []
                level_entities = 0
                
                for current_id in current_level:
                    if current_id in visited:
                        continue
                    
                    visited.add(current_id)
                    
                    # Get entity details
                    entity = await self._get_entity_as_radiating(current_id, depth)
                    if entity:
                        graph.add_node(entity)
                        
                        # Get relationships
                        relationships = await self.get_entity_relationships(
                            current_id, 
                            direction="both",
                            min_confidence=min_relevance
                        )
                        
                        for rel in relationships:
                            # Add relationship to graph
                            try:
                                graph.add_edge(rel)
                            except ValueError:
                                # Target entity might not be in graph yet
                                pass
                            
                            # Add connected entities to next level
                            if depth < max_depth:
                                other_id = (rel.target_entity if rel.source_entity == current_id 
                                          else rel.source_entity)
                                if other_id not in visited and level_entities < max_entities_per_level:
                                    next_level.append(other_id)
                                    level_entities += 1
                
                current_level = next_level[:max_entities_per_level]
                
                logger.debug(f"Depth {depth}: Discovered {level_entities} entities")
            
            logger.info(f"Radiating expansion complete: {graph.total_nodes} nodes, {graph.total_edges} edges")
            return graph
            
        except Exception as e:
            logger.error(f"Failed to expand from entity {entity_id}: {e}")
            return graph
    
    async def _get_entity_as_radiating(self, entity_id: str, 
                                      depth: int = 0) -> Optional[RadiatingEntity]:
        """
        Get entity from Neo4j as RadiatingEntity
        
        Args:
            entity_id: Entity ID
            depth: Current traversal depth
            
        Returns:
            RadiatingEntity or None if not found
        """
        try:
            query = """
            MATCH (n {id: $entity_id})
            RETURN 
                n.id as id,
                n.name as name,
                labels(n)[0] as label,
                n.confidence as confidence,
                properties(n) as properties
            """
            
            with self.driver.session() as session:
                result = session.run(query, {'entity_id': entity_id})
                record = result.single()
                
                if record:
                    entity = RadiatingEntity(
                        text=record.get('name', ''),
                        label=record.get('label', 'UNKNOWN'),
                        start_char=0,
                        end_char=len(record.get('name', '')),
                        confidence=record.get('confidence', 0.5),
                        canonical_form=record.get('name', ''),
                        properties=record.get('properties', {}),
                        traversal_depth=depth,
                        discovery_source="neo4j"
                    )
                    
                    # Ensure ID is in properties
                    entity.properties['id'] = record.get('id')
                    
                    return entity
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get entity {entity_id}: {e}")
            return None
    
    async def find_paths_between_entities(self, start_id: str, end_id: str,
                                        max_depth: int = 5,
                                        max_paths: int = 10) -> List[Dict[str, Any]]:
        """
        Find paths between two entities in Neo4j
        
        Args:
            start_id: Starting entity ID
            end_id: Target entity ID
            max_depth: Maximum path length
            max_paths: Maximum number of paths to return
            
        Returns:
            List of path dictionaries with entities and relationships
        """
        if not self.is_enabled():
            return []
        
        try:
            query = """
            MATCH path = (start {id: $start_id})-[*1..$max_depth]-(end {id: $end_id})
            WITH path, 
                 reduce(score = 1.0, r in relationships(path) | 
                        score * COALESCE(r.confidence, 0.5)) as path_score
            ORDER BY path_score DESC, length(path) ASC
            LIMIT $max_paths
            RETURN 
                [n in nodes(path) | {id: n.id, name: n.name, type: labels(n)[0]}] as nodes,
                [r in relationships(path) | {
                    type: type(r), 
                    confidence: r.confidence,
                    source: startNode(r).id,
                    target: endNode(r).id
                }] as relationships,
                length(path) as path_length,
                path_score
            """
            
            with self.driver.session() as session:
                result = session.run(query, {
                    'start_id': start_id,
                    'end_id': end_id,
                    'max_depth': max_depth,
                    'max_paths': max_paths
                })
                
                paths = []
                for record in result:
                    paths.append({
                        'nodes': record.get('nodes', []),
                        'relationships': record.get('relationships', []),
                        'length': record.get('path_length', 0),
                        'score': record.get('path_score', 0.0)
                    })
                
                return paths
                
        except Exception as e:
            logger.error(f"Failed to find paths between {start_id} and {end_id}: {e}")
            return []
    
    async def get_entity_centrality(self, entity_id: str,
                                   metric: str = "degree") -> float:
        """
        Calculate entity centrality in the knowledge graph
        
        Args:
            entity_id: Entity ID
            metric: Centrality metric ('degree', 'pagerank', 'betweenness')
            
        Returns:
            Centrality score
        """
        if not self.is_enabled():
            return 0.0
        
        try:
            if metric == "degree":
                query = """
                MATCH (n {id: $entity_id})
                OPTIONAL MATCH (n)-[r]-()
                RETURN count(DISTINCT r) as degree
                """
                
                with self.driver.session() as session:
                    result = session.run(query, {'entity_id': entity_id})
                    record = result.single()
                    
                    if record:
                        # Normalize by approximate graph size
                        degree = record.get('degree', 0)
                        # Simple normalization - could be improved with actual graph size
                        return min(1.0, degree / 100.0)
            
            elif metric == "pagerank":
                # Use Neo4j's PageRank algorithm if available
                query = """
                CALL gds.pageRank.stream({
                    nodeQuery: 'MATCH (n) RETURN id(n) as id',
                    relationshipQuery: 'MATCH (n)-[r]-(m) RETURN id(n) as source, id(m) as target'
                })
                YIELD nodeId, score
                WITH gds.util.asNode(nodeId) as node, score
                WHERE node.id = $entity_id
                RETURN score
                """
                
                try:
                    with self.driver.session() as session:
                        result = session.run(query, {'entity_id': entity_id})
                        record = result.single()
                        
                        if record:
                            return record.get('score', 0.0)
                except:
                    # GDS not available, fall back to degree centrality
                    return await self.get_entity_centrality(entity_id, "degree")
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate centrality for {entity_id}: {e}")
            return 0.0
    
    async def store_radiating_results(self, graph: RadiatingGraph,
                                     context_id: str) -> bool:
        """
        Store radiating traversal results back to Neo4j
        
        Args:
            graph: RadiatingGraph with results
            context_id: Unique identifier for this traversal context
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_enabled():
            return False
        
        try:
            with self.driver.session() as session:
                # Store traversal metadata
                session.run("""
                    MERGE (context:RadiatingContext {id: $context_id})
                    SET context.timestamp = datetime(),
                        context.total_nodes = $total_nodes,
                        context.total_edges = $total_edges,
                        context.statistics = $statistics
                """, {
                    'context_id': context_id,
                    'total_nodes': graph.total_nodes,
                    'total_edges': graph.total_edges,
                    'statistics': json.dumps(graph.get_statistics())
                })
                
                # Update entities with radiating metadata
                for node_id, node in graph.nodes.items():
                    entity = node.entity
                    session.run("""
                        MATCH (n {id: $entity_id})
                        SET n.traversal_depth = $depth,
                            n.relevance_score = $relevance,
                            n.discovery_source = $source,
                            n.last_radiating_context = $context_id
                    """, {
                        'entity_id': entity.get_entity_id(),
                        'depth': entity.traversal_depth,
                        'relevance': entity.relevance_score,
                        'source': entity.discovery_source,
                        'context_id': context_id
                    })
                
                logger.info(f"Stored radiating results for context {context_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store radiating results: {e}")
            return False
    
    async def get_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get entity by ID from Neo4j
        
        Args:
            entity_id: Entity ID to fetch
            
        Returns:
            Entity data dictionary or None if not found
        """
        if not self.is_enabled():
            return None
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (n)
                    WHERE n.entity_id = $entity_id OR id(n) = toInteger($entity_id)
                    RETURN n, labels(n) as labels
                    LIMIT 1
                """, {'entity_id': entity_id})
                
                record = result.single()
                if record:
                    node = record['n']
                    labels = record['labels']
                    
                    return {
                        'entity_id': entity_id,
                        'name': node.get('name', node.get('canonical_form', '')),
                        'canonical_form': node.get('canonical_form', node.get('name', '')),
                        'label': labels[0] if labels else 'ENTITY',
                        'confidence': node.get('confidence', 1.0),
                        'relevance_score': node.get('relevance_score', 0.5),
                        'properties': dict(node)
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get entity by ID {entity_id}: {e}")
            return None
    
    async def get_popular_entities(self, domain: str = 'technology', 
                                  limit: int = 30) -> List[Dict[str, Any]]:
        """
        Get popular (highly connected) entities from Neo4j
        
        Args:
            domain: Domain to filter entities (optional)
            limit: Maximum number of entities to return
            
        Returns:
            List of entity data dictionaries
        """
        if not self.is_enabled():
            return []
        
        try:
            with self.driver.session() as session:
                # Query for entities with high relationship count
                query = """
                    MATCH (n)
                    WHERE n.domain = $domain OR $domain = 'general'
                    WITH n, size((n)--()) as rel_count
                    WHERE rel_count > 0
                    RETURN n, labels(n) as labels, rel_count
                    ORDER BY rel_count DESC
                    LIMIT $limit
                """
                
                result = session.run(query, {
                    'domain': domain,
                    'limit': limit
                })
                
                entities = []
                for record in result:
                    node = record['n']
                    labels = record['labels']
                    rel_count = record['rel_count']
                    
                    entities.append({
                        'entity_id': str(node.id),
                        'name': node.get('name', node.get('canonical_form', '')),
                        'canonical_form': node.get('canonical_form', node.get('name', '')),
                        'label': labels[0] if labels else 'ENTITY',
                        'confidence': 1.0,
                        'relevance_score': min(1.0, rel_count / 100),  # Score based on connections
                        'properties': dict(node),
                        'relationship_count': rel_count
                    })
                
                logger.info(f"Found {len(entities)} popular entities in domain '{domain}'")
                return entities
                
        except Exception as e:
            logger.error(f"Failed to get popular entities: {e}")
            return []


# Singleton instance
_radiating_neo4j_service: Optional[RadiatingNeo4jService] = None


def get_radiating_neo4j_service() -> RadiatingNeo4jService:
    """Get or create RadiatingNeo4jService singleton"""
    global _radiating_neo4j_service
    if _radiating_neo4j_service is None:
        _radiating_neo4j_service = RadiatingNeo4jService()
    return _radiating_neo4j_service