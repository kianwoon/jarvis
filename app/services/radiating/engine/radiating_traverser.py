"""
Radiating Traverser

Core traversal engine for the radiating coverage system.
Implements various traversal strategies and manages the discovery process.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
import asyncio
from collections import deque

from app.services.radiating.models.radiating_entity import RadiatingEntity
from app.services.radiating.models.radiating_relationship import RadiatingRelationship
from app.services.radiating.models.radiating_context import RadiatingContext, TraversalStrategy
from app.services.radiating.models.radiating_graph import RadiatingGraph
from app.services.radiating.storage.radiating_neo4j_service import get_radiating_neo4j_service
from app.services.radiating.storage.radiating_cache_manager import get_radiating_cache_manager
from .relevance_scorer import RelevanceScorer
from .path_optimizer import PathOptimizer
from app.services.radiating.extraction.llm_relationship_discoverer import LLMRelationshipDiscoverer

logger = logging.getLogger(__name__)


class RadiatingTraverser:
    """
    Main traversal engine for radiating discovery system.
    Coordinates the exploration of knowledge graphs using radiating patterns.
    """
    
    def __init__(self):
        """Initialize RadiatingTraverser"""
        self.neo4j_service = get_radiating_neo4j_service()
        self.cache_manager = get_radiating_cache_manager()
        self.relevance_scorer = RelevanceScorer()
        self.path_optimizer = PathOptimizer()
        self.llm_discoverer = LLMRelationshipDiscoverer()  # Add LLM discoverer
        
        # Performance metrics
        self.metrics = {
            'entities_processed': 0,
            'relationships_discovered': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'neo4j_queries': 0,
            'llm_discoveries': 0,  # Track LLM discoveries
            'traversal_time_ms': 0
        }
    
    async def traverse(self, context: RadiatingContext, 
                      starting_entities: List[RadiatingEntity]) -> RadiatingGraph:
        """
        Main traversal method that executes radiating discovery
        
        Args:
            context: RadiatingContext with configuration
            starting_entities: Initial entities to start traversal from
            
        Returns:
            RadiatingGraph containing discovered entities and relationships
        """
        start_time = datetime.now()
        graph = RadiatingGraph()
        
        try:
            # Initialize starting entities
            for entity in starting_entities:
                graph.add_node(entity)
                context.add_discovered_entity(entity)
                # Add to traversal queue
                context.add_to_traversal_queue(
                    entity.get_entity_id(),
                    0,  # Starting depth
                    entity.relevance_score
                )
            
            # Execute traversal based on strategy
            if context.traversal_strategy == TraversalStrategy.BREADTH_FIRST:
                graph = await self._breadth_first_traversal(context, graph)
            elif context.traversal_strategy == TraversalStrategy.DEPTH_FIRST:
                graph = await self._depth_first_traversal(context, graph)
            elif context.traversal_strategy == TraversalStrategy.BEST_FIRST:
                graph = await self._best_first_traversal(context, graph)
            elif context.traversal_strategy == TraversalStrategy.HYBRID:
                graph = await self._hybrid_traversal(context, graph)
            
            # Finalize context
            context.finalize()
            
            # Calculate metrics
            end_time = datetime.now()
            self.metrics['traversal_time_ms'] = (end_time - start_time).total_seconds() * 1000
            
            # Populate graph metadata with traversal statistics
            graph.metadata.update({
                'max_depth_reached': context.current_depth,
                'coverage_percentage': min(context.total_entities_discovered * 10, 100.0),  # Estimate based on entities found
                'explored_paths': len(context.traversal_history),
                'pruned_paths': context.pruned_count if hasattr(context, 'pruned_count') else 0,
                'traversal_time_ms': self.metrics['traversal_time_ms'],
                'cache_hit_rate': context.cache_hits / max(1, context.cache_hits + context.cache_misses),
                'entity_type_distribution': context._get_entity_type_distribution()
            })
            
            logger.info(f"Traversal completed: {graph.total_nodes} nodes, {graph.total_edges} edges")
            logger.info(f"Metrics: {self.metrics}")
            
            return graph
            
        except Exception as e:
            logger.error(f"Traversal failed: {e}")
            context.finalize()
            return graph
    
    async def _breadth_first_traversal(self, context: RadiatingContext, 
                                      graph: RadiatingGraph) -> RadiatingGraph:
        """
        Breadth-first traversal implementation
        
        Args:
            context: Traversal context
            graph: Current graph
            
        Returns:
            Updated RadiatingGraph
        """
        logger.debug("Starting breadth-first traversal")
        
        while context.should_continue_traversal():
            # Get next entity from queue
            next_item = context.get_next_entity_to_traverse()
            if not next_item:
                break
            
            entity_id, depth, priority = next_item
            
            # Skip if we've exceeded depth for this path
            if depth >= context.depth_limit:
                continue
            
            # Process entity
            await self._process_entity(entity_id, depth, context, graph)
            
            # Update current depth
            context.current_depth = max(context.current_depth, depth)
        
        return graph
    
    async def _depth_first_traversal(self, context: RadiatingContext,
                                    graph: RadiatingGraph) -> RadiatingGraph:
        """
        Depth-first traversal implementation
        
        Args:
            context: Traversal context
            graph: Current graph
            
        Returns:
            Updated RadiatingGraph
        """
        logger.debug("Starting depth-first traversal")
        
        while context.should_continue_traversal():
            # Get next entity from stack (LIFO)
            next_item = context.get_next_entity_to_traverse()
            if not next_item:
                break
            
            entity_id, depth, priority = next_item
            
            # Skip if we've exceeded depth
            if depth >= context.depth_limit:
                continue
            
            # Process entity
            await self._process_entity(entity_id, depth, context, graph)
            
            # Update current depth
            context.current_depth = max(context.current_depth, depth)
        
        return graph
    
    async def _best_first_traversal(self, context: RadiatingContext,
                                   graph: RadiatingGraph) -> RadiatingGraph:
        """
        Best-first (priority-based) traversal implementation
        
        Args:
            context: Traversal context
            graph: Current graph
            
        Returns:
            Updated RadiatingGraph
        """
        logger.debug("Starting best-first traversal")
        
        while context.should_continue_traversal():
            # Get highest priority entity
            next_item = context.get_next_entity_to_traverse()
            if not next_item:
                break
            
            entity_id, depth, priority = next_item
            
            # Skip if we've exceeded depth
            if depth >= context.depth_limit:
                continue
            
            # Skip low priority entities if we're near limits
            if context.total_entities_discovered > context.max_total_entities * 0.8:
                if priority < context.relevance_threshold * 2:
                    continue
            
            # Process entity
            await self._process_entity(entity_id, depth, context, graph)
            
            # Update current depth
            context.current_depth = max(context.current_depth, depth)
        
        return graph
    
    async def _hybrid_traversal(self, context: RadiatingContext,
                               graph: RadiatingGraph) -> RadiatingGraph:
        """
        Hybrid traversal combining multiple strategies
        
        Args:
            context: Traversal context
            graph: Current graph
            
        Returns:
            Updated RadiatingGraph
        """
        logger.debug("Starting hybrid traversal")
        
        # Use best-first for initial exploration, then switch to breadth-first
        switch_depth = context.depth_limit // 2
        
        while context.should_continue_traversal():
            next_item = context.get_next_entity_to_traverse()
            if not next_item:
                break
            
            entity_id, depth, priority = next_item
            
            # Skip if we've exceeded depth
            if depth >= context.depth_limit:
                continue
            
            # Apply strategy-specific filters
            if depth < switch_depth:
                # Best-first phase: prioritize high-relevance entities
                if priority < context.relevance_threshold:
                    continue
            else:
                # Breadth-first phase: explore more broadly
                pass
            
            # Process entity
            await self._process_entity(entity_id, depth, context, graph)
            
            # Update current depth
            context.current_depth = max(context.current_depth, depth)
        
        return graph
    
    async def _process_entity(self, entity_id: str, depth: int,
                             context: RadiatingContext,
                             graph: RadiatingGraph):
        """
        Process a single entity during traversal
        
        Args:
            entity_id: Entity to process
            depth: Current depth in traversal
            context: Traversal context
            graph: Current graph
        """
        try:
            # Check if already processed
            if entity_id in context.visited_entity_ids:
                return
            
            context.visited_entity_ids.add(entity_id)
            self.metrics['entities_processed'] += 1
            
            # Get entity if not in graph
            if entity_id not in graph.nodes:
                # Try cache first
                cached_entity = await self._get_cached_entity(entity_id)
                if cached_entity:
                    graph.add_node(cached_entity)
                    context.update_cache_stats(hit=True)
                else:
                    # Fetch from Neo4j
                    entity = await self.neo4j_service._get_entity_as_radiating(entity_id, depth)
                    if entity:
                        graph.add_node(entity)
                        context.update_cache_stats(hit=False)
                        self.metrics['neo4j_queries'] += 1
                    else:
                        return
            
            # Get neighbors
            neighbors = await self._get_entity_neighbors(entity_id, depth + 1, context, graph)
            
            # Get relationships
            relationships = await self._get_entity_relationships(entity_id, context, graph)
            
            # Process neighbors and relationships
            for neighbor in neighbors:
                # Apply filters
                if not context.apply_entity_filter(neighbor):
                    continue
                
                # Calculate relevance score
                neighbor.relevance_score = await self.relevance_scorer.calculate_entity_relevance(
                    neighbor, context, graph
                )
                
                # Add to graph
                neighbor_id = neighbor.get_entity_id()
                if neighbor_id not in graph.nodes:
                    graph.add_node(neighbor)
                    context.add_discovered_entity(neighbor)
                
                # Add to traversal queue if relevant
                if neighbor.should_traverse(context.relevance_threshold):
                    priority = neighbor.get_traversal_weight()
                    context.add_to_traversal_queue(neighbor_id, depth + 1, priority)
            
            # Process relationships
            for relationship in relationships:
                # Apply filters
                if not context.apply_relationship_filter(relationship):
                    continue
                
                # Ensure both entities exist in graph
                if (relationship.source_entity in graph.nodes and 
                    relationship.target_entity in graph.nodes):
                    
                    # Calculate relationship score
                    relationship.strength_score = await self.relevance_scorer.calculate_relationship_relevance(
                        relationship, context, graph
                    )
                    
                    # Add to graph
                    try:
                        graph.add_edge(relationship)
                        context.add_discovered_relationship(relationship)
                        self.metrics['relationships_discovered'] += 1
                    except ValueError:
                        # One of the entities doesn't exist yet
                        pass
            
            # Record traversal path if we've discovered new entities
            if neighbors:
                path_entities = [entity_id] + [n.get_entity_id() for n in neighbors[:3]]
                path_relationships = [r.get_relationship_id() for r in relationships[:3]]
                path_score = sum(n.relevance_score for n in neighbors[:3]) / max(1, len(neighbors[:3]))
                
                context.add_traversal_path(path_entities, path_relationships, path_score)
            
        except Exception as e:
            logger.error(f"Error processing entity {entity_id}: {e}")
    
    async def _get_entity_neighbors(self, entity_id: str, depth: int,
                                   context: RadiatingContext,
                                   graph: Optional[RadiatingGraph] = None) -> List[RadiatingEntity]:
        """
        Get neighbors for an entity (with caching and LLM fallback)
        
        Args:
            entity_id: Entity ID
            depth: Current depth
            context: Traversal context
            
        Returns:
            List of neighboring RadiatingEntity objects
        """
        # Try cache first
        cached_neighbors = await self.cache_manager.get_entity_neighbors(entity_id, depth)
        if cached_neighbors:
            self.metrics['cache_hits'] += 1
            return cached_neighbors
        
        self.metrics['cache_misses'] += 1
        
        # Fetch from Neo4j
        neighbors = await self.neo4j_service.get_radiating_neighbors(
            entity_id,
            depth,
            context.relevance_threshold,
            context.max_entities_per_level,
            context.relationship_type_preferences
        )
        
        self.metrics['neo4j_queries'] += 1
        
        # If Neo4j returns empty, try LLM discovery
        if not neighbors and context.enable_llm_discovery:
            logger.info(f"Neo4j returned no neighbors for {entity_id}, attempting LLM discovery")
            
            # Get the current entity from graph first, then fallback to Neo4j
            current_entity = await self._get_entity_by_id(entity_id, graph)
            if current_entity:
                # Get potential entities from context or a broader search
                potential_entities = await self._get_potential_entities_for_discovery(context)
                
                if potential_entities:
                    # Use LLM to discover neighbors
                    neighbor_pairs = await self.llm_discoverer.discover_entity_neighbors(
                        current_entity,
                        potential_entities,
                        max_neighbors=context.max_entities_per_level
                    )
                    
                    if neighbor_pairs:
                        neighbors = [neighbor for neighbor, _ in neighbor_pairs]
                        self.metrics['llm_discoveries'] += len(neighbors)
                        logger.info(f"LLM discovered {len(neighbors)} neighbors for {entity_id}")
        
        # Cache the result
        if neighbors:
            await self.cache_manager.set_entity_neighbors(entity_id, depth, neighbors)
        
        return neighbors
    
    async def _get_entity_relationships(self, entity_id: str,
                                       context: RadiatingContext,
                                       graph: Optional[RadiatingGraph] = None) -> List[RadiatingRelationship]:
        """
        Get relationships for an entity (with caching and LLM fallback)
        
        Args:
            entity_id: Entity ID
            context: Traversal context
            
        Returns:
            List of RadiatingRelationship objects
        """
        # Try cache first
        cached_relationships = await self.cache_manager.get_entity_relationships(
            entity_id, "both"
        )
        if cached_relationships:
            self.metrics['cache_hits'] += 1
            return cached_relationships
        
        self.metrics['cache_misses'] += 1
        
        # Fetch from Neo4j
        relationships = await self.neo4j_service.get_entity_relationships(
            entity_id,
            "both",
            context.relationship_type_preferences,
            context.min_relationship_confidence
        )
        
        self.metrics['neo4j_queries'] += 1
        
        # If Neo4j returns empty, try LLM discovery
        if not relationships and context.enable_llm_discovery:
            logger.info(f"Neo4j returned no relationships for {entity_id}, attempting LLM discovery")
            
            # Get the current entity from graph first, then fallback to Neo4j
            current_entity = await self._get_entity_by_id(entity_id, graph)
            if current_entity:
                # Get entities that might be related
                potential_entities = await self._get_potential_entities_for_discovery(context)
                
                if potential_entities:
                    # Include current entity in the discovery
                    all_entities = [current_entity] + potential_entities
                    
                    # Use LLM to discover relationships
                    discovered_relationships = await self.llm_discoverer.discover_relationships(
                        all_entities,
                        max_relationships_per_pair=3,
                        confidence_threshold=context.min_relationship_confidence
                    )
                    
                    # Filter to only relationships involving the current entity
                    entity_canonical = current_entity.canonical_form.lower()
                    relationships = [
                        rel for rel in discovered_relationships
                        if rel.source_entity.lower() == entity_canonical or 
                           rel.target_entity.lower() == entity_canonical
                    ]
                    
                    if relationships:
                        self.metrics['llm_discoveries'] += len(relationships)
                        logger.info(f"LLM discovered {len(relationships)} relationships for {entity_id}")
        
        # Cache the result
        if relationships:
            await self.cache_manager.set_entity_relationships(
                entity_id, "both", relationships
            )
        
        return relationships
    
    async def _get_cached_entity(self, entity_id: str) -> Optional[RadiatingEntity]:
        """
        Try to get entity from cache
        
        Args:
            entity_id: Entity ID
            
        Returns:
            RadiatingEntity or None
        """
        # This would typically check a separate entity cache
        # For now, return None to fetch from Neo4j
        return None
    
    async def _get_entity_by_id(self, entity_id: str, graph: Optional[RadiatingGraph] = None) -> Optional[RadiatingEntity]:
        """
        Get entity by ID from graph, cache, or Neo4j
        
        Args:
            entity_id: Entity ID to fetch
            graph: Optional graph to check first
            
        Returns:
            RadiatingEntity or None if not found
        """
        # Check current graph first if provided
        if graph and entity_id in graph.nodes:
            # graph.nodes stores GraphNode objects, need to get the entity
            graph_node = graph.nodes[entity_id]
            return graph_node.entity
        
        # Try cache
        cached_entity = await self._get_cached_entity(entity_id)
        if cached_entity:
            return cached_entity
        
        # Fetch from Neo4j
        try:
            entity_data = await self.neo4j_service.get_entity_by_id(entity_id)
            if entity_data:
                # Convert to RadiatingEntity
                entity = RadiatingEntity(
                    canonical_form=entity_data.get('name', entity_data.get('canonical_form', '')),
                    label=entity_data.get('label', 'ENTITY'),
                    confidence=entity_data.get('confidence', 1.0),
                    properties=entity_data.get('properties', {}),
                    relevance_score=entity_data.get('relevance_score', 0.5)
                )
                return entity
        except Exception as e:
            logger.error(f"Failed to get entity {entity_id}: {e}")
        
        return None
    
    async def _get_potential_entities_for_discovery(self, context: RadiatingContext) -> List[RadiatingEntity]:
        """
        Get a pool of potential entities for LLM relationship discovery
        
        Args:
            context: Traversal context with configuration
            
        Returns:
            List of RadiatingEntity objects for discovery
        """
        potential_entities = []
        
        # First, use entities already discovered in context
        if context.expanded_entities:
            # Take up to llm_max_entities_for_discovery entities
            potential_entities.extend(
                context.expanded_entities[:context.llm_max_entities_for_discovery]
            )
        
        # If we have entities pre-loaded for LLM discovery
        if context.llm_discovery_entities:
            # Add unique entities from the pre-loaded pool
            existing_ids = {e.get_entity_id() for e in potential_entities}
            for entity in context.llm_discovery_entities:
                if entity.get_entity_id() not in existing_ids:
                    potential_entities.append(entity)
                    if len(potential_entities) >= context.llm_max_entities_for_discovery:
                        break
        
        # If still not enough entities, try to fetch popular entities from Neo4j
        if len(potential_entities) < 10:  # Minimum threshold
            try:
                # Query Neo4j for highly connected entities in the same domain
                popular_entities = await self.neo4j_service.get_popular_entities(
                    domain=context.query_domain.value,
                    limit=context.llm_max_entities_for_discovery - len(potential_entities)
                )
                
                for entity_data in popular_entities:
                    entity = RadiatingEntity(
                        canonical_form=entity_data.get('name', ''),
                        label=entity_data.get('label', 'ENTITY'),
                        confidence=1.0,
                        properties=entity_data.get('properties', {}),
                        relevance_score=0.5
                    )
                    potential_entities.append(entity)
                    
            except Exception as e:
                logger.debug(f"Could not fetch popular entities: {e}")
        
        # Sort by relevance score
        potential_entities.sort(key=lambda e: e.relevance_score, reverse=True)
        
        return potential_entities[:context.llm_max_entities_for_discovery]
    
    async def explore_paths(self, context: RadiatingContext,
                          start_entities: List[RadiatingEntity],
                          target_entities: List[RadiatingEntity]) -> List[List[str]]:
        """
        Explore paths between start and target entities
        
        Args:
            context: Traversal context
            start_entities: Starting entities
            target_entities: Target entities
            
        Returns:
            List of paths (each path is a list of entity IDs)
        """
        all_paths = []
        
        for start in start_entities:
            for target in target_entities:
                # Check cache first
                cached_path = await self.cache_manager.get_path(
                    start.get_entity_id(),
                    target.get_entity_id()
                )
                
                if cached_path:
                    all_paths.append(cached_path)
                    continue
                
                # Find paths using Neo4j
                paths_data = await self.neo4j_service.find_paths_between_entities(
                    start.get_entity_id(),
                    target.get_entity_id(),
                    context.depth_limit,
                    max_paths=5
                )
                
                for path_data in paths_data:
                    path = [node['id'] for node in path_data['nodes']]
                    all_paths.append(path)
                    
                    # Cache the path
                    if path:
                        await self.cache_manager.set_path(
                            start.get_entity_id(),
                            target.get_entity_id(),
                            path
                        )
        
        # Optimize paths
        optimized_paths = self.path_optimizer.optimize_paths(all_paths, context)
        
        return optimized_paths
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get traversal metrics"""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset traversal metrics"""
        self.metrics = {
            'entities_processed': 0,
            'relationships_discovered': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'neo4j_queries': 0,
            'traversal_time_ms': 0
        }