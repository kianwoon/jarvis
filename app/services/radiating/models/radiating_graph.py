"""
Radiating Graph Model

Graph structure for managing entities and relationships during radiating traversal.
Provides efficient graph operations for pathfinding and neighbor discovery.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict, deque
import heapq
from datetime import datetime

from .radiating_entity import RadiatingEntity
from .radiating_relationship import RadiatingRelationship


@dataclass
class GraphNode:
    """Node in the radiating graph"""
    entity: RadiatingEntity
    outgoing_edges: List[str] = field(default_factory=list)  # Relationship IDs
    incoming_edges: List[str] = field(default_factory=list)  # Relationship IDs
    neighbor_cache: Optional[Dict[str, List[str]]] = None
    last_updated: Optional[datetime] = None


@dataclass
class RadiatingGraph:
    """
    Graph structure for radiating traversal system.
    Manages entities (nodes) and relationships (edges) with efficient operations.
    """
    
    # Core graph data structures
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: Dict[str, RadiatingRelationship] = field(default_factory=dict)
    
    # Adjacency structures for fast lookups
    adjacency_list: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    reverse_adjacency: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    # Entity type indexing
    entities_by_type: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    entities_by_domain: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    # Relationship type indexing
    relationships_by_type: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    # Graph statistics
    total_nodes: int = 0
    total_edges: int = 0
    max_degree_node: Optional[str] = None
    connected_components: List[Set[str]] = field(default_factory=list)
    
    # Performance optimization
    path_cache: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)
    neighbor_cache_ttl: int = 300  # Cache TTL in seconds
    
    # Metadata for tracking graph information
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, entity: RadiatingEntity) -> str:
        """
        Add a node (entity) to the graph
        
        Args:
            entity: RadiatingEntity to add
            
        Returns:
            Entity ID of added node
        """
        entity_id = entity.get_entity_id()
        
        if entity_id not in self.nodes:
            node = GraphNode(
                entity=entity,
                last_updated=datetime.now()
            )
            self.nodes[entity_id] = node
            self.total_nodes += 1
            
            # Update type index
            self.entities_by_type[entity.label].add(entity_id)
            
            # Update domain index
            for domain in entity.domain_types:
                self.entities_by_domain[domain].add(entity_id)
        else:
            # Update existing node
            self.nodes[entity_id].entity = entity
            self.nodes[entity_id].last_updated = datetime.now()
        
        return entity_id
    
    def add_edge(self, relationship: RadiatingRelationship) -> str:
        """
        Add an edge (relationship) to the graph
        
        Args:
            relationship: RadiatingRelationship to add
            
        Returns:
            Relationship ID of added edge
        """
        rel_id = relationship.get_relationship_id()
        source_id = relationship.source_entity
        target_id = relationship.target_entity
        
        # Ensure nodes exist
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError(f"Source or target entity not in graph: {source_id} -> {target_id}")
        
        if rel_id not in self.edges:
            self.edges[rel_id] = relationship
            self.total_edges += 1
            
            # Update adjacency lists
            self.adjacency_list[source_id].add(target_id)
            self.reverse_adjacency[target_id].add(source_id)
            
            # Update node edge lists
            self.nodes[source_id].outgoing_edges.append(rel_id)
            self.nodes[target_id].incoming_edges.append(rel_id)
            
            # Update relationship type index
            self.relationships_by_type[relationship.relationship_type].add(rel_id)
            
            # Handle bidirectional relationships
            if relationship.bidirectional:
                self.adjacency_list[target_id].add(source_id)
                self.reverse_adjacency[source_id].add(target_id)
            
            # Invalidate path cache for affected nodes
            self._invalidate_path_cache(source_id, target_id)
            
            # Update max degree node
            self._update_max_degree_node()
        
        return rel_id
    
    def get_neighbors(self, entity_id: str, 
                     direction: str = "both",
                     relationship_types: Optional[List[str]] = None) -> List[str]:
        """
        Get neighbor entities of a given entity
        
        Args:
            entity_id: Entity ID to get neighbors for
            direction: Direction of relationships ('outgoing', 'incoming', 'both')
            relationship_types: Optional filter for relationship types
            
        Returns:
            List of neighbor entity IDs
        """
        if entity_id not in self.nodes:
            return []
        
        neighbors = set()
        
        # Get outgoing neighbors
        if direction in ["outgoing", "both"]:
            for neighbor_id in self.adjacency_list.get(entity_id, set()):
                if relationship_types:
                    # Check if connection has required relationship type
                    if self._has_relationship_type(entity_id, neighbor_id, relationship_types):
                        neighbors.add(neighbor_id)
                else:
                    neighbors.add(neighbor_id)
        
        # Get incoming neighbors
        if direction in ["incoming", "both"]:
            for neighbor_id in self.reverse_adjacency.get(entity_id, set()):
                if relationship_types:
                    # Check if connection has required relationship type
                    if self._has_relationship_type(neighbor_id, entity_id, relationship_types):
                        neighbors.add(neighbor_id)
                else:
                    neighbors.add(neighbor_id)
        
        return list(neighbors)
    
    def _has_relationship_type(self, source_id: str, target_id: str, 
                              types: List[str]) -> bool:
        """Check if relationship between entities has any of the specified types"""
        node = self.nodes.get(source_id)
        if not node:
            return False
        
        for rel_id in node.outgoing_edges:
            rel = self.edges.get(rel_id)
            if rel and rel.target_entity == target_id:
                if rel.relationship_type in types:
                    return True
        
        return False
    
    def find_path(self, start_id: str, end_id: str, 
                 max_depth: int = 5) -> Optional[List[str]]:
        """
        Find shortest path between two entities using BFS
        
        Args:
            start_id: Starting entity ID
            end_id: Target entity ID
            max_depth: Maximum path depth
            
        Returns:
            List of entity IDs representing path, or None if no path exists
        """
        # Check cache first
        cache_key = (start_id, end_id)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        if start_id not in self.nodes or end_id not in self.nodes:
            return None
        
        if start_id == end_id:
            return [start_id]
        
        # BFS for shortest path
        queue = deque([(start_id, [start_id])])
        visited = {start_id}
        
        while queue:
            current_id, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            for neighbor_id in self.adjacency_list.get(current_id, set()):
                if neighbor_id == end_id:
                    result_path = path + [neighbor_id]
                    self.path_cache[cache_key] = result_path
                    return result_path
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
        
        self.path_cache[cache_key] = None
        return None
    
    def find_all_paths(self, start_id: str, end_id: str, 
                      max_depth: int = 5, 
                      max_paths: int = 10) -> List[List[str]]:
        """
        Find all paths between two entities up to max_depth
        
        Args:
            start_id: Starting entity ID
            end_id: Target entity ID
            max_depth: Maximum path depth
            max_paths: Maximum number of paths to return
            
        Returns:
            List of paths (each path is a list of entity IDs)
        """
        if start_id not in self.nodes or end_id not in self.nodes:
            return []
        
        if start_id == end_id:
            return [[start_id]]
        
        all_paths = []
        
        def dfs(current_id: str, target_id: str, 
               path: List[str], visited: Set[str]):
            if len(all_paths) >= max_paths:
                return
            
            if len(path) > max_depth:
                return
            
            if current_id == target_id:
                all_paths.append(path.copy())
                return
            
            for neighbor_id in self.adjacency_list.get(current_id, set()):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    path.append(neighbor_id)
                    dfs(neighbor_id, target_id, path, visited)
                    path.pop()
                    visited.remove(neighbor_id)
        
        dfs(start_id, end_id, [start_id], {start_id})
        
        return all_paths
    
    def find_weighted_shortest_path(self, start_id: str, end_id: str,
                                   weight_func=None) -> Optional[Tuple[List[str], float]]:
        """
        Find shortest weighted path using Dijkstra's algorithm
        
        Args:
            start_id: Starting entity ID
            end_id: Target entity ID
            weight_func: Optional function to calculate edge weights
            
        Returns:
            Tuple of (path, total_weight) or None if no path exists
        """
        if start_id not in self.nodes or end_id not in self.nodes:
            return None
        
        if start_id == end_id:
            return ([start_id], 0.0)
        
        # Default weight function (uses relationship traversal weight)
        if weight_func is None:
            def weight_func(rel_id):
                rel = self.edges.get(rel_id)
                return 1.0 / rel.calculate_traversal_score() if rel else 1.0
        
        # Dijkstra's algorithm
        distances = {node_id: float('inf') for node_id in self.nodes}
        distances[start_id] = 0
        previous = {}
        priority_queue = [(0, start_id)]
        visited = set()
        
        while priority_queue:
            current_dist, current_id = heapq.heappop(priority_queue)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            if current_id == end_id:
                # Reconstruct path
                path = []
                while current_id in previous:
                    path.insert(0, current_id)
                    current_id = previous[current_id]
                path.insert(0, start_id)
                return (path, current_dist)
            
            # Check neighbors
            node = self.nodes[current_id]
            for rel_id in node.outgoing_edges:
                rel = self.edges.get(rel_id)
                if rel:
                    neighbor_id = rel.target_entity
                    weight = weight_func(rel_id)
                    distance = current_dist + weight
                    
                    if distance < distances[neighbor_id]:
                        distances[neighbor_id] = distance
                        previous[neighbor_id] = current_id
                        heapq.heappush(priority_queue, (distance, neighbor_id))
        
        return None
    
    def get_subgraph(self, entity_ids: Set[str], 
                    include_edges: bool = True) -> 'RadiatingGraph':
        """
        Extract subgraph containing only specified entities
        
        Args:
            entity_ids: Set of entity IDs to include
            include_edges: Whether to include edges between entities
            
        Returns:
            New RadiatingGraph containing subgraph
        """
        subgraph = RadiatingGraph()
        
        # Add nodes
        for entity_id in entity_ids:
            if entity_id in self.nodes:
                subgraph.add_node(self.nodes[entity_id].entity)
        
        # Add edges if requested
        if include_edges:
            for rel_id, rel in self.edges.items():
                if (rel.source_entity in entity_ids and 
                    rel.target_entity in entity_ids):
                    subgraph.add_edge(rel)
        
        return subgraph
    
    def get_connected_component(self, entity_id: str) -> Set[str]:
        """
        Get all entities in the connected component containing the given entity
        
        Args:
            entity_id: Entity ID to start from
            
        Returns:
            Set of entity IDs in connected component
        """
        if entity_id not in self.nodes:
            return set()
        
        component = set()
        queue = deque([entity_id])
        
        while queue:
            current_id = queue.popleft()
            if current_id in component:
                continue
            
            component.add(current_id)
            
            # Add all neighbors (both directions)
            neighbors = self.get_neighbors(current_id, direction="both")
            for neighbor_id in neighbors:
                if neighbor_id not in component:
                    queue.append(neighbor_id)
        
        return component
    
    def calculate_centrality(self, entity_id: str, 
                           metric: str = "degree") -> float:
        """
        Calculate centrality metric for an entity
        
        Args:
            entity_id: Entity ID to calculate centrality for
            metric: Centrality metric ('degree', 'closeness', 'betweenness')
            
        Returns:
            Centrality score
        """
        if entity_id not in self.nodes:
            return 0.0
        
        if metric == "degree":
            # Degree centrality
            in_degree = len(self.reverse_adjacency.get(entity_id, set()))
            out_degree = len(self.adjacency_list.get(entity_id, set()))
            return (in_degree + out_degree) / max(1, self.total_nodes - 1)
        
        elif metric == "closeness":
            # Closeness centrality (simplified)
            total_distance = 0
            reachable_nodes = 0
            
            for other_id in self.nodes:
                if other_id != entity_id:
                    path = self.find_path(entity_id, other_id, max_depth=10)
                    if path:
                        total_distance += len(path) - 1
                        reachable_nodes += 1
            
            if reachable_nodes == 0:
                return 0.0
            
            return reachable_nodes / (total_distance * (self.total_nodes - 1))
        
        elif metric == "betweenness":
            # Betweenness centrality (simplified - computationally expensive)
            betweenness = 0
            
            # Sample pairs for efficiency
            import random
            sample_size = min(50, self.total_nodes)
            sampled_nodes = random.sample(list(self.nodes.keys()), 
                                        min(sample_size, len(self.nodes)))
            
            for source in sampled_nodes:
                for target in sampled_nodes:
                    if source != target and source != entity_id and target != entity_id:
                        all_paths = self.find_all_paths(source, target, 
                                                       max_depth=5, max_paths=5)
                        if all_paths:
                            paths_through_entity = sum(
                                1 for path in all_paths if entity_id in path
                            )
                            betweenness += paths_through_entity / len(all_paths)
            
            return betweenness / max(1, (sample_size - 1) * (sample_size - 2))
        
        return 0.0
    
    def _update_max_degree_node(self):
        """Update the node with maximum degree"""
        max_degree = 0
        max_node = None
        
        for node_id in self.nodes:
            degree = (len(self.adjacency_list.get(node_id, set())) + 
                     len(self.reverse_adjacency.get(node_id, set())))
            if degree > max_degree:
                max_degree = degree
                max_node = node_id
        
        self.max_degree_node = max_node
    
    def _invalidate_path_cache(self, *entity_ids):
        """Invalidate cached paths involving specified entities"""
        keys_to_remove = []
        for key in self.path_cache:
            if any(entity_id in key for entity_id in entity_ids):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.path_cache[key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        avg_degree = 0
        if self.total_nodes > 0:
            total_degrees = sum(
                len(self.adjacency_list.get(node_id, set())) + 
                len(self.reverse_adjacency.get(node_id, set()))
                for node_id in self.nodes
            )
            avg_degree = total_degrees / self.total_nodes
        
        return {
            'total_nodes': self.total_nodes,
            'total_edges': self.total_edges,
            'average_degree': avg_degree,
            'max_degree_node': self.max_degree_node,
            'entity_types': len(self.entities_by_type),
            'relationship_types': len(self.relationships_by_type),
            'connected_components': len(self.connected_components),
            'cache_size': len(self.path_cache)
        }
    
    @property
    def entities(self) -> List[RadiatingEntity]:
        """Get all entities in the graph"""
        return [node.entity for node in self.nodes.values()]
    
    @property
    def relationships(self) -> List[RadiatingRelationship]:
        """Get all relationships in the graph"""
        return list(self.edges.values())
    
    def get_entities_by_depth(self, depth: int) -> List[RadiatingEntity]:
        """Get all entities at a specific depth"""
        return [
            node.entity 
            for node in self.nodes.values() 
            if node.entity.traversal_depth == depth
        ]
    
    def filter_by_relevance(self, threshold: float) -> 'RadiatingGraph':
        """
        Filter graph by relevance score threshold
        
        Args:
            threshold: Minimum relevance score to include
            
        Returns:
            New RadiatingGraph with filtered entities
        """
        filtered_graph = RadiatingGraph()
        
        # Add entities that meet threshold
        for node in self.nodes.values():
            if node.entity.relevance_score >= threshold:
                filtered_graph.add_node(node.entity)
        
        # Add relationships between included entities
        for rel in self.edges.values():
            if (rel.source_entity in filtered_graph.nodes and 
                rel.target_entity in filtered_graph.nodes):
                filtered_graph.add_edge(rel)
        
        return filtered_graph
    
    def merge(self, other: 'RadiatingGraph') -> 'RadiatingGraph':
        """
        Merge another graph into a new graph
        
        Args:
            other: Another RadiatingGraph to merge
            
        Returns:
            New RadiatingGraph containing all entities and relationships from both graphs
        """
        merged_graph = RadiatingGraph()
        
        # Add all entities from this graph
        for node in self.nodes.values():
            merged_graph.add_node(node.entity)
        
        # Add all entities from other graph
        for node in other.nodes.values():
            merged_graph.add_node(node.entity)
        
        # Add all relationships from this graph
        for rel in self.edges.values():
            try:
                merged_graph.add_edge(rel)
            except ValueError:
                # Skip if entities not in merged graph
                pass
        
        # Add all relationships from other graph
        for rel in other.edges.values():
            try:
                merged_graph.add_edge(rel)
            except ValueError:
                # Skip if entities not in merged graph
                pass
        
        return merged_graph