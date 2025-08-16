"""
Path Optimizer

Optimization engine for finding and ranking the most relevant paths
in the radiating traversal system.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import heapq
from collections import defaultdict
import networkx as nx

from app.services.radiating.models.radiating_context import RadiatingContext
from app.services.radiating.models.radiating_graph import RadiatingGraph

logger = logging.getLogger(__name__)


@dataclass
class OptimizedPath:
    """Represents an optimized path with metadata"""
    path: List[str]  # List of entity IDs
    score: float
    length: int
    bottleneck_score: float  # Lowest edge weight in path
    diversity_score: float  # How different from other paths
    metadata: Dict[str, Any]


class PathOptimizer:
    """
    Optimizes paths discovered during radiating traversal.
    Provides algorithms for path ranking, pruning, and selection.
    """
    
    def __init__(self):
        """Initialize PathOptimizer"""
        # Optimization parameters
        self.max_paths_to_consider = 100
        self.min_path_score = 0.1
        self.diversity_weight = 0.2
        self.length_penalty_factor = 0.1
        
        # Path quality metrics weights
        self.metric_weights = {
            'relevance': 0.4,
            'length': 0.2,
            'bottleneck': 0.2,
            'diversity': 0.2
        }
    
    def optimize_paths(self, paths: List[List[str]], 
                      context: RadiatingContext,
                      graph: Optional[RadiatingGraph] = None) -> List[List[str]]:
        """
        Optimize and rank a set of paths
        
        Args:
            paths: List of paths (each path is a list of entity IDs)
            context: Traversal context
            graph: Optional graph for additional scoring
            
        Returns:
            Optimized list of paths, ranked by relevance
        """
        if not paths:
            return []
        
        # Convert to OptimizedPath objects
        optimized_paths = []
        for path in paths[:self.max_paths_to_consider]:
            opt_path = self._create_optimized_path(path, context, graph)
            if opt_path and opt_path.score >= self.min_path_score:
                optimized_paths.append(opt_path)
        
        # Calculate diversity scores
        self._calculate_diversity_scores(optimized_paths)
        
        # Apply multi-criteria optimization
        optimized_paths = self._multi_criteria_optimization(optimized_paths)
        
        # Sort by final score
        optimized_paths.sort(key=lambda p: p.score, reverse=True)
        
        # Return top paths
        return [p.path for p in optimized_paths]
    
    def _create_optimized_path(self, path: List[str],
                              context: RadiatingContext,
                              graph: Optional[RadiatingGraph]) -> Optional[OptimizedPath]:
        """
        Create an OptimizedPath object with scores
        
        Args:
            path: List of entity IDs
            context: Traversal context
            graph: Optional graph for scoring
            
        Returns:
            OptimizedPath object or None
        """
        if not path:
            return None
        
        # Calculate base relevance score
        relevance_score = self._calculate_path_relevance(path, context, graph)
        
        # Calculate bottleneck score (minimum edge weight)
        bottleneck_score = self._calculate_bottleneck_score(path, graph)
        
        # Apply length penalty
        length_penalty = 1.0 / (1 + self.length_penalty_factor * len(path))
        
        # Initial score (without diversity)
        initial_score = (
            relevance_score * self.metric_weights['relevance'] +
            length_penalty * self.metric_weights['length'] +
            bottleneck_score * self.metric_weights['bottleneck']
        )
        
        return OptimizedPath(
            path=path,
            score=initial_score,
            length=len(path),
            bottleneck_score=bottleneck_score,
            diversity_score=0.0,  # Will be calculated later
            metadata={
                'relevance_score': relevance_score,
                'length_penalty': length_penalty
            }
        )
    
    def _calculate_path_relevance(self, path: List[str],
                                 context: RadiatingContext,
                                 graph: Optional[RadiatingGraph]) -> float:
        """
        Calculate relevance score for a path
        
        Args:
            path: List of entity IDs
            context: Traversal context
            graph: Optional graph
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not graph:
            # Simple scoring without graph
            return self._simple_path_scoring(path, context)
        
        # Calculate average entity relevance
        entity_scores = []
        for entity_id in path:
            if entity_id in graph.nodes:
                entity = graph.nodes[entity_id].entity
                entity_scores.append(entity.relevance_score)
        
        if not entity_scores:
            return 0.0
        
        # Use harmonic mean to penalize paths with low-scoring entities
        harmonic_mean = len(entity_scores) / sum(1/s for s in entity_scores if s > 0)
        
        return min(1.0, harmonic_mean)
    
    def _simple_path_scoring(self, path: List[str],
                           context: RadiatingContext) -> float:
        """
        Simple path scoring without graph information
        
        Args:
            path: List of entity IDs
            context: Traversal context
            
        Returns:
            Simple score (0.0 to 1.0)
        """
        # Score based on path properties
        score = 0.5  # Base score
        
        # Bonus for shorter paths
        if len(path) <= 3:
            score += 0.2
        elif len(path) <= 5:
            score += 0.1
        
        # Check if path entities were discovered early (higher priority)
        early_discovery_bonus = 0.0
        for entity_id in path:
            if entity_id in context.visited_entity_ids:
                # Entities discovered early are likely more relevant
                position = list(context.visited_entity_ids).index(entity_id)
                if position < len(context.visited_entity_ids) * 0.2:
                    early_discovery_bonus += 0.05
        
        score += min(0.3, early_discovery_bonus)
        
        return min(1.0, score)
    
    def _calculate_bottleneck_score(self, path: List[str],
                                   graph: Optional[RadiatingGraph]) -> float:
        """
        Calculate bottleneck score (minimum edge weight in path)
        
        Args:
            path: List of entity IDs
            graph: Optional graph
            
        Returns:
            Bottleneck score (0.0 to 1.0)
        """
        if not graph or len(path) < 2:
            return 0.5  # Default score
        
        edge_weights = []
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i + 1]
            
            # Find edge weight
            if source_id in graph.nodes:
                node = graph.nodes[source_id]
                for rel_id in node.outgoing_edges:
                    rel = graph.edges.get(rel_id)
                    if rel and rel.target_entity == target_id:
                        edge_weights.append(rel.calculate_traversal_score())
                        break
        
        if not edge_weights:
            return 0.5
        
        # Return minimum edge weight (bottleneck)
        return min(edge_weights)
    
    def _calculate_diversity_scores(self, paths: List[OptimizedPath]):
        """
        Calculate diversity scores for paths
        
        Args:
            paths: List of OptimizedPath objects
        """
        if len(paths) <= 1:
            if paths:
                paths[0].diversity_score = 1.0
            return
        
        # Calculate pairwise path similarities
        for i, path1 in enumerate(paths):
            min_similarity = 1.0
            
            for j, path2 in enumerate(paths):
                if i != j:
                    similarity = self._calculate_path_similarity(path1.path, path2.path)
                    min_similarity = min(min_similarity, similarity)
            
            # Diversity is inverse of similarity
            path1.diversity_score = 1.0 - min_similarity
    
    def _calculate_path_similarity(self, path1: List[str], 
                                  path2: List[str]) -> float:
        """
        Calculate similarity between two paths
        
        Args:
            path1: First path
            path2: Second path
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not path1 or not path2:
            return 0.0
        
        # Jaccard similarity of entities
        set1 = set(path1)
        set2 = set(path2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        jaccard_sim = intersection / union
        
        # Order similarity (do paths traverse entities in similar order?)
        order_sim = self._calculate_order_similarity(path1, path2)
        
        # Combine similarities
        return 0.7 * jaccard_sim + 0.3 * order_sim
    
    def _calculate_order_similarity(self, path1: List[str],
                                   path2: List[str]) -> float:
        """
        Calculate order similarity between paths
        
        Args:
            path1: First path
            path2: Second path
            
        Returns:
            Order similarity (0.0 to 1.0)
        """
        # Find common entities
        common = set(path1).intersection(set(path2))
        if not common:
            return 0.0
        
        # Check if common entities appear in same relative order
        order_matches = 0
        for entity in common:
            pos1 = path1.index(entity) / len(path1)
            pos2 = path2.index(entity) / len(path2)
            
            # Similar relative position
            if abs(pos1 - pos2) < 0.3:
                order_matches += 1
        
        return order_matches / len(common)
    
    def _multi_criteria_optimization(self, paths: List[OptimizedPath]) -> List[OptimizedPath]:
        """
        Apply multi-criteria optimization to paths
        
        Args:
            paths: List of OptimizedPath objects
            
        Returns:
            Optimized list of paths
        """
        # Update scores with diversity
        for path in paths:
            path.score = (
                path.metadata['relevance_score'] * self.metric_weights['relevance'] +
                (1.0 / (1 + self.length_penalty_factor * path.length)) * self.metric_weights['length'] +
                path.bottleneck_score * self.metric_weights['bottleneck'] +
                path.diversity_score * self.metric_weights['diversity']
            )
        
        # Apply Pareto optimization
        pareto_optimal = self._find_pareto_optimal_paths(paths)
        
        return pareto_optimal
    
    def _find_pareto_optimal_paths(self, paths: List[OptimizedPath]) -> List[OptimizedPath]:
        """
        Find Pareto optimal paths
        
        Args:
            paths: List of OptimizedPath objects
            
        Returns:
            Pareto optimal paths
        """
        pareto_optimal = []
        
        for candidate in paths:
            is_dominated = False
            
            for other in paths:
                if candidate == other:
                    continue
                
                # Check if other dominates candidate
                if (other.score >= candidate.score and
                    other.bottleneck_score >= candidate.bottleneck_score and
                    other.diversity_score >= candidate.diversity_score and
                    other.length <= candidate.length):
                    
                    # Check for strict dominance
                    if (other.score > candidate.score or
                        other.bottleneck_score > candidate.bottleneck_score or
                        other.diversity_score > candidate.diversity_score or
                        other.length < candidate.length):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_optimal.append(candidate)
        
        return pareto_optimal
    
    def find_alternative_paths(self, start_id: str, end_id: str,
                              graph: RadiatingGraph,
                              blocked_entities: Set[str] = None) -> List[List[str]]:
        """
        Find alternative paths avoiding certain entities
        
        Args:
            start_id: Starting entity ID
            end_id: Target entity ID
            graph: RadiatingGraph
            blocked_entities: Set of entity IDs to avoid
            
        Returns:
            List of alternative paths
        """
        if blocked_entities is None:
            blocked_entities = set()
        
        # Create NetworkX graph for advanced path finding
        nx_graph = self._create_networkx_graph(graph, blocked_entities)
        
        if start_id not in nx_graph or end_id not in nx_graph:
            return []
        
        try:
            # Find k-shortest paths
            paths = list(nx.shortest_simple_paths(
                nx_graph, start_id, end_id, weight='weight'
            ))[:10]  # Get up to 10 paths
            
            return paths
            
        except nx.NetworkXNoPath:
            return []
    
    def _create_networkx_graph(self, graph: RadiatingGraph,
                              blocked_entities: Set[str]) -> nx.DiGraph:
        """
        Create NetworkX graph from RadiatingGraph
        
        Args:
            graph: RadiatingGraph
            blocked_entities: Entities to exclude
            
        Returns:
            NetworkX directed graph
        """
        nx_graph = nx.DiGraph()
        
        # Add nodes
        for node_id, node in graph.nodes.items():
            if node_id not in blocked_entities:
                nx_graph.add_node(
                    node_id,
                    entity=node.entity,
                    relevance=node.entity.relevance_score
                )
        
        # Add edges
        for rel_id, rel in graph.edges.items():
            if (rel.source_entity not in blocked_entities and
                rel.target_entity not in blocked_entities):
                
                # Use inverse of traversal score as weight (lower is better)
                weight = 1.0 / max(0.01, rel.calculate_traversal_score())
                
                nx_graph.add_edge(
                    rel.source_entity,
                    rel.target_entity,
                    weight=weight,
                    relationship=rel
                )
                
                # Add reverse edge if bidirectional
                if rel.bidirectional:
                    nx_graph.add_edge(
                        rel.target_entity,
                        rel.source_entity,
                        weight=weight,
                        relationship=rel
                    )
        
        return nx_graph
    
    def merge_similar_paths(self, paths: List[List[str]], 
                          similarity_threshold: float = 0.8) -> List[List[str]]:
        """
        Merge highly similar paths to reduce redundancy
        
        Args:
            paths: List of paths
            similarity_threshold: Threshold for merging
            
        Returns:
            Merged list of paths
        """
        if len(paths) <= 1:
            return paths
        
        merged = []
        used = set()
        
        for i, path1 in enumerate(paths):
            if i in used:
                continue
            
            # Find similar paths
            similar_group = [path1]
            used.add(i)
            
            for j, path2 in enumerate(paths[i+1:], i+1):
                if j not in used:
                    similarity = self._calculate_path_similarity(path1, path2)
                    if similarity >= similarity_threshold:
                        similar_group.append(path2)
                        used.add(j)
            
            # Keep the shortest path from similar group
            representative = min(similar_group, key=len)
            merged.append(representative)
        
        return merged