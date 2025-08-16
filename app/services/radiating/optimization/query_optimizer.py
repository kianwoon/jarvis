"""
Query Optimizer

Optimizes Neo4j queries for the radiating system using cost-based analysis,
index hints, query caching, and batch optimization.
"""

import logging
import hashlib
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio
from enum import Enum

from app.core.redis_client import get_redis_client
from app.services.neo4j_service import get_neo4j_service
from app.core.radiating_settings_cache import get_radiating_settings

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries for optimization"""
    TRAVERSAL = "traversal"
    EXPANSION = "expansion"
    RELATIONSHIP = "relationship"
    ENTITY = "entity"
    PATH = "path"
    AGGREGATION = "aggregation"


@dataclass
class QueryPlan:
    """Represents an optimized query plan"""
    original_query: str
    optimized_query: str
    estimated_cost: float
    index_hints: List[str] = field(default_factory=list)
    execution_strategy: str = "standard"
    cache_key: Optional[str] = None
    batch_size: Optional[int] = None
    parallel_execution: bool = False
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryStatistics:
    """Query execution statistics"""
    execution_time: float
    rows_returned: int
    db_hits: int
    cache_hits: int = 0
    memory_usage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class QueryOptimizer:
    """
    Optimizes Neo4j queries for performance using various techniques including
    cost-based analysis, index hints, caching, and batch processing.
    """
    
    CACHE_PREFIX = "radiating:query:optimized:"
    STATS_PREFIX = "radiating:query:stats:"
    PLAN_CACHE_TTL = 3600  # 1 hour
    
    # Query patterns that benefit from specific optimizations
    OPTIMIZATION_PATTERNS = {
        QueryType.TRAVERSAL: {
            'use_index': True,
            'batch_capable': True,
            'parallel_capable': True,
            'cache_results': True
        },
        QueryType.EXPANSION: {
            'use_index': True,
            'batch_capable': True,
            'parallel_capable': True,
            'cache_results': True
        },
        QueryType.RELATIONSHIP: {
            'use_index': True,
            'batch_capable': False,
            'parallel_capable': True,
            'cache_results': True
        },
        QueryType.ENTITY: {
            'use_index': True,
            'batch_capable': True,
            'parallel_capable': False,
            'cache_results': True
        },
        QueryType.PATH: {
            'use_index': True,
            'batch_capable': False,
            'parallel_capable': True,
            'cache_results': False
        },
        QueryType.AGGREGATION: {
            'use_index': False,
            'batch_capable': False,
            'parallel_capable': False,
            'cache_results': True
        }
    }
    
    def __init__(self):
        """Initialize QueryOptimizer"""
        self.redis_client = get_redis_client()
        self.neo4j_service = get_neo4j_service()
        
        # Query plan cache
        self.plan_cache: Dict[str, QueryPlan] = {}
        
        # Statistics tracking
        self.query_stats: Dict[str, List[QueryStatistics]] = {}
        
        # Cost model parameters
        self.cost_model = {
            'index_scan_cost': 1.0,
            'label_scan_cost': 5.0,
            'all_nodes_scan_cost': 100.0,
            'expand_cost': 2.0,
            'filter_cost': 1.5,
            'sort_cost': 10.0,
            'limit_cost': 0.1,
            'property_access_cost': 0.5
        }
        
        # Index metadata cache
        self.index_metadata: Dict[str, Dict] = {}
        self._refresh_index_metadata()
    
    def _refresh_index_metadata(self):
        """Refresh Neo4j index metadata"""
        try:
            with self.neo4j_service.driver.session() as session:
                # Get index information
                result = session.run("SHOW INDEXES")
                self.index_metadata = {}
                
                for record in result:
                    index_name = record.get('name', '')
                    self.index_metadata[index_name] = {
                        'labels': record.get('labelsOrTypes', []),
                        'properties': record.get('properties', []),
                        'type': record.get('type', ''),
                        'state': record.get('state', '')
                    }
                
                logger.info(f"Refreshed {len(self.index_metadata)} indexes")
        except Exception as e:
            logger.error(f"Error refreshing index metadata: {e}")
    
    async def optimize_query(
        self,
        query: str,
        query_type: QueryType,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """
        Optimize a query based on type and context
        
        Args:
            query: The original Cypher query
            query_type: Type of query for optimization
            parameters: Query parameters
            context: Additional context for optimization
            
        Returns:
            QueryPlan: Optimized query plan
        """
        # Generate cache key
        cache_key = self._generate_cache_key(query, parameters)
        
        # Check if we have a cached plan
        cached_plan = await self._get_cached_plan(cache_key)
        if cached_plan:
            logger.debug(f"Using cached query plan for {cache_key}")
            return cached_plan
        
        # Analyze query structure
        analysis = self._analyze_query(query, query_type)
        
        # Apply optimizations based on query type
        optimizations = self.OPTIMIZATION_PATTERNS.get(query_type, {})
        
        # Generate index hints if applicable
        index_hints = []
        if optimizations.get('use_index', False):
            index_hints = self._generate_index_hints(analysis, context)
        
        # Optimize query structure
        optimized_query = self._optimize_query_structure(query, analysis, index_hints)
        
        # Estimate query cost
        estimated_cost = self._estimate_query_cost(analysis, index_hints)
        
        # Determine execution strategy
        execution_strategy = self._determine_execution_strategy(
            query_type, estimated_cost, context
        )
        
        # Determine batch size if applicable
        batch_size = None
        if optimizations.get('batch_capable', False):
            batch_size = self._calculate_optimal_batch_size(analysis, context)
        
        # Create query plan
        plan = QueryPlan(
            original_query=query,
            optimized_query=optimized_query,
            estimated_cost=estimated_cost,
            index_hints=index_hints,
            execution_strategy=execution_strategy,
            cache_key=cache_key if optimizations.get('cache_results', False) else None,
            batch_size=batch_size,
            parallel_execution=optimizations.get('parallel_capable', False),
            statistics={
                'analysis': analysis,
                'optimization_applied': True,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        # Cache the plan
        await self._cache_plan(cache_key, plan)
        
        return plan
    
    def _analyze_query(self, query: str, query_type: QueryType) -> Dict[str, Any]:
        """Analyze query structure for optimization opportunities"""
        analysis = {
            'type': query_type.value,
            'has_match': 'MATCH' in query.upper(),
            'has_where': 'WHERE' in query.upper(),
            'has_order': 'ORDER BY' in query.upper(),
            'has_limit': 'LIMIT' in query.upper(),
            'has_skip': 'SKIP' in query.upper(),
            'has_aggregation': any(agg in query.upper() for agg in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']),
            'has_with': 'WITH' in query.upper(),
            'has_union': 'UNION' in query.upper(),
            'has_optional': 'OPTIONAL' in query.upper(),
            'estimated_nodes': self._estimate_node_count(query),
            'pattern_complexity': self._analyze_pattern_complexity(query)
        }
        
        # Extract labels and properties
        analysis['labels'] = self._extract_labels(query)
        analysis['properties'] = self._extract_properties(query)
        
        return analysis
    
    def _extract_labels(self, query: str) -> List[str]:
        """Extract node labels from query"""
        import re
        pattern = r'\([\w]*:(\w+)[\s\w\{\}:,]*\)'
        matches = re.findall(pattern, query)
        return list(set(matches))
    
    def _extract_properties(self, query: str) -> List[str]:
        """Extract property names from query"""
        import re
        pattern = r'\.(\w+)\s*[=<>!]'
        matches = re.findall(pattern, query)
        return list(set(matches))
    
    def _estimate_node_count(self, query: str) -> int:
        """Estimate the number of nodes involved in the query"""
        import re
        # Count node patterns
        node_pattern = r'\([^)]*\)'
        nodes = re.findall(node_pattern, query)
        return len(nodes)
    
    def _analyze_pattern_complexity(self, query: str) -> str:
        """Analyze the complexity of graph patterns in the query"""
        import re
        
        # Count relationship patterns
        rel_pattern = r'-\[.*?\]-'
        relationships = re.findall(rel_pattern, query)
        
        # Count variable length paths
        var_length = re.findall(r'\*\d*\.?\.\d*', query)
        
        if var_length:
            return 'complex'
        elif len(relationships) > 3:
            return 'moderate'
        else:
            return 'simple'
    
    def _generate_index_hints(
        self,
        analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate index hints based on query analysis"""
        hints = []
        
        # Get available indexes for the labels in the query
        for label in analysis.get('labels', []):
            for index_name, index_info in self.index_metadata.items():
                if label in index_info.get('labels', []):
                    # Check if query uses indexed properties
                    query_props = set(analysis.get('properties', []))
                    index_props = set(index_info.get('properties', []))
                    
                    if query_props & index_props:
                        hint = f"USING INDEX {label}:{list(query_props & index_props)[0]}"
                        hints.append(hint)
        
        return hints
    
    def _optimize_query_structure(
        self,
        query: str,
        analysis: Dict[str, Any],
        index_hints: List[str]
    ) -> str:
        """Optimize the query structure"""
        optimized = query
        
        # Add index hints after MATCH clause
        if index_hints and 'MATCH' in optimized.upper():
            for hint in index_hints:
                # Insert hint after MATCH clause
                match_pos = optimized.upper().find('MATCH')
                next_clause_pos = len(optimized)
                
                for clause in ['WHERE', 'WITH', 'RETURN', 'ORDER', 'LIMIT']:
                    pos = optimized.upper().find(clause, match_pos)
                    if pos > 0 and pos < next_clause_pos:
                        next_clause_pos = pos
                
                optimized = (
                    optimized[:next_clause_pos] + 
                    f"\n{hint}\n" + 
                    optimized[next_clause_pos:]
                )
        
        # Optimize LIMIT without ORDER BY (add ORDER BY id for consistency)
        if analysis.get('has_limit') and not analysis.get('has_order'):
            limit_pos = optimized.upper().rfind('LIMIT')
            if limit_pos > 0:
                optimized = (
                    optimized[:limit_pos] + 
                    "ORDER BY id(n) " +  # Assuming 'n' is a common node variable
                    optimized[limit_pos:]
                )
        
        # Add PROFILE for complex queries in development mode
        settings = get_radiating_settings()
        if settings.get('debug_mode', False) and analysis.get('pattern_complexity') == 'complex':
            optimized = "PROFILE " + optimized
        
        return optimized
    
    def _estimate_query_cost(
        self,
        analysis: Dict[str, Any],
        index_hints: List[str]
    ) -> float:
        """Estimate the cost of executing the query"""
        cost = 0.0
        
        # Base cost based on scan type
        if index_hints:
            cost += self.cost_model['index_scan_cost']
        elif analysis.get('labels'):
            cost += self.cost_model['label_scan_cost'] * len(analysis['labels'])
        else:
            cost += self.cost_model['all_nodes_scan_cost']
        
        # Add cost for pattern complexity
        complexity_multiplier = {
            'simple': 1.0,
            'moderate': 2.5,
            'complex': 5.0
        }
        cost *= complexity_multiplier.get(analysis.get('pattern_complexity', 'simple'), 1.0)
        
        # Add cost for operations
        if analysis.get('has_where'):
            cost += self.cost_model['filter_cost']
        
        if analysis.get('has_order'):
            cost += self.cost_model['sort_cost']
        
        if analysis.get('has_aggregation'):
            cost += self.cost_model['sort_cost'] * 0.5  # Aggregations often require sorting
        
        # Adjust for estimated node count
        node_count = analysis.get('estimated_nodes', 1)
        cost *= (1 + math.log10(max(1, node_count)))
        
        return cost
    
    def _determine_execution_strategy(
        self,
        query_type: QueryType,
        estimated_cost: float,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Determine the best execution strategy"""
        # High cost queries should use streaming
        if estimated_cost > 100:
            return "streaming"
        
        # Traversal queries benefit from batch processing
        if query_type in [QueryType.TRAVERSAL, QueryType.EXPANSION]:
            return "batch"
        
        # Path queries should use shortest path algorithms when possible
        if query_type == QueryType.PATH:
            return "shortest_path"
        
        # Aggregations should use pipeline
        if query_type == QueryType.AGGREGATION:
            return "pipeline"
        
        return "standard"
    
    def _calculate_optimal_batch_size(
        self,
        analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> int:
        """Calculate optimal batch size for query execution"""
        # Base batch size
        batch_size = 100
        
        # Adjust based on pattern complexity
        if analysis.get('pattern_complexity') == 'complex':
            batch_size = 25
        elif analysis.get('pattern_complexity') == 'moderate':
            batch_size = 50
        
        # Adjust based on estimated node count
        node_count = analysis.get('estimated_nodes', 1)
        if node_count > 10:
            batch_size = max(10, batch_size // 2)
        
        # Consider available memory from context
        if context and 'available_memory' in context:
            memory_gb = context['available_memory'] / (1024 ** 3)
            if memory_gb < 2:
                batch_size = min(batch_size, 25)
        
        return batch_size
    
    def _generate_cache_key(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a cache key for the query plan"""
        # Create a deterministic representation
        cache_data = {
            'query': query.strip(),
            'params': parameters or {}
        }
        
        # Use hash for cache key
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    async def _get_cached_plan(self, cache_key: str) -> Optional[QueryPlan]:
        """Get cached query plan"""
        try:
            cached = await self.redis_client.get(f"{self.CACHE_PREFIX}{cache_key}")
            if cached:
                plan_data = json.loads(cached)
                return QueryPlan(**plan_data)
        except Exception as e:
            logger.debug(f"Error getting cached plan: {e}")
        
        return None
    
    async def _cache_plan(self, cache_key: str, plan: QueryPlan):
        """Cache query plan"""
        try:
            plan_data = {
                'original_query': plan.original_query,
                'optimized_query': plan.optimized_query,
                'estimated_cost': plan.estimated_cost,
                'index_hints': plan.index_hints,
                'execution_strategy': plan.execution_strategy,
                'cache_key': plan.cache_key,
                'batch_size': plan.batch_size,
                'parallel_execution': plan.parallel_execution,
                'statistics': plan.statistics
            }
            
            await self.redis_client.setex(
                f"{self.CACHE_PREFIX}{cache_key}",
                self.PLAN_CACHE_TTL,
                json.dumps(plan_data)
            )
        except Exception as e:
            logger.debug(f"Error caching plan: {e}")
    
    async def batch_optimize_queries(
        self,
        queries: List[Tuple[str, QueryType, Optional[Dict]]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[QueryPlan]:
        """
        Optimize multiple queries in batch
        
        Args:
            queries: List of (query, type, parameters) tuples
            context: Shared context for optimization
            
        Returns:
            List[QueryPlan]: Optimized plans for all queries
        """
        # Process queries concurrently
        tasks = [
            self.optimize_query(query, qtype, params, context)
            for query, qtype, params in queries
        ]
        
        plans = await asyncio.gather(*tasks)
        
        # Analyze for common patterns and further optimization
        self._analyze_batch_patterns(plans)
        
        return plans
    
    def _analyze_batch_patterns(self, plans: List[QueryPlan]):
        """Analyze batch of queries for common patterns"""
        # Track common index usage
        index_usage = {}
        for plan in plans:
            for hint in plan.index_hints:
                index_usage[hint] = index_usage.get(hint, 0) + 1
        
        # Log insights
        if index_usage:
            most_used = max(index_usage, key=index_usage.get)
            logger.info(f"Most used index in batch: {most_used} ({index_usage[most_used]} times)")
    
    async def record_execution_stats(
        self,
        plan: QueryPlan,
        execution_time: float,
        rows_returned: int,
        db_hits: int = 0,
        memory_usage: float = 0.0
    ):
        """Record query execution statistics"""
        stats = QueryStatistics(
            execution_time=execution_time,
            rows_returned=rows_returned,
            db_hits=db_hits,
            memory_usage=memory_usage
        )
        
        # Store in memory
        cache_key = plan.cache_key or self._generate_cache_key(plan.original_query)
        if cache_key not in self.query_stats:
            self.query_stats[cache_key] = []
        
        self.query_stats[cache_key].append(stats)
        
        # Keep only recent stats
        self.query_stats[cache_key] = self.query_stats[cache_key][-100:]
        
        # Store in Redis for persistence
        stats_data = {
            'execution_time': stats.execution_time,
            'rows_returned': stats.rows_returned,
            'db_hits': stats.db_hits,
            'memory_usage': stats.memory_usage,
            'timestamp': stats.timestamp.isoformat()
        }
        
        await self.redis_client.lpush(
            f"{self.STATS_PREFIX}{cache_key}",
            json.dumps(stats_data)
        )
        
        # Trim to keep only recent stats
        await self.redis_client.ltrim(f"{self.STATS_PREFIX}{cache_key}", 0, 99)
    
    async def get_query_statistics(self, query: str) -> Dict[str, Any]:
        """Get statistics for a query"""
        cache_key = self._generate_cache_key(query)
        
        # Get from Redis
        stats_data = await self.redis_client.lrange(
            f"{self.STATS_PREFIX}{cache_key}",
            0,
            -1
        )
        
        if not stats_data:
            return {
                'query': query,
                'no_statistics': True
            }
        
        # Parse statistics
        stats = [json.loads(s) for s in stats_data]
        
        # Calculate aggregates
        exec_times = [s['execution_time'] for s in stats]
        rows = [s['rows_returned'] for s in stats]
        
        return {
            'query': query,
            'executions': len(stats),
            'avg_execution_time': sum(exec_times) / len(exec_times),
            'min_execution_time': min(exec_times),
            'max_execution_time': max(exec_times),
            'avg_rows_returned': sum(rows) / len(rows),
            'last_execution': stats[0]['timestamp'] if stats else None
        }


import math  # Add this import at the top