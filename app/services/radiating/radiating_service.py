"""
Radiating Service

Main service class that orchestrates all radiating components and provides
the primary interface for the radiating coverage system.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import uuid

from app.services.radiating.models.responses import (
    RadiatingCoverage,
    RadiatingSettings,
    RadiatingSystemStatus,
    RadiatingStatus,
    ExpandedEntity,
    DiscoveredRelationship
)
from app.services.radiating.models.radiating_context import RadiatingContext, TraversalStrategy
from app.services.radiating.models.radiating_entity import RadiatingEntity
from app.services.radiating.engine.radiating_traverser import RadiatingTraverser
from app.services.radiating.query_expansion.query_analyzer import QueryAnalyzer
from app.services.radiating.extraction.universal_entity_extractor import UniversalEntityExtractor
from app.services.radiating.storage.radiating_cache_manager import get_radiating_cache_manager
from app.services.neo4j_service import Neo4jService
from app.core.redis_client import get_redis_client
from app.core.db import get_db_session

logger = logging.getLogger(__name__)


def normalize_strategy_string(strategy: str) -> str:
    """Normalize strategy string for TraversalStrategy enum
    
    Handles variations like:
    - "breadth-first" → "breadth_first" 
    - "DEPTH-FIRST" → "depth_first"
    - "Best-First" → "best_first"
    - "hybrid" → "hybrid"
    """
    if not strategy:
        return 'hybrid'
    # Convert hyphens to underscores and lowercase
    return strategy.replace('-', '_').lower()


class RadiatingService:
    """
    Main service class for the radiating coverage system.
    Orchestrates all components and provides the primary API.
    """
    
    # Default settings
    DEFAULT_SETTINGS = RadiatingSettings(
        enabled=True,
        max_depth=3,
        max_entities_per_level=50,
        relevance_threshold=0.3,
        traversal_strategy="ADAPTIVE",
        enable_caching=True,
        cache_ttl_seconds=3600,
        enable_parallel_processing=True,
        max_parallel_workers=4,
        web_first_enabled=True,  # WEB-FIRST: Default to web search for ALL queries
        web_search_timeout=30000  # 30 second timeout for web searches
    )
    
    SETTINGS_KEY = "radiating:settings"
    STATUS_KEY = "radiating:status"
    METRICS_KEY = "radiating:metrics"
    
    def __init__(self):
        """Initialize the RadiatingService"""
        self.traverser = RadiatingTraverser()
        self.query_analyzer = QueryAnalyzer()
        self.entity_extractor = UniversalEntityExtractor()
        self.cache_manager = get_radiating_cache_manager()
        self.redis_client = get_redis_client()
        
        # Service state
        self.service_start_time = datetime.now()
        self.active_queries = {}
        self.metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_processing_time': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'entities_discovered': 0,
            'relationships_found': 0,
            'web_searches_performed': 0,
            'web_entities_discovered': 0
        }
        
        # Initialize settings
        self._initialize_settings()
        
        # Log web-first configuration on startup
        web_first = getattr(self.settings, 'web_first_enabled', True)
        logger.info("="*80)
        logger.info("🌐 RadiatingService initialized with WEB-FIRST approach")
        logger.info(f"Web-first enabled: {web_first}")
        logger.info(f"Web search timeout: {getattr(self.settings, 'web_search_timeout', 30000)}ms")
        logger.info("Web search is the PRIMARY information source for maximum freshness")
        logger.info("="*80)
        
        # Start background tasks
        self._start_background_tasks()
    
    def _initialize_settings(self):
        """Initialize or load settings from Redis"""
        try:
            settings_json = self.redis_client.get(self.SETTINGS_KEY)
            if settings_json:
                self.settings = RadiatingSettings.parse_raw(settings_json)
                logger.info("Loaded radiating settings from Redis")
            else:
                self.settings = self.DEFAULT_SETTINGS
                self._save_settings()
                logger.info("Initialized with default radiating settings")
        except Exception as e:
            logger.error(f"Error initializing settings: {e}")
            self.settings = self.DEFAULT_SETTINGS
    
    def _save_settings(self):
        """Save current settings to Redis"""
        try:
            self.redis_client.set(
                self.SETTINGS_KEY,
                self.settings.json(),
                ex=86400  # Expire after 24 hours
            )
            logger.info("Saved radiating settings to Redis")
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        asyncio.create_task(self._periodic_metrics_save())
        asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_metrics_save(self):
        """Periodically save metrics to Redis"""
        while True:
            try:
                await asyncio.sleep(60)  # Save every minute
                await self._save_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic metrics save: {e}")
    
    async def _periodic_cleanup(self):
        """Periodically cleanup stale data"""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                await self._cleanup_stale_queries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def _save_metrics(self):
        """Save metrics to Redis"""
        try:
            self.redis_client.set(
                self.METRICS_KEY,
                json.dumps(self.metrics),
                ex=3600  # Expire after 1 hour
            )
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    async def _cleanup_stale_queries(self):
        """Cleanup queries that have been running too long"""
        try:
            current_time = datetime.now()
            stale_queries = []
            
            for query_id, query_info in self.active_queries.items():
                if current_time - query_info['start_time'] > timedelta(minutes=10):
                    stale_queries.append(query_id)
            
            for query_id in stale_queries:
                del self.active_queries[query_id]
                logger.warning(f"Cleaned up stale query: {query_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up stale queries: {e}")
    
    async def execute_radiating_query(
        self,
        query: str,
        max_depth: Optional[int] = None,
        strategy: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        include_coverage: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a query with radiating coverage.
        
        Args:
            query: The query to process
            max_depth: Override max depth setting
            strategy: Override traversal strategy
            filters: Filters to apply
            include_coverage: Whether to include coverage data
            
        Returns:
            Query results with optional coverage information
        """
        query_id = f"rq_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()
        
        # Track active query
        self.active_queries[query_id] = {
            'query': query,
            'start_time': start_time,
            'status': 'processing'
        }
        
        try:
            # Check if radiating is enabled
            if not self.settings.enabled:
                logger.info("Radiating is disabled, returning basic response")
                return {
                    'query_id': query_id,
                    'status': 'disabled',
                    'message': 'Radiating coverage is currently disabled',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Update metrics
            self.metrics['total_queries'] += 1
            
            # WEB-FIRST APPROACH: Determine if web search should be used (defaults to True)
            use_web_search = filters.get('use_web_search', self._should_use_web_search(query)) if filters else self._should_use_web_search(query)
            
            # Force web search for all queries unless explicitly disabled
            if filters and 'force_web_search' in filters:
                use_web_search = filters['force_web_search']
                if use_web_search:
                    logger.info("🚀 FORCED WEB-FIRST MODE: Web search override enabled")
            
            # Extract entities from query - WEB-FIRST APPROACH IS PRIMARY
            if use_web_search:
                logger.info("="*60)
                logger.info("🌐 WEB-FIRST APPROACH ACTIVATED")
                logger.info(f"Query: {query[:100]}...")
                logger.info("Using web search as PRIMARY information source")
                logger.info("Reason: Maximum freshness and real-world accuracy")
                logger.info("="*60)
                
                # Track web search metrics
                self.metrics['web_searches_performed'] += 1
                
                # Use the enhanced web search method with force flag
                entities = await self.entity_extractor.extract_entities_with_web_search(
                    query,
                    force_web_search=True  # ALWAYS force web search when determined necessary
                )
                
                # Track web-sourced entities
                web_entity_count = sum(1 for e in entities if hasattr(e, 'source') and e.source == 'web')
                self.metrics['web_entities_discovered'] += web_entity_count
                
                logger.info(f"📊 Web Search Results:")
                logger.info(f"   Total entities discovered: {len(entities)}")
                logger.info(f"   Web-sourced entities: {web_entity_count}")
                logger.info(f"   LLM-augmented entities: {len(entities) - web_entity_count}")
                
            else:
                # Fallback to standard extraction but STILL prefer web search
                logger.info(f"📝 Using LLM extraction with web search preference for: {query[:100]}...")
                logger.info("Note: Web search is still preferred even in standard mode")
                entities = await self.entity_extractor.extract_entities(
                    query,
                    prefer_web_search=True  # ALWAYS prefer web search for freshness
                )
            
            # Store web-discovered entities in Neo4j immediately for persistence
            if use_web_search and entities:
                logger.info("💾 Storing web-discovered entities in Neo4j knowledge graph...")
                try:
                    neo4j_service = Neo4jService()
                    stored_count = 0
                    
                    for entity in entities:
                        # Only store entities with web source or high confidence
                        if (hasattr(entity, 'source') and entity.source == 'web') or \
                           (hasattr(entity, 'confidence') and entity.confidence > 0.8):
                            
                            entity_properties = {
                                'name': entity.text if hasattr(entity, 'text') else '',
                                'confidence': entity.confidence if hasattr(entity, 'confidence') else 0.7,
                                'source': entity.source if hasattr(entity, 'source') else 'extraction',
                                'query_context': query[:200],  # Store query context
                                'discovered_at': datetime.now().isoformat(),
                                'metadata': json.dumps(entity.metadata) if hasattr(entity, 'metadata') else '{}'
                            }
                            
                            # Create entity in Neo4j
                            entity_type = entity.entity_type if hasattr(entity, 'entity_type') else 'Technology'
                            entity_id = neo4j_service.create_entity(entity_type, entity_properties)
                            
                            if entity_id:
                                stored_count += 1
                                # Store the Neo4j ID back in the entity for later use
                                if hasattr(entity, 'metadata'):
                                    entity.metadata['neo4j_id'] = entity_id
                    
                    logger.info(f"✅ Stored {stored_count} web-discovered entities in Neo4j")
                    
                except Exception as e:
                    logger.warning(f"Failed to store entities in Neo4j: {e}")
                    # Continue processing even if storage fails
            
            starting_entities = [
                RadiatingEntity(
                    id=entity.metadata.get('neo4j_id', f"entity_{i}") if hasattr(entity, 'metadata') else f"entity_{i}",
                    type=entity.entity_type if hasattr(entity, 'entity_type') else 'unknown',
                    name=entity.text if hasattr(entity, 'text') else '',
                    properties={'confidence': entity.confidence if hasattr(entity, 'confidence') else 0.7,
                                'context': entity.context if hasattr(entity, 'context') else '',
                                'metadata': entity.metadata if hasattr(entity, 'metadata') else {},
                                'source': entity.source if hasattr(entity, 'source') else 'extraction'}
                )
                for i, entity in enumerate(entities)
            ]
            
            # Create radiating context
            # Normalize the strategy string to handle frontend variations (e.g., "breadth-first" -> "breadth_first")
            strategy_normalized = normalize_strategy_string(strategy or self.settings.traversal_strategy)
            context = RadiatingContext(
                original_query=query,
                depth_limit=max_depth or self.settings.max_depth,
                relevance_threshold=self.settings.relevance_threshold,
                traversal_strategy=TraversalStrategy(strategy_normalized)
            )
            
            # Execute traversal
            graph = await self.traverser.traverse(context, starting_entities)
            
            # Update metrics
            self.metrics['entities_discovered'] += len(graph.entities)
            self.metrics['relationships_found'] += len(graph.relationships)
            
            # Prepare response
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics['total_processing_time'] += processing_time
            self.metrics['successful_queries'] += 1
            
            # Track entity sources in response
            entity_sources = {
                'web': sum(1 for e in graph.entities if e.properties.get('source') == 'web'),
                'llm': sum(1 for e in graph.entities if e.properties.get('source') != 'web'),
                'total': len(graph.entities)
            }
            
            response = {
                'query_id': query_id,
                'status': 'completed',
                'processing_time_ms': processing_time,
                'timestamp': datetime.now().isoformat(),
                'web_search_used': use_web_search,
                'entity_sources': entity_sources
            }
            
            # Add coverage data if requested
            if include_coverage:
                coverage = RadiatingCoverage(
                    total_entities=len(graph.entities),
                    total_relationships=len(graph.relationships),
                    max_depth_reached=graph.metadata.get('max_depth_reached', 0),
                    coverage_percentage=graph.metadata.get('coverage_percentage', 0.0),
                    explored_paths=graph.metadata.get('explored_paths', 0),
                    pruned_paths=graph.metadata.get('pruned_paths', 0),
                    entity_types=self._count_entity_types(graph.entities),
                    relationship_types=self._count_relationship_types(graph.relationships)
                )
                response['coverage'] = coverage.dict()
                
                # Convert entities and relationships to response models
                response['entities'] = [
                    ExpandedEntity(
                        id=e.id,
                        name=e.name,
                        type=e.type,
                        properties=e.properties,
                        relevance_score=e.relevance_score,
                        depth=e.depth
                    ).dict()
                    for e in graph.entities[:20]  # Limit to first 20
                ]
                
                response['relationships'] = [
                    DiscoveredRelationship(
                        id=r.id,
                        source_entity_id=r.source_id,
                        target_entity_id=r.target_id,
                        relationship_type=r.type,
                        properties=r.properties,
                        strength=r.strength
                    ).dict()
                    for r in graph.relationships[:20]  # Limit to first 20
                ]
            
            # Clean up active query
            del self.active_queries[query_id]
            
            return response
            
        except Exception as e:
            logger.error(f"Error executing radiating query: {e}")
            self.metrics['failed_queries'] += 1
            
            # Clean up active query
            if query_id in self.active_queries:
                del self.active_queries[query_id]
            
            return {
                'query_id': query_id,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _should_use_web_search(self, query: str) -> bool:
        """
        Determine if web search should be used for entity extraction.
        
        WEB-FIRST PRINCIPLE: Web search is the PRIMARY source for ALL queries by default.
        Only skip web search for explicitly local/personal queries.
        
        Args:
            query: The user query
            
        Returns:
            True if web search should be used (default: True for maximum freshness)
        """
        # Check if web-first is disabled in settings (but default is True)
        if hasattr(self.settings, 'web_first_enabled') and not self.settings.web_first_enabled:
            logger.warning("Web-first mode is disabled in settings - using legacy behavior")
            return self._should_use_web_search_legacy(query)
        
        query_lower = query.lower()
        
        # Only skip web search for very specific local/personal queries
        skip_web_patterns = [
            # Personal/local file queries
            'my file', 'my document', 'my code', 'my project',
            'local file', 'local database', 'localhost',
            '/home/', '/users/', 'c:\\', 'd:\\',  # File paths
            # Code analysis queries
            'analyze this code', 'review this function', 'debug this',
            'explain this code', 'what does this code do',
            # Abstract/philosophical queries
            'meaning of life', 'philosophical', 'hypothetical',
            # Basic math/logic
            'calculate', 'compute', 'solve equation',
            '2+2', '1+1',  # Simple math
            # Internal system queries
            'system status', 'internal metrics', 'cache status'
        ]
        
        # Check if query is purely local/personal
        is_local_query = any(pattern in query_lower for pattern in skip_web_patterns)
        
        if is_local_query:
            logger.info(f"📍 Detected local/personal query - skipping web search: {query[:50]}...")
            return False
        
        # DEFAULT: Always use web search for maximum information freshness
        logger.info(f"🌐 WEB-FIRST: Using web search as PRIMARY source for: {query[:50]}...")
        logger.info("Web search ensures maximum freshness and accuracy for real-world information")
        return True
    
    def _should_use_web_search_legacy(self, query: str) -> bool:
        """Legacy web search detection for backward compatibility."""
        query_lower = query.lower()
        
        # Technology-related keywords
        is_tech_query = any(term in query_lower for term in [
            'technology', 'technologies', 'tool', 'tools',
            'framework', 'frameworks', 'library', 'libraries',
            'platform', 'platforms', 'system', 'systems',
            'llm', 'ai', 'ml', 'rag', 'vector', 'agent',
            'model', 'models', 'inference', 'embedding',
            'database', 'db', 'api', 'sdk', 'cloud',
            'open source', 'opensource', 'development',
            'software', 'application', 'service', 'engine'
        ])
        
        # Temporal keywords
        temporal_keywords = [
            'latest', 'current', 'recent', 'new', 'newest',
            '2024', '2025', 'this year', 'today', 'modern',
            'emerging', 'cutting edge', 'state of the art', 'now'
        ]
        has_temporal = any(keyword in query_lower for keyword in temporal_keywords)
        
        # Discovery patterns
        has_discovery_pattern = any(pattern in query_lower for pattern in [
            'what are', 'list', 'show', 'find', 'discover',
            'recommend', 'suggest', 'which', 'best', 'top'
        ])
        
        return is_tech_query or has_temporal or has_discovery_pattern
    
    async def execute_radiating_query_with_web_search(
        self,
        query: str,
        max_depth: Optional[int] = None,
        strategy: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        include_coverage: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a query with radiating coverage and forced web search augmentation.
        This method ensures web search is used to discover latest technologies.
        
        Args:
            query: The query to process
            max_depth: Override max depth setting
            strategy: Override traversal strategy
            filters: Filters to apply
            include_coverage: Whether to include coverage data
            
        Returns:
            Query results with entities from both LLM and web search
        """
        logger.info(f"🌐 FORCED WEB-FIRST MODE for query: {query[:100]}...")
        logger.info("Ensuring maximum discovery of latest technologies through web search")
        
        # Force web search in filters
        if filters is None:
            filters = {}
        filters['use_web_search'] = True
        
        # Execute the standard query with web search enabled
        result = await self.execute_radiating_query(
            query=query,
            max_depth=max_depth,
            strategy=strategy,
            filters=filters,
            include_coverage=include_coverage
        )
        
        # Add metadata about web-first mode
        result['web_first_mode'] = True
        result['discovery_method'] = 'web_search_primary'
        
        return result
    
    async def get_radiating_coverage(
        self,
        entity_ids: List[str],
        force_web_search: bool = None
    ) -> RadiatingCoverage:
        """
        Get coverage information for specific entities.
        
        Args:
            entity_ids: List of entity IDs to get coverage for
            force_web_search: Force web search even if settings say otherwise
            
        Returns:
            Coverage information
        """
        try:
            # Check if we should use web search for coverage expansion
            use_web_search = force_web_search if force_web_search is not None else getattr(self.settings, 'web_first_enabled', True)
            
            if use_web_search:
                logger.info("🌐 WEB-FIRST: Using web search for coverage expansion")
                logger.info(f"Expanding coverage for {len(entity_ids)} entities with web augmentation")
            
            # Create minimal context for coverage check
            context = RadiatingContext(
                original_query="coverage_check",
                depth_limit=1,
                relevance_threshold=0.1,
                traversal_strategy=TraversalStrategy.BREADTH_FIRST
            )
            
            # Get entities from Neo4j
            neo4j_service = Neo4jService()
            entities = []
            for entity_id in entity_ids:
                # This would need actual Neo4j query implementation
                entity = RadiatingEntity(
                    id=entity_id,
                    type="unknown",
                    name=entity_id,
                    properties={}
                )
                entities.append(entity)
            
            # Get coverage around these entities
            graph = await self.traverser.traverse(context, entities)
            
            coverage = RadiatingCoverage(
                total_entities=len(graph.entities),
                total_relationships=len(graph.relationships),
                max_depth_reached=graph.metadata.get('max_depth_reached', 0),
                coverage_percentage=graph.metadata.get('coverage_percentage', 0.0),
                explored_paths=graph.metadata.get('explored_paths', 0),
                entity_types=self._count_entity_types(graph.entities),
                relationship_types=self._count_relationship_types(graph.relationships)
            )
            
            return coverage
            
        except Exception as e:
            logger.error(f"Error getting radiating coverage: {e}")
            return RadiatingCoverage(
                total_entities=0,
                total_relationships=0,
                max_depth_reached=0,
                coverage_percentage=0.0,
                explored_paths=0
            )
    
    async def update_settings(
        self,
        settings: RadiatingSettings
    ) -> RadiatingSettings:
        """
        Update radiating system settings.
        
        Args:
            settings: New settings to apply
            
        Returns:
            Updated settings
        """
        try:
            self.settings = settings
            self._save_settings()
            
            # Apply settings to components
            if hasattr(self.traverser, 'update_settings'):
                await self.traverser.update_settings(settings.dict())
            
            logger.info("Updated radiating settings")
            return self.settings
            
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            raise
    
    async def get_system_status(self) -> RadiatingSystemStatus:
        """
        Get current system status and health information.
        
        Returns:
            System status information
        """
        try:
            # Check component health
            neo4j_connected = await self._check_neo4j_connection()
            redis_connected = await self._check_redis_connection()
            
            # Calculate metrics
            avg_processing_time = (
                self.metrics['total_processing_time'] / self.metrics['total_queries']
                if self.metrics['total_queries'] > 0 else 0.0
            )
            
            cache_total = self.metrics['cache_hits'] + self.metrics['cache_misses']
            cache_hit_rate = (
                (self.metrics['cache_hits'] / cache_total * 100)
                if cache_total > 0 else 0.0
            )
            
            # Determine overall status
            if not self.settings.enabled:
                status = RadiatingStatus.INACTIVE
            elif len(self.active_queries) > 0:
                status = RadiatingStatus.PROCESSING
            elif not (neo4j_connected and redis_connected):
                status = RadiatingStatus.ERROR
            else:
                status = RadiatingStatus.ACTIVE
            
            # Get last query timestamp
            last_query_time = None
            if self.active_queries:
                last_query_time = max(
                    q['start_time'] for q in self.active_queries.values()
                )
            
            uptime = (datetime.now() - self.service_start_time).total_seconds()
            
            return RadiatingSystemStatus(
                status=status,
                is_healthy=neo4j_connected and redis_connected,
                active_queries=len(self.active_queries),
                total_queries_processed=self.metrics['total_queries'],
                average_processing_time_ms=avg_processing_time,
                cache_hit_rate=cache_hit_rate,
                neo4j_connected=neo4j_connected,
                redis_connected=redis_connected,
                last_query_timestamp=last_query_time,
                uptime_seconds=uptime,
                errors_last_hour=self.metrics.get('failed_queries', 0)
            )
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return RadiatingSystemStatus(
                status=RadiatingStatus.ERROR,
                is_healthy=False,
                active_queries=0,
                total_queries_processed=0,
                average_processing_time_ms=0.0,
                cache_hit_rate=0.0,
                neo4j_connected=False,
                redis_connected=False,
                uptime_seconds=0.0,
                errors_last_hour=0
            )
    
    async def toggle_radiating(self, enabled: bool) -> bool:
        """
        Enable or disable the radiating system.
        
        Args:
            enabled: Whether to enable or disable
            
        Returns:
            Success status
        """
        try:
            self.settings.enabled = enabled
            self._save_settings()
            
            logger.info(f"Radiating system {'enabled' if enabled else 'disabled'}")
            return True
            
        except Exception as e:
            logger.error(f"Error toggling radiating system: {e}")
            return False
    
    async def preview_expansion(
        self,
        query: str,
        max_depth: int = 2,
        max_entities: int = 20
    ) -> Dict[str, Any]:
        """
        Preview what a radiating expansion would discover.
        
        Args:
            query: Query to preview
            max_depth: Maximum depth for preview
            max_entities: Maximum entities to include
            
        Returns:
            Preview information
        """
        try:
            # Analyze query
            query_analysis = await self.query_analyzer.analyze_query(query)
            
            # WEB-FIRST: Extract potential entities with web search as primary
            if self._should_use_web_search(query):
                logger.info("="*60)
                logger.info("🌐 PREVIEW: WEB-FIRST EXPANSION")
                logger.info(f"Query: {query[:100]}...")
                logger.info("Using web search for maximum discovery potential")
                logger.info("="*60)
                
                entities = await self.entity_extractor.extract_entities_with_web_search(
                    query,
                    force_web_search=True  # Always force for preview to show full potential
                )
            else:
                # Even in fallback, prefer web search
                logger.info(f"📝 Preview using LLM with web preference for: {query[:100]}...")
                entities = await self.entity_extractor.extract_entities(
                    query,
                    prefer_web_search=True  # Always prefer web for freshness
                )
            
            # Create limited context for preview
            context = RadiatingContext(
                original_query=query,
                depth_limit=min(max_depth, 2),  # Limit preview depth
                relevance_threshold=0.1,
                traversal_strategy=TraversalStrategy.BREADTH_FIRST,
                max_entities_per_level=max_entities
            )
            
            # Convert to RadiatingEntity objects
            starting_entities = [
                RadiatingEntity(
                    id=f"preview_{i}",
                    type=entity.entity_type if hasattr(entity, 'entity_type') else 'unknown',
                    name=entity.text if hasattr(entity, 'text') else '',
                    properties={'confidence': entity.confidence if hasattr(entity, 'confidence') else 0.7,
                                'context': entity.context if hasattr(entity, 'context') else '',
                                'metadata': entity.metadata if hasattr(entity, 'metadata') else {}}
                )
                for i, entity in enumerate(entities[:5])  # Limit starting entities
            ]
            
            # Quick traversal for preview
            if starting_entities:
                graph = await self.traverser.traverse(context, starting_entities)
                
                # Estimate full coverage
                estimated_coverage = RadiatingCoverage(
                    total_entities=len(graph.entities) * 3,  # Rough estimate
                    total_relationships=len(graph.relationships) * 4,
                    max_depth_reached=max_depth,
                    coverage_percentage=min(len(graph.entities) * 10, 100),
                    explored_paths=len(graph.entities) * 2,
                    entity_types=self._count_entity_types(graph.entities),
                    relationship_types=self._count_relationship_types(graph.relationships)
                )
                
                # Convert entities for response
                potential_entities = [
                    ExpandedEntity(
                        id=e.id,
                        name=e.name,
                        type=e.type,
                        properties=e.properties,
                        relevance_score=e.relevance_score,
                        depth=e.depth
                    ).dict()
                    for e in graph.entities[:max_entities]
                ]
            else:
                estimated_coverage = RadiatingCoverage(
                    total_entities=0,
                    total_relationships=0,
                    max_depth_reached=0,
                    coverage_percentage=0.0,
                    explored_paths=0
                )
                potential_entities = []
            
            # Add web-first metadata to preview
            web_entity_count = sum(1 for e in potential_entities 
                                 if e.get('properties', {}).get('source') == 'web' or 
                                    e.get('properties', {}).get('metadata', {}).get('source') == 'web_search')
            
            return {
                'query': query,
                'expanded_queries': query_analysis.get('variations', [query]),
                'potential_entities': potential_entities,
                'estimated_coverage': estimated_coverage.dict(),
                'web_first_enabled': getattr(self.settings, 'web_first_enabled', True),
                'web_entities_discovered': web_entity_count,
                'discovery_method': 'web_search_primary' if self._should_use_web_search(query) else 'llm_with_web_preference'
            }
            
        except Exception as e:
            logger.error(f"Error previewing expansion: {e}")
            return {
                'query': query,
                'expanded_queries': [],
                'potential_entities': [],
                'estimated_coverage': RadiatingCoverage(
                    total_entities=0,
                    total_relationships=0,
                    max_depth_reached=0,
                    coverage_percentage=0.0,
                    explored_paths=0
                ).dict(),
                'error': str(e)
            }
    
    async def _check_neo4j_connection(self) -> bool:
        """Check Neo4j connection health"""
        try:
            neo4j_service = Neo4jService()
            # Would need actual health check implementation
            return True
        except Exception:
            return False
    
    async def _check_redis_connection(self) -> bool:
        """Check Redis connection health"""
        try:
            return self.redis_client.ping()
        except Exception:
            return False
    
    def _count_entity_types(self, entities: List[RadiatingEntity]) -> Dict[str, int]:
        """Count entities by type"""
        type_counts = {}
        for entity in entities:
            entity_type = entity.type or 'unknown'
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        return type_counts
    
    def _count_relationship_types(self, relationships: List[Any]) -> Dict[str, int]:
        """Count relationships by type"""
        type_counts = {}
        for rel in relationships:
            rel_type = getattr(rel, 'type', 'unknown')
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
        return type_counts


# Global service instance
_radiating_service = None


def get_radiating_service() -> RadiatingService:
    """Get the global RadiatingService instance"""
    global _radiating_service
    if _radiating_service is None:
        _radiating_service = RadiatingService()
    return _radiating_service