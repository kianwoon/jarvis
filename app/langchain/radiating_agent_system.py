"""
Radiating Agent System

Specialized agent that integrates with the dynamic agent system to provide
radiating coverage capabilities for knowledge graph exploration.
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
import uuid

from app.core.llm_settings_cache import get_llm_settings
from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig
from app.services.radiating.engine.radiating_traverser import RadiatingTraverser
from app.services.radiating.models.radiating_context import RadiatingContext, TraversalStrategy
from app.services.radiating.models.radiating_entity import RadiatingEntity
from app.services.radiating.models.radiating_graph import RadiatingGraph
from app.services.radiating.query_expansion.query_analyzer import QueryAnalyzer
from app.services.radiating.query_expansion.expansion_strategy import (
    ExpansionStrategy,
    SemanticExpansionStrategy,
    HierarchicalExpansionStrategy,
    AdaptiveExpansionStrategy
)
from app.services.radiating.query_expansion.result_synthesizer import ResultSynthesizer, RadiatingResult, OutputFormat
from app.services.radiating.extraction.universal_entity_extractor import UniversalEntityExtractor
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


class RadiatingAgent:
    """
    Agent that specializes in radiating coverage exploration.
    Integrates with the existing dynamic agent system infrastructure.
    """
    
    def __init__(self, trace: Optional[str] = None, radiating_config: Optional[Dict[str, Any]] = None):
        """Initialize the Radiating Agent
        
        Args:
            trace: Optional trace ID for tracking
            radiating_config: Optional configuration including expansion_strategy selection
        """
        self.trace = trace or str(uuid.uuid4())
        self.llm = None
        self.traverser = RadiatingTraverser()
        self.query_analyzer = QueryAnalyzer()
        
        # Initialize expansion strategy based on config or default to AdaptiveExpansionStrategy
        self._initialize_expansion_strategy(radiating_config)
        
        self.result_synthesizer = ResultSynthesizer()
        self.entity_extractor = UniversalEntityExtractor()
        self.redis_client = get_redis_client()
        
        # Agent metadata
        self.agent_id = f"radiating_agent_{self.trace}"
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # Performance tracking
        self.metrics = {
            'queries_processed': 0,
            'entities_discovered': 0,
            'relationships_found': 0,
            'avg_response_time': 0,
            'total_processing_time': 0
        }
        
        # Initialize LLM
        self._initialize_llm()
    
    def _initialize_expansion_strategy(self, radiating_config: Optional[Dict[str, Any]] = None):
        """Initialize the expansion strategy based on configuration
        
        Args:
            radiating_config: Optional configuration dictionary that may contain 'expansion_strategy'
        """
        # Get strategy type from config or use default
        strategy_type = 'adaptive'  # Default strategy
        
        if radiating_config and 'expansion_strategy' in radiating_config:
            strategy_type = radiating_config['expansion_strategy'].lower()
        else:
            # Try to get from radiating settings
            try:
                from app.core.radiating_settings_cache import get_radiating_config
                config = get_radiating_config()
                strategy_type = config.get('expansion_strategy', 'adaptive').lower()
            except Exception as e:
                logger.debug(f"Could not load radiating config, using default: {e}")
        
        # Initialize the appropriate strategy
        try:
            if strategy_type == 'semantic':
                self.expansion_strategy = SemanticExpansionStrategy()
                logger.info("Initialized SemanticExpansionStrategy")
            elif strategy_type == 'hierarchical':
                self.expansion_strategy = HierarchicalExpansionStrategy()
                logger.info("Initialized HierarchicalExpansionStrategy")
            elif strategy_type == 'adaptive':
                self.expansion_strategy = AdaptiveExpansionStrategy()
                logger.info("Initialized AdaptiveExpansionStrategy")
            else:
                # Default to adaptive if unknown strategy specified
                logger.warning(f"Unknown expansion strategy '{strategy_type}', defaulting to AdaptiveExpansionStrategy")
                self.expansion_strategy = AdaptiveExpansionStrategy()
        except Exception as e:
            logger.error(f"Failed to initialize expansion strategy: {e}, using AdaptiveExpansionStrategy as fallback")
            self.expansion_strategy = AdaptiveExpansionStrategy()
    
    def _initialize_llm(self):
        """Initialize the LLM with current settings"""
        try:
            llm_settings = get_llm_settings()
            main_llm_config = llm_settings.get('main_llm', {})
            
            # Store system prompt separately as it's not part of LLMConfig
            self.system_prompt = main_llm_config.get('system_prompt', '')
            
            config = LLMConfig(
                model_name=main_llm_config.get('model', 'qwen2.5:32b'),
                temperature=main_llm_config.get('temperature', 0.7),
                max_tokens=main_llm_config.get('max_tokens', 4096)
            )
            
            self.llm = OllamaLLM(config=config)
            logger.info(f"Initialized RadiatingAgent LLM with model: {config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            # Fallback to default configuration
            self.system_prompt = ''
            config = LLMConfig(
                model_name='qwen2.5:32b',
                temperature=0.7,
                max_tokens=4096
            )
            self.llm = OllamaLLM(config=config)
    
    async def process_with_radiation(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        stream: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main processing method with radiating coverage.
        
        Args:
            query: The user query to process
            context: Optional context information
            stream: Whether to stream responses
            
        Yields:
            Dict containing response chunks and metadata
        """
        start_time = datetime.now()
        self.metrics['queries_processed'] += 1
        self.last_activity = datetime.now()
        
        try:
            # Step 1: Analyze and expand query
            yield {
                'type': 'status',
                'message': 'Analyzing query and preparing radiating exploration...',
                'timestamp': datetime.now().isoformat()
            }
            
            expanded_queries = await self.expand_query(query, context)
            
            # Step 2: Extract entities from query
            entities = await self.entity_extractor.extract_entities(query)
            starting_entities = [
                RadiatingEntity(
                    text=entity.text if hasattr(entity, 'text') else '',
                    label=entity.entity_type if hasattr(entity, 'entity_type') else 'unknown',
                    start_char=entity.start_char if hasattr(entity, 'start_char') else 0,
                    end_char=entity.end_char if hasattr(entity, 'end_char') else len(entity.text if hasattr(entity, 'text') else ''),
                    confidence=entity.confidence if hasattr(entity, 'confidence') else 0.7,
                    canonical_form=entity.text if hasattr(entity, 'text') else '',
                    properties={
                        'id': f"entity_{i}",
                        'context': entity.context if hasattr(entity, 'context') else '',
                        'metadata': entity.metadata if hasattr(entity, 'metadata') else {}
                    }
                )
                for i, entity in enumerate(entities)
            ]
            
            # Step 3: Create radiating context
            # Normalize the strategy string to handle frontend variations (e.g., "breadth-first" -> "breadth_first")
            strategy_normalized = normalize_strategy_string(context.get('strategy', 'hybrid')) if context else 'hybrid'
            radiating_context = RadiatingContext(
                original_query=query,
                depth_limit=context.get('max_depth', 3) if context else 3,
                relevance_threshold=context.get('relevance_threshold', 0.1) if context else 0.1,
                traversal_strategy=TraversalStrategy(strategy_normalized)
            )
            
            # Step 4: Execute radiating traversal
            yield {
                'type': 'status',
                'message': f'Starting radiating traversal with {len(starting_entities)} seed entities...',
                'timestamp': datetime.now().isoformat()
            }
            
            radiating_graph = await self.traverse_knowledge(
                radiating_context,
                starting_entities
            )
            
            self.metrics['entities_discovered'] += len(radiating_graph.entities)
            self.metrics['relationships_found'] += len(radiating_graph.relationships)
            
            # Step 5: Synthesize results
            yield {
                'type': 'status',
                'message': 'Synthesizing discovered knowledge...',
                'timestamp': datetime.now().isoformat()
            }
            
            synthesized_results = await self.synthesize_results(
                query,
                radiating_graph,
                expanded_queries
            )
            
            # Step 6: Generate response using LLM
            if stream:
                async for chunk in self._generate_streaming_response(
                    query,
                    synthesized_results,
                    radiating_graph
                ):
                    yield chunk
            else:
                response = await self._generate_complete_response(
                    query,
                    synthesized_results,
                    radiating_graph
                )
                yield {
                    'type': 'response',
                    'content': response,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate and update metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics['total_processing_time'] += processing_time
            self.metrics['avg_response_time'] = (
                self.metrics['total_processing_time'] / self.metrics['queries_processed']
            )
            
            # Send final metadata
            yield {
                'type': 'metadata',
                'entities_discovered': len(radiating_graph.entities),
                'relationships_found': len(radiating_graph.relationships),
                'processing_time_ms': processing_time,
                'coverage_depth': radiating_graph.metadata.get('max_depth_reached', 0),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in radiating processing: {e}")
            yield {
                'type': 'error',
                'message': f'Error during radiating exploration: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def expand_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Expand the query using the query expansion system.
        
        Args:
            query: Original query
            context: Optional context
            
        Returns:
            List of expanded queries
        """
        try:
            # Analyze query intent
            query_analysis = await self.query_analyzer.analyze_query(query)
            
            # Generate expanded queries based on analysis
            # The expand() method expects only the analyzed_query argument
            expanded_result = await self.expansion_strategy.expand(query_analysis)
            
            # Extract the expanded terms from the result
            expanded_queries = [query]  # Start with original query
            if expanded_result.expanded_terms:
                expanded_queries.extend(expanded_result.expanded_terms)
            
            logger.info(f"Expanded query into {len(expanded_queries)} variations")
            return expanded_queries
            
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return [query]  # Return original query as fallback
    
    async def traverse_knowledge(
        self,
        context: RadiatingContext,
        starting_entities: List[RadiatingEntity]
    ) -> RadiatingGraph:
        """
        Execute knowledge graph traversal with radiating pattern.
        
        Args:
            context: Radiating context with parameters
            starting_entities: Starting entities for traversal
            
        Returns:
            RadiatingGraph with discovered entities and relationships
        """
        try:
            # Execute traversal
            graph = await self.traverser.traverse(context, starting_entities)
            
            logger.info(
                f"Traversal complete: {len(graph.entities)} entities, "
                f"{len(graph.relationships)} relationships discovered"
            )
            
            return graph
            
        except Exception as e:
            logger.error(f"Error during knowledge traversal: {e}")
            # Return empty graph on error
            return RadiatingGraph()
    
    async def synthesize_results(
        self,
        query: str,
        graph: RadiatingGraph,
        expanded_queries: List[str]
    ) -> Dict[str, Any]:
        """
        Synthesize results from radiating traversal.
        
        Args:
            query: Original query
            graph: Discovered knowledge graph
            expanded_queries: Expanded query variations
            
        Returns:
            Synthesized results dictionary
        """
        try:
            # Convert graph entities to RadiatingResult objects
            results = []
            for i, entity in enumerate(graph.entities):
                # Create entity dict from RadiatingEntity
                entity_dict = {
                    'id': getattr(entity, 'entity_id', None) or entity.get_entity_id() if hasattr(entity, 'get_entity_id') else f"entity_{i}",
                    'text': entity.text,
                    'type': entity.label,
                    'confidence': entity.confidence,
                    'properties': getattr(entity, 'properties', {})
                }
                
                # Get relationships for this entity
                entity_relationships = [
                    {
                        'source': rel.source,
                        'target': rel.target,
                        'type': rel.type,
                        'properties': getattr(rel, 'properties', {})
                    }
                    for rel in graph.relationships
                    if hasattr(rel, 'source') and (rel.source == entity_dict['id'] or rel.target == entity_dict['id'])
                ]
                
                result = RadiatingResult(
                    entity=entity_dict,
                    relationships=entity_relationships,
                    relevance_score=getattr(entity, 'relevance_score', 0.5),
                    confidence=entity.confidence,
                    depth=getattr(entity, 'traversal_depth', 0),
                    path=getattr(entity, 'discovery_path', []),
                    source=getattr(entity, 'discovery_source', 'graph_traversal'),
                    metadata=getattr(entity, 'domain_metadata', {})
                )
                results.append(result)
            
            # Use result synthesizer with correct parameters
            synthesized_result = await self.result_synthesizer.synthesize(
                query=query,
                results=results,
                output_format=OutputFormat.SUMMARY
            )
            
            # Convert SynthesizedResult to dictionary format expected by the rest of the code
            return {
                'summary': synthesized_result.summary,
                'key_insights': synthesized_result.key_findings,
                'entity_graph': synthesized_result.entity_graph,
                'relationship_patterns': [
                    f"{rel.get('source', 'Unknown')} -> {rel.get('type', 'relates')} -> {rel.get('target', 'Unknown')}"
                    for rel in synthesized_result.relationships_found[:10]
                ],
                'coverage_metrics': synthesized_result.coverage_metrics,
                'confidence': synthesized_result.confidence,
                'raw_entities': len(graph.entities),
                'raw_relationships': len(graph.relationships)
            }
            
        except Exception as e:
            logger.error(f"Error synthesizing results: {e}")
            return {
                'summary': 'Unable to synthesize results',
                'error': str(e),
                'raw_entities': len(graph.entities),
                'raw_relationships': len(graph.relationships)
            }
    
    async def _generate_streaming_response(
        self,
        query: str,
        synthesized_results: Dict[str, Any],
        graph: RadiatingGraph
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate streaming response using LLM"""
        try:
            # Prepare context for LLM
            llm_context = self._prepare_llm_context(query, synthesized_results, graph)
            
            # Track if we generated any content
            content_generated = False
            
            # Stream response from LLM with system prompt
            async for chunk in self.llm.generate_stream(llm_context, system_prompt=self.system_prompt):
                if chunk and hasattr(chunk, 'text') and chunk.text:
                    content_generated = True
                    yield {
                        'type': 'content',
                        'content': chunk.text,
                        'timestamp': datetime.now().isoformat()
                    }
            
            # If no content was generated, yield the summary as fallback
            if not content_generated:
                fallback_message = synthesized_results.get('summary', 
                    f'Radiating exploration completed. Found {len(graph.entities)} entities and {len(graph.relationships)} relationships.')
                
                logger.warning(f"No content generated from LLM, using fallback summary")
                yield {
                    'type': 'content',
                    'content': fallback_message,
                    'timestamp': datetime.now().isoformat()
                }
                    
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            # Try to provide useful fallback even on error
            fallback_message = synthesized_results.get('summary', 
                f'Radiating exploration found {len(graph.entities)} entities and {len(graph.relationships)} relationships. Error during response generation: {str(e)}')
            yield {
                'type': 'content',
                'content': fallback_message,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _generate_complete_response(
        self,
        query: str,
        synthesized_results: Dict[str, Any],
        graph: RadiatingGraph
    ) -> str:
        """Generate complete response using LLM"""
        try:
            # Prepare context for LLM
            llm_context = self._prepare_llm_context(query, synthesized_results, graph)
            
            # Generate complete response with system prompt
            response = await self.llm.generate(llm_context, system_prompt=self.system_prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            # Fallback to synthesized summary if response is empty
            if not response_text or response_text.strip() == "":
                response_text = synthesized_results.get('summary', 
                    f'Radiating exploration completed. Found {len(graph.entities)} entities and {len(graph.relationships)} relationships.')
                logger.warning(f"Empty response from LLM, using fallback summary")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating complete response: {e}")
            # Provide useful fallback even on error
            fallback_message = synthesized_results.get('summary', 
                f'Radiating exploration found {len(graph.entities)} entities and {len(graph.relationships)} relationships. Error during response generation: {str(e)}')
            return fallback_message
    
    def _prepare_llm_context(
        self,
        query: str,
        synthesized_results: Dict[str, Any],
        graph: RadiatingGraph
    ) -> str:
        """Prepare context string for LLM"""
        context_parts = [
            f"Query: {query}",
            f"\nRadiating Coverage Analysis:",
            f"- Entities discovered: {len(graph.entities)}",
            f"- Relationships found: {len(graph.relationships)}",
            f"- Coverage depth: {graph.metadata.get('max_depth_reached', 0)}",
            f"\nKey Findings:",
        ]
        
        # Add synthesized insights
        if 'key_insights' in synthesized_results:
            for insight in synthesized_results['key_insights'][:5]:
                context_parts.append(f"- {insight}")
        
        # Add entity summary
        if graph.entities:
            context_parts.append(f"\nMain Entities:")
            for entity in graph.entities[:10]:
                context_parts.append(f"- {entity.text} ({entity.label})")
        
        # Add relationship patterns
        if 'relationship_patterns' in synthesized_results:
            context_parts.append(f"\nRelationship Patterns:")
            for pattern in synthesized_results['relationship_patterns'][:5]:
                context_parts.append(f"- {pattern}")
        
        context_parts.append(f"\nBased on this radiating coverage analysis, provide a comprehensive answer to: {query}")
        
        return "\n".join(context_parts)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'agent_id': self.agent_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'metrics': self.metrics,
            'is_active': True,
            'trace': self.trace
        }
    
    async def cleanup(self):
        """Cleanup agent resources"""
        try:
            # Store final metrics in Redis
            await self.redis_client.setex(
                f"radiating_agent_metrics:{self.agent_id}",
                3600,  # Keep for 1 hour
                json.dumps(self.metrics)
            )
            
            logger.info(f"RadiatingAgent {self.agent_id} cleaned up")
            
        except Exception as e:
            logger.error(f"Error during agent cleanup: {e}")


class RadiatingAgentPool:
    """
    Pool manager for RadiatingAgent instances.
    Manages lifecycle and resource allocation.
    """
    
    def __init__(self, max_agents: int = 5):
        """Initialize the agent pool"""
        self.max_agents = max_agents
        self.agents: Dict[str, RadiatingAgent] = {}
        self._lock = asyncio.Lock()
        
    async def get_agent(self, trace: Optional[str] = None, radiating_config: Optional[Dict[str, Any]] = None) -> RadiatingAgent:
        """Get or create a RadiatingAgent
        
        Args:
            trace: Optional trace ID
            radiating_config: Optional configuration for the radiating agent
        """
        async with self._lock:
            # Create new agent if under limit
            if len(self.agents) < self.max_agents:
                agent = RadiatingAgent(trace=trace, radiating_config=radiating_config)
                self.agents[agent.agent_id] = agent
                logger.info(f"Created new RadiatingAgent: {agent.agent_id}")
                return agent
            
            # Find least recently used agent
            lru_agent = min(
                self.agents.values(),
                key=lambda a: a.last_activity
            )
            
            # Reset and return LRU agent
            lru_agent.trace = trace or str(uuid.uuid4())
            lru_agent.last_activity = datetime.now()
            
            logger.info(f"Reusing RadiatingAgent: {lru_agent.agent_id}")
            return lru_agent
    
    async def release_agent(self, agent_id: str):
        """Release an agent back to the pool"""
        async with self._lock:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                await agent.cleanup()
                logger.info(f"Released RadiatingAgent: {agent_id}")
    
    async def cleanup_all(self):
        """Cleanup all agents in the pool"""
        async with self._lock:
            for agent in self.agents.values():
                await agent.cleanup()
            self.agents.clear()
            logger.info("Cleaned up all RadiatingAgents")


# Global agent pool instance
_radiating_agent_pool = None


def get_radiating_agent_pool() -> RadiatingAgentPool:
    """Get the global RadiatingAgent pool"""
    global _radiating_agent_pool
    if _radiating_agent_pool is None:
        _radiating_agent_pool = RadiatingAgentPool()
    return _radiating_agent_pool