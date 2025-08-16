"""
Expansion Strategy for Radiating Coverage System

Implements different strategies for query expansion based on query type and context.
Supports adaptive strategy selection and configuration management.
"""

import logging
from typing import Dict, List, Any, Optional, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.radiating_settings_cache import get_query_expansion_config, get_radiating_settings, get_prompt
from .query_analyzer import QueryIntent, AnalyzedQuery

logger = logging.getLogger(__name__)

@dataclass
class ExpandedQuery:
    """Result of query expansion"""
    original_query: str
    expanded_terms: List[str]
    expanded_entities: List[Dict[str, Any]]
    expansion_type: str
    confidence: float
    metadata: Dict[str, Any]

class ExpansionStrategy(ABC):
    """Abstract base class for expansion strategies"""
    
    def __init__(self):
        self.config = get_query_expansion_config()
        
    @abstractmethod
    async def expand(self, analyzed_query: AnalyzedQuery) -> ExpandedQuery:
        """
        Expand the analyzed query.
        
        Args:
            analyzed_query: The analyzed query to expand
            
        Returns:
            ExpandedQuery with expansion results
        """
        pass
    
    def _merge_expansions(self, *expansion_lists: List[str]) -> List[str]:
        """Merge multiple expansion lists, removing duplicates"""
        seen = set()
        result = []
        
        for expansion_list in expansion_lists:
            for item in expansion_list:
                if item.lower() not in seen:
                    seen.add(item.lower())
                    result.append(item)
        
        return result

class SemanticExpansionStrategy(ExpansionStrategy):
    """
    Semantic expansion using LLM to find related concepts.
    """
    
    def __init__(self, llm_client=None):
        super().__init__()
        self.llm_client = llm_client
        self._init_llm_if_needed()
    
    def _init_llm_if_needed(self):
        """Initialize LLM client if not provided"""
        if not self.llm_client:
            try:
                from app.llm.ollama import JarvisLLM
                settings = get_radiating_settings()
                model_config = settings.get('model_config', {})
                
                # Initialize JarvisLLM with max_tokens from config
                max_tokens = model_config.get('max_tokens', 4096)
                self.llm_client = JarvisLLM(mode='non-thinking', max_tokens=max_tokens)
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                self.llm_client = None
    
    async def expand(self, analyzed_query: AnalyzedQuery) -> ExpandedQuery:
        """Expand query using semantic relationships"""
        
        expanded_terms = []
        expanded_entities = []
        
        # Expand each entity semantically
        for entity in analyzed_query.key_entities:
            related = await self._find_semantic_relations(
                entity['text'], 
                entity.get('type', 'Unknown'),
                analyzed_query.domain_hints
            )
            expanded_terms.extend(related['terms'])
            expanded_entities.extend(related['entities'])
        
        # Expand the overall query concept
        if self.config.get('concept_expansion', True):
            concepts = await self._expand_concepts(
                analyzed_query.original_query,
                analyzed_query.domain_hints
            )
            expanded_terms.extend(concepts)
        
        # Remove duplicates and original terms
        original_terms = {e['text'].lower() for e in analyzed_query.key_entities}
        expanded_terms = [
            term for term in self._merge_expansions(expanded_terms)
            if term.lower() not in original_terms
        ]
        
        return ExpandedQuery(
            original_query=analyzed_query.original_query,
            expanded_terms=expanded_terms[:self.config.get('max_expansions', 5)],
            expanded_entities=expanded_entities[:self.config.get('max_expansions', 5)],
            expansion_type='semantic',
            confidence=analyzed_query.confidence * 0.8,
            metadata={
                'strategy': 'semantic',
                'domains': analyzed_query.domain_hints,
                'intent': analyzed_query.intent.value
            }
        )
    
    async def _find_semantic_relations(
        self, 
        entity: str, 
        entity_type: str,
        domains: List[str]
    ) -> Dict[str, List]:
        """Find semantically related terms and entities"""
        
        if not self.llm_client:
            return {'terms': [], 'entities': []}
        
        domain_context = f"in the context of {', '.join(domains)}" if domains else ""
        
        # Get prompt template from settings
        prompt_template = get_prompt(
            'expansion_strategy',
            'semantic_expansion',
            # Fallback prompt if not in database
            """Find semantically related terms and entities for:
        Entity: {entity}
        Type: {entity_type}
        {domain_context}
        
        Return as JSON..."""
        )
        
        prompt = prompt_template.format(
            entity=entity,
            entity_type=entity_type,
            domain_context=domain_context
        )
        
        try:
            import json
            response = await self.llm_client.invoke(prompt)
            result = json.loads(response)
            return result
        except Exception as e:
            logger.debug(f"Semantic expansion failed for {entity}: {e}")
            return {'terms': [], 'entities': []}
    
    async def _expand_concepts(self, query: str, domains: List[str]) -> List[str]:
        """Expand conceptual understanding of the query"""
        
        if not self.llm_client:
            return []
        
        domain_context = f"focusing on {', '.join(domains)}" if domains else ""
        
        # Get prompt template from settings
        prompt_template = get_prompt(
            'expansion_strategy',
            'concept_expansion',
            # Fallback prompt if not in database
            """Identify related concepts and topics for this query:
        Query: {query}
        {domain_context}
        
        Return as JSON array of related concepts (max 5)..."""
        )
        
        prompt = prompt_template.format(
            query=query,
            domain_context=domain_context
        )
        
        try:
            import json
            response = await self.llm_client.invoke(prompt)
            concepts = json.loads(response)
            return concepts[:5]
        except Exception as e:
            logger.debug(f"Concept expansion failed: {e}")
            return []

class HierarchicalExpansionStrategy(ExpansionStrategy):
    """
    Hierarchical expansion to find parent, child, and sibling concepts.
    """
    
    def __init__(self, llm_client=None):
        super().__init__()
        self.llm_client = llm_client
        self._init_llm_if_needed()
    
    def _init_llm_if_needed(self):
        """Initialize LLM client if not provided"""
        if not self.llm_client:
            try:
                from app.llm.ollama import JarvisLLM
                settings = get_radiating_settings()
                model_config = settings.get('model_config', {})
                
                # Initialize JarvisLLM with max_tokens from config
                max_tokens = model_config.get('max_tokens', 4096)
                self.llm_client = JarvisLLM(mode='non-thinking', max_tokens=max_tokens)
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                self.llm_client = None
    
    async def expand(self, analyzed_query: AnalyzedQuery) -> ExpandedQuery:
        """Expand query using hierarchical relationships"""
        
        expanded_terms = []
        expanded_entities = []
        
        # Expand each entity hierarchically
        for entity in analyzed_query.key_entities:
            hierarchy = await self._find_hierarchy(
                entity['text'],
                entity.get('type', 'Unknown')
            )
            
            expanded_terms.extend(hierarchy.get('parents', []))
            expanded_terms.extend(hierarchy.get('children', []))
            expanded_terms.extend(hierarchy.get('siblings', []))
            
            # Convert to entities
            for parent in hierarchy.get('parents', []):
                expanded_entities.append({
                    'text': parent,
                    'type': entity.get('type', 'Unknown'),
                    'relationship': 'parent_of'
                })
            
            for child in hierarchy.get('children', []):
                expanded_entities.append({
                    'text': child,
                    'type': entity.get('type', 'Unknown'),
                    'relationship': 'child_of'
                })
        
        # Remove duplicates
        expanded_terms = list(set(expanded_terms))
        
        return ExpandedQuery(
            original_query=analyzed_query.original_query,
            expanded_terms=expanded_terms[:self.config.get('max_expansions', 5)],
            expanded_entities=expanded_entities[:self.config.get('max_expansions', 5)],
            expansion_type='hierarchical',
            confidence=analyzed_query.confidence * 0.75,
            metadata={
                'strategy': 'hierarchical',
                'intent': analyzed_query.intent.value
            }
        )
    
    async def _find_hierarchy(self, entity: str, entity_type: str) -> Dict[str, List[str]]:
        """Find hierarchical relationships"""
        
        if not self.llm_client:
            return {'parents': [], 'children': [], 'siblings': []}
        
        # Get prompt template from settings
        prompt_template = get_prompt(
            'expansion_strategy',
            'hierarchical_expansion',
            # Fallback prompt if not in database
            """Find hierarchical relationships for:
        Entity: {entity}
        Type: {entity_type}
        
        Return as JSON..."""
        )
        
        prompt = prompt_template.format(
            entity=entity,
            entity_type=entity_type
        )
        
        try:
            import json
            response = await self.llm_client.invoke(prompt)
            hierarchy = json.loads(response)
            return hierarchy
        except Exception as e:
            logger.debug(f"Hierarchical expansion failed for {entity}: {e}")
            return {'parents': [], 'children': [], 'siblings': []}

class AdaptiveExpansionStrategy(ExpansionStrategy):
    """
    Adaptive strategy that selects the best expansion approach based on query characteristics.
    """
    
    def __init__(self):
        super().__init__()
        self.semantic_strategy = SemanticExpansionStrategy()
        self.hierarchical_strategy = HierarchicalExpansionStrategy()
    
    async def expand(self, analyzed_query: AnalyzedQuery) -> ExpandedQuery:
        """
        Adaptively select and apply the best expansion strategy.
        """
        
        # Select strategy based on query intent
        strategy = self._select_strategy(analyzed_query)
        
        # Apply selected strategy
        if strategy == 'semantic':
            result = await self.semantic_strategy.expand(analyzed_query)
        elif strategy == 'hierarchical':
            result = await self.hierarchical_strategy.expand(analyzed_query)
        elif strategy == 'hybrid':
            # Apply both strategies and merge
            semantic_result = await self.semantic_strategy.expand(analyzed_query)
            hierarchical_result = await self.hierarchical_strategy.expand(analyzed_query)
            result = self._merge_results(semantic_result, hierarchical_result)
        else:
            # Default to semantic
            result = await self.semantic_strategy.expand(analyzed_query)
        
        # Update metadata
        result.metadata['adaptive_strategy'] = strategy
        result.metadata['selection_reason'] = self._get_selection_reason(analyzed_query, strategy)
        
        return result
    
    def _select_strategy(self, analyzed_query: AnalyzedQuery) -> str:
        """Select the best expansion strategy based on query characteristics"""
        
        intent = analyzed_query.intent
        
        # Map intents to strategies
        if intent in [QueryIntent.HIERARCHICAL, QueryIntent.COMPARISON]:
            return 'hierarchical'
        elif intent in [QueryIntent.CONNECTION_FINDING, QueryIntent.CAUSAL]:
            return 'hybrid'
        elif intent in [QueryIntent.EXPLORATION, QueryIntent.COMPREHENSIVE]:
            return 'semantic'
        elif intent == QueryIntent.SPECIFIC:
            # Limited expansion for specific queries
            return 'semantic'
        else:
            # Default to semantic
            return 'semantic'
    
    def _merge_results(
        self, 
        semantic: ExpandedQuery, 
        hierarchical: ExpandedQuery
    ) -> ExpandedQuery:
        """Merge results from multiple strategies"""
        
        # Merge expanded terms
        merged_terms = self._merge_expansions(
            semantic.expanded_terms,
            hierarchical.expanded_terms
        )
        
        # Merge expanded entities
        merged_entities = semantic.expanded_entities + hierarchical.expanded_entities
        
        # Remove duplicate entities based on text
        seen_entities = set()
        unique_entities = []
        for entity in merged_entities:
            if entity['text'] not in seen_entities:
                seen_entities.add(entity['text'])
                unique_entities.append(entity)
        
        # Average confidence
        avg_confidence = (semantic.confidence + hierarchical.confidence) / 2
        
        # Merge metadata
        merged_metadata = {
            **semantic.metadata,
            **hierarchical.metadata,
            'strategy': 'hybrid',
            'strategies_used': ['semantic', 'hierarchical']
        }
        
        return ExpandedQuery(
            original_query=semantic.original_query,
            expanded_terms=merged_terms[:self.config.get('max_expansions', 5)],
            expanded_entities=unique_entities[:self.config.get('max_expansions', 5)],
            expansion_type='hybrid',
            confidence=avg_confidence,
            metadata=merged_metadata
        )
    
    def _get_selection_reason(self, analyzed_query: AnalyzedQuery, strategy: str) -> str:
        """Get explanation for strategy selection"""
        
        intent = analyzed_query.intent
        
        reasons = {
            QueryIntent.HIERARCHICAL: f"Selected {strategy} for hierarchical query intent",
            QueryIntent.COMPARISON: f"Selected {strategy} for comparison query",
            QueryIntent.CONNECTION_FINDING: f"Selected {strategy} to find connections",
            QueryIntent.CAUSAL: f"Selected {strategy} for causal relationship discovery",
            QueryIntent.EXPLORATION: f"Selected {strategy} for exploratory query",
            QueryIntent.COMPREHENSIVE: f"Selected {strategy} for comprehensive search",
            QueryIntent.SPECIFIC: f"Selected {strategy} for specific, targeted query",
            QueryIntent.TEMPORAL: f"Selected {strategy} for temporal query"
        }
        
        return reasons.get(intent, f"Selected {strategy} based on query analysis")