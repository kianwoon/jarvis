"""
Expansion Strategy for Radiating Coverage System

Implements different strategies for query expansion based on query type and context.
Supports adaptive strategy selection and configuration management.
"""

import logging
from typing import Dict, List, Any, Optional, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
import asyncio
import re

from app.core.radiating_settings_cache import get_query_expansion_config, get_radiating_settings, get_prompt
from app.core.timeout_settings_cache import (
    get_concept_expansion_timeout,
    get_radiating_max_retries,
    get_radiating_retry_delay
)
from app.services.radiating.circuit_breaker import CircuitBreakerOpen, get_global_llm_circuit_breaker
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
                from app.core.radiating_settings_cache import get_model_config
                
                # Get model configuration from settings
                model_config = get_model_config()
                
                # Initialize JarvisLLM with configuration from settings
                self.llm_client = JarvisLLM(
                    model=model_config.get('model', 'llama3.1:8b'),
                    mode=model_config.get('llm_mode', 'non-thinking'),  # Use mode from config
                    max_tokens=model_config.get('max_tokens', 4096),
                    temperature=model_config.get('temperature', 0.7),
                    model_server=model_config['model_server']  # Required, no fallback
                )
                
                # Set a JSON-only system prompt for radiating
                self.llm_client.system_prompt = "You are a JSON-only API for query expansion. You must ALWAYS respond with valid JSON and nothing else. Never include explanations, markdown, or any text outside the JSON structure."
                    
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
        """Find semantically related terms and entities with retry logic"""
        
        if not self.llm_client:
            return self._simple_semantic_extraction(entity, entity_type)
        
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

Return ONLY a valid JSON object with this exact structure:
{{
  "terms": ["related term 1", "related term 2"],
  "entities": [
    {{
      "text": "Related Entity",
      "type": "EntityType",
      "confidence": 0.8,
      "relationship": "how it relates"
    }}
  ]
}}

CRITICAL: You MUST return ONLY the JSON object above, with NO additional text, NO explanations, NO markdown formatting. Start your response with { and end with }."""
        )
        
        prompt = prompt_template.format(
            entity=entity,
            entity_type=entity_type,
            domain_context=domain_context
        )
        
        try:
            import json
            # Add JSON instruction to prompt
            json_prompt = prompt + "\n\nREMEMBER: Respond with ONLY the JSON, no other text."
            
            # Use circuit breaker for LLM call
            circuit_breaker = get_global_llm_circuit_breaker()
            try:
                response = await circuit_breaker.call(self.llm_client.invoke, json_prompt)
            except CircuitBreakerOpen:
                logger.warning(f"[ExpansionStrategy] Circuit breaker open for semantic expansion of {entity}")
                return self._simple_semantic_extraction(entity, entity_type)
            
            # Handle empty or invalid response
            if not response or response.strip() == "" or response.strip() == "{}":
                logger.warning(f"[ExpansionStrategy] Empty response for semantic expansion of {entity}")
                return {'terms': [], 'entities': []}
            
            try:
                result = json.loads(response)
                # Ensure result has expected structure
                if not isinstance(result, dict):
                    # If it's a list, wrap it
                    if isinstance(result, list):
                        result = {'terms': result, 'entities': []}
                    else:
                        logger.error(f"[ExpansionStrategy] Expected dict but got {type(result)}")
                        return {'terms': [], 'entities': []}
                # Ensure keys exist
                if 'terms' not in result:
                    result['terms'] = []
                if 'entities' not in result:
                    result['entities'] = []
            except json.JSONDecodeError as e:
                logger.error(f"[ExpansionStrategy] Invalid JSON response: {response[:200]}")
                return {'terms': [], 'entities': []}
            return result
        except Exception as e:
            logger.debug(f"Semantic expansion failed for {entity}: {e}")
            return {'terms': [], 'entities': []}
    
    def _simple_semantic_extraction(self, entity: str, entity_type: str) -> Dict[str, List]:
        """Simple semantic extraction without LLM as fallback"""
        terms = []
        entities = []
        
        # Simple semantic expansion based on common patterns
        entity_lower = entity.lower()
        
        # Technology-related expansions
        tech_expansions = {
            'python': ['programming', 'scripting', 'development'],
            'javascript': ['web development', 'frontend', 'node.js'],
            'react': ['frontend framework', 'ui library', 'components'],
            'database': ['data storage', 'sql', 'nosql'],
            'api': ['interface', 'endpoint', 'rest'],
            'cloud': ['aws', 'azure', 'gcp', 'hosting'],
            'ai': ['machine learning', 'deep learning', 'neural networks'],
            'llm': ['language model', 'gpt', 'transformer'],
        }
        
        for key, expansions in tech_expansions.items():
            if key in entity_lower:
                terms.extend(expansions[:2])
                break
        
        # Type-based expansions
        if entity_type:
            type_lower = entity_type.lower()
            if 'person' in type_lower:
                terms.extend(['professional', 'expert'])
            elif 'company' in type_lower or 'organization' in type_lower:
                terms.extend(['business', 'enterprise'])
            elif 'technology' in type_lower or 'tool' in type_lower:
                terms.extend(['software', 'platform'])
            elif 'concept' in type_lower:
                terms.extend(['idea', 'principle'])
        
        return {'terms': terms[:3], 'entities': entities}
    
    async def _expand_concepts(self, query: str, domains: List[str]) -> List[str]:
        """Expand conceptual understanding of the query with retry logic"""
        
        if not self.llm_client:
            return self._simple_concept_extraction(query, domains)
        
        max_retries = get_radiating_max_retries()
        base_delay = get_radiating_retry_delay()
        timeout = get_concept_expansion_timeout()
        
        for attempt in range(max_retries):
            try:
                # Try LLM expansion with timeout
                result = await asyncio.wait_for(
                    self._expand_concepts_with_llm(query, domains),
                    timeout=timeout
                )
                if result:  # Success
                    return result
                    
            except asyncio.TimeoutError:
                logger.warning(f"[ExpansionStrategy] Concept expansion timeout (attempt {attempt + 1}/{max_retries})")
                
            except Exception as e:
                logger.warning(f"[ExpansionStrategy] Concept expansion failed (attempt {attempt + 1}/{max_retries}): {e}")
            
            # Exponential backoff if not the last attempt
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info(f"[ExpansionStrategy] Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
        
        # All retries failed, fall back to simple extraction
        logger.info("[ExpansionStrategy] Falling back to simple concept extraction")
        return self._simple_concept_extraction(query, domains)
    
    async def _expand_concepts_with_llm(self, query: str, domains: List[str]) -> List[str]:
        """Internal method to expand concepts using LLM with circuit breaker"""
        
        domain_context = f"focusing on {', '.join(domains)}" if domains else ""
        
        # Get prompt template from settings
        prompt_template = get_prompt(
            'expansion_strategy',
            'concept_expansion',
            # Fallback prompt if not in database
            """Identify related concepts and topics for this query:

Query: {query}
{domain_context}

Return ONLY a valid JSON array of related concepts (max 5):
["concept1", "concept2", "concept3"]

CRITICAL: You MUST return ONLY the JSON array above, with NO additional text, NO explanations, NO markdown formatting. Start your response with [ and end with ]."""
        )
        
        prompt = prompt_template.format(
            query=query,
            domain_context=domain_context
        )
        
        import json
        # Add JSON instruction to prompt
        json_prompt = prompt + "\n\nREMEMBER: Respond with ONLY the JSON, no other text."
        
        # Use circuit breaker for LLM call
        circuit_breaker = get_global_llm_circuit_breaker()
        try:
            response = await circuit_breaker.call(self.llm_client.invoke, json_prompt)
        except CircuitBreakerOpen:
            logger.warning("[ExpansionStrategy] Circuit breaker is open, skipping LLM call")
            return []
        
        # Try to extract JSON from response
        concepts = self._extract_json_from_response(response, expected_type='list')
        
        if concepts is None:
            # Try regex extraction as fallback
            concepts = self._extract_json_with_regex(response, 'list')
        
        if not concepts:
            logger.warning("[ExpansionStrategy] Could not extract concepts from LLM response")
            return []
        
        return concepts[:5]
    
    def _simple_concept_extraction(self, query: str, domains: List[str]) -> List[str]:
        """Simple concept extraction without LLM as fallback"""
        concepts = []
        query_lower = query.lower()
        
        # Extract key phrases based on patterns
        # Look for "about X", "related to X", "X and Y", etc.
        patterns = [
            r'about\s+(\w+(?:\s+\w+)*)',
            r'related\s+to\s+(\w+(?:\s+\w+)*)',
            r'regarding\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)',
            r'between\s+(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)',
            r'from\s+(\w+(?:\s+\w+)*)\s+to\s+(\w+(?:\s+\w+)*)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if isinstance(match, tuple):
                    concepts.extend([m.strip() for m in match if m.strip()])
                else:
                    concepts.append(match.strip())
        
        # Add domain-based concepts
        if domains:
            for domain in domains[:2]:
                concepts.append(f"{domain.lower()} concepts")
        
        # Common expansions based on keywords
        keyword_expansions = {
            'ai': ['artificial intelligence', 'machine learning', 'deep learning'],
            'llm': ['large language models', 'language models', 'generative ai'],
            'rag': ['retrieval augmented generation', 'vector search', 'embeddings'],
            'database': ['data storage', 'data management', 'sql'],
            'api': ['web services', 'rest api', 'integration'],
            'cloud': ['cloud computing', 'aws', 'azure', 'gcp'],
            'security': ['cybersecurity', 'data protection', 'encryption'],
            'web': ['web development', 'frontend', 'backend'],
        }
        
        for keyword, expansions in keyword_expansions.items():
            if keyword in query_lower:
                concepts.extend(expansions[:2])
        
        # Remove duplicates and filter
        seen = set()
        unique_concepts = []
        for concept in concepts:
            if concept and concept not in seen and len(concept) > 2:
                seen.add(concept)
                unique_concepts.append(concept)
        
        return unique_concepts[:5]
    
    def _extract_json_from_response(self, response: str, expected_type: str = 'list') -> Optional[Any]:
        """Extract JSON from LLM response with improved handling"""
        if not response or response.strip() == "":
            return None
        
        import json
        
        # Try direct JSON parsing
        try:
            result = json.loads(response)
            if expected_type == 'list' and isinstance(result, list):
                return result
            elif expected_type == 'dict' and isinstance(result, dict):
                return result
            elif expected_type == 'any':
                return result
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in the response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if '```json' in response:
            start = response.find('```json') + 7
            end = response.find('```', start)
            if end != -1:
                response = response[start:end].strip()
        elif '```' in response:
            start = response.find('```') + 3
            end = response.find('```', start)
            if end != -1:
                response = response[start:end].strip()
        
        # Try parsing again
        try:
            result = json.loads(response)
            if expected_type == 'list' and isinstance(result, list):
                return result
            elif expected_type == 'dict' and isinstance(result, dict):
                return result
            elif expected_type == 'any':
                return result
                
            # Handle wrapped responses
            if isinstance(result, dict):
                # Look for common wrapper keys
                for key in ['results', 'items', 'data', 'concepts', 'entities', 'terms']:
                    if key in result and isinstance(result[key], list if expected_type == 'list' else dict):
                        return result[key]
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _extract_json_with_regex(self, response: str, json_type: str = 'list') -> Optional[Any]:
        """Extract JSON using regex as fallback"""
        import json
        
        if json_type == 'list':
            # Look for array pattern
            pattern = r'\[.*?\]'
        else:
            # Look for object pattern
            pattern = r'\{.*?\}'
        
        matches = re.findall(pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                result = json.loads(match)
                if json_type == 'list' and isinstance(result, list):
                    return result
                elif json_type == 'dict' and isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue
        
        return None

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
                from app.core.radiating_settings_cache import get_model_config
                
                # Get model configuration from settings
                model_config = get_model_config()
                
                # Initialize JarvisLLM with configuration from settings
                self.llm_client = JarvisLLM(
                    model=model_config.get('model', 'llama3.1:8b'),
                    mode=model_config.get('llm_mode', 'non-thinking'),  # Use mode from config
                    max_tokens=model_config.get('max_tokens', 4096),
                    temperature=model_config.get('temperature', 0.7),
                    model_server=model_config['model_server']  # Required, no fallback
                )
                
                # Set a JSON-only system prompt for radiating
                self.llm_client.system_prompt = "You are a JSON-only API for query expansion. You must ALWAYS respond with valid JSON and nothing else. Never include explanations, markdown, or any text outside the JSON structure."
                    
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

Return ONLY a valid JSON object with this exact structure:
{{
  "parents": ["parent1", "parent2"],
  "children": ["child1", "child2"],
  "siblings": ["sibling1", "sibling2"]
}}

CRITICAL: You MUST return ONLY the JSON object above, with NO additional text, NO explanations, NO markdown formatting. Start your response with { and end with }."""
        )
        
        prompt = prompt_template.format(
            entity=entity,
            entity_type=entity_type
        )
        
        try:
            import json
            # Add JSON instruction to prompt
            json_prompt = prompt + "\n\nREMEMBER: Respond with ONLY the JSON, no other text."
            
            # Use circuit breaker for LLM call
            circuit_breaker = get_global_llm_circuit_breaker()
            try:
                response = await circuit_breaker.call(self.llm_client.invoke, json_prompt)
            except CircuitBreakerOpen:
                logger.warning(f"[HierarchicalExpansion] Circuit breaker open for {entity}")
                return {'parents': [], 'children': [], 'siblings': []}
            
            # Handle empty or invalid response
            if not response or response.strip() == "" or response.strip() == "{}":
                logger.warning(f"[HierarchicalExpansion] Empty response for {entity}")
                return {'parents': [], 'children': [], 'siblings': []}
            
            try:
                hierarchy = json.loads(response)
                # Ensure result has expected structure
                if not isinstance(hierarchy, dict):
                    # If it's a list, create default structure
                    if isinstance(hierarchy, list):
                        hierarchy = {'parents': [], 'children': hierarchy, 'siblings': []}
                    else:
                        logger.error(f"[HierarchicalExpansion] Expected dict but got {type(hierarchy)}")
                        return {'parents': [], 'children': [], 'siblings': []}
                # Ensure keys exist
                if 'parents' not in hierarchy:
                    hierarchy['parents'] = []
                if 'children' not in hierarchy:
                    hierarchy['children'] = []
                if 'siblings' not in hierarchy:
                    hierarchy['siblings'] = []
                return hierarchy
            except json.JSONDecodeError as e:
                logger.error(f"[HierarchicalExpansion] Invalid JSON response: {response[:200]}")
                return {'parents': [], 'children': [], 'siblings': []}
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