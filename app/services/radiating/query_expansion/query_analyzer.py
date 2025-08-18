"""
Query Analyzer for Radiating Coverage System

Analyzes user queries to extract key entities, identify intent, and detect domain hints.
Uses LLM intelligence for sophisticated query understanding without hardcoded rules.
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from app.core.radiating_settings_cache import get_radiating_settings, get_query_expansion_config, get_prompt
from app.core.timeout_settings_cache import get_query_analysis_timeout

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Types of query intents that affect expansion strategy"""
    EXPLORATION = "exploration"  # Broad, discovery-oriented queries
    CONNECTION_FINDING = "connection_finding"  # Finding relationships between entities
    COMPREHENSIVE = "comprehensive"  # Deep, thorough information gathering
    SPECIFIC = "specific"  # Targeted, narrow queries
    COMPARISON = "comparison"  # Comparing multiple entities
    TEMPORAL = "temporal"  # Time-based queries
    CAUSAL = "causal"  # Cause-effect relationships
    HIERARCHICAL = "hierarchical"  # Parent-child, part-whole relationships

@dataclass
class AnalyzedQuery:
    """Result of query analysis"""
    original_query: str
    key_entities: List[Dict[str, Any]]  # [{"text": str, "type": str, "confidence": float}]
    intent: QueryIntent
    domain_hints: List[str]
    temporal_context: Optional[Dict[str, Any]]
    expansion_hints: Dict[str, Any]
    confidence: float

class QueryAnalyzer:
    """
    Analyzes queries to extract structured information for expansion.
    Uses LLM for intelligent analysis without hardcoded patterns.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the query analyzer.
        
        Args:
            llm_client: Optional LLM client for analysis
        """
        self.config = get_query_expansion_config()
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
                self.llm_client.system_prompt = "You are a JSON-only API that analyzes queries. You must ALWAYS respond with valid JSON and nothing else. Never include explanations, markdown, or any text outside the JSON structure."
                    
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                self.llm_client = None
    
    async def analyze_query(self, query: str) -> AnalyzedQuery:
        """
        Analyze a query to extract key information using parallel processing.
        
        Args:
            query: The user's query text
            
        Returns:
            AnalyzedQuery object with extracted information
        """
        try:
            # Get timeout for analysis
            timeout = get_query_analysis_timeout()
            
            # Run entity extraction, intent identification, and domain extraction in parallel
            # These are independent operations that can be parallelized
            entities_task = asyncio.create_task(self._extract_entities(query))
            domain_hints_task = asyncio.create_task(self._extract_domain_hints(query))
            temporal_context_task = asyncio.create_task(self._extract_temporal_context(query))
            
            # Wait for entities first as intent depends on it (with timeout)
            try:
                entities = await asyncio.wait_for(entities_task, timeout=timeout/3)
            except asyncio.TimeoutError:
                logger.warning("Entity extraction timed out, using fallback")
                entities = self.parse_entities_from_text(query)
            
            # Now identify intent with entities
            intent_task = asyncio.create_task(self._identify_intent(query, entities))
            
            # Wait for remaining tasks with timeout handling
            try:
                domain_hints = await asyncio.wait_for(domain_hints_task, timeout=timeout/3)
            except asyncio.TimeoutError:
                logger.warning("Domain extraction timed out, using empty list")
                domain_hints = []
            
            try:
                temporal_context = await asyncio.wait_for(temporal_context_task, timeout=timeout/3)
            except asyncio.TimeoutError:
                logger.warning("Temporal extraction timed out, using None")
                temporal_context = None
            
            try:
                intent = await asyncio.wait_for(intent_task, timeout=timeout/3)
            except asyncio.TimeoutError:
                logger.warning("Intent identification timed out, using EXPLORATION")
                intent = QueryIntent.EXPLORATION
            
            # Generate expansion hints (depends on previous results)
            expansion_hints = await self._generate_expansion_hints(
                query, entities, intent, domain_hints
            )
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(entities, intent, domain_hints)
            
            return AnalyzedQuery(
                original_query=query,
                key_entities=entities,
                intent=intent,
                domain_hints=domain_hints,
                temporal_context=temporal_context,
                expansion_hints=expansion_hints,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            # Return basic analysis on error
            return AnalyzedQuery(
                original_query=query,
                key_entities=[],
                intent=QueryIntent.EXPLORATION,
                domain_hints=[],
                temporal_context=None,
                expansion_hints={},
                confidence=0.0
            )
    
    async def _extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract key entities from the query using LLM"""
        if not self.llm_client:
            return []
        
        # Get prompt template from settings
        prompt_template = get_prompt(
            'query_analysis',
            'entity_extraction',
            # Fallback prompt if not in database
            """Query: {query}

Extract entities as JSON array:
[{{"text": "React", "type": "Technology", "confidence": 0.9}}]"""
        )
        
        prompt = prompt_template.format(query=query)
        
        try:
            # Add JSON instruction to prompt
            json_prompt = prompt + "\n\nREMEMBER: Respond with ONLY the JSON, no other text."
            response = await self.llm_client.invoke(json_prompt)
            
            # Handle empty or invalid response
            if not response or response.strip() == "" or response.strip() == "[]":
                logger.warning("[QueryAnalyzer] Empty response from LLM for entity extraction")
                return []
            
            try:
                entities = json.loads(response)
                
                # Handle different response formats
                if isinstance(entities, dict):
                    if 'entities' in entities:
                        entities = entities['entities']
                    else:
                        # Try to collect all lists from the dict
                        all_entities = []
                        for key, value in entities.items():
                            if isinstance(value, list) and value:  # Check if list is not empty
                                for item in value:
                                    if isinstance(item, str):
                                        # Convert string to entity dict
                                        all_entities.append({
                                            'text': item,
                                            'type': key.replace('_', ' ').title(),
                                            'confidence': 0.7
                                        })
                                    elif isinstance(item, dict):
                                        all_entities.append(item)
                        if all_entities:
                            entities = all_entities
                        else:
                            # If we got an empty dict structure, return empty list without error
                            logger.debug(f"[QueryAnalyzer] No entities found in response")
                            return []
                
                if not isinstance(entities, list):
                    logger.error(f"[QueryAnalyzer] Expected list but got {type(entities)}")
                    return []
                    
                # Normalize field names
                normalized = []
                for e in entities:
                    normalized.append({
                        'text': e.get('text') or e.get('name', ''),
                        'type': e.get('type', 'Unknown'),
                        'confidence': e.get('confidence', 0.7)
                    })
                entities = normalized
                
            except json.JSONDecodeError as e:
                logger.error(f"[QueryAnalyzer] Invalid JSON response: {response[:200]}")
                return []
            
            # Filter by confidence threshold
            min_confidence = self.config.get('confidence_threshold', 0.6)
            filtered = [e for e in entities if e.get('confidence', 0) >= min_confidence]
            
            return filtered[:self.config.get('max_entities_per_query', 20)]
            
        except Exception as e:
            logger.debug(f"Entity extraction failed: {e}")
            return []
    
    async def _identify_intent(self, query: str, entities: List[Dict]) -> QueryIntent:
        """Identify the intent of the query"""
        if not self.llm_client:
            return QueryIntent.EXPLORATION
        
        # Get prompt template from settings
        prompt_template = get_prompt(
            'query_analysis',
            'intent_identification',
            # Fallback prompt if not in database
            """Identify the primary intent of this query. Choose from:
        - EXPLORATION: Broad discovery queries
        - CONNECTION_FINDING: Finding relationships between things
        - COMPREHENSIVE: Deep, thorough information gathering
        - SPECIFIC: Targeted, narrow queries
        - COMPARISON: Comparing multiple entities
        - TEMPORAL: Time-based queries
        - CAUSAL: Cause-effect relationships
        - HIERARCHICAL: Parent-child or part-whole relationships
        
        Query: {query}
        Entities found: {entities}
        
        Return only the intent type name."""
        )
        
        prompt = prompt_template.format(
            query=query,
            entities=[e['text'] for e in entities]
        )
        
        try:
            # Add JSON instruction to prompt
            json_prompt = prompt + "\n\nREMEMBER: Respond with ONLY the JSON, no other text."
            response = await self.llm_client.invoke(json_prompt)
            intent_str = response.strip().upper()
            
            # Map to enum
            for intent in QueryIntent:
                if intent.name == intent_str:
                    return intent
            
            # Default to exploration if no match
            return QueryIntent.EXPLORATION
            
        except Exception as e:
            logger.debug(f"Intent identification failed: {e}")
            return QueryIntent.EXPLORATION
    
    async def _extract_domain_hints(self, query: str) -> List[str]:
        """Extract domain hints from the query"""
        if not self.llm_client:
            return []
        
        # Get prompt template from settings
        prompt_template = get_prompt(
            'query_analysis',
            'domain_extraction',
            # Fallback prompt if not in database
            """Query: {query}

List relevant domains as JSON array:
["Technology", "Programming"]"""
        )
        
        prompt = prompt_template.format(query=query)
        
        try:
            # Add JSON instruction to prompt
            json_prompt = prompt + "\n\nREMEMBER: Respond with ONLY the JSON, no other text."
            response = await self.llm_client.invoke(json_prompt)
            
            # Handle empty or invalid response
            if not response or response.strip() == "" or response.strip() == "[]":
                logger.warning("[QueryAnalyzer] Empty response from LLM for domain extraction")
                return []
            
            try:
                domains = json.loads(response)
                
                # Handle different response formats
                if isinstance(domains, dict):
                    if 'domains' in domains:
                        domains = domains['domains']
                    else:
                        # Try to extract list from any key
                        for key, value in domains.items():
                            if isinstance(value, list):
                                domains = value
                                break
                        else:
                            logger.error(f"[QueryAnalyzer] Dict response without list: {domains}")
                            return []
                
                if not isinstance(domains, list):
                    logger.error(f"[QueryAnalyzer] Expected list but got {type(domains)}")
                    return []
            except json.JSONDecodeError as e:
                logger.error(f"[QueryAnalyzer] Invalid JSON response: {response[:200]}")
                return []
            return domains[:5]
            
        except Exception as e:
            logger.debug(f"Domain extraction failed: {e}")
            return []
    
    async def _extract_temporal_context(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract temporal context from the query"""
        if not self.llm_client or not self.config.get('temporal_expansion', False):
            return None
        
        # Get prompt template from settings
        prompt_template = get_prompt(
            'query_analysis',
            'temporal_extraction',
            # Fallback prompt if not in database
            """Extract temporal context from this query if present.
        Look for: time periods, dates, relative times (latest, recent, current), etc.
        
        Query: {query}
        
        Return as JSON or null if no temporal context..."""
        )
        
        prompt = prompt_template.format(query=query)
        
        try:
            # Add JSON instruction to prompt
            json_prompt = prompt + "\n\nREMEMBER: Respond with ONLY the JSON, no other text."
            response = await self.llm_client.invoke(json_prompt)
            temporal = json.loads(response)
            return temporal if temporal else None
            
        except Exception as e:
            logger.debug(f"Temporal extraction failed: {e}")
            return None
    
    async def _generate_expansion_hints(
        self, 
        query: str, 
        entities: List[Dict],
        intent: QueryIntent,
        domains: List[str]
    ) -> Dict[str, Any]:
        """Generate hints for query expansion"""
        
        hints = {
            'expand_synonyms': self.config.get('synonym_expansion', True),
            'expand_concepts': self.config.get('concept_expansion', True),
            'expand_hierarchical': self.config.get('hierarchical_expansion', True),
            'suggested_depth': self._suggest_depth(intent),
            'prioritize_entities': self._prioritize_entities(entities),
            'domain_specific': len(domains) > 0,
            'domains': domains
        }
        
        # Adjust based on intent
        if intent == QueryIntent.SPECIFIC:
            hints['suggested_depth'] = min(hints['suggested_depth'], 2)
            hints['expand_concepts'] = False
        elif intent == QueryIntent.COMPREHENSIVE:
            hints['suggested_depth'] = max(hints['suggested_depth'], 4)
            hints['expand_concepts'] = True
        elif intent == QueryIntent.CONNECTION_FINDING:
            hints['focus_on_relationships'] = True
            hints['expand_hierarchical'] = True
        
        return hints
    
    def _suggest_depth(self, intent: QueryIntent) -> int:
        """Suggest radiating depth based on query intent"""
        depth_map = {
            QueryIntent.EXPLORATION: 3,
            QueryIntent.CONNECTION_FINDING: 4,
            QueryIntent.COMPREHENSIVE: 5,
            QueryIntent.SPECIFIC: 2,
            QueryIntent.COMPARISON: 3,
            QueryIntent.TEMPORAL: 3,
            QueryIntent.CAUSAL: 4,
            QueryIntent.HIERARCHICAL: 4
        }
        
        settings = get_radiating_settings()
        suggested = depth_map.get(intent, 3)
        max_depth = settings.get('max_depth', 5)
        
        return min(suggested, max_depth)
    
    def _prioritize_entities(self, entities: List[Dict]) -> List[str]:
        """Prioritize entities for expansion based on confidence"""
        # Sort by confidence and return top entities
        sorted_entities = sorted(
            entities, 
            key=lambda x: x.get('confidence', 0), 
            reverse=True
        )
        
        return [e['text'] for e in sorted_entities[:5]]
    
    def _calculate_confidence(
        self, 
        entities: List[Dict],
        intent: QueryIntent,
        domains: List[str]
    ) -> float:
        """Calculate overall analysis confidence"""
        
        # Base confidence
        confidence = 0.5
        
        # Add confidence based on entities found
        if entities:
            avg_entity_confidence = sum(e.get('confidence', 0) for e in entities) / len(entities)
            confidence += avg_entity_confidence * 0.3
        
        # Add confidence if intent is clear
        if intent != QueryIntent.EXPLORATION:
            confidence += 0.1
        
        # Add confidence if domains identified
        if domains:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def parse_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Simple entity parsing without LLM (fallback method).
        Extracts potential entities based on capitalization and patterns.
        """
        entities = []
        
        # Simple heuristic: look for capitalized sequences
        import re
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, text)
        
        for match in matches:
            # Skip common words
            if match.lower() in ['the', 'this', 'that', 'what', 'when', 'where', 'who', 'how']:
                continue
                
            entities.append({
                'text': match,
                'type': 'Unknown',
                'confidence': 0.5
            })
        
        return entities[:10]  # Limit to 10 entities