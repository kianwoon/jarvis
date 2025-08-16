"""
Query Analyzer for Radiating Coverage System

Analyzes user queries to extract key entities, identify intent, and detect domain hints.
Uses LLM intelligence for sophisticated query understanding without hardcoded rules.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from app.core.radiating_settings_cache import get_radiating_settings, get_query_expansion_config, get_prompt

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
                settings = get_radiating_settings()
                model_config = settings.get('model_config', {})
                
                # Initialize JarvisLLM with max_tokens from config
                max_tokens = model_config.get('max_tokens', 4096)
                self.llm_client = JarvisLLM(mode='non-thinking', max_tokens=max_tokens)
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                self.llm_client = None
    
    async def analyze_query(self, query: str) -> AnalyzedQuery:
        """
        Analyze a query to extract key information.
        
        Args:
            query: The user's query text
            
        Returns:
            AnalyzedQuery object with extracted information
        """
        try:
            # Extract entities using LLM
            entities = await self._extract_entities(query)
            
            # Identify query intent
            intent = await self._identify_intent(query, entities)
            
            # Extract domain hints
            domain_hints = await self._extract_domain_hints(query)
            
            # Extract temporal context if present
            temporal_context = await self._extract_temporal_context(query)
            
            # Generate expansion hints
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
            """Extract key entities from the following query. For each entity, identify:
        1. The entity text
        2. The entity type (Person, Organization, Location, Concept, Event, etc.)
        3. Confidence score (0.0 to 1.0)
        
        Query: {query}
        
        Return as JSON array..."""
        )
        
        prompt = prompt_template.format(query=query)
        
        try:
            response = await self.llm_client.invoke(prompt)
            entities = json.loads(response)
            
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
            response = await self.llm_client.invoke(prompt)
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
            """Identify the knowledge domains relevant to this query.
        Examples: Technology, Business, Science, Medicine, Finance, Education, etc.
        
        Query: {query}
        
        Return as JSON array of domain names (max 5)..."""
        )
        
        prompt = prompt_template.format(query=query)
        
        try:
            response = await self.llm_client.invoke(prompt)
            domains = json.loads(response)
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
            response = await self.llm_client.invoke(prompt)
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