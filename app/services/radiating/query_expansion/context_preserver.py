"""
Context Preserver for Radiating Coverage System

Maintains original query intent during expansion and prevents drift from the original topic.
Tracks expansion history and ensures relevance is maintained throughout the radiating process.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

from app.core.radiating_settings_cache import get_query_expansion_config

logger = logging.getLogger(__name__)

@dataclass
class ExpansionContext:
    """Represents the context of an expansion operation"""
    query_id: str
    original_query: str
    original_intent: str
    original_entities: List[Dict[str, Any]]
    original_domains: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
@dataclass
class ExpansionStep:
    """Represents a single step in the expansion history"""
    step_number: int
    expanded_from: str
    expanded_to: List[str]
    expansion_type: str
    relevance_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ContextPreserver:
    """
    Preserves query context during expansion to prevent topic drift.
    Maintains expansion history and validates relevance at each step.
    """
    
    def __init__(self):
        self.config = get_query_expansion_config()
        self.contexts: Dict[str, ExpansionContext] = {}
        self.expansion_histories: Dict[str, List[ExpansionStep]] = {}
        self.drift_thresholds: Dict[str, float] = {}
        
        # Initialize drift detection settings
        self.max_drift = 0.3  # Maximum allowed drift from original context
        self.relevance_threshold = self.config.get('confidence_threshold', 0.6)
    
    def create_context(
        self,
        query: str,
        intent: str,
        entities: List[Dict[str, Any]],
        domains: List[str]
    ) -> str:
        """
        Create a new expansion context for a query.
        
        Args:
            query: Original query text
            intent: Query intent
            entities: Original entities extracted
            domains: Domain hints
            
        Returns:
            Context ID for tracking
        """
        # Generate unique ID for this context
        query_id = self._generate_query_id(query)
        
        # Create context
        context = ExpansionContext(
            query_id=query_id,
            original_query=query,
            original_intent=intent,
            original_entities=entities,
            original_domains=domains
        )
        
        # Store context
        self.contexts[query_id] = context
        self.expansion_histories[query_id] = []
        self.drift_thresholds[query_id] = self.max_drift
        
        logger.debug(f"Created context {query_id} for query: {query[:50]}...")
        
        return query_id
    
    def validate_expansion(
        self,
        query_id: str,
        expanded_terms: List[str],
        expansion_type: str
    ) -> Tuple[List[str], float]:
        """
        Validate expanded terms against original context.
        
        Args:
            query_id: Context ID
            expanded_terms: Terms to validate
            expansion_type: Type of expansion performed
            
        Returns:
            Tuple of (filtered_terms, average_relevance_score)
        """
        context = self.contexts.get(query_id)
        if not context:
            logger.warning(f"No context found for query_id: {query_id}")
            return expanded_terms, 1.0
        
        # Calculate relevance for each term
        validated_terms = []
        relevance_scores = []
        
        for term in expanded_terms:
            relevance = self._calculate_relevance(term, context)
            
            if relevance >= self.relevance_threshold:
                validated_terms.append(term)
                relevance_scores.append(relevance)
            else:
                logger.debug(f"Filtered out low-relevance term: {term} (score: {relevance})")
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        # Record expansion step
        if validated_terms:
            self._record_expansion(
                query_id,
                context.original_query,
                validated_terms,
                expansion_type,
                avg_relevance
            )
        
        return validated_terms, avg_relevance
    
    def check_drift(self, query_id: str) -> float:
        """
        Check how much the expansion has drifted from original context.
        
        Args:
            query_id: Context ID
            
        Returns:
            Drift score (0.0 = no drift, 1.0 = complete drift)
        """
        context = self.contexts.get(query_id)
        history = self.expansion_histories.get(query_id, [])
        
        if not context or not history:
            return 0.0
        
        # Calculate drift based on expansion history
        total_steps = len(history)
        if total_steps == 0:
            return 0.0
        
        # Average relevance decay over steps
        relevance_decay = 1.0
        for step in history:
            relevance_decay *= step.relevance_score
        
        # Calculate drift as inverse of relevance
        drift = 1.0 - relevance_decay
        
        # Apply exponential penalty for too many steps
        if total_steps > 5:
            drift += (total_steps - 5) * 0.05
        
        return min(drift, 1.0)
    
    def get_expansion_history(self, query_id: str) -> List[ExpansionStep]:
        """
        Get the expansion history for a query.
        
        Args:
            query_id: Context ID
            
        Returns:
            List of expansion steps
        """
        return self.expansion_histories.get(query_id, [])
    
    def should_continue_expansion(self, query_id: str) -> bool:
        """
        Determine if expansion should continue based on drift and history.
        
        Args:
            query_id: Context ID
            
        Returns:
            True if expansion should continue, False otherwise
        """
        # Check drift
        drift = self.check_drift(query_id)
        if drift > self.drift_thresholds.get(query_id, self.max_drift):
            logger.info(f"Stopping expansion due to high drift: {drift}")
            return False
        
        # Check expansion depth
        history = self.expansion_histories.get(query_id, [])
        max_steps = self.config.get('max_expansions', 5)
        
        if len(history) >= max_steps:
            logger.info(f"Stopping expansion after {len(history)} steps")
            return False
        
        # Check if preserve_context is enabled
        if not self.config.get('preserve_context', True):
            return True
        
        # Check relevance trend
        if len(history) >= 3:
            recent_relevance = [step.relevance_score for step in history[-3:]]
            avg_recent = sum(recent_relevance) / len(recent_relevance)
            
            if avg_recent < 0.5:
                logger.info(f"Stopping expansion due to low recent relevance: {avg_recent}")
                return False
        
        return True
    
    def _calculate_relevance(self, term: str, context: ExpansionContext) -> float:
        """
        Calculate relevance of a term to the original context.
        
        This is a simplified implementation. In production, this could use:
        - Semantic similarity with embeddings
        - Domain-specific relevance scoring
        - LLM-based relevance assessment
        """
        
        # Simple heuristic-based relevance
        relevance = 0.5  # Base relevance
        
        # Check if term relates to original entities
        term_lower = term.lower()
        for entity in context.original_entities:
            entity_text = entity.get('text', '').lower()
            if entity_text in term_lower or term_lower in entity_text:
                relevance += 0.3
                break
        
        # Check if term relates to original domains
        for domain in context.original_domains:
            if domain.lower() in term_lower:
                relevance += 0.2
                break
        
        # Check if term appears in original query
        if term_lower in context.original_query.lower():
            relevance += 0.4
        
        # Apply decay based on expansion history length
        history_length = len(self.expansion_histories.get(context.query_id, []))
        decay_factor = 0.95 ** history_length
        relevance *= decay_factor
        
        return min(relevance, 1.0)
    
    def _record_expansion(
        self,
        query_id: str,
        expanded_from: str,
        expanded_to: List[str],
        expansion_type: str,
        relevance_score: float
    ):
        """Record an expansion step in history"""
        
        history = self.expansion_histories.get(query_id, [])
        
        step = ExpansionStep(
            step_number=len(history) + 1,
            expanded_from=expanded_from,
            expanded_to=expanded_to,
            expansion_type=expansion_type,
            relevance_score=relevance_score
        )
        
        history.append(step)
        self.expansion_histories[query_id] = history
        
        logger.debug(f"Recorded expansion step {step.step_number} for query {query_id}")
    
    def _generate_query_id(self, query: str) -> str:
        """Generate a unique ID for a query"""
        timestamp = datetime.now().isoformat()
        hash_input = f"{query}_{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def cleanup_old_contexts(self, max_age_seconds: int = 3600):
        """
        Clean up old contexts to prevent memory buildup.
        
        Args:
            max_age_seconds: Maximum age of contexts to keep
        """
        current_time = datetime.now()
        contexts_to_remove = []
        
        for query_id, context in self.contexts.items():
            age = (current_time - context.timestamp).total_seconds()
            if age > max_age_seconds:
                contexts_to_remove.append(query_id)
        
        for query_id in contexts_to_remove:
            del self.contexts[query_id]
            if query_id in self.expansion_histories:
                del self.expansion_histories[query_id]
            if query_id in self.drift_thresholds:
                del self.drift_thresholds[query_id]
        
        if contexts_to_remove:
            logger.info(f"Cleaned up {len(contexts_to_remove)} old contexts")
    
    def get_context_summary(self, query_id: str) -> Dict[str, Any]:
        """
        Get a summary of the expansion context.
        
        Args:
            query_id: Context ID
            
        Returns:
            Dictionary with context summary
        """
        context = self.contexts.get(query_id)
        history = self.expansion_histories.get(query_id, [])
        
        if not context:
            return {}
        
        # Collect all expanded terms
        all_expanded_terms = []
        for step in history:
            all_expanded_terms.extend(step.expanded_to)
        
        return {
            'query_id': query_id,
            'original_query': context.original_query,
            'original_intent': context.original_intent,
            'original_entities': context.original_entities,
            'original_domains': context.original_domains,
            'expansion_steps': len(history),
            'total_expanded_terms': len(set(all_expanded_terms)),
            'unique_expanded_terms': list(set(all_expanded_terms)),
            'current_drift': self.check_drift(query_id),
            'should_continue': self.should_continue_expansion(query_id),
            'created_at': context.timestamp.isoformat()
        }