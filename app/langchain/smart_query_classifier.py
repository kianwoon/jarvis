"""
Smart Query Classifier for DeepSeek-R1 and other models
Handles simple queries efficiently without triggering unnecessary RAG searches
"""

import re
import yaml
import os
from enum import Enum
from typing import Tuple, Dict, List, Optional
from pathlib import Path


class QueryType(Enum):
    TOOLS = "tools"          # System info, date/time queries
    LLM = "llm"             # General conversation, simple questions
    RAG = "rag"             # Document search, knowledge base queries
    MULTI_AGENT = "multi_agent"  # Complex tasks requiring multiple agents


class SmartQueryClassifier:
    """Pattern-based query classifier for fast, accurate classification"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize classifier with patterns from config file"""
        if config_path is None:
            # Look for config file in same directory as this module
            module_dir = Path(__file__).parent
            config_path = module_dir / "smart_query_patterns.yaml"
        
        self.config_path = config_path
        self.patterns = self._load_patterns()
        self.settings = self.patterns.get('settings', {})
        
    def _load_patterns(self) -> Dict:
        """Load patterns from YAML configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"[WARNING] Failed to load patterns from {self.config_path}: {e}")
            # Return minimal default patterns if file not found
            return {
                "tool_patterns": {
                    "datetime": {
                        "patterns": [r"\b(date|time).*\b(today|now|current)\b"],
                        "confidence": 0.9
                    }
                },
                "llm_patterns": {
                    "general": {
                        "patterns": [r".*"],  # Catch all
                        "confidence": 0.5
                    }
                },
                "settings": {
                    "confidence_threshold": 0.8,
                    "default_classification": "llm",
                    "default_confidence": 0.5
                }
            }
    
    def _compile_patterns(self, pattern_dict: Dict) -> List[Tuple[re.Pattern, float, str]]:
        """Compile patterns from a pattern dictionary"""
        compiled = []
        for category, config in pattern_dict.items():
            if category == 'settings':
                continue
            patterns = config.get('patterns', [])
            confidence = config.get('confidence', 0.8)
            for pattern in patterns:
                try:
                    compiled.append((re.compile(pattern, re.IGNORECASE), confidence, category))
                except re.error as e:
                    print(f"[WARNING] Invalid regex pattern '{pattern}': {e}")
        return compiled

    def classify(self, query: str) -> Tuple[QueryType, float]:
        """
        Classify a query and return type with confidence score
        
        Returns:
            Tuple of (QueryType, confidence_score)
            confidence_score: 0.0 to 1.0, where > 0.8 is high confidence
        """
        query_lower = query.lower().strip()
        
        # Priority order from settings or default
        priority_order = self.settings.get('priority_order', [
            'tool_patterns', 'rag_patterns', 'multi_agent_patterns', 'llm_patterns'
        ])
        
        # Map pattern types to QueryType enum
        type_mapping = {
            'tool_patterns': QueryType.TOOLS,
            'rag_patterns': QueryType.RAG,
            'multi_agent_patterns': QueryType.MULTI_AGENT,
            'llm_patterns': QueryType.LLM
        }
        
        # Check patterns in priority order
        for pattern_type in priority_order:
            if pattern_type not in self.patterns:
                continue
                
            # Compile patterns for this type
            compiled_patterns = self._compile_patterns(self.patterns[pattern_type])
            
            # Check each pattern
            for pattern, confidence, category in compiled_patterns:
                if pattern.search(query_lower):
                    query_type = type_mapping.get(pattern_type, QueryType.LLM)
                    return query_type, confidence
        
        # Default classification based on query characteristics
        default_type = self.settings.get('default_classification', 'llm')
        default_confidence = self.settings.get('default_confidence', 0.5)
        
        # Short queries without document keywords are usually LLM
        if len(query_lower.split()) < 10 and not any(
            keyword in query_lower 
            for keyword in ['document', 'file', 'report', 'search', 'find', 'data']
        ):
            return QueryType.LLM, 0.7
        
        # Map default type string to enum
        default_query_type = QueryType.LLM
        if default_type == 'rag':
            default_query_type = QueryType.RAG
        elif default_type == 'tools':
            default_query_type = QueryType.TOOLS
        
        return default_query_type, default_confidence

    def should_use_llm_classification(self, confidence: float) -> bool:
        """Determine if we should fall back to LLM classification"""
        return confidence < 0.8


# Singleton instance
classifier = SmartQueryClassifier()


def classify_query(query: str) -> Tuple[str, float]:
    """
    Main entry point for query classification
    
    Returns:
        Tuple of (query_type_string, confidence)
    """
    query_type, confidence = classifier.classify(query)
    return query_type.value, confidence


def classify_without_context(query: str) -> Tuple[str, float]:
    """
    Classify query without any conversation context
    This is useful for simple queries that shouldn't be influenced by history
    
    Returns:
        Tuple of (query_type_string, confidence)
    """
    # For simple tool queries, we want high confidence without context
    query_type, confidence = classifier.classify(query)
    
    # Boost confidence for clear tool patterns to avoid LLM overthinking
    if query_type == QueryType.TOOLS and confidence >= 0.9:
        confidence = 0.95  # Very high confidence to bypass LLM
    
    return query_type.value, confidence


def get_classification_explanation(query: str, query_type: str) -> str:
    """Get a brief explanation for why a query was classified a certain way"""
    explanations = {
        "tools": "Query requires system tools or real-time information",
        "llm": "General conversation or simple question that doesn't require document search",
        "rag": "Query requires searching documents or knowledge base",
        "multi_agent": "Complex query requiring coordination of multiple agents"
    }
    return explanations.get(query_type, "Query classification completed")