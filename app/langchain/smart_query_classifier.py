"""
Smart Query Classifier for DeepSeek-R1 and other models
Handles simple queries efficiently without triggering unnecessary RAG searches
"""

import re
from enum import Enum
from typing import Tuple, Dict, List


class QueryType(Enum):
    TOOLS = "tools"          # System info, date/time queries
    LLM = "llm"             # General conversation, simple questions
    RAG = "rag"             # Document search, knowledge base queries
    MULTI_AGENT = "multi_agent"  # Complex tasks requiring multiple agents


class SmartQueryClassifier:
    """Pattern-based query classifier for fast, accurate classification"""
    
    def __init__(self):
        # High-confidence patterns for TOOLS queries
        self.tools_patterns = [
            r"\b(what|tell me|show me|get).*(today|current|now).*(date|time|datetime)\b",
            r"\b(date|time|datetime).*(today|current|now)\b",
            r"\bwhat.*(date|time) is it\b",
            r"\b(system|computer|machine).*(info|information|details|specs)\b",
            r"\b(weather|temperature|forecast)\b",
            r"\b(calculate|compute|math)\b.*\d+",
            # MCP tool patterns
            r"\b(send|compose|draft).*(email|message|mail)\b",
            r"\b(create|update|close).*(ticket|issue|jira)\b",
            r"\b(schedule|book|create).*(meeting|appointment|calendar)\b",
            r"\b(search|find|look).*(email|message|mail)\b",
            r"\b(get|fetch|retrieve).*(calendar|schedule|appointments)\b",
        ]
        
        # High-confidence patterns for simple LLM queries
        self.llm_patterns = [
            r"^(hi|hello|hey|good morning|good afternoon|good evening)\b",
            r"^(thanks|thank you|appreciate|grateful)\b",
            r"^(bye|goodbye|see you|farewell)\b",
            r"\bhow are you\b",
            r"\bwhat is your name\b",
            r"\bwho are you\b",
            r"^(yes|no|okay|ok|sure|alright)\b",
            r"\b(explain|describe|tell me about) (how|what|why|when|where)\b(?!.*\b(document|file|report|data|information|knowledge)\b)",
            r"\bwhat is a?\s+\w+\b(?!.*\b(document|file|report)\b)",  # "what is a cat" but not "what is in document"
            r"\b(joke|funny|humor)\b",
            r"\b(translate|translation)\b",
        ]
        
        # High-confidence patterns for RAG queries
        self.rag_patterns = [
            r"\b(search|find|look for|locate).*(document|file|report|paper|article)\b",
            r"\b(document|file|report|paper|article).*(about|contain|mention|discuss)\b",
            r"\bwhat.*(document|file|report|information|data|knowledge).*(say|contain|mention)\b",
            r"\b(retrieve|fetch|get).*(information|data|details).*(from|about)\b",
            r"\b(query|search).*(database|knowledge base|collection)\b",
            r"\bshow me.*(document|file|report|data)\b",
            r"\b(latest|recent|updated).*(report|document|information)\b",
            r"\bfind.*information about\b",
            r"\bsearch.*for.*in\b",
            # Company-specific patterns
            r"\b(our|company|internal).*(product|service|strategy|roadmap|plan)\b",
            r"\bwhat.*(migration|implementation|deployment).*(timeline|schedule|plan)\b",
            r"\bhow long.*(migration|implementation|deployment|project)\b",
            r"\b(benefits|features|advantages).*(our|company).*(product|service)\b",
        ]
        
        # High-confidence patterns for multi-agent queries
        self.multi_agent_patterns = [
            r"\b(analyze|compare|contrast).*(multiple|several|different).*(document|source|report)\b",
            r"\b(complex|complicated).*(analysis|task|problem)\b",
            r"\b(coordinate|orchestrate).*(multiple|several).*(task|action)\b",
            r"\bcombine.*(information|data).*(from|across).*(source|document)\b",
        ]

    def classify(self, query: str) -> Tuple[QueryType, float]:
        """
        Classify a query and return type with confidence score
        
        Returns:
            Tuple of (QueryType, confidence_score)
            confidence_score: 0.0 to 1.0, where > 0.8 is high confidence
        """
        query_lower = query.lower().strip()
        
        # Check patterns in priority order
        pattern_checks = [
            (self.tools_patterns, QueryType.TOOLS),
            (self.multi_agent_patterns, QueryType.MULTI_AGENT),
            (self.rag_patterns, QueryType.RAG),
            (self.llm_patterns, QueryType.LLM),
        ]
        
        for patterns, query_type in pattern_checks:
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    # High confidence for pattern matches
                    return query_type, 0.9
        
        # Default classification based on query characteristics
        # Short queries without document keywords are usually LLM
        if len(query_lower.split()) < 10 and not any(
            keyword in query_lower 
            for keyword in ['document', 'file', 'report', 'search', 'find', 'data']
        ):
            return QueryType.LLM, 0.7
        
        # Longer queries might need RAG
        return QueryType.RAG, 0.5

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