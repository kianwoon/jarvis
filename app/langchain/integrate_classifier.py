"""
Integration helper to add query classification to existing RAG endpoint.

This module provides a minimal integration that can be added to the existing
service.py without major refactoring.
"""

from typing import Optional, List, Dict, Any
from app.langchain.query_classifier import classify_query, QueryType
import logging

logger = logging.getLogger(__name__)


def should_use_rag(question: str, collections: Optional[List[str]] = None) -> bool:
    """
    Determine if a query should use RAG based on classification.
    
    This is a simple helper that can be called before handle_rag_query
    to avoid unnecessary RAG processing for non-RAG queries.
    """
    context = {
        "has_collections": bool(collections),
        "available_tools": []  # Will be populated if needed
    }
    
    try:
        # Quick classification without LLM fallback for performance
        result = classify_query(question, context, use_llm_fallback=False)
        
        # Use RAG if:
        # 1. Primary type is RAG with good confidence
        # 2. Primary type is HYBRID (might need RAG)
        # 3. Low confidence on other types (might benefit from RAG check)
        
        if result.primary_type == QueryType.RAG and result.confidence > 0.5:
            return True
        
        if result.primary_type == QueryType.HYBRID:
            return True
        
        if result.confidence < 0.4:  # Uncertain classification
            return True  # Check RAG to be safe
        
        # Skip RAG for clear non-RAG queries
        if result.primary_type in [QueryType.DIRECT_LLM, QueryType.TOOLS] and result.confidence > 0.7:
            logger.info(f"Skipping RAG for {result.primary_type.value} query (confidence: {result.confidence:.2f})")
            return False
        
        return True  # Default to checking RAG
        
    except Exception as e:
        logger.error(f"Classification failed: {e}, defaulting to RAG")
        return True  # Default to using RAG on error


def get_query_metadata(question: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get metadata about a query to help with processing decisions.
    
    Returns:
        Dictionary with query metadata including:
        - query_type: The classified type
        - confidence: Classification confidence
        - needs_tools: Whether tools are needed
        - needs_rag: Whether RAG is needed
        - is_large_generation: Whether this needs chunking
        - estimated_tokens: Rough estimate of output size
    """
    from app.core.mcp_tools_cache import get_enabled_mcp_tools
    from app.langchain.service import detect_large_output_potential, get_conversation_history
    
    context = {
        "has_collections": True,  # Assume collections might be available
        "conversation_history": get_conversation_history(conversation_id) if conversation_id else None,
        "available_tools": list(get_enabled_mcp_tools().keys()) if get_enabled_mcp_tools() else []
    }
    
    # Classify with LLM fallback for better accuracy
    result = classify_query(question, context, use_llm_fallback=True)
    
    # Check for large generation
    large_output = detect_large_output_potential(question)
    
    metadata = {
        "query_type": result.primary_type.value,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "needs_tools": result.primary_type == QueryType.TOOLS or (
            result.primary_type == QueryType.HYBRID and 
            any(t[0] == QueryType.TOOLS for t in (result.secondary_types or []))
        ),
        "needs_rag": result.primary_type == QueryType.RAG or (
            result.primary_type == QueryType.HYBRID and
            any(t[0] == QueryType.RAG for t in (result.secondary_types or []))
        ),
        "is_large_generation": result.primary_type == QueryType.LARGE_GENERATION,
        "estimated_items": large_output.get("estimated_items", 0),
        "required_tools": result.metadata.get("required_tools", [])
    }
    
    return metadata


# Minimal patch for existing classify_query_type function
def enhanced_classify_query_type(question: str, llm_cfg: dict) -> str:
    """
    Enhanced version of classify_query_type that uses the new classifier.
    
    This can replace the existing classify_query_type function in service.py
    with minimal changes to the rest of the code.
    """
    try:
        # Get metadata using the new classifier
        metadata = get_query_metadata(question)
        
        # Map to existing type strings
        type_mapping = {
            QueryType.DIRECT_LLM: "LLM",
            QueryType.RAG: "RAG", 
            QueryType.TOOLS: "TOOLS",
            QueryType.LARGE_GENERATION: "LARGE_GENERATION",
            QueryType.HYBRID: "RAG"  # Default hybrid to RAG for compatibility
        }
        
        query_type = metadata.get("query_type", "LLM")
        mapped_type = type_mapping.get(QueryType[query_type.upper()], "LLM")
        
        logger.info(f"Enhanced classification: {query_type} -> {mapped_type} "
                   f"(confidence: {metadata.get('confidence', 0):.2f})")
        
        return mapped_type
        
    except Exception as e:
        logger.error(f"Enhanced classification failed: {e}, falling back to original")
        # Fall back to original implementation
        from app.langchain.service import classify_query_type as original_classify
        return original_classify(question, llm_cfg)


# Simple integration function that can be added to rag_answer
def optimize_rag_processing(question: str, collections: Optional[List[str]] = None) -> Optional[str]:
    """
    Pre-process query to determine optimal handling.
    
    Returns:
        - None: Process normally with RAG
        - "SKIP_RAG": Skip RAG and go directly to LLM
        - "TOOLS_FIRST": Execute tools before RAG
        - "LARGE_GEN": Route to large generation
    """
    metadata = get_query_metadata(question)
    
    # Clear direct LLM queries can skip RAG
    if metadata["query_type"] == "direct_llm" and metadata["confidence"] > 0.8:
        return "SKIP_RAG"
    
    # Large generation should be handled separately
    if metadata["is_large_generation"]:
        return "LARGE_GEN"
    
    # Tool queries might benefit from tool execution first
    if metadata["needs_tools"] and not metadata["needs_rag"]:
        return "TOOLS_FIRST"
    
    return None  # Process normally