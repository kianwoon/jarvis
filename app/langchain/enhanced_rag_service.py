"""
Enhanced RAG Service with Query Classification

This module shows how to integrate the query classifier into the existing
RAG service for intelligent query routing.
"""

import json
import time
from typing import List, Optional, Dict, Any, AsyncGenerator
from app.langchain.query_classifier import classify_query, QueryType
from app.langchain.service import (
    handle_rag_query, build_prompt, get_llm_settings,
    store_conversation_message, get_conversation_history,
    detect_large_output_potential
)
from app.core.mcp_tools_cache import get_enabled_mcp_tools
from app.langchain.multi_agent_system_simple import MultiAgentSystem
import httpx
import logging

logger = logging.getLogger(__name__)


def enhanced_rag_answer(
    question: str,
    thinking: bool = False,
    stream: bool = False,
    conversation_id: str = None,
    use_langgraph: bool = True,
    collections: List[str] = None,
    collection_strategy: str = "auto",
    use_classifier: bool = True
):
    """
    Enhanced RAG answer function with intelligent query classification and routing.
    
    This is a drop-in replacement for the existing rag_answer function that adds
    intelligent query routing based on classification.
    
    Args:
        question: The user's question
        thinking: Whether to enable extended thinking
        stream: Whether to stream the response
        conversation_id: Conversation ID for context
        use_langgraph: Whether to use LangGraph agents
        collections: List of collection names to search
        collection_strategy: Collection search strategy
        use_classifier: Whether to use the query classifier (default: True)
    """
    logger.info(f"Enhanced RAG answer for: {question[:100]}...")
    
    # Store user question in conversation history
    if conversation_id:
        store_conversation_message(conversation_id, "user", question)
    
    # Get settings
    llm_cfg = get_llm_settings()
    
    # Prepare classification context
    context = {
        "has_collections": bool(collections),
        "conversation_history": get_conversation_history(conversation_id) if conversation_id else None,
        "available_tools": list(get_enabled_mcp_tools().keys()) if get_enabled_mcp_tools() else []
    }
    
    # Classify the query
    if use_classifier:
        classification = classify_query(question, context, use_llm_fallback=True)
        logger.info(f"Query classified as: {classification.primary_type.value} "
                   f"(confidence: {classification.confidence:.2f})")
        
        # Store classification in conversation for transparency
        if conversation_id:
            store_conversation_message(
                conversation_id, 
                "system",
                f"Query type: {classification.primary_type.value} ({classification.confidence:.0%} confidence)"
            )
    else:
        # Fall back to existing classification logic
        from app.langchain.service import classify_query_type
        query_type_str = classify_query_type(question, llm_cfg)
        # Map to our QueryType enum
        type_mapping = {
            "LLM": QueryType.DIRECT_LLM,
            "RAG": QueryType.RAG,
            "TOOLS": QueryType.TOOLS,
            "LARGE_GENERATION": QueryType.LARGE_GENERATION
        }
        classification = type(
            'ClassificationResult', 
            (), 
            {
                'primary_type': type_mapping.get(query_type_str, QueryType.DIRECT_LLM),
                'confidence': 0.8,
                'reasoning': f'Legacy classification: {query_type_str}',
                'metadata': {}
            }
        )()
    
    # Route based on classification
    if classification.primary_type == QueryType.LARGE_GENERATION:
        yield from _handle_large_generation_route(
            question, conversation_id, thinking, stream, classification
        )
    
    elif classification.primary_type == QueryType.TOOLS:
        yield from _handle_tools_route(
            question, conversation_id, thinking, stream, classification
        )
    
    elif classification.primary_type == QueryType.RAG:
        yield from _handle_rag_route(
            question, conversation_id, thinking, stream, collections, 
            collection_strategy, classification
        )
    
    elif classification.primary_type == QueryType.HYBRID:
        yield from _handle_hybrid_route(
            question, conversation_id, thinking, stream, collections,
            collection_strategy, classification
        )
    
    else:  # DIRECT_LLM
        yield from _handle_direct_llm_route(
            question, conversation_id, thinking, stream, classification
        )


def _handle_large_generation_route(
    question: str,
    conversation_id: str,
    thinking: bool,
    stream: bool,
    classification: Any
):
    """Handle large generation queries"""
    logger.info("Routing to large generation handler")
    
    # Analyze for chunking requirements
    analysis = detect_large_output_potential(question)
    target_count = analysis.get("estimated_items", 100)
    
    if target_count <= 100:
        # Small enough for single call
        logger.info(f"{target_count} items - using single LLM call")
        yield from _handle_direct_llm_route(
            question, conversation_id, thinking, stream, classification
        )
    else:
        # Use chunked generation
        logger.info(f"{target_count} items - using chunked generation")
        
        if stream:
            yield json.dumps({
                "type": "large_generation_start",
                "estimated_items": target_count,
                "message": "Starting large content generation..."
            }) + "\n"
        
        # Use multi-agent system for chunked generation
        system = MultiAgentSystem(conversation_id=conversation_id)
        
        # This would need to be made async in practice
        for event in system.generate_large_content_sync(
            query=question,
            target_count=target_count
        ):
            if stream:
                yield json.dumps(event) + "\n"


def _handle_tools_route(
    question: str,
    conversation_id: str,
    thinking: bool,
    stream: bool,
    classification: Any
):
    """Handle tool-based queries"""
    logger.info("Routing to tools handler")
    
    required_tools = classification.metadata.get("required_tools", [])
    available_tools = get_enabled_mcp_tools()
    
    if stream:
        yield json.dumps({
            "type": "tools_execution_start",
            "required_tools": required_tools,
            "message": "Executing required tools..."
        }) + "\n"
    
    # Execute tools and collect results
    tool_results = {}
    for tool_name in required_tools:
        if tool_name in available_tools:
            # This would need actual tool execution implementation
            tool_results[tool_name] = f"[{tool_name} results would go here]"
    
    # Build prompt with tool results
    tool_context = "\n".join([
        f"{tool}: {result}" 
        for tool, result in tool_results.items()
    ])
    
    conversation_context = get_conversation_history(conversation_id) if conversation_id else ""
    
    prompt = f"""Answer based on these tool execution results:

Tool Results:
{tool_context}

{conversation_context}

Question: {question}"""
    
    # Generate response
    yield from _stream_llm_response(prompt, thinking, stream)


def _handle_rag_route(
    question: str,
    conversation_id: str,
    thinking: bool,
    stream: bool,
    collections: List[str],
    collection_strategy: str,
    classification: Any
):
    """Handle RAG queries"""
    logger.info("Routing to RAG handler")
    
    if stream:
        yield json.dumps({
            "type": "rag_search_start",
            "collections": collections,
            "message": "Searching documents..."
        }) + "\n"
    
    # Get RAG context
    start_time = time.time()
    rag_context, _ = handle_rag_query(question, thinking, collections, collection_strategy)
    search_time = time.time() - start_time
    
    if stream:
        yield json.dumps({
            "type": "rag_search_complete",
            "found_context": bool(rag_context),
            "search_time": search_time,
            "context_length": len(rag_context) if rag_context else 0
        }) + "\n"
    
    if rag_context:
        # Build RAG prompt
        conversation_context = get_conversation_history(conversation_id) if conversation_id else ""
        
        prompt = f"""Answer based on the following context:

Document Context:
{rag_context}

{conversation_context}

Question: {question}

Provide a comprehensive answer based on the context."""
    else:
        # No context found, fall back to direct LLM
        if stream:
            yield json.dumps({
                "type": "rag_fallback",
                "message": "No relevant documents found, using general knowledge..."
            }) + "\n"
        
        prompt = question
    
    # Generate response
    yield from _stream_llm_response(prompt, thinking, stream)


def _handle_hybrid_route(
    question: str,
    conversation_id: str,
    thinking: bool,
    stream: bool,
    collections: List[str],
    collection_strategy: str,
    classification: Any
):
    """Handle hybrid queries requiring multiple approaches"""
    logger.info("Routing to hybrid handler")
    
    if stream:
        yield json.dumps({
            "type": "hybrid_processing_start",
            "message": "Processing with multiple approaches..."
        }) + "\n"
    
    combined_context = []
    
    # Check secondary types
    if classification.secondary_types:
        for query_type, confidence in classification.secondary_types:
            if query_type == QueryType.RAG and confidence > 0.3:
                # Get RAG context
                rag_context, _ = handle_rag_query(question, thinking, collections, collection_strategy)
                if rag_context:
                    combined_context.append(f"Document Context:\n{rag_context}")
            
            elif query_type == QueryType.TOOLS and confidence > 0.3:
                # Execute tools (simplified)
                combined_context.append("Tool Results:\n[Tool execution would go here]")
    
    # Build comprehensive prompt
    conversation_context = get_conversation_history(conversation_id) if conversation_id else ""
    all_context = "\n\n".join(combined_context)
    
    prompt = f"""Answer using all available information:

{all_context}

{conversation_context}

Question: {question}"""
    
    # Generate response
    yield from _stream_llm_response(prompt, thinking, stream)


def _handle_direct_llm_route(
    question: str,
    conversation_id: str,
    thinking: bool,
    stream: bool,
    classification: Any
):
    """Handle direct LLM queries"""
    logger.info("Routing to direct LLM handler")
    
    if stream:
        yield json.dumps({
            "type": "llm_direct",
            "message": "Generating response..."
        }) + "\n"
    
    # Build simple prompt
    conversation_context = get_conversation_history(conversation_id) if conversation_id else ""
    prompt = f"{conversation_context}\n\n{question}" if conversation_context else question
    
    # Generate response
    yield from _stream_llm_response(prompt, thinking, stream)


def _stream_llm_response(prompt: str, thinking: bool, stream: bool):
    """Stream response from LLM"""
    llm_cfg = get_llm_settings()
    
    # Select model based on thinking mode
    mode_key = "thinking_mode" if thinking else "non_thinking_mode"
    model = llm_cfg.get(mode_key, llm_cfg.get("model"))
    
    # Build full prompt
    full_prompt = build_prompt(prompt, is_internal=False)
    
    llm_api_url = "http://localhost:8000/api/v1/generate_stream"
    payload = {
        "prompt": full_prompt,
        "model": model,
        "temperature": llm_cfg.get("temperature", 0.7),
        "max_tokens": llm_cfg.get("max_tokens", 8192),
        "top_p": llm_cfg.get("top_p", 0.9)
    }
    
    if stream:
        # Stream tokens
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", llm_api_url, json=payload) as response:
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        token = line.replace("data: ", "")
                        yield json.dumps({
                            "type": "token",
                            "content": token
                        }) + "\n"
        
        # Store complete response
        if conversation_id:
            # In practice, you'd accumulate tokens and store the complete response
            store_conversation_message(conversation_id, "assistant", "[Response stored]")
    else:
        # Non-streaming response
        text = ""
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", llm_api_url, json=payload) as response:
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        text += line.replace("data: ", "")
        
        if conversation_id:
            store_conversation_message(conversation_id, "assistant", text)
        
        yield text