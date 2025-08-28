"""
FastAPI router for Notebook management endpoints.
Provides CRUD operations and RAG functionality for notebooks.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, UploadFile, File, Request, Body
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, DatabaseError
from sqlalchemy import and_, func, text
from typing import List, Optional, Dict, Any
import logging
import uuid
import os
import hashlib
import json
import re
import asyncio
from datetime import datetime

from app.core.db import get_db, KnowledgeGraphDocument
from app.models.notebook_models import (
    NotebookCreateRequest, NotebookUpdateRequest, NotebookResponse, NotebookDetailResponse,
    NotebookListResponse, NotebookDocumentAddRequest, NotebookDocumentBulkRequest,
    NotebookRAGRequest, NotebookRAGResponse, NotebookConversationRequest,
    NotebookDocumentBulkResponse, NotebookStatsResponse, NotebookOperationResponse,
    NotebookError, NotebookValidationError, NotebookDocumentResponse, 
    NotebookConversationResponse, DocumentDeleteRequest, DocumentDeleteResponse,
    DocumentUsageInfo, DocumentDeletionSummary, NotebookChatRequest, NotebookChatResponse,
    # Memory models
    MemoryCreateRequest, MemoryUpdateRequest, MemoryResponse, MemoryListResponse,
    # Document update models
    DocumentUpdateRequest,
    # Chunk editing models
    ChunkUpdateRequest, ChunkResponse, ChunkListResponse, BulkChunkReEmbedRequest,
    ChunkOperationResponse, BulkChunkOperationResponse
)
from pydantic import BaseModel
from app.services.notebook_service import NotebookService
from app.services.notebook_rag_service import (
    NotebookRAGService, conversation_context_manager, RetrievalIntensity, RetrievalPlan,
    IntelligentRoutingMetrics, intelligent_routing_metrics, track_execution_time
)
from app.langchain.conversation_context_manager import ConversationContextManager as LangchainContextManager
from app.services.hierarchical_notebook_rag_service import get_hierarchical_notebook_rag_service
from app.services.document_admin_service import DocumentAdminService
from app.services.chunk_management_service import ChunkManagementService
from app.services.ai_task_planner import ai_task_planner, TaskExecutionPlan
from app.services.ai_verification_service import ai_verification_service
from app.services.cache_bypass_detector import cache_bypass_detector
# from app.services.request_execution_state_tracker import (
#     create_request_state, get_request_state, ExecutionPhase
# )  # Removed - causing Redis async/sync errors
from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config
from app.core.notebook_llm_settings_cache import get_notebook_llm_full_config
from app.core.notebook_source_templates_cache import apply_source_templates
from app.core.timeout_settings_cache import get_timeout_value, get_intelligent_plan_timeout
from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig

# Upload response model
class NotebookUploadResponse(BaseModel):
    status: str
    document_id: str
    filename: str
    file_id: str
    total_chunks: int
    unique_chunks: int
    duplicates_filtered: int
    collection: str
    pages_processed: int
    message: str

# Document processing imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from app.core.embedding_settings_cache import get_embedding_settings
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.utils.metadata_extractor import MetadataExtractor
from app.rag.bm25_processor import BM25Processor
from utils.deduplication import hash_text, get_existing_hashes, get_existing_doc_ids, filter_new_chunks
from app.core.document_classifier import get_document_classifier
from app.core.collection_registry_cache import get_collection_config

# Set up logging
logger = logging.getLogger(__name__)

# Global LLM instance cache to prevent wasteful instance creation
_notebook_llm_cache = {}
_llm_cache_lock = asyncio.Lock()

async def get_cached_notebook_llm(llm_config: dict) -> 'OllamaLLM':
    """Get cached LLM instance to avoid creating multiple instances."""
    from app.llm.ollama import OllamaLLM
    from app.llm.base import LLMConfig
    
    config_hash = hashlib.md5(json.dumps(llm_config, sort_keys=True).encode()).hexdigest()
    
    async with _llm_cache_lock:
        if config_hash not in _notebook_llm_cache:
            logger.debug(f"[LLM_CACHE] Creating new cached instance for config {config_hash[:8]}")
            
            # Extract the actual config from nested structure
            # Handle both nested {'notebook_llm': {...}} and flat {...} formats
            if 'notebook_llm' in llm_config:
                config_dict = llm_config['notebook_llm'].copy()
                logger.debug(f"[LLM_CACHE] Extracted nested config from 'notebook_llm' key")
            else:
                config_dict = llm_config.copy()
                logger.debug(f"[LLM_CACHE] Using flat config structure")
            
            # Handle key mapping: database uses 'model' but LLMConfig expects 'model_name'
            if 'model' in config_dict and 'model_name' not in config_dict:
                config_dict['model_name'] = config_dict.pop('model')
                logger.debug(f"[LLM_CACHE] Mapped 'model' to 'model_name': {config_dict['model_name']}")
            
            # Ensure required model_name field exists
            if 'model_name' not in config_dict:
                logger.error(f"[LLM_CACHE] Missing required 'model_name' field in config: {config_dict}")
                raise ValueError("LLM configuration missing required 'model_name' field")
                
            config_obj = LLMConfig(**config_dict)
            _notebook_llm_cache[config_hash] = OllamaLLM(config=config_obj)
        else:
            logger.debug(f"[LLM_CACHE] Reusing cached instance for config {config_hash[:8]}")
        
        return _notebook_llm_cache[config_hash]

async def get_cached_notebook_llm_from_config(llm_config_obj: 'LLMConfig') -> 'OllamaLLM':
    """Get cached LLM instance from LLMConfig object."""
    from app.llm.ollama import OllamaLLM
    
    # Create hash from config object attributes
    config_dict = {
        'model_name': llm_config_obj.model_name,
        'temperature': llm_config_obj.temperature,
        'top_p': llm_config_obj.top_p,
        'max_tokens': llm_config_obj.max_tokens
    }
    config_hash = hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()
    
    async with _llm_cache_lock:
        if config_hash not in _notebook_llm_cache:
            logger.debug(f"[LLM_CACHE] Creating new cached instance for LLMConfig {config_hash[:8]}")
            _notebook_llm_cache[config_hash] = OllamaLLM(llm_config_obj)
        else:
            logger.debug(f"[LLM_CACHE] Reusing cached LLMConfig instance {config_hash[:8]}")
        
        return _notebook_llm_cache[config_hash]

router = APIRouter()

# Initialize the pattern-based context manager for query classification
_query_classifier = LangchainContextManager()

def _classify_query_intent(message: str) -> str:
    """
    Classify query intent using existing ConversationContextManager patterns.
    
    Args:
        message: User's message
        
    Returns:
        str: Query type - 'simple', 'contextual', 'technical', 'general', 'temporal'
    """
    return _query_classifier.classify_query_type(message)

def _should_handle_with_context(message: str, conversation_history: List[Dict]) -> bool:
    """
    Determine if query should be handled with existing conversation context.
    
    Args:
        message: User's message
        conversation_history: Previous conversation exchanges
        
    Returns:
        bool: True if should use conversation context for handling
    """
    return _query_classifier.should_include_history(message, conversation_history)

# Deprecated functions removed - use _classify_query_intent() and _should_handle_with_context() instead


async def _handle_contextual_query(
    message: str,
    conversation_context: Dict[str, Any],
    cached_context: Optional[Dict[str, Any]],
    notebook_id: str,
    conversation_id: str
) -> Optional[Dict[str, Any]]:
    """
    Handle contextual queries using existing conversation data.
    Covers reordering, filtering, formatting, and reference-based operations.
    
    Args:
        message: User's contextual query
        conversation_context: Previous conversation exchange
        cached_context: Cached retrieval context
        notebook_id: Notebook ID
        conversation_id: Conversation ID
        
    Returns:
        Dict with contextual response or None if cannot be handled
    """
    try:
        logger.info(f"[CONTEXTUAL] Processing contextual query: {message[:100]}")
        
        # Extract previous response data
        previous_response = conversation_context.get('ai_response', '')
        previous_sources = conversation_context.get('sources', [])
        
        # Also check cached context for more data
        if cached_context:
            cached_sources = cached_context.get('sources', [])
            if cached_sources:
                previous_sources.extend(cached_sources)
                logger.info(f"[CONTEXTUAL] Enhanced with {len(cached_sources)} cached sources")
        
        if not previous_response and not previous_sources:
            logger.warning("[CONTEXTUAL] No previous response or sources found for contextual query")
            return None
        
        message_lower = message.lower().strip()
        
        # Handle different transformation types
        if any(keyword in message_lower for keyword in ['order by', 'sort by', 'arrange by', 'rank by']):
            return await _handle_reordering_transformation(message, previous_response, previous_sources)
            
        elif any(keyword in message_lower for keyword in ['filter', 'show only', 'limit to', 'from year']):
            return await _handle_filtering_transformation(message, previous_response, previous_sources)
            
        elif any(keyword in message_lower for keyword in ['format as', 'make it a', 'bullet points', 'table']):
            return await _handle_formatting_transformation(message, previous_response, previous_sources)
            
        elif any(keyword in message_lower for keyword in ['summarize', 'summary', 'brief', 'short']):
            return await _handle_summary_transformation(message, previous_response, previous_sources)
            
        elif any(pattern in message_lower for pattern in ['tell me more about', 'about item', 'details about']):
            return await _handle_item_reference_transformation(message, previous_response, previous_sources)
            
        elif any(keyword in message_lower for keyword in ['again', 'show again', 'list again']):
            return await _handle_repeat_transformation(message, previous_response, previous_sources)
            
        else:
            # Generic transformation - try to use LLM with the existing data
            return await _handle_generic_transformation(message, previous_response, previous_sources)
            
    except Exception as e:
        logger.error(f"[TRANSFORM] Error in transformation handler: {str(e)}")
        return None

async def _handle_reordering_transformation(
    message: str, 
    previous_response: str, 
    previous_sources: List[Dict]
) -> Dict[str, Any]:
    """Handle reordering/sorting transformations."""
    logger.info("[TRANSFORM] Handling reordering transformation")
    
    # Simple reordering based on common patterns
    message_lower = message.lower()
    
    # Try to extract structured data from previous response
    if 'year' in message_lower or 'date' in message_lower:
        # Sort by year/date
        response = f"Here's the information reordered by date:\n\n{previous_response}"
    elif 'name' in message_lower or 'alphabetically' in message_lower:
        # Sort alphabetically
        response = f"Here's the information reordered alphabetically:\n\n{previous_response}"
    else:
        # Generic reordering
        response = f"Here's the reordered information:\n\n{previous_response}"
    
    return {
        'response': response,
        'sources': previous_sources,
        'transformation_type': 'reordering',
        'message': f'Applied reordering transformation to {len(previous_sources)} sources'
    }

async def _handle_filtering_transformation(
    message: str, 
    previous_response: str, 
    previous_sources: List[Dict]
) -> Dict[str, Any]:
    """Handle filtering transformations."""
    logger.info("[TRANSFORM] Handling filtering transformation")
    
    message_lower = message.lower()
    
    # Extract filter criteria
    if 'year' in message_lower:
        # Try to extract year
        import re
        year_match = re.search(r'\b(19|20)\d{2}\b', message_lower)
        if year_match:
            year = year_match.group()
            response = f"Filtered results for year {year}:\n\n{previous_response[:500]}..."
        else:
            response = f"Filtered by year criteria:\n\n{previous_response}"
    else:
        # Generic filtering
        response = f"Filtered results based on your criteria:\n\n{previous_response}"
    
    return {
        'response': response,
        'sources': previous_sources,
        'transformation_type': 'filtering',
        'message': f'Applied filtering transformation to {len(previous_sources)} sources'
    }

async def _handle_formatting_transformation(
    message: str, 
    previous_response: str, 
    previous_sources: List[Dict]
) -> Dict[str, Any]:
    """Handle formatting transformations."""
    logger.info("[TRANSFORM] Handling formatting transformation")
    
    message_lower = message.lower()
    
    if 'bullet' in message_lower or 'list' in message_lower:
        # Convert to bullet points
        lines = previous_response.split('\n')
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('•') and not line.startswith('-'):
                formatted_lines.append(f"• {line}")
            else:
                formatted_lines.append(line)
        response = '\n'.join(formatted_lines)
    elif 'table' in message_lower:
        # Simple table format indication
        response = f"Table format requested:\n\n{previous_response}"
    else:
        # Generic formatting
        response = f"Formatted response:\n\n{previous_response}"
    
    return {
        'response': response,
        'sources': previous_sources,
        'transformation_type': 'formatting',
        'message': f'Applied formatting transformation to {len(previous_sources)} sources'
    }

async def _handle_summary_transformation(
    message: str, 
    previous_response: str, 
    previous_sources: List[Dict]
) -> Dict[str, Any]:
    """Handle summary transformations."""
    logger.info("[TRANSFORM] Handling summary transformation")
    
    # Create a simple summary by taking first portion of response
    summary = previous_response[:300]
    if len(previous_response) > 300:
        summary += "...\n\n[Summary of longer response]"
    
    response = f"Summary:\n\n{summary}"
    
    return {
        'response': response,
        'sources': previous_sources,
        'transformation_type': 'summary',
        'message': f'Generated summary from {len(previous_sources)} sources'
    }

async def _handle_item_reference_transformation(
    message: str, 
    previous_response: str, 
    previous_sources: List[Dict]
) -> Dict[str, Any]:
    """Handle item-specific reference transformations."""
    logger.info("[TRANSFORM] Handling item reference transformation")
    
    # Try to extract item number or reference
    import re
    item_match = re.search(r'(?:item|number|#)\s*(\d+)', message.lower())
    
    if item_match:
        item_num = int(item_match.group(1))
        response = f"Details about item #{item_num}:\n\n{previous_response[:400]}..."
    else:
        # Generic item reference
        response = f"Additional details:\n\n{previous_response}"
    
    return {
        'response': response,
        'sources': previous_sources,
        'transformation_type': 'item_reference',
        'message': f'Provided details from {len(previous_sources)} sources'
    }

async def _handle_repeat_transformation(
    message: str, 
    previous_response: str, 
    previous_sources: List[Dict]
) -> Dict[str, Any]:
    """Handle repeat/show again transformations."""
    logger.info("[TRANSFORM] Handling repeat transformation")
    
    response = f"Here's the information again:\n\n{previous_response}"
    
    return {
        'response': response,
        'sources': previous_sources,
        'transformation_type': 'repeat',
        'message': f'Repeated response with {len(previous_sources)} sources'
    }

async def _handle_generic_transformation(
    message: str, 
    previous_response: str, 
    previous_sources: List[Dict]
) -> Dict[str, Any]:
    """Handle generic transformations using the previous response."""
    logger.info("[TRANSFORM] Handling generic transformation")
    
    # For generic transformations, provide the previous response with context
    response = f"Based on the previous information:\n\n{previous_response}"
    
    return {
        'response': response,
        'sources': previous_sources,
        'transformation_type': 'generic',
        'message': f'Applied transformation to {len(previous_sources)} sources'
    }



async def classify_message_intent(message: str, conversation_context: Optional[Dict] = None) -> str:
    """
    LLM-based message intent classification.
    Uses the LLM to intelligently classify user intent instead of hardcoded patterns.
    
    Args:
        message: The user's message to classify
        conversation_context: Optional conversation context for better classification
        
    Returns:
        str: Intent category ('greeting', 'acknowledgment', 'clarification', 
             'retrieval_required', 'domain_query', 'context_reference', 'general_chat')
    """
    try:
        # Get LLM configuration
        llm_config = get_main_llm_full_config()
        if not llm_config:
            # Fallback to basic heuristics if LLM unavailable
            return 'retrieval_required' if len(message.strip()) > 20 else 'general_chat'
        
        # Create LLM instance
        llm = OllamaLLM(LLMConfig(**llm_config))
        
        # Build classification prompt
        context_info = ""
        if conversation_context:
            context_info = f"\nConversation context: User has been discussing {conversation_context.get('topic', 'various topics')}"
        
        classification_prompt = f"""Classify this user message into one of these categories:

Categories:
- greeting: Simple hello, hi, good morning etc.
- acknowledgment: Thanks, appreciate, great job etc. (but NOT requests for work)
- clarification: Short questions like "what?" "how?" when unclear
- retrieval_required: Requests for data, lists, information, updates, tables, reports
- domain_query: Questions about specific topics, projects, work, documents
- context_reference: References to previous conversation ("that", "this", "it")  
- general_chat: General conversation that doesn't need data

User message: "{message}"{context_info}

IMPORTANT: If the user is asking for data, updates, tables, lists, or information - classify as "retrieval_required" or "domain_query".

Reply with just the category name, nothing else:"""

        # Get classification from LLM with short timeout
        response = await llm.generate(classification_prompt, timeout=10)
        intent = response.text.strip().lower()
        
        # Validate response and map to known categories
        valid_intents = ['greeting', 'acknowledgment', 'clarification', 'retrieval_required', 'domain_query', 'context_reference', 'general_chat']
        
        if intent in valid_intents:
            return intent
        
        # If LLM gives invalid response, default to retrieval for safety
        # Better to retrieve unnecessarily than miss a data request
        return 'retrieval_required'
        
    except Exception as e:
        # If LLM fails completely, always default to retrieval
        # This ensures we don't miss any data requests due to classification errors
        return 'retrieval_required'


def enrich_response_metadata(
    base_metadata: Dict[str, Any],
    routing_decision: str,
    intent: str,
    retrieval_triggered: bool,
    cache_hit: bool = False,
    execution_time: float = None,
    retrieval_plan: RetrievalPlan = None
) -> Dict[str, Any]:
    """
    Enrich response metadata with comprehensive routing and performance information.
    
    Args:
        base_metadata: Base metadata dict to enrich
        routing_decision: The routing decision made
        intent: Classified intent
        retrieval_triggered: Whether retrieval was triggered
        cache_hit: Whether cache was hit
        execution_time: Execution time in seconds
        retrieval_plan: Optional retrieval plan details
        
    Returns:
        Enhanced metadata dictionary
    """
    enriched = {
        **base_metadata,
        "routing": {
            "intent_classification": intent,
            "routing_decision": routing_decision,
            "retrieval_triggered": retrieval_triggered,
            "cache_hit": cache_hit
        },
        "intelligence": {
            "understand_think_plan_do": True,
            "ai_powered_routing": True,
            "intelligent_message_handling": True
        }
    }
    
    # Add performance data if available
    if execution_time is not None:
        enriched["performance"] = {
            "execution_time": execution_time,
            "routing_efficiency": "high" if not retrieval_triggered else "normal"
        }
    
    # Add retrieval plan details if available
    if retrieval_plan:
        enriched["retrieval_plan"] = {
            "intensity": retrieval_plan.intensity.value,
            "max_sources": retrieval_plan.max_sources,
            "use_multiple_strategies": retrieval_plan.use_multiple_strategies,
            "reasoning": retrieval_plan.reasoning[:100]  # Truncated
        }
    
    return enriched


@track_execution_time("simple_response")
async def simple_llm_response(message: str, notebook_id: str, conversation_id: str = None) -> Dict[str, Any]:
    """
    Generate simple response without retrieval for greetings/acknowledgments
    
    Args:
        message: The user's message
        notebook_id: The notebook ID for context
        conversation_id: Optional conversation ID
        
    Returns:
        Dict containing simple response without retrieval
    """
    try:
        # Get basic notebook info for context - this is lightweight, no full retrieval
        notebook_name = "your notebook"  # Could get actual name from DB if needed
        
        # Simple, contextual responses based on message classification
        message_lower = message.lower()
        
        if any(greeting in message_lower for greeting in ['hello', 'hi', 'hey']):
            response = f"Hello! I'm here to help you explore and discuss the content in {notebook_name}. What would you like to know?"
        elif any(thanks in message_lower for thanks in ['thanks', 'thank']):
            response = "You're welcome! Is there anything else you'd like to explore in your notebook?"
        elif 'good morning' in message_lower:
            response = f"Good morning! Ready to dive into {notebook_name}? What can I help you discover today?"
        elif 'good afternoon' in message_lower:
            response = f"Good afternoon! I'm here to help you with {notebook_name}. What would you like to explore?"
        elif 'good evening' in message_lower:
            response = f"Good evening! How can I assist you with {notebook_name} today?"
        else:
            response = f"I'm ready to help you with {notebook_name}. What would you like to discuss?"
        
        return {
            "response": response,
            "conversation_id": conversation_id or f"notebook-{notebook_id}",
            "sources": [],
            "metadata": {
                "routing": "simple_response",
                "retrieval_triggered": False,
                "message_classification": "non_retrieval_intent"
            }
        }
    except Exception as e:
        logger.error(f"Simple response failed: {e}")
        # Fallback - let it proceed to full pipeline
        return None


@track_execution_time("cached_response") 
async def respond_with_cached_sources(message: str, cached_context: Dict[str, Any], notebook_id: str, conversation_id: str) -> Dict[str, Any]:
    """
    Generate response using cached retrieval results without new retrieval.
    
    Args:
        message: The user's follow-up message
        cached_context: Previously cached retrieval context
        notebook_id: The notebook ID
        conversation_id: The conversation ID
        
    Returns:
        Dict containing response with cached sources and cache metadata
    """
    try:
        # Extract cached data
        cached_sources = cached_context.get('sources', [])
        original_query = cached_context.get('query', '')
        cached_at = cached_context.get('cached_at', '')
        
        if not cached_sources:
            logger.warning(f"[CACHE] No cached sources found for conversation {conversation_id}")
            return None
            
        # Get notebook LLM configuration
        llm_config = get_notebook_llm_full_config()
        
        if not llm_config:
            logger.error("Failed to get notebook LLM configuration for cached response")
            return None
            
        # Use cached LLM instance to avoid creating new connections
        llm = await get_cached_notebook_llm(llm_config)
        
        # Build context from cached sources
        context_parts = []
        for idx, source in enumerate(cached_sources[:10]):  # Limit to top 10 sources
            context_parts.append(f"Source {idx + 1}: {source.get('content', '')[:500]}...")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for cached response
        system_prompt = f"""You are an AI assistant helping with notebook content. You're answering a follow-up question using previously retrieved information.

Original query that retrieved this context: "{original_query}"
Current follow-up question: "{message}"

Use the provided context to answer the follow-up question. Be clear that you're using previously retrieved information if relevant."""
        
        user_prompt = f"""Context from notebook:
{context}

Question: {message}

Please provide a helpful response based on the context above."""
        
        # Generate response using cached context
        response = await llm.agenerate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=2000
        )
        
        # Convert cached sources to proper format
        formatted_sources = []
        for source in cached_sources:
            if isinstance(source, dict):
                formatted_sources.append({
                    'content': source.get('content', ''),
                    'metadata': source.get('metadata', {}),
                    'score': source.get('score', 0.0)
                })
        
        logger.info(f"[CACHE] Generated response using {len(formatted_sources)} cached sources")
        
        return {
            "response": response,
            "conversation_id": conversation_id,
            "sources": formatted_sources,
            "metadata": {
                "routing": "cached_response",
                "retrieval_triggered": False,
                "cache_hit": True,
                "original_query": original_query,
                "cached_at": cached_at,
                "cache_age_info": f"Using context from: {original_query}",
                "source_count": len(formatted_sources)
            }
        }
        
    except Exception as e:
        logger.error(f"[CACHE] Failed to generate cached response: {str(e)}")
        # Return None to fall back to regular retrieval
        return None


async def respond_with_llm_intelligence_streaming(message: str, cached_context: Dict[str, Any], conversation_context: Dict[str, Any], notebook_id: str, conversation_id: str):
    """
    Generate streaming response using LLM-driven conversation intelligence.
    
    The LLM sees both the cached sources and the previous conversation context,
    allowing it to make intelligent decisions about data reuse vs new retrieval.
    
    Args:
        message: The user's current message
        cached_context: Previously cached retrieval context (if available)
        conversation_context: Last conversation exchange context
        notebook_id: The notebook ID
        conversation_id: The conversation ID
        
    Yields:
        JSON-formatted streaming chunks or raises exception if LLM requests new retrieval
    """
    try:
        # Get notebook LLM configuration
        llm_config = get_notebook_llm_full_config()
        
        if not llm_config:
            logger.error("Failed to get notebook LLM configuration for intelligent response")
            raise ValueError("No LLM configuration available")
            
        # Use cached LLM instance to avoid creating new connections
        llm = await get_cached_notebook_llm(llm_config)
        
        # Build context information - prioritize extracted entities over raw sources
        available_sources = []
        
        # First, try to use extracted/structured entities (e.g., projects)
        if cached_context and cached_context.get('extracted_entities'):
            logger.info(f"[CACHE_ENTITIES] Using {len(cached_context['extracted_entities'])} extracted entities from cache")
            for idx, entity in enumerate(cached_context['extracted_entities']):
                if isinstance(entity, dict):
                    # Handle structured project data
                    title = entity.get('title', entity.get('name', f'Entity {idx + 1}'))
                    description = entity.get('description', entity.get('summary', ''))
                    available_sources.append(f"Project {idx + 1}: {title} - {description}")
                else:
                    available_sources.append(f"Entity {idx + 1}: {str(entity)}")
        
        # Fallback to raw sources if no extracted entities available
        elif cached_context and cached_context.get('sources'):
            logger.info(f"[CACHE_SOURCES] Using {len(cached_context['sources'])} raw sources from cache (no extracted entities)")
            for idx, source in enumerate(cached_context['sources'][:10]):
                # Preserve significantly more context for reordering/filtering decisions
                content = source.get('content', '')
                truncated_content = content[:1500] + ('...' if len(content) > 1500 else '')
                available_sources.append(f"Source {idx + 1}: {truncated_content}")
        
        sources_context = "\n\n".join(available_sources) if available_sources else "No previous sources available."
        
        # Build conversation memory context
        conversation_memory = ""
        if conversation_context:
            last_user_msg = conversation_context.get('user_message', '')
            last_ai_response = conversation_context.get('ai_response', '')
            
            if last_user_msg and last_ai_response:
                conversation_memory = f"""
RECENT CONVERSATION:
User: {last_user_msg}
Assistant: {last_ai_response[:500]}{'...' if len(last_ai_response) > 500 else ''}
"""
        
        # Create intelligent conversation prompt with CLEAR examples and CONSERVATIVE retrieval logic
        system_prompt = f"""You are an intelligent AI assistant helping with notebook content analysis. Your role is to provide helpful responses while being conversationally aware.

CONVERSATION INTELLIGENCE INSTRUCTIONS:
1. You have access to previously retrieved information from the notebook
2. You can see the recent conversation context 
3. Your job is to determine if you can answer the current question with existing information OR if you need new data retrieval

DECISION LOGIC - BE CONSERVATIVE ABOUT NEW RETRIEVAL:
✅ ANSWER WITH AVAILABLE DATA if the user asks for:
  - Reordering: "order by date", "sort by name", "arrange by year"
  - Filtering: "only show 2023", "just the Python projects", "remove duplicates"
  - Analysis: "summarize that list", "what's the pattern", "compare these"
  - References: "give me that list again", "show me those results", "the previous data"
  - Clarification: "explain the first one", "tell me more about X from the list"

❌ REQUEST NEW RETRIEVAL only if:
  - User asks for completely different information not in sources
  - User requests new topics/entities not mentioned in available data
  - Sources are clearly insufficient for the specific question

EXAMPLE SCENARIOS:
- "order by end year, give me list again" → ANSWER (reorder existing data)
- "show me only projects from 2023" → ANSWER (filter existing data)  
- "tell me about quantum computing projects" (when sources are about web development) → NEED_NEW_RETRIEVAL
- "show that list again" → ANSWER (display existing data)

AVAILABLE INFORMATION:
{sources_context}

{conversation_memory}

Current user question: "{message}"

Respond naturally and conversationally. PRIORITIZE using available data. Only request retrieval if you absolutely cannot answer with current information."""

        user_prompt = f"""Based on the available information and conversation context, please respond to: {message}

CONTEXT AWARENESS - CRITICAL:
- Display references like "that list", "those results", "the data", "show again" refer to previously shown information
- Ordering/sorting requests ("order by", "sort by", "arrange") should use existing data
- Filtering requests ("only show", "just the", "remove") should use existing data  
- Simple continuation phrases ("give me list again", "show that") mean reuse previous results

FRESH DATA REQUESTS - REQUEST NEW RETRIEVAL for compound phrases indicating fresh searches:
- "find ... again", "query ... again", "search ... again" (action + again = fresh search)

ONLY request new retrieval if you need completely different information not available in the sources above OR if the user explicitly requests fresh data using the patterns above."""

        # Generate LLM decision and response with streaming
        response_stream = llm.chat_stream(
            user_prompt,
            system_prompt=system_prompt,
            max_tokens=2000
        )
        
        # Format existing sources if using cached context
        formatted_sources = []
        if cached_context and cached_context.get('sources'):
            for source in cached_context['sources']:
                if isinstance(source, dict):
                    formatted_sources.append({
                        'content': source.get('content', ''),
                        'metadata': source.get('metadata', {}),
                        'score': source.get('score', 0.0)
                    })
        
        # First yield the sources to match existing pattern
        if formatted_sources:
            yield json.dumps({
                "sources": formatted_sources,
                "notebook_id": notebook_id,
                "conversation_id": conversation_id
            }) + "\n"
        
        # Collect complete response for retrieval decision check
        collected_response = ""
        
        # Stream the LLM response chunks
        async for chunk in response_stream:
            if chunk and chunk.text:
                collected_response += chunk.text
                
                # Stream each chunk to frontend
                yield json.dumps({
                    "chunk": chunk.text,
                    "notebook_id": notebook_id,
                    "conversation_id": conversation_id
                }) + "\n"
        
        # Check if LLM requests new retrieval after collecting full response
        if collected_response and "NEED_NEW_RETRIEVAL" in collected_response:
            logger.info(f"[LLM_INTELLIGENCE_STREAMING] LLM requests new retrieval: {collected_response}")
            raise ValueError("LLM requests new retrieval")  # Signal that new retrieval is needed
        
        # Enhanced logging for debugging continuation query failures
        if not collected_response or len(collected_response.strip()) < 10:
            logger.warning(f"[LLM_INTELLIGENCE_STREAMING] LLM provided insufficient response for message: '{message}' - Response: '{collected_response}'")
            raise ValueError("LLM provided insufficient response")
            
        logger.info(f"[LLM_INTELLIGENCE_STREAMING] Successfully generated streaming intelligent response for continuation query: '{message[:50]}...')")
        
        # Yield final response with complete answer and metadata
        final_response = {
            "answer": collected_response,
            "conversation_id": conversation_id,
            "sources": formatted_sources,
            "metadata": {
                "routing": "llm_intelligent_streaming_response",
                "retrieval_triggered": False,
                "conversation_aware": True,
                "had_conversation_context": bool(conversation_context),
                "had_cached_sources": bool(cached_context and cached_context.get('sources')),
                "source_count": len(formatted_sources)
            }
        }
        
        yield json.dumps(final_response) + "\n"
        
        logger.info(f"[LLM_INTELLIGENCE_STREAMING] Generated intelligent streaming response using conversation context")
        
    except Exception as e:
        logger.error(f"[LLM_INTELLIGENCE_STREAMING] Failed to generate intelligent streaming response: {str(e)}")
        # Re-raise to indicate failure - caller handles fallback to regular retrieval
        raise


# Keep the original function for backward compatibility
async def respond_with_llm_intelligence(message: str, cached_context: Dict[str, Any], conversation_context: Dict[str, Any], notebook_id: str, conversation_id: str) -> Dict[str, Any]:
    """
    Generate response using LLM-driven conversation intelligence.
    
    DEPRECATED: Use respond_with_llm_intelligence_streaming instead for better React state handling.
    
    The LLM sees both the cached sources and the previous conversation context,
    allowing it to make intelligent decisions about data reuse vs new retrieval.
    
    Args:
        message: The user's current message
        cached_context: Previously cached retrieval context (if available)
        conversation_context: Last conversation exchange context
        notebook_id: The notebook ID
        conversation_id: The conversation ID
        
    Returns:
        Dict containing response or None if LLM requests new retrieval
    """
    try:
        # Get notebook LLM configuration
        llm_config = get_notebook_llm_full_config()
        
        if not llm_config:
            logger.error("Failed to get notebook LLM configuration for intelligent response")
            return None
            
        # Use cached LLM instance to avoid creating new connections
        llm = await get_cached_notebook_llm(llm_config)
        
        # Build context information - prioritize extracted entities over raw sources
        available_sources = []
        
        # First, try to use extracted/structured entities (e.g., projects)
        if cached_context and cached_context.get('extracted_entities'):
            logger.info(f"[CACHE_ENTITIES] Using {len(cached_context['extracted_entities'])} extracted entities from cache")
            for idx, entity in enumerate(cached_context['extracted_entities']):
                if isinstance(entity, dict):
                    # Handle structured project data
                    title = entity.get('title', entity.get('name', f'Entity {idx + 1}'))
                    description = entity.get('description', entity.get('summary', ''))
                    available_sources.append(f"Project {idx + 1}: {title} - {description}")
                else:
                    available_sources.append(f"Entity {idx + 1}: {str(entity)}")
        
        # Fallback to raw sources if no extracted entities available
        elif cached_context and cached_context.get('sources'):
            logger.info(f"[CACHE_SOURCES] Using {len(cached_context['sources'])} raw sources from cache (no extracted entities)")
            for idx, source in enumerate(cached_context['sources'][:10]):
                # Preserve significantly more context for reordering/filtering decisions
                content = source.get('content', '')
                truncated_content = content[:1500] + ('...' if len(content) > 1500 else '')
                available_sources.append(f"Source {idx + 1}: {truncated_content}")
        
        sources_context = "\n\n".join(available_sources) if available_sources else "No previous sources available."
        
        # Build conversation memory context
        conversation_memory = ""
        if conversation_context:
            last_user_msg = conversation_context.get('user_message', '')
            last_ai_response = conversation_context.get('ai_response', '')
            
            if last_user_msg and last_ai_response:
                conversation_memory = f"""
RECENT CONVERSATION:
User: {last_user_msg}
Assistant: {last_ai_response[:500]}{'...' if len(last_ai_response) > 500 else ''}
"""
        
        # Create intelligent conversation prompt with CLEAR examples and CONSERVATIVE retrieval logic
        system_prompt = f"""You are an intelligent AI assistant helping with notebook content analysis. Your role is to provide helpful responses while being conversationally aware.

CONVERSATION INTELLIGENCE INSTRUCTIONS:
1. You have access to previously retrieved information from the notebook
2. You can see the recent conversation context 
3. Your job is to determine if you can answer the current question with existing information OR if you need new data retrieval

DECISION LOGIC - BE CONSERVATIVE ABOUT NEW RETRIEVAL:
✅ ANSWER WITH AVAILABLE DATA if the user asks for:
  - Reordering: "order by date", "sort by name", "arrange by year"
  - Filtering: "only show 2023", "just the Python projects", "remove duplicates"
  - Analysis: "summarize that list", "what's the pattern", "compare these"
  - References: "give me that list again", "show me those results", "the previous data"
  - Clarification: "explain the first one", "tell me more about X from the list"

❌ REQUEST NEW RETRIEVAL only if:
  - User asks for completely different information not in sources
  - User requests new topics/entities not mentioned in available data
  - Sources are clearly insufficient for the specific question

EXAMPLE SCENARIOS:
- "order by end year, give me list again" → ANSWER (reorder existing data)
- "show me only projects from 2023" → ANSWER (filter existing data)  
- "tell me about quantum computing projects" (when sources are about web development) → NEED_NEW_RETRIEVAL
- "show that list again" → ANSWER (display existing data)

AVAILABLE INFORMATION:
{sources_context}

{conversation_memory}

Current user question: "{message}"

Respond naturally and conversationally. PRIORITIZE using available data. Only request retrieval if you absolutely cannot answer with current information."""

        user_prompt = f"""Based on the available information and conversation context, please respond to: {message}

CONTEXT AWARENESS - CRITICAL:
- Display references like "that list", "those results", "the data", "show again" refer to previously shown information
- Ordering/sorting requests ("order by", "sort by", "arrange") should use existing data
- Filtering requests ("only show", "just the", "remove") should use existing data  
- Simple continuation phrases ("give me list again", "show that") mean reuse previous results

FRESH DATA REQUESTS - REQUEST NEW RETRIEVAL for compound phrases indicating fresh searches:
- "find ... again", "query ... again", "search ... again" (action + again = fresh search)

ONLY request new retrieval if you need completely different information not available in the sources above OR if the user explicitly requests fresh data using the patterns above."""

        # Generate LLM decision and response
        response = await llm.chat(
            user_prompt,
            system_prompt=system_prompt,
            max_tokens=2000
        )
        
        # Check if LLM requests new retrieval
        if response and "NEED_NEW_RETRIEVAL" in response.text:
            logger.info(f"[LLM_INTELLIGENCE] LLM requests new retrieval: {response.text}")
            return None  # Signal that new retrieval is needed
        
        # Enhanced logging for debugging continuation query failures
        if not response or len(response.text.strip()) < 10:
            logger.warning(f"[LLM_INTELLIGENCE] LLM provided insufficient response for message: '{message}' - Response: '{response.text}'")
            return None
            
        logger.info(f"[LLM_INTELLIGENCE] Successfully generated intelligent response for continuation query: '{message[:50]}...')")
        
        # Format existing sources if using cached context
        formatted_sources = []
        if cached_context and cached_context.get('sources'):
            for source in cached_context['sources']:
                if isinstance(source, dict):
                    formatted_sources.append({
                        'content': source.get('content', ''),
                        'metadata': source.get('metadata', {}),
                        'score': source.get('score', 0.0)
                    })
        
        logger.info(f"[LLM_INTELLIGENCE] Generated intelligent response using conversation context")
        
        return {
            "answer": response.text,
            "conversation_id": conversation_id,
            "sources": formatted_sources,
            "metadata": {
                "routing": "llm_intelligent_response",
                "retrieval_triggered": False,
                "conversation_aware": True,
                "had_conversation_context": bool(conversation_context),
                "had_cached_sources": bool(cached_context and cached_context.get('sources')),
                "source_count": len(formatted_sources)
            }
        }
        
    except Exception as e:
        logger.error(f"[LLM_INTELLIGENCE] Failed to generate intelligent response: {str(e)}")
        # Return None to fall back to regular retrieval
        return None


def _should_use_intelligent_planning(query: str, intent_analysis: dict, total_available: int) -> bool:
    """
    Detect when to use intelligent AI planning pipeline.
    
    Uses query analysis to determine if the request requires intelligent
    planning, execution, and verification for complete, accurate results.
    
    Args:
        query: User query string
        intent_analysis: Results from query intent analysis
        total_available: Total items available in notebook
        
    Returns:
        bool: True if intelligent pipeline should be used
    """
    query_lower = query.lower()
    
    # Analysis query indicators - complex analytical requests
    analysis_indicators = [
        'analyze', 'table format', 'in table', 
        'as table', 'table form', 'complete list', 'full list',
        'comprehensive', 'overview', 'summary'
    ]
    
    # Complex analytical indicators - requiring structured analysis
    analytical_indicators = [
        'analyze', 'compare', 'categorize', 'organize', 'group by',
        'break down', 'summarize by', 'pattern', 'trend', 'relationship',
        'correlation', 'distribution', 'statistics'
    ]
    
    # Quality-critical indicators - high precision requirements
    quality_indicators = [
        'count', 'total', 'exactly', 'precise', 'accurate', 'complete',
        'comprehensive', 'thorough', 'detailed', 'exhaustive'
    ]
    
    # Check for analysis queries
    has_analysis = any(indicator in query_lower for indicator in analysis_indicators)
    
    # Check for analytical complexity  
    has_analytical = any(indicator in query_lower for indicator in analytical_indicators)
    
    # Check for quality requirements
    needs_precision = any(indicator in query_lower for indicator in quality_indicators)
    
    # Check intent analysis results
    wants_comprehensive = intent_analysis.get('wants_comprehensive', False)
    query_type = intent_analysis.get('query_type', 'filtered')
    confidence = intent_analysis.get('confidence', 0.8)
    
    # Use intelligent pipeline if:
    # 1. Explicit analysis request
    # 2. High-complexity analytical query
    # 3. Quality-critical requirements
    # 4. Intent analysis indicates comprehensive need
    # 5. Large dataset with structured requirements
    
    should_use = (
        has_analysis or
        has_analytical or 
        needs_precision or
        wants_comprehensive or
        (query_type == 'comprehensive' and confidence > 0.7) or
        (total_available > 50 and (has_analysis or wants_comprehensive))
    )
    
    if should_use:
        logger.info(f"[AI_PIPELINE] Intelligent planning triggered: analysis={has_analysis}, "
                   f"analytical={has_analytical}, precision={needs_precision}, comprehensive={wants_comprehensive}")
    
    return should_use

async def _analyze_query_intent(query: str) -> dict:
    """
    Analyze query intent using AI-powered QueryIntentAnalyzer for intelligent understanding.
    
    Provides comprehensive intent analysis including:
    - Query type detection (comprehensive, filtered, specific)
    - Quantity intent analysis (all, limited, few, single)
    - Confidence scoring and semantic understanding
    - Context-aware categorization
    
    Args:
        query: The user query string
        
    Returns:
        dict: Query analysis results with intent classification and confidence
    """
    try:
        from app.services.query_intent_analyzer import analyze_query_intent
        
        # Use AI-powered intent analysis with notebook LLM config
        notebook_llm_config = get_notebook_llm_full_config()
        intent_result = await analyze_query_intent(query, llm_config=notebook_llm_config)
        
        # Transform to expected format while preserving AI insights
        return {
            "wants_comprehensive": intent_result.get('quantity_intent') == 'all' or intent_result.get('scope') == 'comprehensive',
            "confidence": intent_result.get('confidence', 0.8),
            "query_type": intent_result.get('scope', 'filtered'),
            "quantity_intent": intent_result.get('quantity_intent', 'limited'),
            "user_type": intent_result.get('user_type', 'casual'),
            "completeness_preference": intent_result.get('completeness_preference', 'balanced'),
            "context": intent_result.get('context', {}),
            "reasoning": intent_result.get('reasoning', 'AI semantic analysis'),
            "ai_powered": True
        }
        
    except Exception as e:
        logger.warning(f"AI intent analysis failed, using semantic fallback: {str(e)}")
        # Intelligent fallback based on query characteristics
        query_lower = query.lower()
        
        # Semantic indicators for comprehensive queries
        comprehensive_indicators = (
            any(word in query_lower for word in ['all', 'every', 'complete', 'comprehensive', 'overview', 'summary']) or
            len(query.split()) > 8 or  # Complex queries often want comprehensive results
            any(phrase in query_lower for phrase in ['give me', 'show me', 'list all', 'find everything'])
        )
        
        return {
            "wants_comprehensive": comprehensive_indicators,
            "confidence": 0.7,  # Moderate confidence for semantic fallback
            "query_type": "comprehensive" if comprehensive_indicators else "filtered",
            "quantity_intent": "all" if comprehensive_indicators else "limited",
            "fallback_reason": str(e),
            "ai_powered": False
        }


# Initialize services
notebook_service = NotebookService()
notebook_rag_service = NotebookRAGService()
chunk_management_service = ChunkManagementService()

# HTTP embedding function for document processing
import requests

class HTTPEmbeddingFunction:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        if not self.endpoint:
            raise ValueError("Embedding endpoint must be provided - no hardcoding allowed")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            normalized_text = text.lower().strip()
            payload = {"texts": [normalized_text]}
            
            try:
                resp = requests.post(
                    self.endpoint, 
                    json=payload,
                    timeout=30
                )
                resp.raise_for_status()
                embedding_data = resp.json()["embeddings"][0]
                embeddings.append(embedding_data)
                
            except requests.exceptions.Timeout as e:
                logger.error(f"Embedding service timeout: {str(e)}")
                raise HTTPException(status_code=504, detail=f"Embedding service timeout: {str(e)}")
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Embedding service connection error: {str(e)}")
                raise HTTPException(status_code=503, detail=f"Embedding service unavailable: {str(e)}")
            except Exception as e:
                logger.error(f"Embedding error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")
                
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# Milvus utilities
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

def ensure_milvus_collection(collection_name: str, vector_dim: int, uri: str, token: str):
    """Ensure Milvus collection exists with proper schema"""
    connections.connect(uri=uri, token=token)
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="uploaded_at", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="bm25_tokens", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="bm25_term_count", dtype=DataType.INT64),
            FieldSchema(name="bm25_unique_terms", dtype=DataType.INT64),
            FieldSchema(name="bm25_top_terms", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="creation_date", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="last_modified_date", dtype=DataType.VARCHAR, max_length=100),
        ]
        schema = CollectionSchema(fields, description="Knowledge base with metadata, deduplication support")
        collection = Collection(collection_name, schema)
    
    # Create index for vector field if not exists
    has_index = False
    for idx in collection.indexes:
        if idx.field_name == "vector":
            has_index = True
            break
    if not has_index:
        collection.create_index(
            field_name="vector",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 1024}
            }
        )
    return collection

@router.post("/", response_model=NotebookResponse, status_code=201)
async def create_notebook(
    request: NotebookCreateRequest,
    db: Session = Depends(get_db)
):
    """
    Create a new notebook.
    
    Args:
        request: Notebook creation parameters
        db: Database session
        
    Returns:
        Created notebook details
        
    Raises:
        HTTPException: If creation fails
    """
    try:
        logger.info(f"Creating notebook: {request.name}")
        
        notebook = await notebook_service.create_notebook(
            db=db,
            name=request.name,
            description=request.description,
            user_id=request.user_id,
            source_filter=request.source_filter,
            metadata=request.metadata
        )
        
        logger.info(f"Successfully created notebook {notebook.id}")
        return notebook
        
    except ValueError as e:
        logger.error(f"Validation error creating notebook: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        logger.error(f"Database integrity error creating notebook: {str(e)}")
        raise HTTPException(status_code=409, detail="Notebook with this name may already exist")
    except DatabaseError as e:
        logger.error(f"Database error creating notebook: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    except Exception as e:
        logger.error(f"Unexpected error creating notebook: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/", response_model=NotebookListResponse)
async def list_notebooks(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    search: Optional[str] = Query(None, description="Search in name or description"),
    db: Session = Depends(get_db)
):
    """
    List notebooks with pagination and filtering.
    
    Args:
        page: Page number (1-based)
        page_size: Number of items per page
        user_id: Filter by user ID
        search: Search query for name/description
        db: Database session
        
    Returns:
        Paginated list of notebooks
    """
    try:
        logger.info(f"Listing notebooks: page={page}, size={page_size}, user_id={user_id}")
        
        result = await notebook_service.list_notebooks(
            db=db,
            page=page,
            page_size=page_size,
            user_id=user_id,
            search=search
        )
        
        return result
        
    except ValueError as e:
        logger.error(f"Validation error listing notebooks: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except DatabaseError as e:
        logger.error(f"Database error listing notebooks: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    except Exception as e:
        logger.error(f"Unexpected error listing notebooks: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{notebook_id}", response_model=NotebookDetailResponse)
async def get_notebook(
    notebook_id: str = Path(..., description="Notebook ID"),
    db: Session = Depends(get_db)
):
    """
    Get detailed notebook information including documents and conversations.
    
    Args:
        notebook_id: Notebook ID
        db: Database session
        
    Returns:
        Detailed notebook information
        
    Raises:
        HTTPException: If notebook not found
    """
    try:
        logger.info(f"Getting notebook details: {notebook_id}")
        
        notebook = await notebook_service.get_notebook_detail(db=db, notebook_id=notebook_id)
        
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        return notebook
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error getting notebook: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except DatabaseError as e:
        logger.error(f"Database error getting notebook: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    except Exception as e:
        logger.error(f"Unexpected error getting notebook: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/{notebook_id}", response_model=NotebookResponse)
async def update_notebook(
    notebook_id: str = Path(..., description="Notebook ID"),
    request: NotebookUpdateRequest = None,
    db: Session = Depends(get_db)
):
    """
    Update notebook information.
    
    Args:
        notebook_id: Notebook ID
        request: Update parameters
        db: Database session
        
    Returns:
        Updated notebook details
        
    Raises:
        HTTPException: If notebook not found or update fails
    """
    try:
        logger.info(f"Updating notebook: {notebook_id}")
        
        if not request:
            raise HTTPException(status_code=400, detail="Update request body is required")
        
        notebook = await notebook_service.update_notebook(
            db=db,
            notebook_id=notebook_id,
            name=request.name,
            description=request.description,
            user_id=request.user_id,
            source_filter=request.source_filter,
            metadata=request.metadata
        )
        
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        logger.info(f"Successfully updated notebook {notebook_id}")
        return notebook
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error updating notebook: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        logger.error(f"Database integrity error updating notebook: {str(e)}")
        raise HTTPException(status_code=409, detail="Update would violate constraints")
    except DatabaseError as e:
        logger.error(f"Database error updating notebook: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    except Exception as e:
        logger.error(f"Unexpected error updating notebook: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{notebook_id}", response_model=NotebookOperationResponse)
async def delete_notebook(
    notebook_id: str = Path(..., description="Notebook ID"),
    db: Session = Depends(get_db)
):
    """
    Delete a notebook and all associated data.
    
    Args:
        notebook_id: Notebook ID
        db: Database session
        
    Returns:
        Operation result
        
    Raises:
        HTTPException: If notebook not found or deletion fails
    """
    try:
        logger.info(f"Deleting notebook: {notebook_id}")
        
        success = await notebook_service.delete_notebook(db=db, notebook_id=notebook_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        logger.info(f"Successfully deleted notebook {notebook_id}")
        return NotebookOperationResponse(
            success=True,
            message=f"Notebook {notebook_id} deleted successfully"
        )
        
    except HTTPException:
        raise
    except DatabaseError as e:
        logger.error(f"Database error deleting notebook: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    except Exception as e:
        logger.error(f"Unexpected error deleting notebook: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{notebook_id}/documents", response_model=NotebookDocumentResponse, status_code=201)
async def add_document_to_notebook(
    notebook_id: str = Path(..., description="Notebook ID"),
    request: NotebookDocumentAddRequest = None,
    db: Session = Depends(get_db)
):
    """
    Add a document to a notebook.
    
    Args:
        notebook_id: Notebook ID
        request: Document addition parameters
        db: Database session
        
    Returns:
        Added document details
        
    Raises:
        HTTPException: If notebook not found or addition fails
    """
    try:
        logger.info(f"Adding document {request.document_id} to notebook {notebook_id}")
        
        if not request:
            raise HTTPException(status_code=400, detail="Document request body is required")
        
        document = await notebook_service.add_document_to_notebook(
            db=db,
            notebook_id=notebook_id,
            document_id=request.document_id,
            document_name=request.document_name,
            document_type=request.document_type,
            milvus_collection=request.milvus_collection,
            metadata=request.metadata
        )
        
        if not document:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        logger.info(f"Successfully added document to notebook {notebook_id}")
        return document
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error adding document: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        logger.error(f"Database integrity error adding document: {str(e)}")
        raise HTTPException(status_code=409, detail="Document may already exist in notebook")
    except DatabaseError as e:
        logger.error(f"Database error adding document: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    except Exception as e:
        logger.error(f"Unexpected error adding document: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{notebook_id}/documents/{document_id}", response_model=NotebookOperationResponse)
async def remove_document_from_notebook(
    notebook_id: str = Path(..., description="Notebook ID"),
    document_id: str = Path(..., description="Document ID"),
    db: Session = Depends(get_db)
):
    """
    Remove a document from a notebook.
    
    Args:
        notebook_id: Notebook ID
        document_id: Document ID to remove
        db: Database session
        
    Returns:
        Operation result
        
    Raises:
        HTTPException: If notebook or document not found
    """
    try:
        logger.info(f"Removing document {document_id} from notebook {notebook_id}")
        
        success = await notebook_service.remove_document_from_notebook(
            db=db,
            notebook_id=notebook_id,
            document_id=document_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Notebook or document not found")
        
        logger.info(f"Successfully removed document from notebook {notebook_id}")
        return NotebookOperationResponse(
            success=True,
            message=f"Document {document_id} removed from notebook"
        )
        
    except HTTPException:
        raise
    except DatabaseError as e:
        logger.error(f"Database error removing document: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    except Exception as e:
        logger.error(f"Unexpected error removing document: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{notebook_id}/query", response_model=NotebookRAGResponse)
async def query_notebook(
    notebook_id: str = Path(..., description="Notebook ID"),
    request: NotebookRAGRequest = None,
    db: Session = Depends(get_db)
):
    """
    Query notebook documents using RAG.
    
    Args:
        notebook_id: Notebook ID
        request: Query parameters
        db: Database session
        
    Returns:
        RAG query results
        
    Raises:
        HTTPException: If notebook not found or query fails
    """
    try:
        logger.info(f"Querying notebook {notebook_id} with query: {request.query[:50]}...")
        
        if not request:
            raise HTTPException(status_code=400, detail="Query request body is required")
        
        # Verify notebook exists
        notebook_exists = await notebook_service.notebook_exists(db=db, notebook_id=notebook_id)
        if not notebook_exists:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        result = await notebook_rag_service.query_notebook(
            db=db,
            notebook_id=notebook_id,
            query=request.query,
            top_k=request.top_k,
            include_metadata=request.include_metadata,
            collection_filter=request.collection_filter
        )
        
        logger.info(f"Successfully queried notebook {notebook_id}, found {len(result.sources)} sources")
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error querying notebook: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error querying notebook: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{notebook_id}/query-optimized", response_model=NotebookRAGResponse)
async def notebook_rag_query_optimized(
    notebook_id: str = Path(..., description="Notebook ID"),
    request: NotebookRAGRequest = Body(..., description="RAG query request"),
    db: Session = Depends(get_db)
):
    """
    Enhanced RAG query with hierarchical retrieval and context optimization.
    Uses Google NotebookLM-like strategies to manage context window efficiently.
    """
    try:
        logger.info(f"Optimized RAG query for notebook {notebook_id}: '{request.query[:100]}...'")
        
        # Verify notebook exists
        notebook_exists = await notebook_service.notebook_exists(db=db, notebook_id=notebook_id)
        if not notebook_exists:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        # Use hierarchical RAG service
        hierarchical_rag_service = get_hierarchical_notebook_rag_service()
        result = await hierarchical_rag_service.query_with_context_optimization(
            notebook_id=notebook_id,
            query=request.query,
            top_k=request.top_k,
            include_metadata=request.include_metadata,
            collection_filter=request.collection_filter,
            db=db
        )

        logger.info(f"Optimized RAG query completed: {len(result.sources)} sources, strategy: {result.metadata.get('strategy', 'unknown')}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during optimized notebook RAG query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{notebook_id}/context-stats")
async def get_notebook_context_stats(
    notebook_id: str = Path(..., description="Notebook ID"),
    query: str = Query(..., description="Sample query to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get context usage statistics for notebook RAG queries.
    Helps understand token usage and optimization opportunities.
    """
    try:
        # Verify notebook exists
        notebook_exists = await notebook_service.notebook_exists(db=db, notebook_id=notebook_id)
        if not notebook_exists:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        # Get context statistics
        hierarchical_rag_service = get_hierarchical_notebook_rag_service()
        stats = await hierarchical_rag_service.get_context_stats(notebook_id, query)
        
        return {
            "notebook_id": notebook_id,
            "query_sample": query[:100] + "..." if len(query) > 100 else query,
            "context_analysis": stats,
            "recommendations": {
                "use_hierarchical": stats.get("optimization_needed", False),
                "recommended_max_chunks": stats.get("recommended_max_chunks", 10),
                "current_efficiency": "low" if stats.get("would_exceed_budget", False) else "good"
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting context stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{notebook_id}/conversations", response_model=NotebookConversationResponse, status_code=201)
async def start_notebook_conversation(
    notebook_id: str = Path(..., description="Notebook ID"),
    request: NotebookConversationRequest = None,
    db: Session = Depends(get_db)
):
    """
    Start a new conversation with the notebook.
    
    Args:
        notebook_id: Notebook ID
        request: Conversation parameters
        db: Database session
        
    Returns:
        Conversation details
        
    Raises:
        HTTPException: If notebook not found or conversation creation fails
    """
    try:
        logger.info(f"Starting conversation {request.conversation_id} for notebook {notebook_id}")
        
        if not request:
            raise HTTPException(status_code=400, detail="Conversation request body is required")
        
        conversation = await notebook_service.start_conversation(
            db=db,
            notebook_id=notebook_id,
            conversation_id=request.conversation_id
        )
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        logger.info(f"Successfully started conversation for notebook {notebook_id}")
        return conversation
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error starting conversation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        logger.error(f"Database integrity error starting conversation: {str(e)}")
        raise HTTPException(status_code=409, detail="Conversation may already exist")
    except DatabaseError as e:
        logger.error(f"Database error starting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    except Exception as e:
        logger.error(f"Unexpected error starting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{notebook_id}/stats", response_model=NotebookStatsResponse)
async def get_notebook_stats(
    notebook_id: str = Path(..., description="Notebook ID"),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive statistics for a notebook.
    
    Args:
        notebook_id: Notebook ID
        db: Database session
        
    Returns:
        Notebook statistics
        
    Raises:
        HTTPException: If notebook not found
    """
    try:
        logger.info(f"Getting stats for notebook {notebook_id}")
        
        stats = await notebook_service.get_notebook_stats(db=db, notebook_id=notebook_id)
        
        if not stats:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        return stats
        
    except HTTPException:
        raise
    except DatabaseError as e:
        logger.error(f"Database error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    except Exception as e:
        logger.error(f"Unexpected error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{notebook_id}/upload", response_model=NotebookUploadResponse, status_code=201)
async def upload_file_to_notebook(
    notebook_id: str = Path(..., description="Notebook ID"),
    file: UploadFile = File(..., description="File to upload"),
    db: Session = Depends(get_db)
):
    """
    Upload a file directly to a notebook.
    
    All notebook documents are automatically stored in the 'notebooks' collection for
    consistent organization and retrieval.
    
    This endpoint accepts file upload, processes it (extracts text, creates embeddings),
    adds it to the vector store, and then links it to the specified notebook.
    
    Args:
        notebook_id: Notebook ID to add the document to
        file: File to upload (PDF, TXT, etc.)
        db: Database session
        
    Returns:
        Upload result with document information, collection set to 'notebooks'
        
    Raises:
        HTTPException: If notebook not found, notebooks collection missing, or processing fails
    """
    try:
        logger.info(f"Uploading file {file.filename} to notebook {notebook_id}")
        
        # 1. Verify notebook exists
        notebook_exists = await notebook_service.notebook_exists(db=db, notebook_id=notebook_id)
        if not notebook_exists:
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        # 2. Save file temporarily
        temp_path = f"/tmp/{file.filename}"
        try:
            with open(temp_path, "wb") as f:
                f.write(await file.read())
            logger.info(f"📁 Saved {file.filename} to {temp_path}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to save file: {str(e)}")
        
        # 3. Load document based on file type
        try:
            if file.filename.lower().endswith('.pdf'):
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                logger.info(f"✅ PDF loaded: {len(docs)} pages from {file.filename}")
            else:
                # For non-PDF files, read as text
                with open(temp_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                from langchain.schema import Document
                docs = [Document(page_content=content, metadata={"source": file.filename, "page": 0})]
                logger.info(f"✅ Text file loaded: {file.filename}")
                
        except Exception as e:
            logger.error(f"❌ Error loading file: {e}")
            os.remove(temp_path) if os.path.exists(temp_path) else None
            raise HTTPException(status_code=400, detail=f"Failed to load file: {str(e)}")
        
        # 4. Text splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False
        )
        
        chunks = splitter.split_documents(docs)
        total_chunks = len(chunks)
        logger.info(f"✅ Created {total_chunks} chunks")
        
        # 5. Generate file ID and metadata
        with open(temp_path, "rb") as f:
            file_content = f.read()
            file_id = hashlib.sha256(file_content).hexdigest()[:12]
            file_size_bytes = len(file_content)
        
        logger.info(f"🔑 File ID: {file_id}")
        
        # Initialize BM25 processor
        bm25_processor = BM25Processor()
        
        # Extract file metadata
        file_metadata = MetadataExtractor.extract_metadata(temp_path, file.filename)
        
        # 6. Enhanced metadata setting
        for i, chunk in enumerate(chunks):
            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                chunk.metadata = {}
            
            original_page = chunk.metadata.get('page', 0)
            
            chunk.metadata.update({
                'source': file.filename.lower(),
                'page': original_page,
                'doc_type': 'pdf' if file.filename.lower().endswith('.pdf') else 'text',
                'uploaded_at': datetime.now().isoformat(),
                'section': f"chunk_{i}",
                'author': '',
                'chunk_index': i,
                'file_id': file_id,
                'creation_date': file_metadata['creation_date'],
                'last_modified_date': file_metadata['last_modified_date']
            })
            
            # Add hash and doc_id
            chunk_hash = hash_text(chunk.page_content)
            chunk.metadata['hash'] = chunk_hash
            chunk.metadata['doc_id'] = f"{file_id}_p{original_page}_c{i}"
            
            # Add BM25 preprocessing
            bm25_metadata = bm25_processor.prepare_document_for_bm25(chunk.page_content, chunk.metadata)
            chunk.metadata.update(bm25_metadata)
        
        # 7. Determine target collection - notebooks always use 'notebooks' collection
        target_collection = "notebooks"
        logger.info(f"📚 Using dedicated notebooks collection: {target_collection}")
        
        # Validate that the notebooks collection exists
        collection_info = get_collection_config(target_collection)
        if not collection_info:
            raise HTTPException(
                status_code=500, 
                detail=f"Notebooks collection '{target_collection}' not found. Please create it first."
            )
        
        # Add collection info to chunk metadata
        for chunk in chunks:
            chunk.metadata['collection_name'] = target_collection
        
        # 8. Vector DB storage
        from app.utils.vector_db_migration import migrate_vector_db_settings
        vector_db_cfg = migrate_vector_db_settings(get_vector_db_settings())
        
        # Find active Milvus database
        milvus_db = None
        for db_config in vector_db_cfg.get("databases", []):
            if db_config.get("id") == "milvus" and db_config.get("enabled"):
                milvus_db = db_config
                break
        
        if not milvus_db:
            os.remove(temp_path) if os.path.exists(temp_path) else None
            raise HTTPException(status_code=500, detail="No active Milvus configuration found")
        
        logger.info("🔄 Using Milvus vector database")
        milvus_cfg = milvus_db.get("config", {})
        uri = milvus_cfg.get("MILVUS_URI")
        token = milvus_cfg.get("MILVUS_TOKEN")
        collection = target_collection
        vector_dim = int(milvus_cfg.get("dimension", 2560))
        
        # Ensure collection exists
        collection_obj = ensure_milvus_collection(collection, vector_dim=vector_dim, uri=uri, token=token)
        
        # Get embedding configuration
        embedding_cfg = get_embedding_settings()
        embedding_model = embedding_cfg.get('embedding_model')
        embedding_endpoint = embedding_cfg.get('embedding_endpoint')
        
        if embedding_endpoint:
            embeddings = HTTPEmbeddingFunction(embedding_endpoint)
            logger.info("✅ Using HTTP embedding endpoint")
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            logger.info(f"✅ Using HuggingFace embeddings: {embedding_model}")
        
        # Enhanced deduplication check
        collection_obj.load()
        existing_hashes = get_existing_hashes(collection_obj)
        existing_doc_ids = get_existing_doc_ids(collection_obj)
        
        logger.info(f"📊 Deduplication analysis:")
        logger.info(f"   - Total chunks to process: {len(chunks)}")
        logger.info(f"   - Existing hashes in DB: {len(existing_hashes)}")
        logger.info(f"   - Existing doc_ids in DB: {len(existing_doc_ids)}")
        
        # Filter duplicates
        unique_chunks = []
        duplicate_count = 0
        
        for chunk in chunks:
            chunk_hash = chunk.metadata.get('hash')
            doc_id = chunk.metadata.get('doc_id')
            
            is_duplicate = (chunk_hash in existing_hashes or doc_id in existing_doc_ids)
            
            if is_duplicate:
                duplicate_count += 1
            else:
                unique_chunks.append(chunk)
        
        logger.info(f"📊 After deduplication:")
        logger.info(f"   - Unique chunks to insert: {len(unique_chunks)}")
        logger.info(f"   - Duplicates filtered: {duplicate_count}")
        
        if not unique_chunks:
            # All chunks are duplicates, but we can still link existing document to notebook
            logger.info(f"📋 Document already processed, checking if linked to notebook...")
            os.remove(temp_path) if os.path.exists(temp_path) else None
            
            # Try to add existing document to notebook
            try:
                result = await notebook_service.add_document_to_notebook(
                    db=db,
                    notebook_id=notebook_id,
                    document_id=file_id,
                    document_name=file.filename,
                    document_type=file.filename.split('.')[-1].lower() if '.' in file.filename else 'unknown',
                    milvus_collection=target_collection,
                    metadata={
                        'file_size_bytes': file_size_bytes,
                        'total_chunks': total_chunks,
                        'unique_chunks': 0,
                        'processing_status': 'already_exists'
                    }
                )
                
                if result:
                    logger.info(f"✅ Linked existing document {file_id} to notebook {notebook_id}")
                    return NotebookUploadResponse(
                        status="success",
                        document_id=file_id,
                        filename=file.filename,
                        file_id=file_id,
                        total_chunks=total_chunks,
                        unique_chunks=0,
                        duplicates_filtered=total_chunks,
                        collection=target_collection,
                        pages_processed=total_chunks,  # Approximate
                        message=f"Document already exists and has been linked to notebook"
                    )
                else:
                    raise HTTPException(status_code=409, detail="Document already exists in this notebook")
                    
            except Exception as link_error:
                logger.error(f"❌ Error linking existing document: {link_error}")
                if "already exist" in str(link_error).lower():
                    raise HTTPException(status_code=409, detail="Document already exists in this notebook")
                else:
                    raise HTTPException(status_code=500, detail=f"Failed to link document: {str(link_error)}")
        
        # Generate embeddings and insert into Milvus
        unique_ids = [str(uuid.uuid4()) for _ in unique_chunks]
        unique_texts = [chunk.page_content for chunk in unique_chunks]
        
        logger.info(f"🔄 Generating embeddings for {len(unique_texts)} chunks...")
        try:
            embeddings_list = embeddings.embed_documents(unique_texts)
            logger.info(f"✅ Generated {len(embeddings_list)} embeddings")
        except Exception as e:
            logger.error(f"❌ Error generating embeddings: {e}")
            os.remove(temp_path) if os.path.exists(temp_path) else None
            raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {str(e)}")
        
        # Prepare data for insertion
        data = [
            unique_ids,
            embeddings_list,
            unique_texts,
            [chunk.metadata.get('source', '') for chunk in unique_chunks],
            [chunk.metadata.get('page', 0) for chunk in unique_chunks],
            [chunk.metadata.get('doc_type', 'text') for chunk in unique_chunks],
            [chunk.metadata.get('uploaded_at', '') for chunk in unique_chunks],
            [chunk.metadata.get('section', '') for chunk in unique_chunks],
            [chunk.metadata.get('author', '') for chunk in unique_chunks],
            [chunk.metadata.get('hash', '') for chunk in unique_chunks],
            [chunk.metadata.get('doc_id', '') for chunk in unique_chunks],
            [chunk.metadata.get('bm25_tokens', '') for chunk in unique_chunks],
            [chunk.metadata.get('bm25_term_count', 0) for chunk in unique_chunks],
            [chunk.metadata.get('bm25_unique_terms', 0) for chunk in unique_chunks],
            [chunk.metadata.get('bm25_top_terms', '') for chunk in unique_chunks],
            [chunk.metadata.get('creation_date', '') for chunk in unique_chunks],
            [chunk.metadata.get('last_modified_date', '') for chunk in unique_chunks],
        ]
        
        logger.info(f"🔄 Inserting {len(unique_chunks)} chunks into Milvus collection '{collection}'...")
        try:
            insert_result = collection_obj.insert(data)
            collection_obj.flush()
            logger.info(f"✅ Successfully inserted {len(unique_chunks)} chunks")
        except Exception as e:
            logger.error(f"❌ Error inserting into Milvus: {e}")
            os.remove(temp_path) if os.path.exists(temp_path) else None
            raise HTTPException(status_code=500, detail=f"Failed to insert into Milvus: {str(e)}")
        
        # 9. Create document record in database (handle duplicates)
        try:
            document = KnowledgeGraphDocument(
                document_id=file_id,
                filename=file.filename,
                file_hash=hashlib.sha256(file_content).hexdigest(),
                file_size_bytes=file_size_bytes,
                file_type=file.filename.split('.')[-1].lower() if '.' in file.filename else 'unknown',
                milvus_collection=collection,
                processing_status='completed',
                chunks_processed=len(unique_chunks),
                total_chunks=len(chunks),
                processing_completed_at=datetime.now()
            )
            
            db.add(document)
            db.commit()
            db.refresh(document)
            logger.info(f"✅ Created new document record: {file_id}")
            
        except Exception as e:
            db.rollback()
            if "duplicate key value violates unique constraint" in str(e):
                # Document already exists, get the existing one
                logger.info(f"📋 Document {file_id} already exists, using existing record")
                document = db.query(KnowledgeGraphDocument).filter(
                    KnowledgeGraphDocument.document_id == file_id
                ).first()
                if not document:
                    raise HTTPException(status_code=500, detail="Document exists but could not retrieve it")
            else:
                logger.error(f"❌ Failed to create document record: {e}")
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
        # 10. Add document to notebook
        try:
            await notebook_service.add_document_to_notebook(
                db=db,
                notebook_id=notebook_id,
                document_id=file_id,
                document_name=file.filename,
                document_type=file.filename.split('.')[-1].lower() if '.' in file.filename else 'unknown',
                milvus_collection=collection,
                metadata={
                    'file_size_bytes': file_size_bytes,
                    'total_chunks': len(chunks),
                    'unique_chunks': len(unique_chunks),
                    'processing_status': 'completed'
                }
            )
            logger.info(f"✅ Added document to notebook {notebook_id}")
        except Exception as e:
            logger.error(f"❌ Error adding document to notebook: {e}")
            # Document is in vector store but not linked to notebook
            raise HTTPException(
                status_code=500, 
                detail=f"Document processed but failed to link to notebook: {str(e)}"
            )
        
        # Cleanup temporary file
        try:
            os.remove(temp_path)
            logger.info(f"🧹 Cleaned up temporary file: {temp_path}")
        except Exception as e:
            logger.warning(f"⚠️  Could not remove temp file: {e}")
        
        return NotebookUploadResponse(
            status="success",
            document_id=file_id,
            filename=file.filename,
            file_id=file_id,
            total_chunks=len(chunks),
            unique_chunks=len(unique_chunks),
            duplicates_filtered=duplicate_count,
            collection=collection,
            pages_processed=len(docs),
            message=f"Successfully uploaded {file.filename} to notebook {notebook_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error uploading file to notebook: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Document Admin Endpoints

@router.get("/documents/{document_id}/usage", response_model=DocumentUsageInfo)
async def get_document_usage_info(
    document_id: str = Path(..., description="Document ID to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get usage information for a document before deletion.
    Shows which notebooks use it and what the impact of deletion will be.
    """
    try:
        logger.info(f"Getting usage info for document: {document_id}")
        
        admin_service = DocumentAdminService()
        usage_info = await admin_service.get_document_usage_info(db, document_id)
        
        if 'error' in usage_info:
            if usage_info['error'] == 'Document not found':
                raise HTTPException(status_code=404, detail="Document not found")
            else:
                raise HTTPException(status_code=500, detail=usage_info['error'])
        
        return DocumentUsageInfo(**usage_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document usage info: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/documents/permanent", response_model=DocumentDeleteResponse)
async def delete_documents_permanently(
    request: DocumentDeleteRequest,
    db: Session = Depends(get_db)
):
    """
    Permanently delete documents from all systems (Milvus, database, notebooks).
    This operation cannot be undone!
    """
    try:
        logger.info(f"Permanent deletion requested for {len(request.document_ids)} documents")
        
        # Additional safety check
        if not request.confirm_permanent_deletion:
            raise HTTPException(
                status_code=400, 
                detail="Must confirm permanent deletion by setting confirm_permanent_deletion=true"
            )
        
        admin_service = DocumentAdminService()
        
        if len(request.document_ids) == 1:
            # Single document deletion
            result = await admin_service.delete_document_permanently(
                db, 
                request.document_ids[0],
                request.remove_from_notebooks
            )
            
            response_data = {
                'success': result.get('success', False),
                'message': f"Document {'successfully' if result.get('success') else 'failed to be'} deleted permanently",
                'total_requested': 1,
                'successful_deletions': 1 if result.get('success') else 0,
                'failed_deletions': 0 if result.get('success') else 1,
                'deletion_details': [DocumentDeletionSummary(**result)],
                'overall_errors': result.get('errors', [])
            }
        else:
            # Bulk document deletion
            result = await admin_service.bulk_delete_documents(
                db,
                request.document_ids,
                request.remove_from_notebooks
            )
            
            response_data = {
                'success': result.get('success', False),
                'message': f"Bulk deletion completed: {result.get('successful_deletions', 0)} successful, {result.get('failed_deletions', 0)} failed",
                'total_requested': result.get('total_requested', 0),
                'successful_deletions': result.get('successful_deletions', 0),
                'failed_deletions': result.get('failed_deletions', 0),
                'deletion_details': [DocumentDeletionSummary(**detail) for detail in result.get('deletion_details', [])],
                'overall_errors': result.get('overall_errors', [])
            }
        
        logger.info(f"Permanent deletion completed: {response_data['successful_deletions']} successful, {response_data['failed_deletions']} failed")
        return DocumentDeleteResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in permanent document deletion: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/documents/{document_id}/permanent", response_model=DocumentDeleteResponse)
async def delete_single_document_permanently(
    document_id: str = Path(..., description="Document ID to delete permanently"),
    remove_from_notebooks: bool = Query(True, description="Remove from all notebooks"),
    confirm: bool = Query(..., description="Confirmation that deletion is permanent"),
    db: Session = Depends(get_db)
):
    """
    Permanently delete a single document from all systems.
    Convenience endpoint for single document deletion.
    """
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Must confirm permanent deletion with confirm=true query parameter"
            )
        
        # Convert to DocumentDeleteRequest format
        request = DocumentDeleteRequest(
            document_ids=[document_id],
            remove_from_notebooks=remove_from_notebooks,
            confirm_permanent_deletion=confirm
        )
        
        # Use the bulk deletion endpoint logic
        return await delete_documents_permanently(request, db)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in single document permanent deletion: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# System-wide Documents Management Endpoint

@router.get("/system/documents", response_model=Dict[str, Any])
async def get_all_system_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Page size"),
    search: Optional[str] = Query(None, description="Search in filename or document_id"),
    file_type: Optional[str] = Query(None, description="Filter by file type"),
    status: Optional[str] = Query(None, description="Filter by processing status"),
    collection: Optional[str] = Query(None, description="Filter by Milvus collection"),
    sort_by: str = Query("created_at", description="Sort field: created_at, filename, file_size_bytes"),
    sort_order: str = Query("desc", description="Sort order: asc, desc"),
    db: Session = Depends(get_db)
):
    """
    Get ALL documents across the entire system with notebook relationships.
    This is the system-wide admin view for document management.
    
    Returns documents with:
    - Document metadata
    - List of notebooks using each document
    - Processing status and collection information
    - File size, type, and creation timestamps
    
    Args:
        page: Page number (1-based)
        page_size: Number of items per page (max 200 for admin)
        search: Search query for filename or document_id
        file_type: Filter by file type (pdf, txt, etc.)
        status: Filter by processing status
        collection: Filter by Milvus collection
        sort_by: Field to sort by
        sort_order: Sort direction (asc/desc)
        db: Database session
        
    Returns:
        Comprehensive system document list with notebook relationships
    """
    try:
        logger.info(f"System documents query: page={page}, size={page_size}, search={search}")
        
        # Build document query with filters using raw SQL for consistency
        where_conditions = []
        params = {"offset": (page - 1) * page_size, "limit": page_size}
        
        if search:
            where_conditions.append("(filename ILIKE :search OR document_id ILIKE :search)")
            params["search"] = f"%{search}%"
            
        if file_type:
            where_conditions.append("file_type = :file_type")
            params["file_type"] = file_type
            
        if status:
            where_conditions.append("processing_status = :status")
            params["status"] = status
            
        if collection:
            where_conditions.append("milvus_collection = :collection")
            params["collection"] = collection
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        # Determine sort field and order
        valid_sort_fields = {"created_at", "filename", "file_size_bytes"}
        sort_field = sort_by if sort_by in valid_sort_fields else "created_at"
        sort_direction = "ASC" if sort_order.lower() == "asc" else "DESC"
        
        # Get total count
        count_query = text(f"""
            SELECT COUNT(*) FROM knowledge_graph_documents 
            WHERE {where_clause}
        """)
        total_count = db.execute(count_query, params).scalar() or 0
        
        # Get documents with pagination
        documents_query = text(f"""
            SELECT * FROM knowledge_graph_documents
            WHERE {where_clause}
            ORDER BY {sort_field} {sort_direction}
            LIMIT :limit OFFSET :offset
        """)
        documents = db.execute(documents_query, params).fetchall()
        
        # Get document IDs for notebook relationship lookup
        document_ids = [doc.document_id for doc in documents]
        
        # Get notebook details efficiently with a join query
        doc_to_notebooks = {}
        if document_ids:
            notebook_query = text("""
                SELECT 
                    nd.document_id,
                    nd.notebook_id,
                    nd.added_at,
                    n.name as notebook_name
                FROM notebook_documents nd
                JOIN notebooks n ON nd.notebook_id = n.id
                WHERE nd.document_id = ANY(:document_ids)
                ORDER BY nd.document_id, nd.added_at DESC
            """)
            
            notebook_results = db.execute(notebook_query, {"document_ids": document_ids}).fetchall()
            
            for row in notebook_results:
                if row.document_id not in doc_to_notebooks:
                    doc_to_notebooks[row.document_id] = []
                
                doc_to_notebooks[row.document_id].append({
                    "id": row.notebook_id,
                    "name": row.notebook_name,
                    "added_at": row.added_at.isoformat() if row.added_at else None
                })
        
        # Build comprehensive response
        system_documents = []
        for doc in documents:
            notebooks_using = doc_to_notebooks.get(doc.document_id, [])
            
            system_documents.append({
                "document_id": doc.document_id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "file_size_bytes": doc.file_size_bytes or 0,
                "processing_status": doc.processing_status,
                "milvus_collection": doc.milvus_collection,
                "created_at": doc.created_at.isoformat() if doc.created_at else "",
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else "",
                "chunks_processed": getattr(doc, 'chunks_processed', 0),
                "total_chunks": getattr(doc, 'total_chunks', 0),
                "file_hash": getattr(doc, 'file_hash', ''),
                
                # Notebook relationship information
                "notebook_count": len(notebooks_using),
                "notebooks_using": notebooks_using,
                "is_orphaned": len(notebooks_using) == 0,
                
                # Admin metadata
                "processing_completed_at": doc.processing_completed_at.isoformat() if getattr(doc, 'processing_completed_at', None) else None,
                "can_be_deleted": True,  # All system documents can be deleted by admin
            })
        
        # Get summary statistics
        stats_query = text("""
            SELECT 
                COUNT(*) as total_documents,
                COUNT(DISTINCT file_type) as unique_file_types,
                COUNT(DISTINCT milvus_collection) as unique_collections,
                COALESCE(SUM(file_size_bytes), 0) as total_size_bytes,
                COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as completed_documents,
                COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed_documents
            FROM knowledge_graph_documents
        """)
        stats_result = db.execute(stats_query).fetchone()
        
        # Count orphaned documents (not in any notebook)
        orphaned_query = text("""
            SELECT COUNT(*)
            FROM knowledge_graph_documents kgd
            LEFT JOIN notebook_documents nd ON kgd.document_id = nd.document_id
            WHERE nd.document_id IS NULL
        """)
        orphaned_count = db.execute(orphaned_query).scalar() or 0
        
        response_data = {
            "documents": system_documents,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": (total_count + page_size - 1) // page_size,
                "has_next": page * page_size < total_count,
                "has_prev": page > 1
            },
            "summary_stats": {
                "total_documents": stats_result.total_documents or 0,
                "unique_file_types": stats_result.unique_file_types or 0,
                "unique_collections": stats_result.unique_collections or 0,
                "total_size_bytes": stats_result.total_size_bytes or 0,
                "completed_documents": stats_result.completed_documents or 0,
                "failed_documents": stats_result.failed_documents or 0,
                "orphaned_documents": orphaned_count
            },
            "filters_applied": {
                "search": search,
                "file_type": file_type,
                "status": status,
                "collection": collection,
                "sort_by": sort_by,
                "sort_order": sort_order
            }
        }
        
        logger.info(f"System documents query completed: {len(system_documents)} documents returned")
        return response_data
        
    except Exception as e:
        logger.error(f"Error getting system documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{notebook_id}/chat-debug")
async def notebook_chat_debug(
    request: Request,
    notebook_id: str = Path(..., description="Notebook ID"),
    db: Session = Depends(get_db)
):
    """Debug endpoint to see raw request body"""
    try:
        body = await request.body()
        body_str = body.decode('utf-8')
        logger.info(f"🔍 Raw request body: {body_str}")
        
        import json
        json_data = json.loads(body_str)
        logger.info(f"📋 Parsed JSON: {json_data}")
        
        # Try to validate manually
        chat_request = NotebookChatRequest(**json_data)
        logger.info(f"✅ Validation successful: {chat_request}")
        
        return {
            "status": "validation_success", 
            "parsed": json_data,
            "validated": {
                "message": chat_request.message,
                "conversation_id": chat_request.conversation_id,
                "include_context": chat_request.include_context,
                "max_sources": chat_request.max_sources
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "validation_failed", "error": str(e), "traceback": traceback.format_exc()}

@router.post("/test-chat-validation")
async def test_chat_validation(request: NotebookChatRequest):
    """Simple test endpoint to validate NotebookChatRequest"""
    return {
        "status": "validation_success",
        "message": request.message,
        "conversation_id": request.conversation_id,
        "include_context": request.include_context,
        "max_sources": request.max_sources
    }


@router.post("/{notebook_id}/chat")
async def notebook_chat(
    request: NotebookChatRequest,
    notebook_id: str = Path(..., description="Notebook ID"),
    db: Session = Depends(get_db)
):
    """
    Chat with notebook using RAG-powered AI assistant.
    
    Provides streaming responses based on notebook documents and context.
    Uses NotebookRAGService to query relevant content and generates
    conversational responses using the configured LLM.
    
    Args:
        notebook_id: Notebook ID to chat with
        request: Chat request parameters
        db: Database session
        
    Returns:
        StreamingResponse: Real-time chat response
        
    Raises:
        HTTPException: If notebook not found or chat fails
    """
    # Get configurable timeout for intelligent plan execution
    base_timeout = get_intelligent_plan_timeout()
    
    # Calculate dynamic timeout based on query characteristics and expected source count
    # For comprehensive queries, use higher timeout
    is_comprehensive_query = any(keyword in request.message.lower() for keyword in 
                                 ['all projects', 'all experiences', 'complete list', 'entire', 'comprehensive'])
    
    if is_comprehensive_query:
        plan_timeout = max(base_timeout, 480)  # 8 minutes for comprehensive queries
        logger.info(f"[TIMEOUT_DYNAMIC] Using extended timeout {plan_timeout}s for comprehensive query")
    else:
        plan_timeout = base_timeout
    
    # === REMOVED EXECUTION STATE TRACKING ===
    # Removed to fix Redis async/sync errors and over-complication
    # execution_state = await create_request_state(request.conversation_id, request.message)
    # request_id = execution_state.request_id
    # logger.info(f"[EXECUTION_STATE] Created state tracking for request {request_id[:16]}...")
    
    
    logger.info(f"📨 Notebook chat request received for {notebook_id}")
    logger.info(f"🔍 Request details: message='{request.message}', conversation_id='{request.conversation_id}', include_context={request.include_context}, max_sources={request.max_sources}")
    
    async def stream_chat_response():
        try:
            logger.info(f"Starting notebook chat for {notebook_id}: {request.message[:50]}...")
            logger.info(f"Chat request details: message='{request.message}', conversation_id='{request.conversation_id}', include_context={request.include_context}, max_sources={request.max_sources}")
            
            # Initialize variables early to prevent UnboundLocalError
            execution_plan = None
            verification_result = None
            
            if not request or not request.message:
                yield json.dumps({
                    "error": "Chat request validation failed: message is required",
                    "error_details": {
                        "error_type": "ValidationError",
                        "notebook_id": notebook_id,
                        "request_provided": bool(request),
                        "message_provided": bool(request and request.message) if request else False
                    },
                    "troubleshooting": "Ensure request body contains a 'message' field with non-empty text"
                }) + "\n"
                return
            
            # Verify notebook exists
            notebook_exists = await notebook_service.notebook_exists(db=db, notebook_id=notebook_id)
            if not notebook_exists:
                yield json.dumps({
                    "error": "Notebook access failed: notebook not found or inaccessible",
                    "error_details": {
                        "error_type": "NotFoundError",
                        "notebook_id": notebook_id,
                        "notebook_exists": False
                    },
                    "troubleshooting": "Verify notebook ID exists and you have access permissions"
                }) + "\n"
                return
            
            # === PHASE 1: CONVERSATION INTELLIGENCE FIRST ===
            # Check conversation context FIRST before any classification or routing
            logger.info("[CONVERSATION_FIRST] Checking for conversation context before classification")
            
            # Classify query intent to determine appropriate handling strategy
            query_type = _classify_query_intent(request.message)
            logger.info(f"[QUERY_CLASSIFICATION] Query classified as: {query_type}")
            
            # Get conversation context (previous exchange)
            conversation_context = await conversation_context_manager.get_conversation_context(request.conversation_id)
            
            # === CACHE BYPASS DETECTION: Check if user query requires fresh data retrieval ===
            bypass_decision = cache_bypass_detector.should_bypass_cache(request.message, request.conversation_id)
            logger.info(f"[CACHE_BYPASS] Decision for conversation {request.conversation_id}: "
                       f"bypass={bypass_decision['should_bypass']}, confidence={bypass_decision['confidence']:.2f}, "
                       f"reason={bypass_decision['reason']}")
            
            # Get cached retrieval context if available (unless bypassing)
            cached_context = None
            if not bypass_decision['should_bypass']:
                cached_context = await conversation_context_manager.get_cached_context(request.conversation_id)
            else:
                logger.info(f"[CACHE_BYPASS] Skipping cache retrieval due to bypass decision")
            
            # === TRANSFORMATION HANDLING: Check for transformation requests first ===
            if conversation_context and query_type == 'contextual':
                logger.info(f"[TRANSFORMATION_HANDLER] Transformation request detected: {request.message[:100]}")
                
                try:
                    # Generate transformation response using existing conversation data
                    transformation_response = await _handle_contextual_query(
                        request.message,
                        conversation_context,
                        cached_context,
                        notebook_id,
                        request.conversation_id
                    )
                    
                    if transformation_response:
                        logger.info(f"[CONTEXTUAL_HANDLER] Successfully generated contextual response")
                        
                        # Log routing decision
                        await intelligent_routing_metrics.log_routing_decision(request.conversation_id, {
                            'message': request.message[:100],
                            'intent': 'contextual_query',
                            'query_type': query_type,
                            'routing_decision': 'contextual_handled',
                            'retrieval_triggered': False,
                            'conversation_aware': True,
                            'notebook_id': notebook_id
                        })
                        
                        yield json.dumps({
                            "status": "completed",
                            "message": "Transformation applied to previous response data",
                            "notebook_id": notebook_id,
                            "conversation_id": request.conversation_id,
                            "routing_decision": "transformation_complete"
                        }) + "\n"
                        
                        yield json.dumps(transformation_response) + "\n"
                        
                        # Store this contextual response for future reference
                        await conversation_context_manager.store_conversation_response(
                            request.conversation_id,
                            request.message,
                            transformation_response.get('response', ''),
                            transformation_response.get('sources', [])
                        )
                        
                        return
                    else:
                        logger.info(f"[TRANSFORMATION_HANDLER] Could not handle transformation - falling back to retrieval")
                        
                except Exception as e:
                    logger.error(f"[CONTEXTUAL_HANDLER] Error handling contextual query: {str(e)}")
                    # Fall through to normal retrieval on error
            
            # Handle different query types with appropriate strategies
            if query_type == 'simple':
                logger.info(f"[SIMPLE_QUERY] Simple query detected - using minimal retrieval")
                request.max_sources = min(request.max_sources, 10)  # Limit sources for speed
            elif query_type == 'temporal':
                logger.info(f"[TEMPORAL_QUERY] Time-related query detected - using fresh data retrieval")
                # Temporal queries need fresh data, minimal context
                request.include_context = False
            elif conversation_context and query_type == 'contextual':  # Contextual queries with existing data
                logger.info(f"[CONTEXTUAL_BYPASS] Contextual query detected - optimizing retrieval path")
                
                # Set request parameters for fastest possible retrieval
                request.max_sources = min(request.max_sources, 10)  # Limit sources for speed
                request.include_context = True  # Ensure we get data
                
                # Skip the conversation intelligence and force direct to simple retrieval
                logger.info(f"[CONTEXTUAL_BYPASS] Bypassing conversation intelligence - proceeding to fast retrieval")
                # DO NOT return here - let it fall through to retrieval with bypass flag
            
            # Always try LLM-driven intelligent response FIRST - let LLM decide cache vs fresh retrieval
            conversation_history = [conversation_context] if conversation_context else []
            should_use_context = _should_handle_with_context(request.message, conversation_history) or bool(cached_context)
            
            # Try LLM-driven intelligent response FIRST - LLM will decide whether to use cached data or request fresh retrieval
            if should_use_context and (conversation_context or cached_context):
                logger.info(f"[CONVERSATION_FIRST] Using context - conversation: {bool(conversation_context)}, cached: {bool(cached_context)}, query_type: {query_type}")
                
                yield json.dumps({
                    "status": "analyzing_conversation",
                    "message": "AI analyzing conversation context and available data",
                    "notebook_id": notebook_id,
                    "conversation_id": request.conversation_id,
                    "routing_decision": "conversation_intelligence_first"
                }) + "\n"
                
                # Let LLM make intelligent decision with streaming BEFORE any intent classification
                intelligent_stream = respond_with_llm_intelligence_streaming(
                    request.message,
                    cached_context,
                    conversation_context,
                    notebook_id,
                    request.conversation_id
                )
                
                # Try to use streaming intelligence
                intelligent_used = False
                collected_response = ""
                collected_sources = []
                
                try:
                    # Process the streaming response
                    async for chunk in intelligent_stream:
                        if not intelligent_used:
                            # First chunk indicates success
                            intelligent_used = True
                            logger.info(f"[CONVERSATION_FIRST] LLM generated streaming response using existing data")
                            
                            # Log intelligent routing decision
                            await intelligent_routing_metrics.log_routing_decision(request.conversation_id, {
                                'message': request.message[:100],
                                'intent': 'conversation_intelligent_reuse',
                                'routing_decision': 'conversation_first_streaming_success',
                                'retrieval_triggered': False,
                                'conversation_aware': True,
                                'notebook_id': notebook_id
                            })
                            
                            yield json.dumps({
                                "status": "completed",
                                "message": "AI providing intelligent streaming response using conversation context",
                                "notebook_id": notebook_id,
                                "conversation_id": request.conversation_id,
                                "routing_decision": "conversation_intelligence_streaming"
                            }) + "\n"
                        
                        # Parse and collect response data from chunks
                        try:
                            chunk_data = json.loads(chunk.rstrip('\n'))
                            
                            # Collect sources from source chunks
                            if 'sources' in chunk_data:
                                collected_sources = chunk_data['sources']
                            
                            # Collect response text from answer chunks
                            elif 'answer' in chunk_data:
                                collected_response = chunk_data['answer']
                            
                            # Also collect from streaming text chunks
                            elif 'chunk' in chunk_data:
                                collected_response += chunk_data['chunk']
                                
                        except (json.JSONDecodeError, AttributeError):
                            # Handle non-JSON chunks gracefully
                            pass
                        
                        # Stream the chunk
                        yield chunk
                        
                except Exception as e:
                    logger.info(f"[CONVERSATION_FIRST] Streaming intelligence indicated retrieval needed or failed: {str(e)}")
                    intelligent_used = False
                
                if intelligent_used:
                    
                    # Store this conversation exchange for future context using collected data
                    await conversation_context_manager.store_conversation_response(
                        request.conversation_id,
                        request.message,
                        collected_response,
                        collected_sources
                    )
                    
                    return
                else:
                    # LLM requested new retrieval - continue to intent classification
                    logger.info(f"[CONVERSATION_FIRST] LLM determined new retrieval is needed - proceeding to classification")
            else:
                logger.info(f"[CONVERSATION_FIRST] No conversation or cached context available - proceeding to classification")
            
            # === PHASE 2: INTELLIGENT MESSAGE CLASSIFICATION ===
            # Only classify intent if conversation intelligence didn't handle the request
            intent = await classify_message_intent(request.message)
            logger.info(f"[INTELLIGENT_ROUTING] Message classified as: {intent}")
            
            # Log routing decision for metrics
            await intelligent_routing_metrics.log_routing_decision(request.conversation_id, {
                'message': request.message[:100],  # Truncated for privacy
                'intent': intent,
                'routing_phase': 'post_conversation_classification',
                'notebook_id': notebook_id
            })
            
            # Handle non-retrieval intents with simple responses
            if intent in ['greeting', 'acknowledgment']:
                logger.info(f"[INTELLIGENT_ROUTING] Using simple response for {intent} - NO RETRIEVAL triggered")
                
                # Log simple response routing decision
                await intelligent_routing_metrics.log_routing_decision(request.conversation_id, {
                    'message': request.message[:100],
                    'intent': intent,
                    'routing_decision': f'simple_response_for_{intent}',
                    'retrieval_triggered': False,
                    'notebook_id': notebook_id
                })
                
                simple_response = await simple_llm_response(request.message, notebook_id, request.conversation_id)
                
                if simple_response:
                    # Yield the simple response directly
                    yield json.dumps({
                        "status": "completed",
                        "message": "Response ready - no retrieval needed",
                        "notebook_id": notebook_id,
                        "conversation_id": request.conversation_id,
                        "routing_decision": f"simple_response_for_{intent}"
                    }) + "\n"
                    
                    yield json.dumps(simple_response) + "\n"
                    return
            
            elif intent in ['clarification', 'general_chat'] and len(request.message) < 20:
                logger.info(f"[INTELLIGENT_ROUTING] Short {intent} message - NO RETRIEVAL triggered")
                yield json.dumps({
                    "status": "completed",
                    "message": "Clarification response ready",
                    "notebook_id": notebook_id,
                    "conversation_id": request.conversation_id,
                    "routing_decision": "clarification_prompt"
                }) + "\n"
                
                yield json.dumps({
                    "response": "Could you please provide more details about what you'd like to know?",
                    "conversation_id": request.conversation_id or f"notebook-{notebook_id}",
                    "sources": [],
                    "metadata": {
                        "routing": "clarification_prompt",
                        "retrieval_triggered": False,
                        "message_classification": "short_clarification"
                    }
                }) + "\n"
                return
            
            # === PHASE 3: INTELLIGENT RETRIEVAL PLANNING ===
            # If we reach here, either no context existed or LLM requested new retrieval
            logger.info(f"[RETRIEVAL_PLANNING] Proceeding to retrieval planning for intent: {intent}")
            intent = 'retrieval_required'  # Force retrieval since conversation intelligence didn't handle it
            
            # === PHASE 4: INTELLIGENT RETRIEVAL PLANNING ===
            # Determine HOW to retrieve based on query complexity
            import time
            plan_start_time = time.time()
            retrieval_plan = await notebook_rag_service.plan_retrieval_strategy(request.message, intent)
            plan_execution_time = time.time() - plan_start_time
            
            logger.info(f"[RETRIEVAL_PLANNING] Using {retrieval_plan.intensity.value} strategy: {retrieval_plan.reasoning}")
            
            # Log retrieval planning metrics
            await intelligent_routing_metrics.log_retrieval_plan(
                conversation_id=request.conversation_id,
                plan=retrieval_plan,
                execution_time=plan_execution_time
            )
            
            # Log full retrieval routing decision
            await intelligent_routing_metrics.log_routing_decision(request.conversation_id, {
                'message': request.message[:100],
                'intent': intent,
                'routing_decision': f'full_retrieval_for_{intent}',
                'retrieval_triggered': True,
                'retrieval_intensity': retrieval_plan.intensity.value,
                'max_sources': retrieval_plan.max_sources,
                'notebook_id': notebook_id
            })
            
            # For intents that need retrieval, use planned approach
            logger.info(f"[INTELLIGENT_ROUTING] Intent '{intent}' requires retrieval - using {retrieval_plan.intensity.value} approach")
            
            # Start by yielding initial status
            yield json.dumps({
                "status": "searching",
                "message": "Searching notebook documents...",
                "notebook_id": notebook_id,
                "conversation_id": request.conversation_id,
                "routing_decision": f"full_retrieval_for_{intent}"
            }) + "\n"
            
            # Query relevant documents using RAG service
            rag_response = None
            # DEBUG: Context inclusion check
            if request.include_context:
                try:
                    # Check if progressive loading would be beneficial for large datasets
                    intent_analysis = await notebook_rag_service._analyze_query_intent(request.message)
                    total_available = await notebook_rag_service.get_notebook_total_items(notebook_id)
                    
                    # Override max_sources for comprehensive queries early in the pipeline
                    if intent_analysis and intent_analysis.get('query_type') == 'comprehensive':
                        original_max_sources = request.max_sources
                        request.max_sources = min(total_available, 100)  # Increase limit for comprehensive queries
                        logger.info(f"[COMPREHENSIVE_OVERRIDE_EARLY] Increased max_sources from {original_max_sources} to {request.max_sources} for comprehensive query")
                    
                    # === INTELLIGENT AI PIPELINE: Understand → Plan → Execute → Verify → Respond ===
                    
                    # Step 1: Understand - Detect when to use intelligent planning
                    should_use_intelligent_pipeline = _should_use_intelligent_planning(request.message, intent_analysis, total_available)
                    
                    if should_use_intelligent_pipeline:
                        logger.info(f"[AI_PIPELINE] Activating intelligent pipeline for complex query")
                        
                        yield json.dumps({
                            "status": "ai_planning",
                            "message": "Creating intelligent execution plan...",
                            "notebook_id": notebook_id,
                            "conversation_id": request.conversation_id
                        }) + "\n"
                        
                        # Step 2: Plan - Create AI task plan
                        try:
                            execution_plan = await ai_task_planner.understand_and_plan(request.message)
                            logger.info(f"[AI_PIPELINE] Task plan created - Strategies: {len(execution_plan.retrieval_strategies)}, "
                                       f"Expected: {execution_plan.data_requirements.expected_count}")
                            logger.info(f"[AI_PIPELINE] Plan details: {execution_plan.data_requirements.entities}, "
                                       f"Format: {execution_plan.presentation.format}")
                            
                            yield json.dumps({
                                "status": "ai_executing",
                                "message": f"Executing intelligent plan ({len(execution_plan.retrieval_strategies)} strategies)...",
                                "notebook_id": notebook_id,
                                "conversation_id": request.conversation_id,
                                "plan_metadata": {
                                    "entities": execution_plan.data_requirements.entities,
                                    "completeness": execution_plan.data_requirements.completeness,
                                    "format": execution_plan.presentation.format
                                }
                            }) + "\n"
                            
                        except ImportError as e:
                            logger.error(f"[AI_PIPELINE] Missing dependency for task planning: {str(e)}")
                            execution_plan = None
                        except ValueError as e:
                            logger.error(f"[AI_PIPELINE] Invalid parameters for task planning: {str(e)}, query='{request.message[:100]}...'")
                            execution_plan = None
                        except ConnectionError as e:
                            logger.error(f"[AI_PIPELINE] Connection failed during task planning: {str(e)}")
                            raise HTTPException(status_code=503, detail="AI planning service unavailable")
                        except asyncio.TimeoutError as e:
                            logger.error(f"[AI_PIPELINE] Task planning timeout: {str(e)}")
                            raise HTTPException(status_code=504, detail="AI planning timeout")
                        except Exception as e:
                            logger.error(f"[AI_PIPELINE] Unexpected task planning error: {type(e).__name__}: {str(e)}")
                            logger.error(f"[AI_PIPELINE] Query context: notebook_id={notebook_id}, message_length={len(request.message)}")
                            import traceback
                            logger.error(f"[AI_PIPELINE] Full traceback: {traceback.format_exc()}")
                            execution_plan = None
                    
                    should_use_progressive = await notebook_rag_service.should_use_progressive_loading(
                        query=request.message,
                        intent_analysis=intent_analysis,
                        total_available=total_available,
                        max_sources=request.max_sources
                    )
                    
                    if should_use_progressive:
                        logger.info(f"[REGULAR_CHAT_PROGRESSIVE] Redirecting to progressive loading due to large dataset: "
                                   f"available={total_available}, requested={request.max_sources}")
                        
                        # Stream a message about switching to progressive mode
                        yield json.dumps({
                            "status": "switching_to_progressive",
                            "message": f"Large dataset detected ({total_available} items). Switching to progressive loading for better performance...",
                            "notebook_id": notebook_id,
                            "conversation_id": request.conversation_id,
                            "progressive_loading_recommended": True
                        }) + "\n"
                        
                        # Use progressive streaming instead
                        async for progress_chunk in stream_progressive_notebook_chat(
                            notebook_id=notebook_id,
                            query=request.message,
                            intent_analysis=intent_analysis,
                            rag_service=notebook_rag_service,
                            max_sources=request.max_sources
                        ):
                            yield progress_chunk
                        
                        # End the regular chat stream here since progressive handling is complete
                        return
                    
                        
                        logger.info(f"[DIRECT_MODE] Fast extraction completed: {len(rag_response.sources) if rag_response else 0} sources")
                    
                    # Step 3: Execute - Use intelligent plan or fall back to standard RAG
                    elif execution_plan:
                        # Override max_sources for comprehensive queries to ensure all data is retrieved
                        if intent_analysis and intent_analysis.get('query_type') == 'comprehensive':
                            original_max_sources = request.max_sources
                            request.max_sources = min(total_available, 100)  # Increase limit for comprehensive queries
                            logger.info(f"[COMPREHENSIVE_OVERRIDE] Increased max_sources from {original_max_sources} to {request.max_sources} for comprehensive query")
                        
                        logger.info(f"[AI_PIPELINE] Executing intelligent plan with {len(execution_plan.retrieval_strategies)} strategies (WITH {plan_timeout}s TIMEOUT)")
                        
                        try:
                            # Execute with timeout to prevent infinite processing
                            rag_response = await asyncio.wait_for(
                                notebook_rag_service.execute_intelligent_plan(
                                    db=db,
                                    notebook_id=notebook_id,
                                    plan=execution_plan,
                                    include_metadata=True
                                ),
                                timeout=plan_timeout
                            )
                        except asyncio.TimeoutError as e:
                            logger.error(f"[TIMEOUT_CRITICAL] Intelligent plan execution timeout after {plan_timeout}s")
                            logger.error(f"[TIMEOUT_CRITICAL] This may indicate: database deadlock, memory leak, or system resource exhaustion")
                            logger.error(f"[TIMEOUT_CRITICAL] Notebook: {notebook_id}, Query length: {len(request.message)}, Strategies: {len(execution_plan.retrieval_strategies)}")
                            
                            yield json.dumps({
                                "status": "timeout_error", 
                                "message": f"Processing timeout after {plan_timeout}s. This may indicate a system issue - please check with administrators if this persists.",
                                "notebook_id": notebook_id,
                                "conversation_id": request.conversation_id,
                                "error_type": "timeout",
                                "timeout_duration": plan_timeout
                            }) + "\n"
                            
                            # Try simple RAG but with shorter timeout to avoid cascading timeouts
                            try:
                                rag_response = await asyncio.wait_for(
                                    notebook_rag_service.query_notebook_adaptive(
                                        db=db,
                                        notebook_id=notebook_id,
                                        query=request.message,
                                        max_sources=min(total_available, 20),  # Limit sources to reduce processing time
                                        include_metadata=True,
                                        force_simple_retrieval=True,
                                    ), timeout=60  # Increased timeout for emergency fallback to prevent cascade
                                )
                            except asyncio.TimeoutError:
                                logger.error(f"[TIMEOUT_CRITICAL] Even simple RAG timed out - system may be under severe load")
                                raise HTTPException(status_code=504, detail="System timeout - please try again later or contact support")
                        except MemoryError as e:
                            logger.error(f"[MEMORY_CRITICAL] Out of memory during plan execution: {str(e)}")
                            logger.error(f"[MEMORY_CRITICAL] Notebook: {notebook_id}, Available memory may be insufficient")
                            raise HTTPException(status_code=507, detail="Insufficient memory to process request")
                        except ConnectionError as e:
                            logger.error(f"[CONNECTION_CRITICAL] Database/service connection failed during plan execution: {str(e)}")
                            raise HTTPException(status_code=503, detail="Database connection failed")
                        except Exception as e:
                            logger.error(f"[PLAN_EXECUTION_ERROR] Plan execution failed with {type(e).__name__}: {str(e)}")
                            logger.error(f"[PLAN_EXECUTION_ERROR] Notebook: {notebook_id}, Strategies: {len(execution_plan.retrieval_strategies)}")
                            import traceback
                            logger.error(f"[PLAN_EXECUTION_ERROR] Full traceback: {traceback.format_exc()}")
                            
                            # Fallback to simple RAG
                            rag_response = await notebook_rag_service.query_notebook_adaptive(
                                db=db,
                                notebook_id=notebook_id,
                                query=request.message,
                                max_sources=total_available,
                                include_metadata=True,
                                force_simple_retrieval=True,
# removed request_id parameter
                            )
                        
                        # Step 4: Verify - Check completeness and trigger self-correction if needed
                        yield json.dumps({
                            "status": "ai_verifying",
                            "message": "Verifying result completeness...",
                            "notebook_id": notebook_id,
                            "conversation_id": request.conversation_id
                        }) + "\n"
                        
                        verification_result = await ai_verification_service.verify_completeness(
                            results=rag_response,
                            plan=execution_plan,
                            notebook_id=notebook_id
                        )
                        
                        logger.info(f"[AI_PIPELINE] Verification - Confidence: {verification_result.confidence:.2f}, "
                                   f"Completeness: {verification_result.completeness_score:.2f}, "
                                   f"Needs correction: {verification_result.needs_correction}")
                        
                        # CIRCUIT BREAKER: Skip correction for simple queries to prevent infinite loops
                        if intent_analysis and intent_analysis.get('query_type') == 'simple':
                            logger.info("[CIRCUIT_BREAKER] Simple query detected - skipping verification correction to prevent infinite loops")
                        elif verification_result.needs_correction and verification_result.correction_strategies:
                            logger.info(f"[AI_PIPELINE] Triggering self-correction with {len(verification_result.correction_strategies)} strategies (LIMITED TO 1 ATTEMPT)")
                            
                            yield json.dumps({
                                "status": "ai_correcting",
                                "message": f"Self-correcting to improve completeness (confidence: {verification_result.confidence:.1%}) - single attempt...",
                                "notebook_id": notebook_id,
                                "conversation_id": request.conversation_id,
                                "verification_metadata": {
                                    "original_confidence": verification_result.confidence,
                                    "correction_strategies_count": len(verification_result.correction_strategies)
                                }
                            }) + "\n"
                            
                            # Execute correction strategies with timeout
                            try:
                                correction_response = await asyncio.wait_for(
                                    notebook_rag_service.execute_correction_strategies(
                                        db=db,
                                        notebook_id=notebook_id,
                                        original_response=rag_response,
                                        correction_strategies=verification_result.correction_strategies[:1]  # LIMIT TO 1 STRATEGY
                                    ),
                                    timeout=30.0  # 30 second timeout
                                )
                                
                                if correction_response and len(correction_response.sources) > len(rag_response.sources):
                                    logger.info(f"[AI_PIPELINE] Self-correction successful: {len(rag_response.sources)} → {len(correction_response.sources)} sources")
                                    rag_response = correction_response
                                else:
                                    logger.info("[AI_PIPELINE] Self-correction did not improve results, using original response")
                                    
                            except asyncio.TimeoutError:
                                logger.warning("[CIRCUIT_BREAKER] Correction strategies timed out after 30s, proceeding with original results")
                            except Exception as e:
                                logger.error(f"[CIRCUIT_BREAKER] Correction strategies failed: {str(e)}, proceeding with original results")
                        
                    else:
                        # === PHASE 3 EXECUTION: Use planned retrieval approach ===
                        logger.info(f"[PLANNED_RETRIEVAL] Using {retrieval_plan.intensity.value} strategy with {retrieval_plan.max_sources} max sources")
                        
                        # Always include chronological ordering instruction since data is ALWAYS sorted chronologically
                        enhanced_query = request.message + "\n\nNOTE: Project data is sorted chronologically by start year - maintain this order in tables."
                        logger.info("[CHRONOLOGICAL_ORDER] Enhanced query with chronological order preservation instruction")
                        
                        # Execute based on planned retrieval intensity
                        if retrieval_plan.intensity == RetrievalIntensity.MINIMAL:
                            rag_response = await notebook_rag_service.execute_minimal_retrieval(
                                message=enhanced_query,
                                notebook_id=notebook_id,
                                db=db
                            )
                        elif retrieval_plan.intensity == RetrievalIntensity.BALANCED:
                            rag_response = await notebook_rag_service.execute_balanced_retrieval(
                                message=enhanced_query,
                                notebook_id=notebook_id,
                                db=db
                            )
                        else:  # COMPREHENSIVE
                            rag_response = await notebook_rag_service.execute_comprehensive_retrieval(
                                message=enhanced_query,
                                notebook_id=notebook_id,
                                db=db,
                                max_sources=retrieval_plan.max_sources
                            )
                        
                        logger.info(f"[PLANNED_RETRIEVAL] {retrieval_plan.intensity.value} retrieval completed: {len(rag_response.sources)} sources")
                    
                    # RAG Query Execution completed
                    
                    # Build comprehensive status with AI pipeline metadata
                    status_data = {
                        "status": "context_found",
                        "sources_count": len(rag_response.sources),
                        "collections_searched": rag_response.collections_searched,
                        "notebook_id": notebook_id,
                        "conversation_id": request.conversation_id
                    }
                    
                    # Add AI pipeline metadata if available
                    if execution_plan:
                        status_data["ai_pipeline_metadata"] = {
                            "used_intelligent_pipeline": True,
                            "strategies_executed": len(execution_plan.retrieval_strategies),
                            "data_completeness": execution_plan.data_requirements.completeness,
                            "expected_format": execution_plan.presentation.format
                        }
                        
                        if verification_result:
                            status_data["ai_pipeline_metadata"]["verification"] = {
                                "confidence": verification_result.confidence,
                                "completeness_score": verification_result.completeness_score,
                                "used_self_correction": verification_result.needs_correction
                            }
                    
                    yield json.dumps(status_data) + "\n"
                    
                except Exception as e:
                    logger.warning(f"RAG query failed for notebook {notebook_id}: {str(e)}")
                    yield json.dumps({
                        "status": "context_warning",
                        "message": "Could not retrieve full context, proceeding with general response",
                        "notebook_id": notebook_id,
                        "conversation_id": request.conversation_id
                    }) + "\n"
            
            # Get notebook LLM configuration
            llm_config_full = get_notebook_llm_full_config()
            if not llm_config_full:
                yield json.dumps({
                    "error": "Notebook LLM configuration not available",
                    "notebook_id": notebook_id,
                    "conversation_id": request.conversation_id
                }) + "\n"
                return
            
            # Extract the notebook_llm config from the nested structure
            llm_config = llm_config_full.get('notebook_llm', {})
            if not llm_config:
                yield json.dumps({
                    "error": "Notebook LLM configuration not properly structured in database",
                    "notebook_id": notebook_id,
                    "conversation_id": request.conversation_id
                }) + "\n"
                return
            
            yield json.dumps({
                "status": "generating",
                "message": "Generating response...",
                "notebook_id": notebook_id,
                "conversation_id": request.conversation_id
            }) + "\n"
            
            # Initialize LLM with database configuration - NO HARDCODED FALLBACKS
            llm_config_obj = LLMConfig(
                model_name=llm_config['model'],
                temperature=float(llm_config['temperature']),
                top_p=float(llm_config['top_p']),
                max_tokens=int(llm_config['max_tokens'])
            )
            llm = await get_cached_notebook_llm_from_config(llm_config_obj)
            
            # Build context-aware prompt using configurable system prompt from database
            base_system_prompt = llm_config['system_prompt']
            system_prompt = base_system_prompt + " "
            context_info = ""
            
            if rag_response and rag_response.sources:
                # Count different source types for better prompt customization
                document_count = sum(1 for src in rag_response.sources if src.source_type != "memory")
                memory_count = sum(1 for src in rag_response.sources if src.source_type == "memory")
                
                # Apply configurable source integration templates
                source_integration_prompt = apply_source_templates(
                    total_sources=len(rag_response.sources),
                    document_count=document_count,
                    memory_count=memory_count
                )
                system_prompt += source_integration_prompt
                
                # === INTELLIGENT AI ENHANCEMENTS: Add structured AI planning instructions ===
                
                # Step 5: Enhance LLM with AI planning context
                if execution_plan:
                    # Add AI task plan context to system prompt
                    ai_enhancement_prompt = f"\n\nINTELLIGENT AI CONTEXT:\n"
                    ai_enhancement_prompt += f"This query has been processed through intelligent AI planning with the following understanding:\n"
                    ai_enhancement_prompt += f"- Required entities: {', '.join(execution_plan.data_requirements.entities)}\n"
                    ai_enhancement_prompt += f"- Required attributes: {', '.join(execution_plan.data_requirements.attributes)}\n"
                    ai_enhancement_prompt += f"- Completeness requirement: {execution_plan.data_requirements.completeness}\n"
                    ai_enhancement_prompt += f"- Expected format: {execution_plan.presentation.format}\n"
                    
                    if execution_plan.data_requirements.expected_count:
                        ai_enhancement_prompt += f"- Expected count: {execution_plan.data_requirements.expected_count}\n"
                    
                    # Add format-specific instructions
                    if execution_plan.presentation.format == "table":
                        ai_enhancement_prompt += f"\nFORMAT REQUIREMENT: Present results in a well-formatted table with columns: {', '.join(execution_plan.presentation.fields_to_show)}\n"
                        
                        if execution_plan.presentation.sorting:
                            sort_field = execution_plan.presentation.sorting.get('field', 'default')
                            sort_order = execution_plan.presentation.sorting.get('order', 'asc')
                            ai_enhancement_prompt += f"Sort by {sort_field} in {sort_order} order.\n"
                            
                    elif execution_plan.presentation.format == "list":
                        ai_enhancement_prompt += f"\nFORMAT REQUIREMENT: Present results as a structured list with clear organization.\n"
                    
                    # Add verification context if available
                    if verification_result:
                        ai_enhancement_prompt += f"\nQUALITY ASSURANCE:\n"
                        ai_enhancement_prompt += f"- Confidence level: {verification_result.confidence:.1%}\n"
                        ai_enhancement_prompt += f"- Completeness score: {verification_result.completeness_score:.1%}\n"
                        
                        if verification_result.needs_correction:
                            ai_enhancement_prompt += f"- Self-correction was applied to improve completeness\n"
                        
                        if verification_result.quality_issues:
                            ai_enhancement_prompt += f"- Note potential gaps: {'; '.join(verification_result.quality_issues[:2])}\n"
                    
                    ai_enhancement_prompt += f"\nPlease provide a response that meets these intelligent requirements and leverages the comprehensive analysis performed.\n"
                    
                    system_prompt += ai_enhancement_prompt
                    logger.info(f"[AI_PIPELINE] Enhanced LLM prompt with intelligent planning context ({len(ai_enhancement_prompt)} chars)")
                
                # === COMPREHENSIVE ANALYSIS ENHANCEMENT ===
                
                # Detect comprehensive analysis queries that require complete results
                query_lower = request.message.lower()
                is_comprehensive_analysis = (
                    ("analyze" in query_lower) or
                    ("summarize" in query_lower and "all" in query_lower) or
                    (("table" in query_lower or "format" in query_lower) and 
                     ("projects" in query_lower or "project" in query_lower or "work" in query_lower))
                )
                
                if is_comprehensive_analysis:
                    source_count = len(rag_response.sources) if rag_response else 0
                    system_prompt += f"""

ANALYSIS TASK:
Process all {source_count} sources and provide a comprehensive analysis.
Present the findings in the requested format (table, list, etc.).
"""
                    
                    logger.info(f"[COMPLETENESS] Enhanced prompt for comprehensive analysis - {source_count} sources")
                
                
                context_info = "\n\n--- RELEVANT CONTEXT FROM NOTEBOOK ---\n"
                context_info += "The following sources contain information relevant to your question. "
                context_info += "Please integrate information from ALL sources in your response:\n"
                
                for i, source in enumerate(rag_response.sources, 1):
                    # Format source name with type prefix for clarity
                    source_name = source.document_name or 'Unknown'
                    source_type_prefix = "Personal Memory" if source.source_type == "memory" else "Document"
                    
                    # DEBUG: Context Building Process
                    
                    context_info += f"\n[Source {i} - {source_type_prefix}: {source_name}]:\n"
                    context_info += source.content[:1000] + ("..." if len(source.content) > 1000 else "")
                    context_info += "\n"
                
                context_info += "--- END CONTEXT ---\n\n"
                
                # Enhanced prompt with structured project data for analysis queries
                if hasattr(rag_response, 'extracted_projects') and rag_response.extracted_projects:
                    # Detect if this is a structured query that could benefit from structured data
                    query_lower = request.message.lower()
                    structured_indicators = ['table format', 'in table', 'as table', 'table form', 'analyze', 'summarize']
                    project_indicators = ['project', 'projects', 'work', 'experience']
                    
                    is_structured_query = (
                        any(indicator in query_lower for indicator in structured_indicators) and 
                        any(indicator in query_lower for indicator in project_indicators)
                    )
                    
                    if is_structured_query:
                        project_count = len(rag_response.extracted_projects)
                        logger.info(f"[STRUCTURED_RESPONSE] Injecting {project_count} structured projects into prompt for analysis query")
                        
                        # Add structured project data to context
                        context_info += "--- STRUCTURED PROJECT DATA ---\n"
                        context_info += f"IMPORTANT: Here are ALL {project_count} projects extracted from the notebook content. "
                        context_info += "You MUST include every single project listed below, even if some fields show 'N/A':\n\n"
                        
                        for i, project in enumerate(rag_response.extracted_projects, 1):
                            context_info += f"Project {i}:\n"
                            context_info += f"  Name: {project.name}\n"
                            context_info += f"  Company: {project.company}\n"
                            context_info += f"  Year: {project.year}\n"
                            context_info += f"  Description: {project.description[:200]}{'...' if len(project.description) > 200 else ''}\n\n"
                        
                        context_info += "--- END STRUCTURED DATA ---\n\n"
                        
                        # Add explicit formatting instructions to system prompt
                        if any(table_word in query_lower for table_word in ['table', 'format']):
                            system_prompt += f" CRITICAL INSTRUCTION: The user has requested table format. You must create a table that includes ALL {project_count} projects listed above. Do not exclude any projects due to missing information - use 'N/A' for missing fields. Ensure your table shows exactly {project_count} rows of projects."
                        else:
                            system_prompt += f" CRITICAL INSTRUCTION: You must mention ALL {project_count} projects listed above. Do not exclude any projects due to missing information. Include every project even if some details are marked as 'N/A'."
                
                # DEBUG: Final Context Analysis
                
                # DEBUG: Check if memory content is in final context
            else:
                system_prompt += "No specific context was found in this notebook for this question. Provide a helpful general response and suggest ways the user might find relevant information."
            
            # Build the full prompt
            full_prompt = f"{system_prompt}\n\n{context_info}User Question: {request.message}\n\nResponse:"
            
            # DEBUG: LLM Prompt Construction
            logger.debug(f"[PROMPT_DEBUG] Full prompt length: {len(full_prompt)} chars")
            logger.debug(f"[PROMPT_DEBUG] Prompt preview: {full_prompt[:300]}...")
            
            # DEBUG: Verify memory context reaches LLM
            if "Memory" in full_prompt:
                logger.debug(f"[PROMPT_DEBUG] Memory content confirmed in LLM prompt")
            else:
                logger.debug(f"[PROMPT_DEBUG] WARNING: No memory content found in LLM prompt")
            
            # FAILSAFE: Direct table generation for structured queries
            if (hasattr(rag_response, 'extracted_projects') and 
                rag_response.extracted_projects and 
                len(rag_response.extracted_projects) > 10 and
                any(word in request.message.lower() for word in ['table', 'list all', 'counter'])):
                
                logger.info(f"[DIRECT_TABLE] Generating direct table for {len(rag_response.extracted_projects)} projects")
                
                # Create table directly without LLM
                table_response = f"Here are all {len(rag_response.extracted_projects)} projects with company and year information:\n\n"
                table_response += "| # | Project | Company | Year |\n"
                table_response += "|---|---------|---------|------|\n"
                
                for i, project in enumerate(rag_response.extracted_projects, 1):
                    name = project.name[:50] if project.name else "N/A"
                    company = project.company[:30] if project.company else "Not specified"
                    year = project.year if project.year else "Not specified"
                    table_response += f"| {i} | {name} | {company} | {year} |\n"
                
                # Stream the table response
                yield json.dumps({
                    "chunk": table_response,
                    "notebook_id": notebook_id,
                    "conversation_id": request.conversation_id
                }) + "\n"
                
                collected_response = table_response
                
            else:
                    # Generate streaming response via LLM with timeout
                    collected_response = ""
                    try:
                        async with asyncio.timeout(300.0):  # 5 minute timeout for LLM generation
                            async for response_chunk in llm.generate_stream(full_prompt):
                                if response_chunk.text.strip():
                                    collected_response += response_chunk.text
                                    
                                    # Stream the response chunk
                                    yield json.dumps({
                                        "chunk": response_chunk.text,
                                        "notebook_id": notebook_id,
                                        "conversation_id": request.conversation_id
                                    }) + "\n"
                    except asyncio.TimeoutError:
                        logger.error("[RESPONSE_TIMEOUT] LLM generation timed out")
                        error_message = "Response generation timed out. Please try again."
                        yield json.dumps({
                            "chunk": error_message,
                            "notebook_id": notebook_id,
                            "conversation_id": request.conversation_id
                        }) + "\n"
                        collected_response = error_message
                    except Exception as e:
                        logger.error(f"[RESPONSE_ERROR] LLM generation failed: {str(e)}")
                        error_response = f"Response generation failed. Extracted {len(rag_response.sources) if rag_response else 0} sources but couldn't format response."
                        yield json.dumps({
                            "chunk": error_response,
                            "notebook_id": notebook_id,
                            "conversation_id": request.conversation_id
                        }) + "\n"
                        collected_response = error_response
            
            # GENERAL PURPOSE: Verify completeness for comprehensive queries
            if execution_plan and execution_plan.data_requirements.completeness == "all":
                try:
                    from app.services.completeness_verifier import completeness_verifier
                    completeness_check = completeness_verifier.verify_completeness(
                        response_text=collected_response,
                        sources=rag_response.sources if rag_response else [],
                        query=request.message,
                        expected_entity_count=len(rag_response.sources) if rag_response else None
                    )
                    
                    logger.info(f"[COMPLETENESS] Verification: complete={completeness_check.is_complete}, "
                               f"confidence={completeness_check.confidence:.2f}, "
                               f"found={completeness_check.found_count}, "
                               f"gaps={len(completeness_check.gaps_detected)}")
                    
                except Exception as e:
                    logger.warning(f"[COMPLETENESS] Verification failed: {e}")
            
            # Send final response with complete answer and sources
            final_response = {
                "answer": collected_response,
                "sources": [
                    {
                        "document_id": source.document_id,
                        "document_name": source.document_name,
                        "content": source.content[:500] + ("..." if len(source.content) > 500 else ""),
                        "score": source.score,
                        "collection": source.collection
                    }
                    for source in (rag_response.sources if rag_response else [])
                ],
                "notebook_id": notebook_id,
                "conversation_id": request.conversation_id,
                "status": "complete"
            }
            
            # === INTELLIGENT AI PIPELINE: Include comprehensive metadata in response ===
            
            # Add AI pipeline metadata to final response for transparency and debugging
            if execution_plan:
                final_response["ai_pipeline_metadata"] = {
                    "used_intelligent_pipeline": True,
                    "pipeline_version": "1.0",
                    "task_plan": {
                        "entities": execution_plan.data_requirements.entities,
                        "attributes": execution_plan.data_requirements.attributes,
                        "completeness_requirement": execution_plan.data_requirements.completeness,
                        "expected_count": execution_plan.data_requirements.expected_count,
                        "format": execution_plan.presentation.format,
                        "fields_shown": execution_plan.presentation.fields_to_show,
                        "sorting": execution_plan.presentation.sorting,
                        "strategies_executed": len(execution_plan.retrieval_strategies)
                    }
                }
                
                # Add verification results if available
                if verification_result:
                    final_response["ai_pipeline_metadata"]["verification"] = {
                        "confidence": verification_result.confidence,
                        "completeness_score": verification_result.completeness_score,
                        "needs_correction": verification_result.needs_correction,
                        "result_count": verification_result.result_count,
                        "unique_sources": verification_result.unique_sources,
                        "diversity_score": verification_result.diversity_score,
                        "expected_vs_actual": verification_result.expected_vs_actual,
                        "used_self_correction": verification_result.needs_correction,
                        "quality_issues": verification_result.quality_issues[:3] if verification_result.quality_issues else [],
                        "reasoning": verification_result.reasoning[:200] + "..." if len(verification_result.reasoning) > 200 else verification_result.reasoning
                    }
                
                logger.info(f"[AI_PIPELINE] Complete intelligent pipeline executed successfully - "
                           f"Plan: {len(execution_plan.retrieval_strategies)} strategies, "
                           f"Verification: {verification_result.confidence:.1%} confidence" if verification_result else "No verification")
            else:
                # Mark as using traditional RAG approach
                final_response["ai_pipeline_metadata"] = {
                    "used_intelligent_pipeline": False,
                    "pipeline_version": "traditional_rag",
                    "fallback_reason": "Query did not meet intelligent pipeline criteria"
                }
            
            # Include structured project data in response if available
            if rag_response and hasattr(rag_response, 'extracted_projects') and rag_response.extracted_projects:
                final_response["extracted_projects"] = [
                    {
                        "name": project.name,
                        "company": project.company,
                        "year": project.year,
                        "description": project.description,
                        "confidence_score": project.confidence_score
                    }
                    for project in rag_response.extracted_projects
                ]
                final_response["extracted_projects_count"] = len(rag_response.extracted_projects)
            
            # DEBUG: Final Response Analysis
            logger.debug(f"[RESPONSE_DEBUG] Generated response length: {len(collected_response)} chars")
            logger.debug(f"[RESPONSE_DEBUG] Response preview: {collected_response[:200]}...")
            logger.debug(f"[RESPONSE_DEBUG] Final response includes {len(final_response['sources'])} sources")
            
            # DEBUG: Check if response mentions employment/memory content
            if any(keyword in collected_response.lower() for keyword in ['employment', 'job', 'work', 'career', 'company']):
                logger.debug(f"[RESPONSE_DEBUG] Response contains employment-related content")
            else:
                logger.debug(f"[RESPONSE_DEBUG] Response does NOT contain employment-related content")
            
            # DEBUG: Validate structured project data usage
            if rag_response and hasattr(rag_response, 'extracted_projects') and rag_response.extracted_projects:
                extracted_count = len(rag_response.extracted_projects)
                query_lower = request.message.lower()
                
                # Count project mentions in response for validation
                project_mentions = collected_response.lower().count('project')
                table_format_requested = any(table_word in query_lower for table_word in ['table', 'format'])
                
                logger.info(f"[PROJECT_VALIDATION] Query: '{request.message[:50]}...'")
                logger.info(f"[PROJECT_VALIDATION] Extracted projects: {extracted_count}")
                logger.info(f"[PROJECT_VALIDATION] Project mentions in response: {project_mentions}")
                logger.info(f"[PROJECT_VALIDATION] Table format requested: {table_format_requested}")
                
                # Warn if significantly fewer projects mentioned than extracted
                if table_format_requested and project_mentions < extracted_count * 0.8:
                    logger.warning(f"[PROJECT_VALIDATION] Potential project loss: {extracted_count} extracted but only {project_mentions} mentions in response")
                    
            else:
                logger.debug(f"[PROJECT_VALIDATION] No extracted projects available for validation")
            
            # === PHASE 2: CACHE RETRIEVAL RESULTS ===
            # Cache the successful retrieval for follow-up questions
            if rag_response and len(rag_response.sources) > 0 and request.conversation_id:
                try:
                    # Prepare context for caching
                    cache_context = {
                        'sources': [
                            {
                                'content': source.content,
                                'metadata': {
                                    'document_id': source.document_id,
                                    'document_name': source.document_name,
                                    'collection': source.collection
                                },
                                'score': source.score
                            }
                            for source in rag_response.sources
                        ],
                        'query': request.message,
                        'extracted_entities': getattr(rag_response, 'extracted_projects', []),
                        'timestamp': datetime.now().isoformat(),
                        'metadata': {
                            'total_sources': len(rag_response.sources),
                            'queried_documents': rag_response.queried_documents,
                            'collections_searched': getattr(rag_response, 'collections_searched', []),
                            'ai_pipeline_used': bool(execution_plan)
                        }
                    }
                    
                    # Cache the context asynchronously (don't block response)
                    cache_success = await conversation_context_manager.cache_retrieval_context(
                        request.conversation_id,
                        cache_context
                    )
                    
                    if cache_success:
                        logger.info(f"[CACHE] Successfully cached context for conversation {request.conversation_id}: "
                                  f"{len(rag_response.sources)} sources")
                    
                except Exception as e:
                    logger.warning(f"[CACHE] Failed to cache retrieval context: {str(e)}")
                    # Don't let caching errors affect the response
            
            # Store conversation response for future LLM-driven intelligence
            try:
                await conversation_context_manager.store_conversation_response(
                    request.conversation_id,
                    request.message,
                    collected_response,  # The complete AI response
                    final_response.get('sources', [])
                )
                logger.info(f"[CONVERSATION_MEMORY] Stored new conversation exchange for future context")
            except Exception as e:
                logger.warning(f"[CONVERSATION_MEMORY] Failed to store conversation: {str(e)}")
                # Don't let conversation storage errors affect the response
            
            yield json.dumps(final_response) + "\n"
            
            logger.info(f"Successfully completed notebook chat for {notebook_id}")
            
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"Notebook chat error for {notebook_id}: {error_type}: {str(e)}")
            logger.error(f"Chat context: query_length={len(request.message) if request and request.message else 0}, max_sources={getattr(request, 'max_sources', 'unknown')}")
            import traceback
            logger.error(f"Chat error traceback: {traceback.format_exc()}")
            
            yield json.dumps({
                "error": f"Chat failed with {error_type}: {str(e)}",
                "error_details": {
                    "error_type": error_type,
                    "notebook_id": notebook_id,
                    "query_provided": bool(request and request.message),
                    "query_length": len(request.message) if request and request.message else 0,
                    "conversation_id": request.conversation_id if request else None,
                    "max_sources_requested": getattr(request, 'max_sources', None)
                },
                "status": "error",
                "troubleshooting": "Check server logs for detailed traceback. Common causes: database connection, memory issues, or malformed request."
            }) + "\n"
    
    return StreamingResponse(stream_chat_response(), media_type="application/json")


async def stream_progressive_notebook_chat(
    notebook_id: str,
    query: str,
    intent_analysis: dict,
    rag_service: NotebookRAGService,
    max_sources: int,
    retrieval_plan: Optional['RetrievalPlan'] = None
):
    """
    Handle progressive streaming for large result sets.
    
    Streams initial results quickly, then continues with background loading
    and provides progress updates to frontend.
    
    Args:
        notebook_id: Notebook ID
        query: User query string
        intent_analysis: Query intent analysis results
        rag_service: RAG service instance
        
    Yields:
        JSON chunks with progressive results and progress updates
    """
    try:
        logger.info(f"[PROGRESSIVE_CHAT] Starting progressive streaming for notebook {notebook_id}")
        logger.info(f"[PROGRESSIVE_CHAT] Query: '{query[:100]}...', comprehensive: {intent_analysis.get('wants_comprehensive', False)}")
        
        # === PHASE 3: Use planned retrieval approach ===
        if retrieval_plan is None:
            # Fallback: plan retrieval if not provided
            retrieval_plan = await rag_service.plan_retrieval_strategy(query, 'retrieval_required')
            
        logger.info(f"[PROGRESSIVE_PLANNED] Using {retrieval_plan.intensity.value} strategy: {retrieval_plan.reasoning}")
        
        # Get total available content for progress tracking (if needed for comprehensive)
        total_available = await rag_service.get_notebook_total_items(notebook_id) if retrieval_plan.intensity == RetrievalIntensity.COMPREHENSIVE else 0
        
        # Use planned retrieval approach instead of deciding progressive loading
        if retrieval_plan.intensity != RetrievalIntensity.COMPREHENSIVE:
            logger.info(f"[PROGRESSIVE_CHAT] Using {retrieval_plan.intensity.value} retrieval instead of progressive loading")
            # Use planned retrieval approach
            from app.core.db import SessionLocal
            db = SessionLocal()
            try:
                # Always include chronological ordering instruction since data is ALWAYS sorted chronologically
                enhanced_query = query + "\n\nNOTE: Project data is sorted chronologically by start year - maintain this order in tables."
                logger.info("[CHRONOLOGICAL_ORDER] Enhanced progressive fallback query with chronological order preservation instruction (default)")
                
                # Execute based on planned retrieval intensity
                if retrieval_plan.intensity == RetrievalIntensity.MINIMAL:
                    response = await rag_service.execute_minimal_retrieval(
                        message=enhanced_query,
                        notebook_id=notebook_id,
                        db=db
                    )
                else:  # BALANCED
                    response = await rag_service.execute_balanced_retrieval(
                        message=enhanced_query,
                        notebook_id=notebook_id,
                        db=db
                    )
                
                # Send as single complete response
                yield json.dumps({
                    "stage": "complete",
                    "sources": [
                        {
                            "document_id": source.document_id,
                            "document_name": source.document_name,
                            "content": source.content[:1000] + ("..." if len(source.content) > 1000 else ""),
                            "score": source.score,
                            "collection": source.collection,
                            "source_type": source.source_type
                        }
                        for source in response.sources
                    ],
                    "total_sources": response.total_sources,
                    "progress_percent": 100.0,
                    "more_available": False,
                    "progressive_loading_used": False,
                    "notebook_id": notebook_id
                }) + "\n"
            finally:
                db.close()
            return
        
        # Use multi-stage retrieval for large datasets
        stage_count = 0
        total_retrieved = 0
        all_sources = []
        
        logger.info(f"[PROGRESSIVE_CHAT] Using progressive loading: max_sources={max_sources}, available={total_available}")
        
        # Stream each stage of retrieval
        async for stage_response in rag_service.multi_stage_retrieval(
            notebook_id=notebook_id,
            query=query,
            intent_analysis=intent_analysis,
            total_available=total_available,
            max_sources=max_sources
        ):
            stage_count += 1
            stage_sources = stage_response.sources
            all_sources.extend(stage_sources)
            total_retrieved += len(stage_sources)
            
            # Get stage metadata
            stage_metadata = stage_response.metadata.get("multi_stage", {})
            is_initial_stage = stage_metadata.get("stage") == "initial"
            progress_percent = stage_metadata.get("progress_percent", 0)
            more_available = stage_metadata.get("more_available", False)
            
            logger.info(f"[PROGRESSIVE_CHAT] Stage {stage_count}: {len(stage_sources)} sources, "
                       f"total: {total_retrieved}, progress: {progress_percent}%")
            
            # Stream stage results
            stage_data = {
                "stage": "initial" if is_initial_stage else "progressive",
                "stage_number": stage_count,
                "sources": [
                    {
                        "document_id": source.document_id,
                        "document_name": source.document_name,
                        "content": source.content[:1000] + ("..." if len(source.content) > 1000 else ""),
                        "score": source.score,
                        "collection": source.collection,
                        "source_type": source.source_type
                    }
                    for source in stage_sources
                ],
                "batch_size": len(stage_sources),
                "total_retrieved_so_far": total_retrieved,
                "progress_percent": progress_percent,
                "more_available": more_available,
                "progressive_loading_used": True,
                "notebook_id": notebook_id
            }
            
            # Add timing information for initial stage
            if is_initial_stage:
                stage_data["initial_response_time"] = "2-3 seconds"
                stage_data["message"] = "Quick initial results - more loading in background"
            
            yield json.dumps(stage_data) + "\n"
            
            # Add small delay between stages for better UX
            if not is_initial_stage:
                import asyncio
                await asyncio.sleep(0.1)
        
        # Send completion signal
        yield json.dumps({
            "stage": "complete",
            "total_stages": stage_count,
            "final_source_count": total_retrieved,
            "progress_percent": 100.0,
            "more_available": False,
            "progressive_loading_used": True,
            "notebook_id": notebook_id,
            "message": f"Progressive loading complete: retrieved {total_retrieved} sources in {stage_count} stages"
        }) + "\n"
        
        logger.info(f"[PROGRESSIVE_CHAT] Completed progressive streaming: {total_retrieved} sources in {stage_count} stages")
        
    except Exception as e:
        logger.error(f"[PROGRESSIVE_CHAT] Error in progressive streaming: {str(e)}")
        yield json.dumps({
            "stage": "error",
            "error": str(e),
            "progressive_loading_used": True,
            "notebook_id": notebook_id
        }) + "\n"


@router.post("/{notebook_id}/chat-progressive")
async def notebook_chat_progressive(
    request: NotebookChatRequest,
    notebook_id: str = Path(..., description="Notebook ID"),
    db: Session = Depends(get_db)
):
    """
    Progressive notebook chat with multi-stage retrieval for large datasets.
    
    This endpoint provides:
    - Quick initial results (50-100 items) within 2-3 seconds
    - Progressive loading of additional results in background
    - Real-time progress updates
    - Optimized performance for very large notebooks (500+ items)
    
    Args:
        request: Chat request with message and configuration
        notebook_id: Target notebook ID
        db: Database session
        
    Returns:
        Streaming response with progressive results
    """
    # Initialize services
    notebook_service = NotebookService(db)
    notebook_rag_service = NotebookRAGService()
    
    # === REMOVED EXECUTION STATE TRACKING FOR PROGRESSIVE ===
    # Removed to fix Redis async/sync errors and over-complication
    # execution_state = await create_request_state(request.conversation_id, request.message)
    # request_id = execution_state.request_id
    # logger.info(f"[EXECUTION_STATE] Created progressive state tracking for request {request_id[:16]}...")
    
    # === PRE-CLASSIFICATION FOR PROGRESSIVE ENDPOINT ===
    
    async def progressive_chat_stream():
        try:
            logger.info(f"[PROGRESSIVE_ENDPOINT] Starting progressive chat for {notebook_id}: {request.message[:50]}...")
            
            if not request or not request.message:
                yield json.dumps({
                    "error": "Chat request validation failed: message is required",
                    "error_details": {
                        "error_type": "ValidationError",
                        "notebook_id": notebook_id,
                        "request_provided": bool(request),
                        "message_provided": bool(request and request.message) if request else False
                    },
                    "troubleshooting": "Ensure request body contains a 'message' field with non-empty text"
                }) + "\n"
                return
            
            # Verify notebook exists
            notebook_exists = await notebook_service.notebook_exists(db=db, notebook_id=notebook_id)
            if not notebook_exists:
                yield json.dumps({
                    "error": "Notebook access failed: notebook not found or inaccessible",
                    "error_details": {
                        "error_type": "NotFoundError",
                        "notebook_id": notebook_id,
                        "notebook_exists": False
                    },
                    "troubleshooting": "Verify notebook ID exists and you have access permissions"
                }) + "\n"
                return
            
            # === PHASE 1: CONVERSATION INTELLIGENCE FIRST (PROGRESSIVE) ===
            # Check conversation context FIRST before any classification or routing
            logger.info("[PROGRESSIVE_CONVERSATION_FIRST] Checking for conversation context before classification")
            
            # Get conversation context (previous exchange)
            conversation_context = await conversation_context_manager.get_conversation_context(request.conversation_id)
            
            # === CACHE BYPASS DETECTION: Check if user query requires fresh data retrieval ===
            bypass_decision = cache_bypass_detector.should_bypass_cache(request.message, request.conversation_id)
            logger.info(f"[CACHE_BYPASS] Progressive endpoint decision for conversation {request.conversation_id}: "
                       f"bypass={bypass_decision['should_bypass']}, confidence={bypass_decision['confidence']:.2f}, "
                       f"reason={bypass_decision['reason']}")
            
            # Get cached retrieval context if available (unless bypassing)
            cached_context = None
            if not bypass_decision['should_bypass']:
                cached_context = await conversation_context_manager.get_cached_context(request.conversation_id)
            else:
                logger.info(f"[CACHE_BYPASS] Progressive endpoint skipping cache retrieval due to bypass decision")
            
            # === TRANSFORMATION HANDLING FOR PROGRESSIVE: Check for transformation requests first ===
            if conversation_context and _classify_query_intent(request.message) == 'contextual':
                logger.info(f"[PROGRESSIVE_TRANSFORMATION_HANDLER] Transformation request detected: {request.message[:100]}")
                
                try:
                    # Generate transformation response using existing conversation data
                    transformation_response = await _handle_contextual_query(
                        request.message,
                        conversation_context,
                        cached_context,
                        notebook_id,
                        request.conversation_id
                    )
                    
                    if transformation_response:
                        logger.info(f"[PROGRESSIVE_CONTEXTUAL_HANDLER] Successfully generated contextual response")
                        
                        yield json.dumps({
                            "status": "completed",
                            "message": "Progressive contextual handling applied to previous response data",
                            "notebook_id": notebook_id,
                            "conversation_id": request.conversation_id,
                            "routing_decision": "progressive_contextual_complete",
                            "query_type": query_type
                        }) + "\n"
                        
                        yield json.dumps(transformation_response) + "\n"
                        
                        # Store this contextual response for future reference
                        await conversation_context_manager.store_conversation_response(
                            request.conversation_id,
                            request.message,
                            transformation_response.get('response', ''),
                            transformation_response.get('sources', [])
                        )
                        
                        return
                    else:
                        logger.info(f"[PROGRESSIVE_CONTEXTUAL_HANDLER] Could not handle contextual query - falling back to retrieval")
                        
                except Exception as e:
                    logger.error(f"[PROGRESSIVE_CONTEXTUAL_HANDLER] Error handling contextual query: {str(e)}")
                    # Fall through to normal retrieval on error
            
            # Handle different query types with appropriate strategies (progressive)
            if query_type == 'simple':
                logger.info(f"[PROGRESSIVE_SIMPLE_QUERY] Simple query detected - using minimal retrieval")
            elif query_type == 'temporal':
                logger.info(f"[PROGRESSIVE_TEMPORAL_QUERY] Time-related query detected - using fresh data retrieval")
            
            # Always try LLM-driven intelligent response FIRST - let LLM decide cache vs fresh retrieval
            conversation_history = [conversation_context] if conversation_context else []
            should_use_context = _should_handle_with_context(request.message, conversation_history) or bool(cached_context)
            
            # Try LLM-driven intelligent response FIRST - LLM will decide whether to use cached data or request fresh retrieval
            if should_use_context and (conversation_context or cached_context):
                logger.info(f"[PROGRESSIVE_CONVERSATION_FIRST] Using context - conversation: {bool(conversation_context)}, cached: {bool(cached_context)}, query_type: {query_type}")
                
                yield json.dumps({
                    "status": "progressive_analyzing_conversation",
                    "message": "AI analyzing conversation context and available data",
                    "notebook_id": notebook_id,
                    "conversation_id": request.conversation_id,
                    "routing_decision": "progressive_conversation_intelligence_first"
                }) + "\n"
                
                # Let LLM make intelligent decision with streaming BEFORE any intent classification
                intelligent_stream = respond_with_llm_intelligence_streaming(
                    request.message,
                    cached_context,
                    conversation_context,
                    notebook_id,
                    request.conversation_id
                )
                
                # Try to use streaming intelligence
                intelligent_used = False
                collected_response = ""
                collected_sources = []
                
                try:
                    # Process the streaming response
                    async for chunk in intelligent_stream:
                        if not intelligent_used:
                            # First chunk indicates success
                            intelligent_used = True
                            logger.info(f"[PROGRESSIVE_CONVERSATION_FIRST] LLM generated streaming response using existing data")
                            
                            yield json.dumps({
                                "status": "completed",
                                "message": "AI providing intelligent streaming response using conversation context",
                                "notebook_id": notebook_id,
                                "conversation_id": request.conversation_id,
                                "routing_decision": "progressive_conversation_intelligence_streaming"
                            }) + "\n"
                        
                        # Parse and collect response data from chunks
                        try:
                            chunk_data = json.loads(chunk.rstrip('\n'))
                            
                            # Collect sources from source chunks
                            if 'sources' in chunk_data:
                                collected_sources = chunk_data['sources']
                            
                            # Collect response text from answer chunks
                            elif 'answer' in chunk_data:
                                collected_response = chunk_data['answer']
                            
                            # Also collect from streaming text chunks
                            elif 'chunk' in chunk_data:
                                collected_response += chunk_data['chunk']
                                
                        except (json.JSONDecodeError, AttributeError):
                            # Handle non-JSON chunks gracefully
                            pass
                        
                        # Stream the chunk
                        yield chunk
                        
                except Exception as e:
                    logger.info(f"[PROGRESSIVE_CONVERSATION_FIRST] Streaming intelligence indicated retrieval needed or failed: {str(e)}")
                    intelligent_used = False
                
                if intelligent_used:
                    
                    # Store this conversation exchange for future context using collected data
                    await conversation_context_manager.store_conversation_response(
                        request.conversation_id,
                        request.message,
                        collected_response,
                        collected_sources
                    )
                    
                    return
                else:
                    # LLM requested new retrieval - continue to intent classification
                    logger.info(f"[PROGRESSIVE_CONVERSATION_FIRST] LLM determined new retrieval is needed - proceeding to classification")
            else:
                logger.info(f"[PROGRESSIVE_CONVERSATION_FIRST] No conversation or cached context available - proceeding to classification")
            
            # === PHASE 2: INTELLIGENT MESSAGE CLASSIFICATION FOR PROGRESSIVE ===
            # Only classify intent if conversation intelligence didn't handle the request
            intent = await classify_message_intent(request.message)
            logger.info(f"[PROGRESSIVE_INTELLIGENT_ROUTING] Processing {intent} message")
            
            # Handle non-retrieval intents with simple responses
            if intent in ['greeting', 'acknowledgment']:
                logger.info(f"[PROGRESSIVE_INTELLIGENT_ROUTING] Using simple response for {intent} - NO RETRIEVAL triggered")
                simple_response = await simple_llm_response(request.message, notebook_id, request.conversation_id)
                
                if simple_response:
                    yield json.dumps({
                        "status": "completed",
                        "message": "Response ready - no retrieval needed",
                        "notebook_id": notebook_id,
                        "conversation_id": request.conversation_id,
                        "routing_decision": f"progressive_simple_response_for_{intent}"
                    }) + "\n"
                    
                    yield json.dumps(simple_response) + "\n"
                    return
            
            elif intent in ['clarification', 'general_chat'] and len(request.message) < 20:
                logger.info(f"[PROGRESSIVE_INTELLIGENT_ROUTING] Short {intent} message - NO RETRIEVAL triggered")
                yield json.dumps({
                    "status": "completed",
                    "message": "Clarification response ready",
                    "notebook_id": notebook_id,
                    "conversation_id": request.conversation_id,
                    "routing_decision": "progressive_clarification_prompt"
                }) + "\n"
                
                yield json.dumps({
                    "response": "Could you please provide more details about what you'd like to know?",
                    "conversation_id": request.conversation_id or f"notebook-{notebook_id}",
                    "sources": [],
                    "metadata": {
                        "routing": "progressive_clarification_prompt",
                        "retrieval_triggered": False,
                        "message_classification": "short_clarification"
                    }
                }) + "\n"
                return
            
            # === PHASE 3: FALLBACK TO RETRIEVAL PLANNING (PROGRESSIVE) ===
            # If we reach here, either no context existed or LLM requested new retrieval
            logger.info(f"[PROGRESSIVE_RETRIEVAL_PLANNING] Proceeding to retrieval planning for intent: {intent}")
            intent = 'retrieval_required'  # Force retrieval since conversation intelligence didn't handle it
            
            # === PHASE 4: INTELLIGENT RETRIEVAL PLANNING (Progressive) ===
            # Determine HOW to retrieve based on query complexity
            retrieval_plan = await notebook_rag_service.plan_retrieval_strategy(request.message, intent)
            logger.info(f"[PROGRESSIVE_RETRIEVAL_PLANNING] Using {retrieval_plan.intensity.value} strategy: {retrieval_plan.reasoning}")
            
            # For intents that need retrieval, use planned approach
            logger.info(f"[PROGRESSIVE_INTELLIGENT_ROUTING] Intent '{intent}' requires retrieval - using {retrieval_plan.intensity.value} approach")
            
            # Analyze query intent for progressive loading decision (only for retrieval intents)
            intent_analysis = await notebook_rag_service._analyze_query_intent(request.message)
            intent_analysis["estimated_sources_needed"] = request.max_sources
            
            logger.info(f"[PROGRESSIVE_ENDPOINT] Intent analysis: comprehensive={intent_analysis.get('wants_comprehensive', False)}, "
                       f"quantity={intent_analysis.get('quantity_intent', 'limited')}")
            
            # Send initial status (only for retrieval intents)
            yield json.dumps({
                "status": "analyzing",
                "message": "Analyzing query and determining retrieval strategy...",
                "notebook_id": notebook_id,
                "conversation_id": request.conversation_id,
                "routing_decision": f"progressive_full_retrieval_for_{intent}",
                "intent_analysis": {
                    "message_classification": intent,
                    "wants_comprehensive": intent_analysis.get("wants_comprehensive", False),
                    "quantity_intent": intent_analysis.get("quantity_intent", "limited"),
                    "confidence": intent_analysis.get("confidence", 0.5)
                }
            }) + "\n"
            
            if request.include_context:
                # Stream progressive retrieval results (only for retrieval intents)
                yield json.dumps({
                    "status": "retrieving",
                    "message": f"Starting progressive document retrieval for {intent} query...",
                    "notebook_id": notebook_id,
                    "conversation_id": request.conversation_id,
                    "message_classification": intent
                }) + "\n"
                
                # Use progressive streaming function with retrieval plan
                async for progress_chunk in stream_progressive_notebook_chat(
                    notebook_id=notebook_id,
                    query=request.message,
                    intent_analysis=intent_analysis,
                    rag_service=notebook_rag_service,
                    max_sources=request.max_sources,
                    retrieval_plan=retrieval_plan
                ):
                    yield progress_chunk
            
            # After retrieval is complete, send final status
            yield json.dumps({
                "status": "complete",
                "message": "Progressive retrieval complete",
                "notebook_id": notebook_id,
                "conversation_id": request.conversation_id
            }) + "\n"
            
            logger.info(f"[PROGRESSIVE_ENDPOINT] Successfully completed progressive chat for {notebook_id}")
            
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"[PROGRESSIVE_ENDPOINT] Progressive chat error for {notebook_id}: {error_type}: {str(e)}")
            logger.error(f"[PROGRESSIVE_ENDPOINT] Context: query_length={len(request.message) if request and request.message else 0}")
            import traceback
            logger.error(f"[PROGRESSIVE_ENDPOINT] Progressive error traceback: {traceback.format_exc()}")
            
            yield json.dumps({
                "error": f"Progressive chat failed with {error_type}: {str(e)}",
                "error_details": {
                    "error_type": error_type,
                    "notebook_id": notebook_id,
                    "conversation_id": request.conversation_id if request else None,
                    "progressive_loading_attempted": True,
                    "query_length": len(request.message) if request and request.message else 0
                },
                "status": "error",
                "troubleshooting": "Progressive loading failed. Check system resources and database connectivity."
            }) + "\n"
    
    return StreamingResponse(progressive_chat_stream(), media_type="application/json")


# Memory Management Endpoints

@router.post("/{notebook_id}/memories", response_model=MemoryResponse, status_code=201)
async def create_memory(
    notebook_id: str = Path(..., description="Notebook ID"),
    request: MemoryCreateRequest = ...,
    db: Session = Depends(get_db)
):
    """
    Create a new memory for a notebook.
    
    Args:
        notebook_id: Target notebook ID
        request: Memory creation parameters
        db: Database session
        
    Returns:
        Created memory details
        
    Raises:
        HTTPException: If creation fails
    """
    try:
        logger.info(f"Creating memory '{request.name}' for notebook {notebook_id}")
        
        memory = await notebook_service.create_memory(
            db=db,
            notebook_id=notebook_id,
            name=request.name,
            content=request.content,
            description=request.description,
            metadata=request.metadata
        )
        
        logger.info(f"Successfully created memory {memory.memory_id}")
        return memory
        
    except ValueError as e:
        logger.error(f"Validation error creating memory: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create memory: {str(e)}")

@router.get("/{notebook_id}/memories", response_model=MemoryListResponse)
async def get_memories(
    notebook_id: str = Path(..., description="Notebook ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    db: Session = Depends(get_db)
):
    """
    Get memories for a notebook with pagination.
    
    Args:
        notebook_id: Target notebook ID
        page: Page number (1-based)
        page_size: Number of memories per page
        db: Database session
        
    Returns:
        List of memories with pagination info
        
    Raises:
        HTTPException: If retrieval fails
    """
    try:
        logger.info(f"Getting memories for notebook {notebook_id} (page {page})")
        
        memories = await notebook_service.get_memories(
            db=db,
            notebook_id=notebook_id,
            page=page,
            page_size=page_size
        )
        
        return memories
        
    except ValueError as e:
        logger.error(f"Validation error getting memories: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting memories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get memories: {str(e)}")

@router.get("/{notebook_id}/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    notebook_id: str = Path(..., description="Notebook ID"),
    memory_id: str = Path(..., description="Memory ID"),
    db: Session = Depends(get_db)
):
    """
    Get a specific memory by ID.
    
    Args:
        notebook_id: Target notebook ID
        memory_id: Memory ID
        db: Database session
        
    Returns:
        Memory details
        
    Raises:
        HTTPException: If memory not found
    """
    try:
        logger.info(f"Getting memory {memory_id} for notebook {notebook_id}")
        
        memory = await notebook_service.get_memory(
            db=db,
            notebook_id=notebook_id,
            memory_id=memory_id
        )
        
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        return memory
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory: {str(e)}")

@router.put("/{notebook_id}/memories/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    notebook_id: str = Path(..., description="Notebook ID"),
    memory_id: str = Path(..., description="Memory ID"),
    request: MemoryUpdateRequest = ...,
    db: Session = Depends(get_db)
):
    """
    Update a memory.
    
    Args:
        notebook_id: Target notebook ID
        memory_id: Memory ID
        request: Memory update parameters
        db: Database session
        
    Returns:
        Updated memory details
        
    Raises:
        HTTPException: If update fails
    """
    try:
        logger.info(f"Updating memory {memory_id} for notebook {notebook_id}")
        
        memory = await notebook_service.update_memory(
            db=db,
            notebook_id=notebook_id,
            memory_id=memory_id,
            name=request.name,
            description=request.description,
            content=request.content,
            metadata=request.metadata
        )
        
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        logger.info(f"Successfully updated memory {memory_id}")
        return memory
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error updating memory: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update memory: {str(e)}")

@router.put("/{notebook_id}/documents/{document_id}", response_model=NotebookDocumentResponse)
async def update_document(
    notebook_id: str = Path(..., description="Notebook ID"),
    document_id: str = Path(..., description="Document ID"),
    request: DocumentUpdateRequest = ...,
    db: Session = Depends(get_db)
):
    """
    Update a document's name and metadata.
    
    Args:
        notebook_id: Target notebook ID
        document_id: Document ID
        request: Document update parameters
        db: Database session
        
    Returns:
        Updated document details
        
    Raises:
        HTTPException: If update fails
    """
    try:
        logger.info(f"Updating document {document_id} for notebook {notebook_id}")
        logger.info(f"Request data - name: '{request.name}', metadata: {request.metadata}")
        
        document = await notebook_service.update_document(
            db=db,
            notebook_id=notebook_id,
            document_id=document_id,
            name=request.name,
            metadata=request.metadata
        )
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"Successfully updated document {document_id}")
        return document
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error updating document: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update document: {str(e)}")

@router.delete("/{notebook_id}/memories/{memory_id}")
async def delete_memory(
    notebook_id: str = Path(..., description="Notebook ID"),
    memory_id: str = Path(..., description="Memory ID"),
    db: Session = Depends(get_db)
):
    """
    Delete a memory and all its associated data.
    
    Args:
        notebook_id: Target notebook ID
        memory_id: Memory ID
        db: Database session
        
    Returns:
        Success confirmation
        
    Raises:
        HTTPException: If deletion fails
    """
    try:
        logger.info(f"Deleting memory {memory_id} for notebook {notebook_id}")
        
        deleted = await notebook_service.delete_memory(
            db=db,
            notebook_id=notebook_id,
            memory_id=memory_id
        )
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        logger.info(f"Successfully deleted memory {memory_id}")
        return {"message": "Memory deleted successfully", "deleted": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {str(e)}")

# Chunk Management Endpoints

@router.get("/chunks/{collection_name}/{document_id}", response_model=ChunkListResponse)
async def get_chunks_for_document(
    collection_name: str = Path(..., description="Milvus collection name"),
    document_id: str = Path(..., description="Document or memory ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    db: Session = Depends(get_db)
):
    """
    Get all chunks for a specific document or memory.
    
    Args:
        collection_name: Milvus collection name
        document_id: Document or memory ID
        page: Page number (1-based)
        page_size: Number of chunks per page
        db: Database session
        
    Returns:
        List of chunks with pagination info
        
    Raises:
        HTTPException: If retrieval fails
    """
    try:
        logger.info(f"Getting chunks for document {document_id} from collection {collection_name}")
        
        chunks = await chunk_management_service.get_chunks_for_document(
            db=db,
            collection_name=collection_name,
            document_id=document_id,
            page=page,
            page_size=page_size
        )
        
        return chunks
        
    except ValueError as e:
        logger.error(f"Validation error getting chunks: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get chunks: {str(e)}")

@router.get("/chunks/{collection_name}/chunk/{chunk_id}", response_model=ChunkResponse)
async def get_chunk_by_id(
    collection_name: str = Path(..., description="Milvus collection name"),
    chunk_id: str = Path(..., description="Chunk ID"),
    db: Session = Depends(get_db)
):
    """
    Get a specific chunk by ID.
    
    Args:
        collection_name: Milvus collection name
        chunk_id: Chunk ID to retrieve
        db: Database session
        
    Returns:
        Chunk details with edit history
        
    Raises:
        HTTPException: If chunk not found
    """
    try:
        logger.info(f"Getting chunk {chunk_id} from collection {collection_name}")
        
        chunk = await chunk_management_service.get_chunk_by_id(
            db=db,
            collection_name=collection_name,
            chunk_id=chunk_id
        )
        
        return chunk
        
    except ValueError as e:
        logger.error(f"Validation error getting chunk: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting chunk: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get chunk: {str(e)}")

@router.put("/chunks/{collection_name}/chunk/{chunk_id}", response_model=ChunkOperationResponse)
async def update_chunk(
    collection_name: str = Path(..., description="Milvus collection name"),
    chunk_id: str = Path(..., description="Chunk ID"),
    request: ChunkUpdateRequest = ...,
    user_id: Optional[str] = Query(None, description="User ID making the edit"),
    db: Session = Depends(get_db)
):
    """
    Update a chunk's content and optionally re-embed it.
    
    Args:
        collection_name: Milvus collection name
        chunk_id: Chunk ID to update
        request: Chunk update parameters
        user_id: User making the edit
        db: Database session
        
    Returns:
        Operation result
        
    Raises:
        HTTPException: If update fails
    """
    try:
        logger.info(f"Updating chunk {chunk_id} in collection {collection_name}")
        
        result = await chunk_management_service.update_chunk(
            db=db,
            collection_name=collection_name,
            chunk_id=chunk_id,
            new_content=request.content,
            re_embed=request.re_embed,
            user_id=user_id,
            edit_metadata=request.metadata
        )
        
        return result
        
    except ValueError as e:
        logger.error(f"Validation error updating chunk: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating chunk: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update chunk: {str(e)}")

@router.post("/chunks/{collection_name}/bulk-re-embed", response_model=BulkChunkOperationResponse)
async def bulk_re_embed_chunks(
    collection_name: str = Path(..., description="Milvus collection name"),
    request: BulkChunkReEmbedRequest = ...,
    user_id: Optional[str] = Query(None, description="User ID performing the operation"),
    db: Session = Depends(get_db)
):
    """
    Re-embed multiple chunks in bulk.
    
    Args:
        collection_name: Milvus collection name
        request: Bulk re-embed parameters
        user_id: User performing the operation
        db: Database session
        
    Returns:
        Bulk operation results
        
    Raises:
        HTTPException: If operation fails
    """
    try:
        logger.info(f"Bulk re-embedding {len(request.chunk_ids)} chunks in collection {collection_name}")
        
        result = await chunk_management_service.bulk_re_embed_chunks(
            db=db,
            collection_name=collection_name,
            chunk_ids=request.chunk_ids,
            user_id=user_id
        )
        
        return result
        
    except ValueError as e:
        logger.error(f"Validation error bulk re-embedding: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error bulk re-embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to bulk re-embed: {str(e)}")

@router.get("/{notebook_id}/efficiency-metrics")
async def get_notebook_efficiency_metrics(
    notebook_id: str = Path(..., description="Notebook ID"),
    hours: int = Query(24, description="Number of hours to analyze", ge=1, le=168),
    db: Session = Depends(get_db)
):
    """
    Get efficiency metrics for intelligent routing system.
    
    Provides comprehensive analysis of routing decisions, cache effectiveness,
    retrieval planning optimization, and performance improvements over the
    specified time period.
    
    Args:
        notebook_id: Notebook ID to get metrics for
        hours: Number of hours to analyze (1-168, default 24)
        db: Database session
        
    Returns:
        Dict containing comprehensive efficiency metrics and analysis
        
    Raises:
        HTTPException: If notebook not found or metrics unavailable
    """
    try:
        # Verify notebook exists
        notebook_exists = await notebook_service.notebook_exists(db=db, notebook_id=notebook_id)
        if not notebook_exists:
            raise HTTPException(status_code=404, detail=f"Notebook {notebook_id} not found")
        
        logger.info(f"[METRICS] Getting efficiency metrics for notebook {notebook_id} (last {hours} hours)")
        
        # Get efficiency summary from metrics service
        metrics = IntelligentRoutingMetrics()
        summary = await metrics.get_efficiency_summary(hours)
        
        # Add notebook-specific context
        response = {
            "notebook_id": notebook_id,
            "time_period_hours": hours,
            "efficiency_summary": summary,
            "system_info": {
                "intelligent_routing_active": True,
                "understand_think_plan_do_framework": True,
                "metrics_collection_enabled": True,
                "cache_ttl_hours": 24
            },
            "generated_at": datetime.now().isoformat()
        }
        
        # Add interpretation if we have data
        if summary.get('status') == 'success':
            total_messages = summary.get('total_messages_analyzed', 0)
            efficiency_rate = summary.get('routing_efficiency', {}).get('efficiency_rate', '0%')
            cache_hit_rate = summary.get('cache_effectiveness', {}).get('hit_rate', '0%')
            
            response["insights"] = {
                "efficiency_summary": f"Analyzed {total_messages} messages with {efficiency_rate} routing efficiency",
                "cache_performance": f"Cache hit rate of {cache_hit_rate} providing significant time savings",
                "intelligent_routing_benefit": "Intelligent routing avoids expensive operations when possible",
                "performance_optimization": "System automatically optimizes based on query complexity"
            }
        
        logger.info(f"[METRICS] Efficiency metrics retrieved successfully for notebook {notebook_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[METRICS] Failed to get efficiency metrics for notebook {notebook_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve efficiency metrics: {str(e)}"
        )


@router.get("/{notebook_id}/cache-status/{conversation_id}")
async def get_notebook_cache_status(
    notebook_id: str = Path(..., description="Notebook ID"),
    conversation_id: str = Path(..., description="Conversation ID"),
    db: Session = Depends(get_db)
):
    """
    Get cache status and metadata for a specific notebook conversation.
    
    Returns detailed information about cached context including source count,
    cache expiration time, and query preview for debugging and monitoring.
    
    Args:
        notebook_id: Notebook ID to check cache for
        conversation_id: Conversation ID to get cache status
        db: Database session
        
    Returns:
        Dict containing cache status, metadata, and statistics
        
    Raises:
        HTTPException: If notebook not found or cache unavailable
    """
    try:
        # Verify notebook exists
        notebook_exists = await notebook_service.notebook_exists(db=db, notebook_id=notebook_id)
        if not notebook_exists:
            logger.warning(f"[CACHE_STATUS] Notebook not found: {notebook_id}")
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        # Get cache statistics
        cache_stats = await notebook_rag_service.get_cache_stats(conversation_id)
        
        logger.info(f"[CACHE_STATUS] Retrieved cache status for notebook {notebook_id}, "
                   f"conversation {conversation_id}: {cache_stats.get('status', 'unknown')}")
        
        return {
            "notebook_id": notebook_id,
            "conversation_id": conversation_id,
            "cache_status": cache_stats,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CACHE_STATUS] Failed to get cache status for notebook {notebook_id}, "
                    f"conversation {conversation_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve cache status: {str(e)}"
        )


@router.delete("/{notebook_id}/cache/{conversation_id}")
async def clear_notebook_cache(
    notebook_id: str = Path(..., description="Notebook ID"),
    conversation_id: str = Path(..., description="Conversation ID"),
    db: Session = Depends(get_db)
):
    """
    Clear cached context for a specific notebook conversation.
    
    Removes all cached conversation context and metadata for the specified
    notebook and conversation. Useful for forcing fresh context retrieval
    or clearing stale cache entries.
    
    Args:
        notebook_id: Notebook ID to clear cache for
        conversation_id: Conversation ID to clear cache
        db: Database session
        
    Returns:
        Dict containing operation result and metadata
        
    Raises:
        HTTPException: If notebook not found or cache clearing fails
    """
    try:
        # Verify notebook exists
        notebook_exists = await notebook_service.notebook_exists(db=db, notebook_id=notebook_id)
        if not notebook_exists:
            logger.warning(f"[CACHE_CLEAR] Notebook not found: {notebook_id}")
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        # Clear cache
        cache_cleared = await notebook_rag_service.invalidate_context(conversation_id)
        
        if cache_cleared:
            logger.info(f"[CACHE_CLEAR] Successfully cleared cache for notebook {notebook_id}, "
                       f"conversation {conversation_id}")
            return {
                "notebook_id": notebook_id,
                "conversation_id": conversation_id,
                "status": "cleared",
                "message": "Cache successfully cleared",
                "timestamp": datetime.now().isoformat()
            }
        else:
            logger.warning(f"[CACHE_CLEAR] No cache found or failed to clear for notebook {notebook_id}, "
                          f"conversation {conversation_id}")
            return {
                "notebook_id": notebook_id,
                "conversation_id": conversation_id,
                "status": "no_cache",
                "message": "No cache found to clear",
                "timestamp": datetime.now().isoformat()
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CACHE_CLEAR] Failed to clear cache for notebook {notebook_id}, "
                    f"conversation {conversation_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to clear cache: {str(e)}"
        )


# Error handlers are handled at the app level, not router level
# The individual endpoints already have proper try/catch blocks with HTTPException