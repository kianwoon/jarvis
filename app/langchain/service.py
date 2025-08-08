import requests
import re
import httpx
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config, get_query_classifier_full_config
from app.core.embedding_settings_cache import get_embedding_settings
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.core.mcp_tools_cache import get_enabled_mcp_tools
from app.core.collection_registry_cache import get_all_collections
from app.core.rag_tool_config import get_rag_tool_config
from app.api.v1.endpoints.document import HTTPEmbeddingFunction
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from app.rag.bm25_processor import BM25Processor, BM25CorpusManager

logger = logging.getLogger(__name__)

# Enhanced conversation management with Redis fallback

# Cache RAG tool name to avoid repeated config calls
_rag_tool_name_cache = None

def get_rag_tool_name():
    """Get the configurable RAG tool name (cached)"""
    global _rag_tool_name_cache
    if _rag_tool_name_cache is None:
        rag_config = get_rag_tool_config()
        _rag_tool_name_cache = rag_config.get_tool_name()
    return _rag_tool_name_cache

def reload_rag_tool_name_cache():
    """Reload the RAG tool name cache"""
    global _rag_tool_name_cache
    _rag_tool_name_cache = None

def is_rag_tool(tool_name: str) -> bool:
    """Check if a tool name matches the configurable RAG tool"""
    return tool_name.lower() == get_rag_tool_name().lower() or tool_name.lower() in get_rag_tool_name().lower()
_conversation_cache = {}  # In-memory fallback

# Simple query cache for RAG results (in-memory, expires after 5 minutes)
import time
_rag_cache = {}  # {query_hash: (result, timestamp)}

def get_rag_cache_settings():
    """Get RAG cache settings"""
    from app.core.rag_settings_cache import get_performance_settings
    perf_settings = get_performance_settings()
    return {
        'ttl': perf_settings.get('cache_ttl_hours', 2) * 3600,  # Convert to seconds
        'max_size': get_document_retrieval_settings().get('cache_max_size', 100)
    }

def get_document_retrieval_settings():
    """Get document retrieval settings"""
    from app.core.rag_settings_cache import get_document_retrieval_settings as _get_settings
    return _get_settings()

def _cache_rag_result(query_hash: int, result: tuple, current_time: float):
    """Helper function to cache RAG results with cleanup"""
    _rag_cache[query_hash] = (result, current_time)
    # Clean up cache if it gets too large
    cache_settings = get_rag_cache_settings()
    max_size = cache_settings['max_size']
    if len(_rag_cache) > max_size:
        # Remove oldest entries
        sorted_cache = sorted(_rag_cache.items(), key=lambda x: x[1][1])
        entries_to_remove = min(20, len(_rag_cache) - max_size + 20)  # Remove extra entries
        for old_key, _ in sorted_cache[:entries_to_remove]:
            del _rag_cache[old_key]

def get_redis_conversation_client():
    """Get Redis client specifically for conversation storage"""
    try:
        from app.core.redis_client import get_redis_client
        return get_redis_client()
    except Exception as e:
        print(f"[DEBUG] Redis client not available for conversations: {e}")
        return None

def build_enhanced_system_prompt(base_prompt: str = None) -> str:
    """Build system prompt with real MCP tools, RAG collections, and temporal context from cache"""
    
    
    # Get base system prompt from settings if not provided
    if not base_prompt:
        llm_settings = get_llm_settings()
        base_prompt = llm_settings.get('main_llm', {}).get('system_prompt', 'You are Jarvis, an AI assistant.')
    
    # Enhance with temporal context first (before other enhancements)
    try:
        from app.core.temporal_context_manager import get_temporal_context_manager
        temporal_manager = get_temporal_context_manager()
        base_prompt = temporal_manager.enhance_system_prompt_with_time_context(base_prompt)
        logger.debug("Successfully enhanced system prompt with temporal context")
    except Exception as e:
        logger.warning(f"Failed to enhance system prompt with temporal context: {e}")
    
    
    # Check if system prompt already has format instructions
    has_format_instructions = any(phrase in base_prompt.lower() for phrase in [
        'format at the end', 'in this format', 'opinion:', 'provide your point of view',
        'end with', 'conclude with', 'finish with'
    ])
    
    # Get available MCP tools from cache
    tools_info = []
    try:
        mcp_tools = get_enabled_mcp_tools()
        if mcp_tools:
            tools_info.append("\n**Available Tools:**")
            for tool_name, tool_info in mcp_tools.items():
                # Extract description from manifest if available
                description = "Available for use"
                if isinstance(tool_info, dict):
                    manifest = tool_info.get('manifest', {})
                    if manifest and 'tools' in manifest:
                        for tool_def in manifest['tools']:
                            if tool_def.get('name') == tool_name:
                                description = tool_def.get('description', description)
                                break
                tools_info.append(f"- **{tool_name}**: {description}")
    except Exception as e:
        logger.error(f"Failed to load MCP tools for prompt: {e}")
    
    # Get available RAG collections from cache
    collections_info = []
    try:
        collections = get_all_collections()
        if collections:
            collections_info.append("\n**Available Knowledge Collections:**")
            for collection in collections:
                name = collection.get('collection_name', '')
                description = collection.get('description', 'No description')
                collection_type = collection.get('collection_type', 'Unknown')
                stats = collection.get('statistics', {})
                doc_count = stats.get('document_count', 0)
                
                collections_info.append(
                    f"- **{name}**: {description} (Type: {collection_type}, Documents: {doc_count})"
                )
    except Exception as e:
        print(f"Failed to load RAG collections for prompt: {e}")
    
    # Build enhanced prompt
    enhanced_prompt = base_prompt
    
    if tools_info:
        enhanced_prompt += "\n" + "\n".join(tools_info)
        enhanced_prompt += "\n\nYou can use these tools when needed by including tool calls in your response."
    
    if collections_info:
        enhanced_prompt += "\n" + "\n".join(collections_info)
        enhanced_prompt += "\n\nYou can search these collections to find relevant information for the user's query."
    
    # Add minimal guidance only if no format instructions in base prompt
    if (tools_info or collections_info) and not has_format_instructions:
        enhanced_prompt += "\n\n**Guidelines:**"
        enhanced_prompt += "\n- Use tools for real-time data, calculations, or external information"
        enhanced_prompt += "\n- Search collections for documented knowledge and historical information"
        enhanced_prompt += "\n- Combine multiple sources when appropriate for comprehensive answers"
        enhanced_prompt += "\n- Always provide accurate, helpful responses based on available resources"
        
    # Add critical factual data prioritization instructions (always include)
    enhanced_prompt += "\n\n**CRITICAL: FACTUAL DATA PRIORITIZATION:**"
    enhanced_prompt += "\n- When provided with CURRENT FACTUAL DATA (web search, tool results), always prioritize it over conversation history"
    enhanced_prompt += "\n- If fresh search results contradict previous conversation context, trust the fresh data"
    enhanced_prompt += "\n- Conversation history provides context, but factual data provides truth"
    enhanced_prompt += "\n- Always base your answers on the most current and verified information available"
        
    # Add tool calling format instructions
    if tools_info:
        enhanced_prompt += "\n\n**Tool Usage Format:**"
        enhanced_prompt += "\nWhen you need to use a tool, format your tool call as:"
        enhanced_prompt += "\n<tool>tool_name(parameters)</tool>"
        enhanced_prompt += "\n\nExamples:"
        
        # Build dynamic examples based on actual available tools
        try:
            mcp_tools = get_enabled_mcp_tools()
            example_tools = []
            
            # Add RAG search example if available (using configurable tool name)
            rag_tool_name = get_rag_tool_name()
            if rag_tool_name in mcp_tools:
                example_tools.append(f'- <tool>{rag_tool_name}({{"query": "partnership information"}})</tool>')
            
            # Add datetime tool example if available
            datetime_tools = [name for name in mcp_tools.keys() if 'datetime' in name.lower() or 'time' in name.lower()]
            if datetime_tools:
                example_tools.append(f'- <tool>{datetime_tools[0]}()</tool>')
            
            # Add search tool example if available (but not RAG tool)
            rag_tool_name = get_rag_tool_name()
            search_tools = [name for name in mcp_tools.keys() if 'search' in name.lower() and name != rag_tool_name]
            if search_tools:
                example_tools.append(f'- <tool>{search_tools[0]}({{"query": "latest news"}})</tool>')
            
            # If no specific tools found, use generic examples
            if not example_tools:
                example_tools = [
                    f'- <tool>{get_rag_tool_name()}({{"query": "search query"}})</tool>',
                    '- <tool>get_datetime()</tool>'
                ]
            
            for example in example_tools:
                enhanced_prompt += f"\n{example}"
                
        except Exception as e:
            # Fallback to safe examples if tool loading fails
            enhanced_prompt += f'\n- <tool>{get_rag_tool_name()}({{"query": "search query"}})</tool>'
            enhanced_prompt += "\n- <tool>get_datetime()</tool>"
        
        enhanced_prompt += "\n\nIMPORTANT: Use ONLY the exact tool names listed above. Actually include these tool calls in your response when needed, don't just describe what you would do."
    
    
    return enhanced_prompt

def get_conversation_history(conversation_id: str) -> str:
    """Get formatted conversation history with Redis support"""
    if not conversation_id:
        return ""
    
    from app.core.large_generation_utils import get_config_accessor
    config = get_config_accessor()
    
    try:
        # Try Redis first
        redis_client = get_redis_conversation_client()
        if redis_client:
            try:
                # Get conversation from Redis
                redis_key = f"conversation:{conversation_id}"
                history_json = redis_client.lrange(redis_key, -config.conversation_history_display, -1)  # Last N messages
                
                if history_json:
                    import json
                    history = [json.loads(msg) for msg in history_json]
                    
                    # Format messages
                    formatted = []
                    for msg in history:
                        role = "User" if msg.get("role") == "user" else "Assistant"
                        formatted.append(f"{role}: {msg.get('content', '')}")
                    
                    return "\n".join(formatted)
            except Exception as e:
                print(f"[DEBUG] Redis conversation retrieval failed: {e}")
        
        # Fallback to in-memory cache
        history = _conversation_cache.get(conversation_id, [])
        if not history:
            return ""
        
        # Format last N exchanges
        formatted = []
        for msg in history[-config.conversation_history_display:]:  # Last N messages
            role = "User" if msg.get("role") == "user" else "Assistant"
            formatted.append(f"{role}: {msg.get('content', '')}")
        
        return "\n".join(formatted)
    except Exception as e:
        print(f"[ERROR] Failed to get conversation history: {e}")
        return ""

def get_limited_conversation_history(conversation_id: str, max_messages: int = 2, current_query: str = None) -> str:
    """Get limited conversation history for simple factual queries with temporal awareness"""
    if not conversation_id:
        return ""
    
    # Use smart conversation history if enabled and query is provided
    if current_query:
        try:
            from app.core.large_generation_utils import get_config_accessor
            config = get_config_accessor()
            
            if config.enable_smart_context_filtering:
                from app.langchain.conversation_context_manager import get_smart_conversation_history
                return get_smart_conversation_history(
                    conversation_id=conversation_id,
                    current_query=current_query,
                    max_messages=config.smart_context_max_messages or max_messages
                )
        except Exception as e:
            print(f"[DEBUG] Smart conversation history failed, falling back: {e}")
    
    # Apply temporal filtering if current query is provided
    if current_query:
        try:
            from app.core.temporal_context_manager import get_temporal_context_manager
            temporal_manager = get_temporal_context_manager()
            
            # Get full history first
            full_history = get_full_conversation_history(conversation_id)
            if full_history:
                # Apply temporal filtering
                filtered_history = temporal_manager.get_time_aware_conversation_window(
                    full_history, current_query
                )
                
                # Format the filtered history
                formatted = []
                for msg in filtered_history:
                    role = "User" if msg.get("role") == "user" else "Assistant"
                    content = msg.get('content', '')[:100]  # Truncate long content
                    formatted.append(f"{role}: {content}")
                
                if formatted:
                    logger.debug(f"Applied temporal filtering to conversation history: {len(filtered_history)} messages")
                    return "\n".join(formatted)
                    
        except Exception as e:
            logger.debug(f"Temporal conversation filtering failed, using fallback: {e}")
    
    # Original implementation as fallback
    try:
        # Try Redis first
        redis_client = get_redis_conversation_client()
        if redis_client:
            try:
                # Get conversation from Redis - only last few messages
                redis_key = f"conversation:{conversation_id}"
                history_json = redis_client.lrange(redis_key, -max_messages, -1)  # Last max_messages only
                
                if history_json:
                    import json
                    history = [json.loads(msg) for msg in history_json]
                    
                    # Filter out unrelated content (long responses, complex topics)
                    filtered_history = []
                    for msg in history:
                        content = msg.get('content', '')
                        # Skip very long responses or responses about complex topics that might bleed
                        if len(content) > 200 or any(word in content.lower() for word in [
                            'interview questions', 'generate', 'strategy', 'analysis', 'comprehensive',
                            'questions', 'tailored', 'director', 'bank', '50', 'below are',
                            'chunk', 'chunked', 'continuation', 'generate items', 'item number'
                        ]):
                            continue
                        filtered_history.append(msg)
                    
                    # Format filtered messages
                    formatted = []
                    for msg in filtered_history:
                        role = "User" if msg.get("role") == "user" else "Assistant"
                        content = msg.get('content', '')[:100]  # Truncate long content
                        formatted.append(f"{role}: {content}")
                    
                    return "\n".join(formatted)
            except Exception as e:
                print(f"[DEBUG] Redis limited conversation retrieval failed: {e}")
        
        # Fallback to in-memory cache
        history = _conversation_cache.get(conversation_id, [])
        if not history:
            return ""
        
        # Get only recent, short exchanges
        recent_history = history[-max_messages:]
        formatted = []
        for msg in recent_history:
            content = msg.get('content', '')
            # Skip long or complex responses
            if len(content) > 200:
                continue
            role = "User" if msg.get("role") == "user" else "Assistant"
            formatted.append(f"{role}: {content[:100]}")
        
        return "\n".join(formatted)
    except Exception as e:
        print(f"[ERROR] Failed to get limited conversation history: {e}")
        return ""

def store_conversation_message(conversation_id: str, role: str, content: str):
    """Store a message in conversation history with Redis support and temporal context"""
    if not conversation_id:
        return
    
    from app.core.large_generation_utils import get_config_accessor
    config = get_config_accessor()
    
    # Add temporal context to user messages when appropriate
    enhanced_content = content
    if role == "user":
        try:
            from app.core.temporal_context_manager import get_temporal_context_manager
            temporal_manager = get_temporal_context_manager()
            enhanced_content = temporal_manager.add_temporal_context_to_message(content, role)
            if enhanced_content != content:
                logger.debug("Added temporal context to user message")
        except Exception as e:
            logger.debug(f"Failed to add temporal context to message: {e}")
    
    message = {
        "role": role,
        "content": enhanced_content,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Try Redis first
        redis_client = get_redis_conversation_client()
        if redis_client:
            try:
                import json
                redis_key = f"conversation:{conversation_id}"
                
                # Store message in Redis list
                redis_client.lpush(redis_key, json.dumps(message))
                
                # Keep only last N messages (more than in-memory for persistence)
                redis_client.ltrim(redis_key, 0, config.max_redis_messages - 1)
                
                # Set expiration (configurable TTL for conversations)
                redis_client.expire(redis_key, config.redis_conversation_ttl)
                
                print(f"[DEBUG] Stored message in Redis for conversation {conversation_id}")
                
                # Also update in-memory cache for immediate access
                if conversation_id not in _conversation_cache:
                    _conversation_cache[conversation_id] = []
                _conversation_cache[conversation_id].append(message)
                if len(_conversation_cache[conversation_id]) > config.max_memory_messages:
                    _conversation_cache[conversation_id] = _conversation_cache[conversation_id][-config.max_memory_messages:]
                
                return
            except Exception as e:
                print(f"[DEBUG] Redis conversation storage failed: {e}")
        
        # Fallback to in-memory storage
        if conversation_id not in _conversation_cache:
            _conversation_cache[conversation_id] = []
        
        _conversation_cache[conversation_id].append(message)
        
        # Keep only last N messages
        if len(_conversation_cache[conversation_id]) > config.max_memory_messages:
            _conversation_cache[conversation_id] = _conversation_cache[conversation_id][-config.max_memory_messages:]
            
        print(f"[DEBUG] Stored message in memory cache for conversation {conversation_id}")
        
    except Exception as e:
        print(f"[ERROR] Failed to store conversation message: {e}")


def clear_conversation_context_bleeding(conversation_id: str):
    """Clear conversation context to prevent bleeding between different query types"""
    if not conversation_id:
        return
    
    try:
        # Clear from in-memory cache
        if conversation_id in _conversation_cache:
            print(f"[DEBUG] Clearing conversation cache for {conversation_id}")
            del _conversation_cache[conversation_id]
        
        # Optionally clear from Redis (only if we're sure it's needed)
        # This is commented out to preserve legitimate conversation history
        # redis_client = get_redis_conversation_client()
        # if redis_client:
        #     redis_key = f"conversation:{conversation_id}"
        #     redis_client.delete(redis_key)
        
    except Exception as e:
        print(f"[ERROR] Failed to clear conversation context: {e}")


def get_full_conversation_history(conversation_id: str) -> list:
    """Get full conversation history as list for chunked processing"""
    if not conversation_id:
        return []
    
    try:
        # Try Redis first for full history
        redis_client = get_redis_conversation_client()
        if redis_client:
            try:
                import json
                redis_key = f"conversation:{conversation_id}"
                history_json = redis_client.lrange(redis_key, 0, -1)  # All messages
                
                if history_json:
                    # Redis stores newest first, so reverse for chronological order
                    return [json.loads(msg) for msg in reversed(history_json)]
            except Exception as e:
                print(f"[DEBUG] Redis full conversation retrieval failed: {e}")
        
        # Fallback to in-memory cache
        return _conversation_cache.get(conversation_id, [])
        
    except Exception as e:
        print(f"[ERROR] Failed to get full conversation history: {e}")
        return []

def build_prompt(prompt: str, thinking: bool = False, is_internal: bool = False, include_tools: bool = True) -> str:
    """
    Build a comprehensive prompt with optional tool context injection
    
    Args:
        prompt: Base prompt text
        thinking: Whether to include thinking instructions
        is_internal: Whether this is an internal system prompt
        include_tools: Whether to inject MCP tools context
    """
    final_prompt = prompt
    
    # Add thinking instructions if requested
    if thinking:
        final_prompt = (
            "Please show your reasoning step by step before giving the final answer.\n"
            + final_prompt
        )
    
    # Inject MCP tools context for non-internal prompts (unless explicitly disabled)
    if include_tools and not is_internal:
        # Check if this prompt might benefit from tools
        prompt_lower = prompt.lower()
        tool_indicators = [
            'latest', 'recent', 'current', 'search', 'find', 'look up',
            'what time', 'date', 'email', 'send', 'gmail', 'jira',
            'online', 'internet', 'web', 'news', 'update'
        ]
        
        # Only inject tools context if the prompt suggests tool usage might be helpful
        if any(indicator in prompt_lower for indicator in tool_indicators):
            tools_context = get_mcp_tools_context(include_examples=False)  # Compact version for injection
            final_prompt = f"""{final_prompt}

{tools_context}

IMPORTANT: If any of the available tools above can help provide a better answer to this question, use them by formatting your response as:
<tool>tool_name(parameters)</tool>

Only use tools when they would genuinely improve your response with real-time or specific information."""
    
    return final_prompt

def detect_contradictory_information(tool_context: str, conversation_history: str) -> dict:
    """
    Detect when tool_context contains information that contradicts conversation_history.
    
    Returns:
        dict with keys: 'has_contradiction', 'contradiction_type', 'filtered_history', 'override_instructions'
    """
    import re
    
    if not tool_context or not conversation_history:
        return {
            'has_contradiction': False,
            'contradiction_type': None,
            'filtered_history': conversation_history,
            'override_instructions': ""
        }
    
    tool_lower = tool_context.lower()
    history_lower = conversation_history.lower()
    
    # Define contradiction patterns - company/entity + action combinations
    contradiction_patterns = [
        # CRITICAL FIX: OpenAI GPT-5 / ChatGPT-5 specific patterns
        {
            'entity': ['openai', 'open ai', 'chatgpt.*5', 'gpt.*5', 'chatgpt-5', 'gpt-5', 'chatgpt 5', 'gpt 5'],
            'positive_actions': ['released', 'launches', 'introduces', 'announces', 'open-source', 'open source', 'is out', 'available', 'exists', 'launched'],
            'negative_phrases': ['has not released', 'no.*open-source', 'no.*open source', 'not released any', 'does not exist', 'no official release', 'persistent myth', 'myth', 'not exist', 'there is no'],
            'topic': ['models', 'gpt', 'language model', 'chatgpt.*5', 'gpt.*5', 'chatgpt-5', 'gpt-5', 'chatgpt 5', 'gpt 5']
        },
        # Enhanced OpenAI general patterns  
        {
            'entity': ['openai', 'open ai'],
            'positive_actions': ['released', 'launches', 'introduces', 'announces', 'open-source', 'open source', 'is out', 'available'],
            'negative_phrases': ['has not released', 'no.*open-source', 'no.*open source', 'not released any', 'does not exist', 'no official release', 'persistent myth'],
            'topic': ['models', 'gpt', 'language model']
        },
        # Generic patterns for other companies/entities
        {
            'entity': ['google', 'microsoft', 'meta', 'facebook', 'apple', 'amazon'],
            'positive_actions': ['released', 'launched', 'introduced', 'announced', 'available', 'supports', 'is out'],
            'negative_phrases': ['has not', 'no.*available', 'not released', 'does not support', 'does not exist', 'no official release', 'persistent myth'],
            'topic': ['model', 'service', 'feature', 'product', 'api']
        },
        # ENHANCED: General existence vs non-existence patterns with stronger detection
        {
            'entity': ['.*'],  # Any entity
            'positive_actions': ['exists', 'available', 'released', 'launched', 'supports', 'offers', 'is out', 'has been released', 'now available'],
            'negative_phrases': ['does not exist', 'not available', 'no.*available', 'not released', 'not supported', 'no official release', 'persistent myth', 'myth', 'not exist', 'there is no'],
            'topic': ['.*']  # Any topic
        }
    ]
    
    detected_contradictions = []
    
    for pattern in contradiction_patterns:
        # ENHANCED SEMANTIC MATCHING: Check if any entity is mentioned in both contexts
        # Also handle cases where tool mentions product (ChatGPT-5) but history mentions company (OpenAI)
        entity_found = False
        
        for entity in pattern['entity']:
            if entity == '.*':
                entity_found = True
                break
                
            # Direct entity match in both contexts
            if re.search(entity, tool_lower) and re.search(entity, history_lower):
                entity_found = True
                break
            
            # CRITICAL FIX: Semantic entity matching for ChatGPT-5 <-> OpenAI connection
            # Handle case where tool mentions ChatGPT-5/GPT-5 but history mentions OpenAI
            chatgpt_patterns = ['chatgpt.*5', 'gpt.*5', 'chatgpt-5', 'gpt-5', 'chatgpt 5', 'gpt 5']
            openai_patterns = ['openai', 'open ai']
            
            if entity in chatgpt_patterns:
                # Tool has ChatGPT-5, check if history has OpenAI or ChatGPT-5
                if (re.search(entity, tool_lower) and 
                    (any(re.search(openai_pattern, history_lower) for openai_pattern in openai_patterns) or
                     any(re.search(chatgpt_pattern, history_lower) for chatgpt_pattern in chatgpt_patterns))):
                    entity_found = True
                    break
            
            if entity in openai_patterns:
                # Tool has OpenAI, check if history has ChatGPT-5 patterns
                if (re.search(entity, tool_lower) and 
                    any(re.search(chatgpt_pattern, history_lower) for chatgpt_pattern in chatgpt_patterns)):
                    entity_found = True
                    break
        
        if not entity_found:
            continue
        
        # Check for positive action in tool_context
        positive_in_tool = any(re.search(action, tool_lower) for action in pattern['positive_actions'])
        
        # Check for negative phrases in history
        negative_in_history = any(re.search(phrase, history_lower) for phrase in pattern['negative_phrases'])
        
        # Check if topic is relevant
        topic_relevant = any(re.search(topic, tool_lower) or re.search(topic, history_lower) 
                           for topic in pattern['topic'])
        
        if positive_in_tool and negative_in_history and topic_relevant:
            # ENHANCED CONFIDENCE SCORING: ChatGPT-5 existence vs non-existence gets CRITICAL confidence
            confidence = 'high'
            contradiction_type = 'positive_vs_negative'
            
            # Check for ChatGPT-5 specific existence contradictions
            chatgpt5_existence_indicators = ['chatgpt.*5', 'gpt.*5', 'chatgpt-5', 'gpt-5']
            existence_positive = ['is out', 'released', 'available', 'exists', 'launched']
            existence_negative = ['does not exist', 'persistent myth', 'myth', 'no official release', 'there is no']
            
            has_chatgpt5_reference = any(re.search(indicator, tool_lower) or re.search(indicator, history_lower) 
                                       for indicator in chatgpt5_existence_indicators)
            has_existence_positive = any(re.search(pos, tool_lower) for pos in existence_positive)
            has_existence_negative = any(re.search(neg, history_lower) for neg in existence_negative)
            
            if has_chatgpt5_reference and has_existence_positive and has_existence_negative:
                confidence = 'critical'
                contradiction_type = 'existence_vs_nonexistence'
            
            detected_contradictions.append({
                'type': contradiction_type,
                'confidence': confidence,
                'tool_evidence': [action for action in pattern['positive_actions'] if re.search(action, tool_lower)],
                'history_evidence': [phrase for phrase in pattern['negative_phrases'] if re.search(phrase, history_lower)],
                'entity_match_type': 'semantic' if has_chatgpt5_reference else 'direct'
            })
    
    if detected_contradictions:
        # Generate filtered history that minimizes contradictory statements
        filtered_history = conversation_history
        
        # For high-confidence contradictions, mark contradictory portions
        highest_confidence = 'low'
        for contradiction in detected_contradictions:
            if contradiction['confidence'] in ['high', 'critical']:
                if contradiction['confidence'] == 'critical':
                    highest_confidence = 'critical'
                elif highest_confidence != 'critical':
                    highest_confidence = 'high'
                    
                for negative_phrase in contradiction['history_evidence']:
                    # Add warning markers around contradictory statements
                    marker = "[OUTDATED INFORMATION]" if contradiction['confidence'] == 'critical' else "[POTENTIALLY OUTDATED: {m.group()}]"
                    pattern = re.compile(negative_phrase, re.IGNORECASE)
                    if contradiction['confidence'] == 'critical':
                        filtered_history = pattern.sub(lambda m: f"[OUTDATED INFORMATION: {m.group()}]", filtered_history)
                    else:
                        filtered_history = pattern.sub(lambda m: f"[POTENTIALLY OUTDATED: {m.group()}]", filtered_history)
        
        # ENHANCED OVERRIDE INSTRUCTIONS: Stronger language for critical contradictions
        if highest_confidence == 'critical':
            override_instructions = f"""
🚨 CRITICAL CONTRADICTION DETECTED: Existence vs Non-existence conflict found!

The search results contain CURRENT, VERIFIED information about something that was previously stated as non-existent or mythical.

ABSOLUTE OVERRIDE RULES:
1. The search results are the DEFINITIVE source of truth - they contain current facts
2. COMPLETELY IGNORE all contradictory historical statements - they are now obsolete
3. Answer based EXCLUSIVELY on the search results
4. DO NOT acknowledge, mention, or reference any contradictory historical claims
5. Present the current information as fact without hedging or uncertainty
6. Do not use phrases like "despite previous claims" or "contrary to earlier statements"

The user needs current, accurate information. Provide it with confidence based on the search results."""
        else:
            override_instructions = f"""
🚨 CONTRADICTION DETECTED: The search results contain current information that contradicts previous statements in the conversation history.

MANDATORY OVERRIDE RULES:
1. The search results contain CURRENT, VERIFIED information that supersedes any previous statements
2. IGNORE any contradictory information from the conversation history
3. Base your answer EXCLUSIVELY on the fresh search results
4. Do NOT mention or reference the contradictory historical statements
5. Treat the search results as the single source of truth for this response

The user is asking for current information - provide it based on the search results."""

        # Determine the most severe contradiction type
        contradiction_types = [c['type'] for c in detected_contradictions]
        primary_type = 'existence_vs_nonexistence' if 'existence_vs_nonexistence' in contradiction_types else contradiction_types[0]
        
        # Map confidence levels for proper comparison
        confidence_order = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        max_confidence = max(detected_contradictions, key=lambda x: confidence_order.get(x['confidence'], 0))['confidence']
        
        return {
            'has_contradiction': True,
            'contradiction_type': primary_type,
            'filtered_history': filtered_history,
            'override_instructions': override_instructions,
            'confidence': max_confidence,
            'contradictions_found': len(detected_contradictions),
            'highest_confidence_level': highest_confidence
        }
    
    return {
        'has_contradiction': False,
        'contradiction_type': None,
        'filtered_history': conversation_history,
        'override_instructions': ""
    }

def detect_and_resolve_conflicts(info_sources: list, question: str, conversation_id: str = None) -> dict:
    """
    Detect conflicts between information sources and prepare resolution.
    Enhanced with automatic cache invalidation for conflicting messages.
    
    Args:
        info_sources: List of information sources with priority and freshness
        question: User's current question
        conversation_id: ID for conversation cache cleanup
    
    Returns:
        dict: Analysis results with conflicts, resolution instructions, and cache cleanup actions
    """
    if len(info_sources) < 2:
        return {'conflicts_detected': False, 'resolution_summary': '', 'resolved_conflicts': []}
    
    import re
    
    conflicts = []
    resolved_conflicts = []
    
    # Define conflict detection patterns
    conflict_patterns = [
        {
            'type': 'existence_conflict',
            'positive_indicators': [
                r'\b(exists?|available|released?|launched?|announced?|is out|now available)\b',
                r'\b(has been|was|have|got|can be|are)\b.*\b(released?|made available|published)\b'
            ],
            'negative_indicators': [
                r'\b(does not exist|doesn\'t exist|not exist|no.*available|not.*released?)\b',
                r'\b(not.*real|myth|fictional|doesn\'t.*have|does not.*have)\b'
            ]
        },
        {
            'type': 'date_conflict', 
            'patterns': [
                r'\b(\d{4})\b',  # Years
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',
                r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b'
            ]
        },
        {
            'type': 'version_conflict',
            'patterns': [
                r'\b(version|v\.?)\s*(\d+(?:\.\d+)*)\b',
                r'\b([a-zA-Z]+-\d+(?:\.\d+)*)\b',  # ChatGPT-5, GPT-4 etc
                r'\b(\d+(?:\.\d+)*(?:\.\d+)?)\b.*\b(latest|current|new|old)\b'
            ]
        }
    ]
    
    # Compare each pair of sources
    for i in range(len(info_sources)):
        for j in range(i + 1, len(info_sources)):
            source_high = info_sources[i]  # Higher priority (lower number)
            source_low = info_sources[j]   # Lower priority (higher number)
            
            content_high = source_high['content'].lower()
            content_low = source_low['content'].lower()
            
            # Check for existence conflicts
            for pattern in conflict_patterns:
                if pattern['type'] == 'existence_conflict':
                    # Check if high priority source indicates existence
                    high_positive = any(re.search(p, content_high) for p in pattern['positive_indicators'])
                    high_negative = any(re.search(p, content_high) for p in pattern['negative_indicators'])
                    
                    # Check if low priority source indicates non-existence
                    low_positive = any(re.search(p, content_low) for p in pattern['positive_indicators'])
                    low_negative = any(re.search(p, content_low) for p in pattern['negative_indicators'])
                    
                    # Detect existence conflict
                    if (high_positive and low_negative) or (high_negative and low_positive):
                        conflict_type = "Existence conflict detected"
                        resolution = f"Using {source_high['label']} (Priority {source_high['priority']}) over {source_low['label']} (Priority {source_low['priority']})"
                        
                        conflicts.append({
                            'type': pattern['type'],
                            'high_priority_source': source_high['label'],
                            'low_priority_source': source_low['label'],
                            'high_priority_claim': 'exists/available' if high_positive else 'does not exist',
                            'low_priority_claim': 'exists/available' if low_positive else 'does not exist',
                            'resolution': resolution
                        })
                        
                        resolved_conflicts.append(f"{conflict_type}: {resolution}")
    
    # Identify messages for cache cleanup
    messages_to_remove = []
    cache_cleanup_needed = False
    
    if conflicts:
        # Find conversation sources that were overridden by higher priority sources
        for conflict in conflicts:
            if any(source['source_type'] in ['conversation_fresh', 'conversation_stale', 'conversation_legacy'] 
                   for source in info_sources):
                # Mark conversation messages containing conflicting information for removal
                for source in info_sources:
                    if (source['source_type'] in ['conversation_fresh', 'conversation_stale', 'conversation_legacy'] and
                        source['label'] == conflict['low_priority_source']):
                        messages_to_remove.append({
                            'source_type': source['source_type'],
                            'content_snippet': source['content'][:200] + '...' if len(source['content']) > 200 else source['content'],
                            'conflict_type': conflict['type'],
                            'superseded_by': conflict['high_priority_source']
                        })
                        cache_cleanup_needed = True
    
    # Generate resolution summary
    if conflicts:
        resolution_summary = f"Detected {len(conflicts)} conflicts between information sources. "
        resolution_summary += "Applying information hierarchy: newer, higher-priority sources override older information."
        
        if cache_cleanup_needed:
            resolution_summary += f" Marked {len(messages_to_remove)} outdated messages for cache removal."
        
        return {
            'conflicts_detected': True,
            'resolution_summary': resolution_summary,
            'resolved_conflicts': resolved_conflicts,
            'detailed_conflicts': conflicts,
            'cache_cleanup_needed': cache_cleanup_needed,
            'messages_to_remove': messages_to_remove,
            'conversation_id': conversation_id
        }
    
    return {'conflicts_detected': False, 'resolution_summary': '', 'resolved_conflicts': [], 'cache_cleanup_needed': False}

async def purge_conflicting_messages(conversation_manager, conversation_id: str, messages_to_remove: list):
    """
    Intelligently remove conflicting messages from conversation cache with full metadata tracking.
    
    Args:
        conversation_manager: Instance of conversation manager with Redis access
        conversation_id: ID of the conversation to clean
        messages_to_remove: List of message metadata to remove
    """
    try:
        import logging
        import re
        from datetime import datetime
        
        logger = logging.getLogger(__name__)
        logger.info(f"[CACHE_CLEANUP] Starting intelligent message purging for conversation {conversation_id}")
        
        # Get current conversation messages including superseded ones for analysis
        current_messages = await conversation_manager.get_conversation_history(
            conversation_id, limit=50, include_superseded=True
        )
        
        if not current_messages:
            logger.info(f"[CACHE_CLEANUP] No messages found in conversation {conversation_id}")
            return
        
        messages_marked = 0
        conflict_patterns_found = []
        
        # Enhanced content matching with conflict pattern detection
        for msg_to_remove in messages_to_remove:
            content_snippet = msg_to_remove['content_snippet'].replace('...', '').strip()
            conflict_type = msg_to_remove['conflict_type']
            superseded_by = msg_to_remove['superseded_by']
            
            for i, msg in enumerate(current_messages):
                msg_content = msg.get('content', '')
                
                # Multiple matching strategies for robust detection
                content_match = False
                
                # Strategy 1: Direct content snippet matching
                if len(content_snippet) > 30 and content_snippet.lower() in msg_content.lower():
                    content_match = True
                
                # Strategy 2: Conflict pattern matching for existence conflicts
                elif conflict_type == 'existence_conflict':
                    # Look for contradictory existence claims
                    negative_patterns = [
                        r'does not exist', r'doesn\'t exist', r'not exist', r'no.*available',
                        r'not.*released', r'myth', r'fictional', r'not.*real'
                    ]
                    if any(re.search(pattern, msg_content.lower()) for pattern in negative_patterns):
                        # Check if this contradicts what we now know exists
                        if any(keyword in superseded_by.lower() for keyword in ['search', 'results', 'current']):
                            content_match = True
                
                # Strategy 3: Temporal conflict detection
                elif 'date' in conflict_type or 'version' in conflict_type:
                    # Look for outdated version or date information
                    old_version_patterns = [
                        r'latest.*is.*gpt-4', r'current.*model.*is', r'as of.*2024',
                        r'no.*gpt-5', r'gpt-5.*not.*available'
                    ]
                    if any(re.search(pattern, msg_content.lower()) for pattern in old_version_patterns):
                        content_match = True
                
                if content_match:
                    # Use the enhanced conversation manager methods
                    conflict_metadata = {
                        'type': conflict_type,
                        'superseded_by': superseded_by,
                        'detected_pattern': content_snippet[:100],
                        'action': 'cache_cleanup',
                        'confidence': 'high' if len(content_snippet) > 30 else 'medium'
                    }
                    
                    # Mark message as superseded using conversation manager
                    success = await conversation_manager.mark_message_superseded(
                        conversation_id, i, conflict_metadata
                    )
                    
                    if success:
                        messages_marked += 1
                        conflict_patterns_found.append({
                            'message_index': i,
                            'content_preview': msg_content[:100] + '...',
                            'conflict_type': conflict_type
                        })
                        logger.info(f"[CACHE_CLEANUP] Marked message {i} as superseded: {msg_content[:100]}...")
        
        # Remove all superseded messages from cache
        messages_removed = await conversation_manager.remove_superseded_messages(conversation_id)
        
        # Add conflict resolution log for tracking and prevention
        if messages_marked > 0:
            conflict_log = {
                'type': 'batch_conflict_resolution',
                'action': 'selective_cache_cleanup',
                'affected_count': messages_removed,
                'patterns_detected': len(set(p['conflict_type'] for p in conflict_patterns_found)),
                'superseded_by': 'real_time_search_results'
            }
            await conversation_manager.add_conflict_resolution_log(conversation_id, conflict_log)
        
        # Summary logging
        if messages_removed > 0:
            logger.info(f"[CACHE_CLEANUP] Successfully removed {messages_removed} conflicting messages from conversation {conversation_id}")
            logger.info(f"[CACHE_CLEANUP] Conflict patterns detected: {[p['conflict_type'] for p in conflict_patterns_found]}")
        else:
            logger.info(f"[CACHE_CLEANUP] No conflicting messages found for removal in conversation {conversation_id}")
            
    except Exception as e:
        logger.error(f"[CACHE_CLEANUP] Error during intelligent message purging: {e}")
        # Don't raise - cache cleanup failure shouldn't break synthesis

def build_messages_for_synthesis(
    question: str,
    query_type: str,
    rag_context: str = "",
    tool_context: str = "",
    conversation_history: str = "",
    thinking: bool = False,
    rag_sources: list = None,
    conversation_messages: list = None,
    conversation_id: str = None
) -> tuple[list[dict[str, str]], str, str, str]:
    """
    Build messages array for chat-based LLM synthesis with information hierarchy.
    
    Implements fundamental principles:
    1. Fresh tool results > RAG knowledge > conversation history
    2. Newer information overrides older information
    3. Source credibility and freshness scoring
    
    Args:
        conversation_messages: List of conversation message dicts with freshness_score
    
    Returns:
        tuple: (messages, source_label, context_for_metadata, system_prompt)
    """
    
    # Get enhanced system prompt with information hierarchy principles
    llm_settings = get_llm_settings()
    base_prompt = llm_settings.get('main_llm', {}).get('system_prompt', 'You are Jarvis, an AI assistant.')
    
    if tool_context or rag_context or conversation_messages or conversation_history:
        # INFORMATION HIERARCHY SYSTEM PROMPT
        system_prompt = f"""{base_prompt}

📊 INFORMATION HIERARCHY MODE: Multiple information sources provided with priority ranking.

FUNDAMENTAL PRINCIPLES:
1. **Fresh information overrides old information** - Always prioritize newer, more current data
2. **Source hierarchy matters** - Follow the priority levels shown in each source label
3. **Information validity periods** - Older information may be outdated and should be used cautiously

SOURCE PRIORITY RULES (in order of authority):
• PRIORITY 1 (🔍): Real-time search results - MOST AUTHORITATIVE, always current
• PRIORITY 2 (📚): Knowledge base - Generally current, authoritative for internal topics  
• PRIORITY 3 (💬): Recent conversation - Contextual, use for understanding user intent
• PRIORITY 4 (⚠️): Older conversation - POTENTIALLY OUTDATED, verify against higher priority sources

AUTOMATIC CONFLICT RESOLUTION PROTOCOL:
When information sources contradict each other, apply these rules automatically:

1. **Priority Override**: Higher priority source ALWAYS wins
   - Priority 1 (🔍 search) > Priority 2 (📚 knowledge) > Priority 3 (💬 recent) > Priority 4 (⚠️ older)
   
2. **Freshness Within Priority**: Among same priority sources, fresher information wins
   - Check 'freshness_score' and 'age_hours' metadata
   - Real-time data > Recent data > Archived data
   
3. **Conflict Acknowledgment**: When overriding information, explicitly state:
   - "While [lower priority source] indicated X, current [higher priority source] shows Y"
   - "Previous information has been updated: [new information]"
   - "Based on the latest search results: [corrected information]"
   
4. **Source Transparency**: Always indicate which source provided the information:
   - "According to current search results..."
   - "Based on knowledge base documentation..."
   - "As discussed in our recent conversation..."
   
5. **Automatic Fact Correction**: Proactively identify and correct outdated facts:
   - Compare factual claims across priority levels
   - Flag temporal inconsistencies (dates, versions, availability)
   - Highlight when "does not exist" conflicts with "now available"

REQUIREMENTS:
- Base ALL factual statements on the highest priority source available
- Automatically detect and resolve conflicts without asking for clarification
- DO NOT generate tool calls like <tool>...</tool>
- Be explicit about which source informed each claim

Answer using this automatic conflict resolution protocol while maintaining transparency about information sources and any corrections made."""
    else:
        system_prompt = build_enhanced_system_prompt()
    
    # Check if system prompt contains specific format instructions
    has_format_instructions = any(phrase in system_prompt.lower() for phrase in [
        'format at the end', 'in this format', 'opinion:', 'provide your point of view'
    ])
    
    # Build user message content - SIMPLIFIED APPROACH
    user_content_parts = []
    
    # INFORMATION HIERARCHY IMPLEMENTATION
    # Fundamental principle: Fresh info overrides old info, with explicit source ranking
    
    # Calculate information freshness and priority scores
    info_sources = []
    
    # 1. Tool results = Priority 1 (highest freshness, most authoritative)
    if tool_context:
        from datetime import datetime
        current_time = datetime.now()
        info_sources.append({
            'priority': 1,
            'freshness_score': 1.0,  # Always fresh
            'source_type': 'tool_results',
            'content': tool_context,
            'label': f'🔍 SEARCH RESULTS (PRIORITY 1 - MOST AUTHORITATIVE - {current_time.strftime("%H:%M UTC")})',
            'description': f'Real-time search results fetched at {current_time.strftime("%Y-%m-%d %H:%M UTC")}',
            'metadata': {
                'fetch_time': current_time.isoformat(),
                'age_seconds': 0,
                'credibility': 'high',
                'update_frequency': 'real_time'
            }
        })
    
    # 2. RAG context = Priority 2 (knowledge base)
    if rag_context:
        # Calculate RAG freshness based on source metadata if available
        rag_freshness = 0.8  # Default
        rag_age_info = "unknown age"
        rag_sources_info = []
        
        if rag_sources:
            # Analyze RAG sources for freshness indicators
            for source in rag_sources:
                source_score = source.get('score', 0.5)
                collection = source.get('collection', 'unknown')
                file_path = source.get('file', '')
                
                # Determine freshness based on collection and file indicators
                if 'temp' in collection.lower() or '[TEMP]' in file_path:
                    source_freshness = 0.9  # Recent uploads are fresher
                elif 'recent' in collection.lower():
                    source_freshness = 0.8
                elif 'archive' in collection.lower():
                    source_freshness = 0.4
                else:
                    source_freshness = 0.6
                    
                rag_sources_info.append({
                    'collection': collection,
                    'file': file_path,
                    'freshness': source_freshness,
                    'score': source_score
                })
            
            # Use weighted average of source freshness
            if rag_sources_info:
                rag_freshness = sum(s['freshness'] * s['score'] for s in rag_sources_info) / sum(s['score'] for s in rag_sources_info)
                rag_age_info = f"from {len(rag_sources_info)} documents"
        
        info_sources.append({
            'priority': 2,
            'freshness_score': rag_freshness,
            'source_type': 'knowledge_base',
            'content': rag_context,
            'label': f'📚 KNOWLEDGE BASE (PRIORITY 2 - INTERNAL DOCS - {rag_age_info})',
            'description': f'Information from internal knowledge base (freshness: {rag_freshness:.1f})',
            'metadata': {
                'source_count': len(rag_sources) if rag_sources else 1,
                'avg_relevance': sum(s.get('score', 0.5) for s in rag_sources) / len(rag_sources) if rag_sources else 0.5,
                'collections': list(set(s.get('collection', 'unknown') for s in rag_sources)) if rag_sources else ['unknown'],
                'credibility': 'medium-high',
                'update_frequency': 'periodic'
            }
        })
    
    # 3. Conversation history = Priority 3 (lowest, age-dependent)
    if conversation_messages:
        # Filter and score conversation messages by freshness
        fresh_messages = []
        stale_messages = []
        
        for msg in conversation_messages:
            freshness = msg.get('freshness_score', 0.5)
            age_hours = msg.get('age_hours', 0)
            
            if freshness > 0.5 and age_hours < 6:  # Fresh within 6 hours
                fresh_messages.append(msg)
            elif freshness > 0.2:  # Somewhat fresh
                stale_messages.append(msg)
            # Messages with freshness <= 0.2 are filtered out as too stale
        
        if fresh_messages:
            # Format fresh conversation context with enhanced metadata
            fresh_content = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in fresh_messages])
            max_freshness = max(msg.get('freshness_score', 0) for msg in fresh_messages)
            max_age = max(msg.get('age_hours', 0) for msg in fresh_messages)
            avg_freshness = sum(msg.get('freshness_score', 0) for msg in fresh_messages) / len(fresh_messages)
            
            info_sources.append({
                'priority': 3,
                'freshness_score': avg_freshness,  # Use average freshness for more accurate scoring
                'source_type': 'conversation_fresh',
                'content': fresh_content,
                'label': f'💬 RECENT CONVERSATION (PRIORITY 3 - {max_age:.1f}h ago)',
                'description': f'Fresh conversation context (avg freshness: {avg_freshness:.2f})',
                'metadata': {
                    'message_count': len(fresh_messages),
                    'max_age_hours': max_age,
                    'avg_freshness': avg_freshness,
                    'freshness_range': [min(msg.get('freshness_score', 0) for msg in fresh_messages), max_freshness],
                    'credibility': 'medium',
                    'update_frequency': 'conversational'
                }
            })
        
        if stale_messages:
            # Format stale conversation context with enhanced warning metadata
            stale_content = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in stale_messages])
            avg_stale_freshness = sum(msg.get('freshness_score', 0) for msg in stale_messages) / len(stale_messages)
            max_stale_age = max(msg.get('age_hours', 0) for msg in stale_messages)
            
            info_sources.append({
                'priority': 4,
                'freshness_score': avg_stale_freshness,
                'source_type': 'conversation_stale',
                'content': stale_content,
                'label': f'⚠️  OLDER CONVERSATION (PRIORITY 4 - {max_stale_age:.1f}h old)',
                'description': f'Potentially outdated context (freshness: {avg_stale_freshness:.2f})',
                'metadata': {
                    'message_count': len(stale_messages),
                    'max_age_hours': max_stale_age,
                    'avg_freshness': avg_stale_freshness,
                    'staleness_warning': True,
                    'credibility': 'low',
                    'update_frequency': 'historical'
                }
            })
    elif conversation_history:  # Fallback for legacy string-based history
        info_sources.append({
            'priority': 3,
            'freshness_score': 0.3,  # Unknown age, assume somewhat stale
            'source_type': 'conversation_legacy',
            'content': conversation_history,
            'label': '💬 PREVIOUS CONVERSATION (PRIORITY 3 - LEGACY FORMAT)',
            'description': 'Previous conversation context (freshness unknown)',
            'metadata': {
                'format': 'legacy_string',
                'freshness_unknown': True,
                'credibility': 'medium',
                'update_frequency': 'unknown'
            }
        })
    
    # Sort by priority (ascending) and freshness (descending)
    info_sources.sort(key=lambda x: (x['priority'], -x['freshness_score']))
    
    # AUTOMATIC CONFLICT DETECTION AND RESOLUTION
    conflict_analysis = detect_and_resolve_conflicts(info_sources, question, conversation_id)
    if conflict_analysis['conflicts_detected']:
        # Add conflict resolution summary to user content
        conflict_summary = f"""
🔍 AUTOMATIC CONFLICT RESOLUTION APPLIED:
{conflict_analysis['resolution_summary']}

RESOLVED CONFLICTS:
{chr(10).join([f'• {conflict}' for conflict in conflict_analysis['resolved_conflicts']])}
"""
        user_content_parts.append(conflict_summary)
        
        # PROACTIVE CACHE CLEANUP: Remove conflicting messages from cache
        if conflict_analysis['cache_cleanup_needed'] and conversation_id:
            try:
                # Import conversation manager for cache cleanup
                from app.core.simple_conversation_manager import conversation_manager
                import asyncio
                
                # Schedule cache cleanup asynchronously
                cleanup_task = asyncio.create_task(
                    purge_conflicting_messages(
                        conversation_manager, 
                        conversation_id, 
                        conflict_analysis['messages_to_remove']
                    )
                )
                
                # Log cache cleanup action
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"[CACHE_CLEANUP] Scheduled removal of {len(conflict_analysis['messages_to_remove'])} conflicting messages from conversation {conversation_id}")
                
            except Exception as e:
                # Don't fail synthesis if cache cleanup fails
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"[CACHE_CLEANUP] Failed to schedule cache cleanup: {e}")
    
    # Build user content with clear hierarchy
    if info_sources:
        # Information hierarchy mode - use ranked sources
        for source in info_sources:
            user_content_parts.append(f"""{source['label']}:
{source['content']}""")
    else:
        # Fallback modes when no enhanced sources available
        if rag_context:
            # RAG-only fallback mode
            user_content_parts.append(f"""📚 CURRENT FACTUAL DATA (Knowledge Base - PRIMARY SOURCE):
{rag_context}

SYNTHESIS INSTRUCTIONS: Use the above information as your primary source. Answer based on this factual data.""")
            
            # Add conversation history for context (existing behavior)
            if conversation_history:
                user_content_parts.append(f"""Previous conversation context (IMPORTANT: If this conflicts with the factual data above, prioritize the factual data):
{conversation_history}""")
        
        else:
            # Pure LLM fallback mode
            # Add conversation history normally since no factual data to prioritize
            if conversation_history:
                user_content_parts.append(f"Previous conversation:\n{conversation_history}")
            
            # Pure LLM - just note the context without instructions
            if query_type == "TOOLS":
                user_content_parts.append("The user is asking for information that would benefit from real-time tools, but no tools were executed.")
            elif query_type == "RAG":
                user_content_parts.append("No specific documents were found in our knowledge base for this query.")
    
    # Add the actual question
    user_content_parts.append(f"Question: {question}")
    
    # If system prompt has format instructions, add a subtle reminder at the end
    if has_format_instructions:
        # Extract potential format from system prompt
        format_hints = []
        if 'opinion:' in system_prompt.lower():
            format_hints.append("Remember to include your opinion in the specified format.")
        if 'end with' in system_prompt.lower() or 'conclude with' in system_prompt.lower():
            format_hints.append("Remember to follow the specified ending format.")
        if 'format at the end' in system_prompt.lower():
            format_hints.append("Remember to use the specified format at the end of your response.")
        
        if format_hints:
            user_content_parts.append("\n".join(format_hints))
    
    # Join all user content
    user_content = "\n\n".join(user_content_parts)
    
    
    # Apply thinking wrapper if needed (but be more subtle if format instructions exist)
    if thinking:
        if has_format_instructions:
            # More subtle thinking request that doesn't override format
            user_content = "Show your thinking process.\n\n" + user_content
        else:
            user_content = "Please show your reasoning step by step before giving the final answer.\n\n" + user_content
    
    # Build messages array
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # Determine source label for metadata
    sources = []
    if rag_context or rag_sources:
        has_temp_docs = False
        if rag_sources:
            for source in rag_sources:
                if (source.get("file", "").startswith("[TEMP]") or 
                    source.get("is_temporary", False) or
                    source.get("collection", "").lower() == "temp_documents" or
                    "temp" in source.get("collection", "").lower()):
                    has_temp_docs = True
                    break
        
        if has_temp_docs:
            sources.append("RAG_TEMP")
        elif rag_context or rag_sources:
            sources.append("RAG")
    if tool_context or query_type in ["TOOLS", "HYBRID_LLM_TOOLS"]:
        sources.append("TOOLS")
    sources.append("LLM")
    source_label = "+".join(sources)
    
    # Build context for metadata
    full_context = ""
    if rag_context and tool_context:
        full_context = f"Internal Knowledge:\n{rag_context}\n\nTool Results:\n{tool_context}"
    elif rag_context:
        full_context = rag_context
    elif tool_context:
        full_context = tool_context
    
    print(f"[DEBUG] Messages synthesis - Source: {source_label}, Query type: {query_type}")
    print(f"[DEBUG] Messages synthesis - Has RAG: {bool(rag_context)}, Has tools: {bool(tool_context)}")
    print(f"[DEBUG] Messages synthesis - Has format instructions: {has_format_instructions}")
    print(f"[DEBUG] Messages synthesis - System prompt preview: {system_prompt[:100]}...")
    print(f"[DEBUG] Messages synthesis - Number of messages: {len(messages)}")
    
    return messages, source_label, full_context, system_prompt

def unified_llm_synthesis(
    question: str,
    query_type: str,
    rag_context: str = "",
    tool_context: str = "",
    conversation_history: str = "",
    thinking: bool = False,
    rag_sources: list = None
) -> tuple[str, str, str]:
    """
    Unified LLM synthesis function that generates responses regardless of query classification.
    All queries flow through this function for consistent, high-quality responses.
    
    This function now internally uses the messages format but returns a single prompt
    for backward compatibility.
    
    Args:
        question: Original user question
        query_type: Classification result (TOOLS, RAG, LLM, etc.)
        rag_context: Retrieved document context (if any)
        tool_context: Tool execution results (if any)
        conversation_history: Previous conversation context
        thinking: Whether to enable extended thinking
        rag_sources: List of RAG source documents (if any)
    
    Returns:
        tuple: (prompt, source_label, context_for_metadata)
    """
    # Use the new message builder
    messages, source_label, full_context, system_prompt = build_messages_for_synthesis(
        question=question,
        query_type=query_type,
        rag_context=rag_context,
        tool_context=tool_context,
        conversation_history=conversation_history,
        thinking=thinking,
        rag_sources=rag_sources,
        conversation_messages=None,  # Legacy call doesn't have enhanced messages
        conversation_id=None  # Legacy call doesn't have conversation_id context
    )
    
    # For backward compatibility, concatenate messages into a single prompt
    # This will be passed to the generate method which now uses chat internally
    prompt_parts = []
    for msg in messages:
        if msg["role"] == "system":
            prompt_parts.append(msg["content"])
        elif msg["role"] == "user":
            prompt_parts.append(msg["content"])
    
    final_prompt = "\n\n".join(prompt_parts)
    
    return final_prompt, source_label, full_context

def calculate_relevance_score(query: str, context: str, corpus_stats=None) -> float:
    """Enhanced relevance score using BM25 with fallback to existing TF-IDF logic"""
    if not context or not query:
        return 0.0
    
    # Initialize BM25 processor
    bm25_processor = BM25Processor()
    
    # Use enhanced BM25 scoring
    enhanced_score = bm25_processor.enhance_existing_relevance_score(
        query=query,
        content=context,
        corpus_stats=corpus_stats,
        existing_score=_calculate_legacy_relevance_score(query, context)
    )
    
    return enhanced_score

def _calculate_legacy_relevance_score(query: str, context: str) -> float:
    """Legacy TF-IDF-like calculation as fallback"""
    if not context or not query:
        return 0.0
        
    import math
    from collections import Counter
    
    # Convert to lowercase for comparison
    query_lower = query.lower()
    context_lower = context.lower()
    
    # More comprehensive stop words list
    stop_words = {
        'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
        'a', 'an', 'what', 'how', 'when', 'where', 'why', 'like', 'it', 'this', 'that', 'these', 
        'those', 'can', 'could', 'would', 'should', 'may', 'might', 'must', 'shall', 'will',
        'do', 'does', 'did', 'have', 'has', 'had', 'be', 'been', 'being', 'was', 'were',
        'find', 'get', 'show', 'tell', 'about', 'info', 'information', 'news', 'give', 'provide', 
        'need', 'want', 'please', 'help'
    }
    
    # Extract meaningful words
    query_words = [w for w in query_lower.split() if w not in stop_words and len(w) > 2]
    context_words_list = [w for w in context_lower.split() if len(w) > 2]
    
    if not query_words:
        query_words = query_lower.split()
    
    # Calculate term frequency (TF) for context
    context_word_freq = Counter(context_words_list)
    total_words = len(context_words_list)
    
    # Calculate TF-IDF-like score
    tfidf_score = 0.0
    matched_terms = 0
    
    for word in query_words:
        # Direct match
        if word in context_word_freq:
            tf = context_word_freq[word] / total_words if total_words > 0 else 0
            # Simulate IDF with a bonus for rare terms
            idf_bonus = 1.0 if context_word_freq[word] <= 2 else 0.5
            tfidf_score += tf * idf_bonus
            matched_terms += 1
        # Partial match (stemming-like)
        else:
            partial_matches = [w for w in context_word_freq if word in w or w in word]
            if partial_matches:
                best_match = max(partial_matches, key=lambda w: context_word_freq[w])
                tf = context_word_freq[best_match] / total_words if total_words > 0 else 0
                tfidf_score += tf * 0.7  # Lower weight for partial matches
                matched_terms += 0.7
    
    # Normalize by query length
    base_score = matched_terms / len(query_words) if query_words else 0.0
    
    # BM25-like saturation (diminishing returns for high term frequency)
    bm25_score = tfidf_score / (tfidf_score + 0.5)
    
    # Exact phrase match bonus
    phrase_bonus = 0.0
    if len(query_words) > 1 and query_lower in context_lower:
        phrase_bonus = 0.3
    
    # Proximity scoring - reward documents where query terms appear close together
    proximity_bonus = 0.0
    if len(query_words) > 1:
        positions = {}
        for word in query_words:
            positions[word] = [i for i, w in enumerate(context_words_list) if word in w or w in word]
        
        if all(positions.get(w) for w in query_words):
            # Calculate average distance between consecutive query terms
            total_distance = 0
            distance_count = 0
            
            query_word_list = list(query_words)
            for i in range(len(query_word_list) - 1):
                if positions.get(query_word_list[i]) and positions.get(query_word_list[i + 1]):
                    for p1 in positions[query_word_list[i]]:
                        for p2 in positions[query_word_list[i + 1]]:
                            if p2 > p1:  # Ensure order
                                total_distance += (p2 - p1)
                                distance_count += 1
                                break
            
            if distance_count > 0:
                avg_distance = total_distance / distance_count
                # Closer terms get higher bonus
                if avg_distance < 5:
                    proximity_bonus = 0.2
                elif avg_distance < 10:
                    proximity_bonus = 0.1
    
    # Coverage bonus - what percentage of query terms appear in the document
    coverage = matched_terms / len(query_words) if query_words else 0.0
    coverage_bonus = 0.1 if coverage >= 0.8 else 0.0
    
    # Combine all scores
    final_score = (
        base_score * 0.3 +      # Basic term matching
        bm25_score * 0.3 +      # TF-IDF with saturation
        phrase_bonus +          # Exact phrase match
        proximity_bonus +       # Term proximity
        coverage_bonus          # Query coverage
    )
    
    # Ensure score is between 0 and 1
    return min(1.0, max(0.0, final_score))

def detect_large_output_potential(question: str) -> dict:
    """Detect if question will likely produce large output requiring chunked processing"""
    # Skip large generation detection for questions that already contain tool results
    from app.langchain.smart_query_classifier import classify_query
    query_type, confidence = classify_query(question)
    if query_type == "tools":
        return {
            "likely_large": False,
            "estimated_items": 1,
            "confidence": 0.0,
            "score": 0,
            "max_number": 0,
            "matched_indicators": [],
            "pattern_matches": []
        }
    
    from app.core.large_generation_utils import get_config_accessor
    config = get_config_accessor()
    
    large_output_indicators = config.large_output_indicators
    
    # Count indicator patterns
    score = 0
    question_lower = question.lower()
    matched_indicators = []
    
    for indicator in large_output_indicators:
        if indicator in question_lower:
            score += 1
            matched_indicators.append(indicator)
    
    # Extract numbers that might indicate quantity (exclude years, URLs, and contextual numbers)
    import re
    
    # First remove URLs to avoid extracting numbers from them
    question_no_urls = re.sub(r'https?://[^\s]+', '', question)
    all_numbers = re.findall(r'\b(\d+)\b', question_no_urls)
    
    # Filter out years and other contextual numbers that don't indicate quantity
    quantity_numbers = []
    for num_str in all_numbers:
        num = int(num_str)
        # Exclude common year ranges (1900-2100) and other contextual numbers
        if num >= 1900 and num <= 2100:
            continue  # Skip years
        # Exclude small contextual numbers that might be version numbers, IDs, etc.
        if num <= 5 and not any(quantity_word in question_lower for quantity_word in [
            "generate", "create", "list", "write", "questions", "items", "examples", "steps"
        ]):
            continue  # Skip small numbers without generation context
        quantity_numbers.append(num)
    
    max_number = max(quantity_numbers, default=0)
    
    print(f"[DEBUG] Number filtering - All numbers: {all_numbers}, Quantity numbers: {quantity_numbers}, Max: {max_number}")
    
    # Additional patterns that suggest large output
    large_patterns = config.large_patterns
    
    pattern_matches = []
    for pattern in large_patterns:
        matches = re.findall(pattern, question_lower)
        if matches:
            pattern_matches.extend(matches)
            score += config.pattern_score_weight  # Patterns get higher weight
    
    # Calculate confidence and estimated items
    base_confidence = min(1.0, score / config.max_score_for_confidence)
    number_confidence = min(1.0, max_number / config.max_number_for_confidence) if max_number > 0 else 0
    final_confidence = max(base_confidence, number_confidence)
    
    # Estimate number of items to generate (only use filtered quantity numbers)
    if max_number > 10:
        estimated_items = max_number
    elif score >= 3:
        estimated_items = score * config.score_multiplier  # Heuristic: more indicators = more items
    elif any(keyword in question_lower for keyword in config.comprehensive_keywords):
        estimated_items = config.default_comprehensive_items  # Default for comprehensive requests
    else:
        estimated_items = config.min_estimated_items
    
    # More refined logic for determining if it's a large generation request
    is_likely_large = False
    
    # Check for factual question patterns that should NOT trigger large generation
    factual_patterns = [
        r'\bwho\s+(is|are|was|were)\b',  # "who is/are/was/were"
        r'\bwhat\s+(is|are|was|were)\b',  # "what is/are/was/were"
        r'\bwhen\s+(is|are|was|were|did|will)\b',  # "when is/are/was/were/did/will"
        r'\bwhere\s+(is|are|was|were)\b',  # "where is/are/was/were"
        r'\bhow\s+(much|many|long|old)\b',  # "how much/many/long/old"
        r'\bwinner[s]?\s+(of|for)\b',  # "winner of/for"
        r'\bresult[s]?\s+(of|for)\b',  # "result of/for"
        r'\bchampion[s]?\s+(of|for)\b',  # "champion of/for"
    ]
    
    is_factual_query = any(re.search(pattern, question_lower) for pattern in factual_patterns)
    
    # If it's a factual query, don't trigger large generation even with numbers
    if is_factual_query:
        print(f"[DEBUG] Detected factual query pattern, skipping large generation: {question}")
        is_likely_large = False
    # Strong indicators: explicit large numbers (but only if not a factual query)
    elif max_number >= config.strong_number_threshold:
        is_likely_large = True
    # Medium indicators: moderate numbers + generation keywords
    elif max_number >= config.medium_number_threshold and score >= config.min_score_for_medium_numbers:
        is_likely_large = True
    # Pattern-based indicators: multiple strong keywords suggesting comprehensive content
    elif score >= config.min_score_for_keywords and any(keyword in question_lower for keyword in config.comprehensive_keywords):
        is_likely_large = True
    # Don't trigger for small numbers even with keywords
    elif max_number > 0 and max_number < config.small_number_threshold:
        is_likely_large = False
    
    # Adjust estimated items for small number requests and factual queries
    if max_number > 0 and max_number < config.small_number_threshold:
        estimated_items = max_number
    elif is_factual_query:
        estimated_items = 1  # Factual queries typically return single answers
    
    result = {
        "likely_large": is_likely_large,
        "estimated_items": estimated_items,
        "confidence": final_confidence,
        "score": score,
        "max_number": max_number,
        "matched_indicators": matched_indicators,
        "pattern_matches": pattern_matches
    }
    
    print(f"[DEBUG] Large output detection for '{question}': {result}")
    return result

async def classify_query_type_efficient(question: str, llm_cfg) -> str:
    """
    EFFICIENT query classification using direct OllamaLLM instead of HTTP.
    Only does RAG if the question requires internal corporate knowledge.
    
    Categories:
    - LARGE_GENERATION: Requires chunked processing
    - RAG: Needs internal company documents/knowledge base  
    - TOOLS: Needs function calls (time, calculations, web search, etc.)
    - LLM: Can be answered with general knowledge
    """
    print(f"[DEBUG] classify_query_type_efficient: question = {question}")
    
    # First check for large generation (keep existing logic)
    large_output_analysis = detect_large_output_potential(question)
    from app.core.large_generation_utils import get_config_accessor
    config = get_config_accessor()
    
    if large_output_analysis["likely_large"] and large_output_analysis["estimated_items"] >= config.min_items_for_chunking:
        print(f"[DEBUG] classify_query_type_efficient: Detected large generation requirement")
        return "LARGE_GENERATION"
    
    # Try smart pattern classifier first (fast, no LLM call)
    try:
        from app.langchain.smart_query_classifier import classify_without_context
        query_type, confidence = classify_without_context(question)
        print(f"[DEBUG] Smart classifier result: {query_type} (confidence: {confidence})")
        
        if confidence >= 0.7:  # High confidence, trust the pattern classifier
            return query_type.upper()  # Normalize to uppercase for consistency
    except Exception as e:
        print(f"[DEBUG] Smart classifier failed: {e}")
    
    # Use efficient LLM classification (direct OllamaLLM, low tokens)
    return await _llm_classify_efficient(question, llm_cfg)

async def _llm_classify_efficient(question: str, llm_cfg) -> str:
    """Use direct OllamaLLM for fast classification with minimal tokens"""
    try:
        from app.llm.ollama import OllamaLLM
        from app.llm.base import LLMConfig
        from app.core.mcp_tools_cache import get_enabled_mcp_tools
        import os
        
        # Get available tools for context
        available_tools = get_enabled_mcp_tools()
        tool_names = list(available_tools.keys()) if available_tools else []
        
        # Use configurable tokens for classification (default 10 for one word response)
        classifier_config = llm_cfg.get("query_classifier", {})
        max_tokens = int(classifier_config.get("classifier_max_tokens", 10))
        
        # Create LLM config for classification using query classifier
        query_classifier_config = get_query_classifier_full_config(llm_cfg)
        llm_config = LLMConfig(
            model_name=query_classifier_config["model"],
            temperature=0.1,  # Low temperature for consistent classification
            top_p=0.9,
            max_tokens=max_tokens
        )
        
        # Create LLM instance
        ollama_url = query_classifier_config.get("model_server", "http://ollama:11434")
        # If localhost, change to host.docker.internal for Docker containers
        if "localhost" in ollama_url:
            ollama_url = ollama_url.replace("localhost", "host.docker.internal")
        llm = OllamaLLM(llm_config, base_url=ollama_url)
        
        # Build messages array for chat endpoint
        # Check if query_classifier has a custom system prompt
        query_classifier_system_prompt = query_classifier_config.get("system_prompt", "")
        
        # Build classification system message
        classification_instructions = f"""You are a query classifier. Analyze questions and classify them into exactly ONE category:

RAG: Question asks about internal company information, policies, processes, or specific corporate knowledge
TOOLS: Question needs real-time data, calculations, or external information. Available tools: {', '.join(tool_names[:5])}
LLM: Question can be answered with general knowledge

Answer with exactly one word: RAG, TOOLS, or LLM"""

        # Combine custom system prompt with classification instructions if present
        if query_classifier_system_prompt:
            system_content = f"{query_classifier_system_prompt}\n\n{classification_instructions}"
        else:
            system_content = classification_instructions
            
        # Build messages for chat endpoint
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f'Question: "{question}"'}
        ]
        
        print(f"[DEBUG] Using chat endpoint for classification with {len(messages)} messages")
        print(f"[DEBUG] System message includes custom prompt: {bool(query_classifier_system_prompt)}")
        
        # Get classification using chat endpoint
        response_text = ""
        async for chunk in llm.chat_stream(messages):
            response_text += chunk.text
        
        # Parse classification
        classification = response_text.strip().upper()
        if classification in ["RAG", "TOOLS", "LLM"]:
            print(f"[DEBUG] LLM classification: {classification}")
            return classification
        else:
            print(f"[DEBUG] Invalid LLM response '{classification}', defaulting to LLM")
            return "LLM"  # Safe default for general questions
            
    except Exception as e:
        print(f"[DEBUG] LLM classification failed: {e}, defaulting to LLM")
        return "LLM"  # Safe default

def classify_query_type(question: str, llm_cfg) -> str:
    """
    Classify the query into one of four categories:
    - LARGE_GENERATION: Requires chunked processing for large outputs
    - RAG: Needs internal company documents/knowledge base
    - TOOLS: Needs function calls (current time, calculations, API calls, etc.)
    - LLM: Can be answered with general knowledge
    """
    print(f"[DEBUG] classify_query_type: question = {question}")
    
    # First check for large generation requirements
    large_output_analysis = detect_large_output_potential(question)
    from app.core.large_generation_utils import get_config_accessor
    config = get_config_accessor()
    
    if large_output_analysis["likely_large"] and large_output_analysis["estimated_items"] >= config.min_items_for_chunking:
        print(f"[DEBUG] classify_query_type: Detected large generation requirement")
        return "LARGE_GENERATION"
    
    # Use smart pattern-based classifier first for high-confidence cases
    try:
        from app.langchain.smart_query_classifier import classify_without_context, classifier
        query_type, confidence = classify_without_context(question)
        print(f"[DEBUG] Smart classifier result: {query_type} (confidence: {confidence})")
        
        # If high confidence, use smart classifier result
        # Use lower threshold (0.7) to catch more patterns without LLM
        if confidence >= 0.7:
            # Map smart classifier types to expected return values
            type_mapping = {
                "tools": "TOOLS",
                "llm": "LLM",
                "rag": "RAG",
                "multi_agent": "RAG"  # Multi-agent queries often need RAG
            }
            mapped_type = type_mapping.get(query_type, "LLM")
            print(f"[DEBUG] Using smart classifier result: {mapped_type}")
            return mapped_type
    except Exception as e:
        print(f"[DEBUG] Smart classifier failed: {e}, falling back to LLM classification")
    
    # Get available tools for better classification
    available_tools = get_enabled_mcp_tools()
    tool_descriptions = []
    for tool_name, tool_info in available_tools.items():
        tool_descriptions.append(f"- {tool_name}: {tool_info['description']}")
    
    tools_list = "\n".join(tool_descriptions) if tool_descriptions else "No tools available"
    
    router_prompt = f"""NO_THINK
User question: "{question}"

Available tools:
{tools_list}

IMPORTANT: Only classify as TOOLS if the available tools can actually help answer the question.

Classify this question into exactly ONE category:
- RAG: Question about internal company data, documents, business-specific information, OR questions about specific entities (companies/products/people) that might be covered in internal documents
- TOOLS: Question that can be answered using the available tools listed above
- LLM: Question that can be answered with pure general knowledge without referencing specific entities or time-sensitive information

Classification guidelines:
1. Check if available tools can actually help with the question
2. Look for company-specific indicators (our, internal, company name if it matches your organization)
3. Questions about specific entities (companies, products, people) with topics like progress, development, strategy, performance, or updates should check RAG first for any relevant documents
4. Only use LLM for pure general knowledge questions that don't reference specific entities or time-sensitive information

Examples:
- "What's our Q4 revenue?" → RAG (internal company data)
- "What time is it now?" → TOOLS (if time/date tools available)
- "Explain machine learning" → LLM (general knowledge)
- "How's Apple's AI progress?" → RAG (check for market research/competitor analysis docs first)
- "How's DBS progress in AI?" → RAG (check for industry reports/analysis first)
- "What is photosynthesis?" → LLM (pure general knowledge)
- "Send email to John" → TOOLS (if email tools available)
- "Find the meeting notes from last week" → RAG (internal documents)

Answer with exactly one word: RAG, TOOLS, or LLM"""

    print(f"[DEBUG] classify_query_type: router_prompt = {router_prompt}")
    prompt = build_prompt(router_prompt, is_internal=True)
    
    llm_api_url = "http://localhost:8000/api/v1/generate_stream"
    payload = {
        "prompt": prompt,
        "temperature": 0.1,  # Lower temperature for more consistent classification
        "top_p": 0.5,
        "max_tokens": 50
    }
    
    text = ""
    # Use centralized timeout configuration
    from app.core.timeout_settings_cache import get_timeout_value
    http_timeout = get_timeout_value("api_network", "http_request_timeout", 30)
    
    with httpx.Client(timeout=http_timeout) as client:
        with client.stream("POST", llm_api_url, json=payload) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.startswith("data: "):
                    token = line.replace("data: ", "")
                    text += token
    
    print(f"[DEBUG] classify_query_type: LLM raw output = {text}")
    print(f"[DEBUG] classify_query_type: Available tools = {list(available_tools.keys()) if available_tools else 'None'}")
    
    # Extract classification with fallback logic
    text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip().upper()
    
    if "RAG" in text_clean:
        classification = "RAG"
    elif "TOOLS" in text_clean or "TOOL" in text_clean:
        classification = "TOOLS"
    elif "LLM" in text_clean:
        classification = "LLM"
    else:
        print(f"[DEBUG] classify_query_type: No clear classification found, using fallback logic")
        # Fallback logic based on keywords and tool capability
        question_lower = question.lower()
        
        # Check if available tools can actually help with this question
        tool_can_help = False
        if available_tools:
            # Dynamic tool matching - check if any available tools could help with the query
            for tool_name, tool_info in available_tools.items():
                tool_desc = tool_info.get("description", "").lower()
                # Check if the tool description matches the query intent
                if any(keyword in question_lower for keyword in ["time", "date", "now"]) and any(word in tool_desc for word in ["time", "date", "datetime"]):
                    tool_can_help = True
                    break
                elif any(keyword in question_lower for keyword in ["email", "gmail", "mail"]) and any(word in tool_desc for word in ["email", "gmail", "mail"]):
                    tool_can_help = True
                    break
                elif any(keyword in question_lower for keyword in ["search", "find"]) and any(word in tool_desc for word in ["search", "find", "list"]):
                    tool_can_help = True
                    break
        
        if tool_can_help:
            classification = "TOOLS"
            print(f"[DEBUG] classify_query_type: Fallback classified as TOOLS (tool can help)")
        elif any(keyword in question_lower for keyword in ["our", "company", "internal", "document", "meeting", "policy"]):
            classification = "RAG"
            print(f"[DEBUG] classify_query_type: Fallback classified as RAG (company-specific)")
        elif any(keyword in question_lower for keyword in ["progress", "development", "strategy", "performance", "update", "latest", "recent"]) and \
             any(char.isupper() for char in question if char.isalpha()):  # Check for proper nouns (capitalized words)
            classification = "RAG"
            print(f"[DEBUG] classify_query_type: Fallback classified as RAG (entity-specific with progress/development keywords)")
        else:
            classification = "LLM"
            print(f"[DEBUG] classify_query_type: Fallback classified as LLM (general knowledge)")
    
    print(f"[DEBUG] classify_query_type: Final classification = {classification}")
    return classification

async def execute_intelligent_planning(question: str, thinking: bool = False, trace=None) -> tuple:
    """
    Enhanced intelligent planning system using IntelligentToolPlanner
    Returns: (tool_results, search_context)
    """
    print(f"[DEBUG] execute_intelligent_planning: question = {question}, thinking = {thinking}")
    
    try:
        from app.langchain.intelligent_tool_planner import get_tool_planner
        from app.langchain.intelligent_tool_executor import IntelligentToolExecutor
        from app.core.langfuse_integration import get_tracer
        
        # Create intelligent planning span
        planning_span = None
        if trace:
            tracer = get_tracer()
            if tracer.is_enabled():
                planning_span = tracer.create_span(
                    trace,
                    name="intelligent-tool-planning",
                    metadata={
                        "operation": "intelligent_planning",
                        "question": question,
                        "thinking": thinking
                    }
                )
        
        # Get tool planner and executor
        planner = get_tool_planner()
        executor = IntelligentToolExecutor(trace=trace)
        
        print(f"[DEBUG] execute_intelligent_planning: Creating execution plan...")
        
        # Create execution plan
        execution_plan = await planner.plan_tool_execution(
            task=question,
            context={"user_question": question, "thinking_mode": thinking},
            mode="standard"
        )
        
        print(f"[DEBUG] execute_intelligent_planning: Plan created with {len(execution_plan.tools)} tools: {[t.tool_name for t in execution_plan.tools]}")
        
        if not execution_plan.tools:
            print(f"[DEBUG] execute_intelligent_planning: No tools planned, returning empty results")
            if planning_span:
                tracer = get_tracer()
                tracer.end_span_with_result(planning_span, {"tools": [], "reasoning": execution_plan.reasoning}, True)
            return [], ""
        
        # Execute planned tools
        tool_results = []
        search_context = ""
        
        async for update in executor.execute_task_intelligently(
            task=question,
            context={"user_question": question, "thinking_mode": thinking},
            mode="standard"
        ):
            if update.get("type") == "tool_result":
                result = update.get("result")
                if result and result.success:
                    tool_results.append({
                        "tool": result.tool_name,
                        "success": True,
                        "result": result.result,
                        "execution_time": result.execution_time or 0.5
                    })
                    
                    # Build search context
                    search_context += f"\n{result.tool_name}: {json.dumps(result.result, indent=2) if isinstance(result.result, dict) else result.result}\n"
        
        print(f"[DEBUG] execute_intelligent_planning: Executed {len(tool_results)} tools successfully")
        
        if planning_span:
            tracer = get_tracer()
            tracer.end_span_with_result(planning_span, {
                "tools_executed": len(tool_results),
                "tools": [r["tool"] for r in tool_results],
                "reasoning": execution_plan.reasoning
            }, True)
        
        return tool_results, search_context
        
    except Exception as e:
        print(f"[DEBUG] execute_intelligent_planning: Failed with error: {e}")
        logger.error(f"Intelligent planning failed: {e}", exc_info=True)
        
        if planning_span:
            tracer = get_tracer()
            tracer.end_span_with_result(planning_span, {"error": str(e)}, False)
        
        # Fallback to legacy system
        return execute_tools_first_legacy(question, thinking)

def execute_tools_first_legacy(question: str, thinking: bool = False) -> tuple:
    """
    Execute tools first, then generate answer with tool results
    Returns: (tool_results, updated_question_with_context)
    """
    print(f"[DEBUG] execute_tools_first: question = {question}, thinking = {thinking}")
    
    # Get tools context
    mcp_tools_context = get_mcp_tools_context()
    print(f"[DEBUG] execute_tools_first: got MCP tools context, length = {len(mcp_tools_context)}")
    
    # First, ask LLM to identify which tools to use with enhanced prompt
    tool_selection_prompt = f"""USER QUESTION: "{question}"

{mcp_tools_context}

INSTRUCTIONS: Analyze the question and output EXACTLY ONE of these options:

1. If you need to search the internet/web: <tool>google_search({{"query": "search terms here", "num_results": 10}})</tool>
2. If you need current date/time: <tool>get_datetime({{}})</tool>
3. If no tools needed: NO_TOOLS_NEEDED

For the question "{question}" - this clearly needs web search.

Output the tool call now:"""

    llm_cfg = get_llm_settings()
    prompt = build_prompt(tool_selection_prompt, thinking=False)
    
    # Get tool calls from LLM
    llm_api_url = "http://localhost:8000/api/v1/generate_stream"
    payload = {
        "prompt": prompt,
        "temperature": 0.5,  # Slightly higher to encourage tool output
        "top_p": 0.8,
        "max_tokens": 200
    }
    
    tool_selection_text = ""
    # Use centralized timeout configuration
    from app.core.timeout_settings_cache import get_timeout_value
    http_timeout = get_timeout_value("api_network", "http_request_timeout", 30)
    
    with httpx.Client(timeout=http_timeout) as client:
        with client.stream("POST", llm_api_url, json=payload) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.startswith("data: "):
                    token = line.replace("data: ", "")
                    tool_selection_text += token
    
    print(f"[DEBUG] execute_tools_first: tool_selection_text = {tool_selection_text}")
    print(f"[DEBUG] execute_tools_first: First 500 chars of LLM response: {tool_selection_text[:500]}")
    
    # Execute identified tools
    tool_results = extract_and_execute_tool_calls(tool_selection_text)
    
    # Build context with tool results
    if tool_results:
        results_context = "\n\nTool Results:\n"
        for result in tool_results:
            if "error" in result:
                results_context += f"- {result['tool']}: Error - {result['error']}\n"
            else:
                results_context += f"- {result['tool']}: {json.dumps(result['result'], indent=2)}\n"
        
        updated_question = f"{question}\n{results_context}\n\nPlease provide a complete answer using the tool results above."
    else:
        updated_question = question
        results_context = ""
    
    return tool_results, results_context

def llm_expand_query(question: str, llm_cfg: dict) -> list:
    """Use LLM to generate alternative queries for better retrieval"""
    logger.debug(f"[QUERY_EXPANSION] Starting query expansion for: {question[:100]}...")
    print(f"[DEBUG] Starting query expansion for: {question[:100]}...")  # Keep print for immediate visibility
    
    # Check if llm_cfg is valid
    if not isinstance(llm_cfg, dict):
        logger.error(f"[QUERY_EXPANSION] Invalid llm_cfg type: {type(llm_cfg)}")
        print(f"[ERROR] llm_expand_query: Invalid llm_cfg type: {type(llm_cfg)}")
        return [question]  # Return original query only
    
    # Extract important terms (proper nouns, acronyms, etc.)
    import re
    words = question.split()
    important_terms = []
    
    for word in words:
        cleaned = re.sub(r'[^\w]', '', word)
        if len(cleaned) > 1:
            # Detect acronyms (all caps) or proper nouns (capitalized)
            if cleaned.isupper() or (cleaned[0].isupper() and cleaned.lower() not in 
                                   ['find', 'get', 'show', 'tell', 'give', 'provide']):
                important_terms.append(cleaned)
    
    # Always include original query
    expanded_queries = [question]
    
    # If we have specific entities, add focused variations
    if important_terms:
        # Add just the important terms as a focused query
        expanded_queries.append(" ".join(important_terms))
        
        # Add combinations with common related terms found in the original query
        content_terms = []
        for word in question.lower().split():
            if word in ['issue', 'issues', 'problem', 'problems', 'outage', 'outages', 
                       'disruption', 'failure', 'down', 'error', 'bug', 'incident']:
                content_terms.append(word)
        
        # Create combinations of important terms with content terms
        for important in important_terms:
            for content in content_terms:
                if len(expanded_queries) < 4:
                    expanded_queries.append(f"{important} {content}")
    
    # If we don't have enough variations, use LLM expansion
    if len(expanded_queries) < 3:
        logger.debug(f"[QUERY_EXPANSION] Need more variations, using LLM expansion")
        print(f"[DEBUG] Need more variations, using LLM expansion")
        
        expand_prompt = f"""Generate 2 alternative ways to ask this question for better document search. 
Keep the same intent but vary the phrasing and keywords.
Original question: {question}

Provide ONLY the 2 alternatives, one per line, no numbering or explanations."""
        
        llm_api_url = "http://localhost:8000/api/v1/generate_stream"
        payload = {
            "prompt": expand_prompt,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 150
        }
        
        logger.debug(f"[QUERY_EXPANSION] Calling LLM API at {llm_api_url}")
        print(f"[DEBUG] Calling LLM API for query expansion")
        
        try:
            text = ""
            # Use centralized timeout configuration with a shorter timeout for query expansion
            from app.core.timeout_settings_cache import get_timeout_value
            # Use a shorter timeout specifically for query expansion to prevent hanging
            http_timeout = min(get_timeout_value("api_network", "http_request_timeout", 30), 10)
            logger.debug(f"[QUERY_EXPANSION] Using timeout: {http_timeout} seconds")
            
            with httpx.Client(timeout=httpx.Timeout(http_timeout, connect=5.0)) as client:
                logger.debug(f"[QUERY_EXPANSION] Sending request with timeout={http_timeout}s")
                with client.stream("POST", llm_api_url, json=payload) as response:
                    logger.debug(f"[QUERY_EXPANSION] Got response, processing stream...")
                    line_count = 0
                    for line in response.iter_lines():
                        line_count += 1
                        if not line:
                            continue
                        if isinstance(line, bytes):
                            line = line.decode("utf-8")
                        
                        # Ensure line is a string before calling startswith
                        if not isinstance(line, str):
                            print(f"[ERROR] Unexpected line type in LLM response: {type(line)}")
                            continue
                            
                        if line.startswith("data: "):
                            token = line.replace("data: ", "")
                            text += token
                    logger.debug(f"[QUERY_EXPANSION] Processed {line_count} lines from stream")
            
            # Parse alternatives
            logger.debug(f"[QUERY_EXPANSION] Raw LLM response: {text[:200]}...")
            alternatives = [q.strip() for q in text.strip().split('\n') if q.strip() and len(q.strip()) > 5]
            expanded_queries.extend(alternatives[:2])  # Take up to 2 alternatives
            logger.debug(f"[QUERY_EXPANSION] Added {len(alternatives[:2])} LLM-generated variations")
            print(f"[DEBUG] LLM query expansion successful, added {len(alternatives[:2])} variations")
            
        except httpx.TimeoutException as e:
            logger.warning(f"[QUERY_EXPANSION] LLM expansion timed out after {http_timeout}s: {str(e)}")
            print(f"[WARNING] LLM query expansion timed out after {http_timeout}s, using basic expansion only")
        except Exception as e:
            logger.error(f"[QUERY_EXPANSION] LLM expansion failed: {str(e)}")
            print(f"[DEBUG] LLM query expansion failed: {str(e)}")
    
    # Normalize all queries to lowercase for consistent searching
    normalized_queries = [query.lower().strip() for query in expanded_queries]
    
    logger.debug(f"[QUERY_EXPANSION] Final expanded queries: {normalized_queries}")
    print(f"[DEBUG] Expanded queries: {normalized_queries}")
    return normalized_queries

def keyword_search_milvus(question: str, collection_name: str, uri: str, token: str) -> list:
    """Enhanced keyword search with smart term extraction"""
    from pymilvus import Collection, connections
    import re
    
    try:
        print(f"[DEBUG] keyword_search_milvus: Connecting to Milvus...")
        connections.connect(uri=uri, token=token, alias="keyword_search")
        print(f"[DEBUG] keyword_search_milvus: Connected, accessing collection {collection_name}...")
        collection = Collection(collection_name, using="keyword_search")
        print(f"[DEBUG] keyword_search_milvus: Loading collection...")
        collection.load()
        print(f"[DEBUG] keyword_search_milvus: Collection loaded successfully")
        
        # Generic stop words for any query
        stop_words = {
            'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an',
            'find', 'get', 'show', 'tell', 'about', 'info', 'information', 'news', 'what', 'how', 'when', 'where',
            'why', 'can', 'could', 'would', 'should', 'give', 'provide', 'need', 'want', 'like', 'please', 'help',
            'me', 'out', 'up', 'do', 'does', 'did', 'has', 'have', 'had', 'will', 'was', 'were', 'been', 'being'
        }
        
        # Intelligently extract search terms
        words = question.lower().split()
        search_terms = []
        
        # Use lightweight keyword extraction (no external dependencies)
        try:
            from app.langchain.lightweight_keyword_extractor import LightweightKeywordExtractor
            extractor = LightweightKeywordExtractor()
            extracted_terms = extractor.extract_key_terms(question)
            
            # Get prioritized search terms
            important_terms = extractor.get_search_terms(extracted_terms, max_terms=5)
            
            # Get all extracted terms
            all_extracted = []
            for terms in extracted_terms.values():
                all_extracted.extend(terms)
            all_extracted = list(set(all_extracted))
            
            print(f"[DEBUG] Keyword extraction - patterns: {extracted_terms.get('pattern_matches', [])}")
            print(f"[DEBUG] Keyword extraction - entities: {extracted_terms.get('potential_entities', [])}")
            print(f"[DEBUG] Keyword extraction - phrases: {extracted_terms.get('important_phrases', [])}")
            print(f"[DEBUG] Keyword extraction - final search terms: {important_terms}")
            
        except Exception as e:
            print(f"[WARNING] Keyword extraction failed: {e}")
            # Fallback to empty
            important_terms = []
            all_extracted = []
        
        # Basic extraction as fallback or complement
        content_terms = []
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) < 2:
                continue
            
            # Skip if already extracted by smart extractor
            if clean_word in all_extracted:
                continue
            
            # Check if it's an important term by looking at original casing
            original_words = question.split()
            for orig in original_words:
                orig_clean = re.sub(r'[^\w]', '', orig)
                if orig_clean.lower() == clean_word:
                    # Important if it's an acronym (all caps) or proper noun (capitalized, not common word)
                    if (orig_clean.isupper() and len(orig_clean) > 1) or \
                       (orig_clean[0].isupper() and orig_clean.lower() not in stop_words and 
                        orig_clean not in ['Find', 'Get', 'Show', 'Tell', 'Give', 'Provide']):
                        if clean_word not in important_terms:
                            important_terms.append(clean_word)
                    break
            
            # Also collect content terms (not stop words, length > 2)
            if clean_word not in stop_words and len(clean_word) > 2:
                content_terms.append(clean_word)
        
        # Strategy: prioritize important terms, but include relevant content terms
        if important_terms:
            search_terms = important_terms
            # Add a few most relevant content terms that aren't already included
            for term in content_terms:
                if term not in important_terms and len(search_terms) < 4:
                    search_terms.append(term)
        else:
            # If no important terms, use content terms
            search_terms = content_terms[:4]  # Limit to avoid overly complex queries
        
        print(f"[DEBUG] Keyword search - original query: {question}")
        print(f"[DEBUG] Keyword search - important terms: {important_terms}")
        print(f"[DEBUG] Keyword search - search terms: {search_terms}")
        
        # Build expressions - try different strategies
        all_results = []
        
        # Strategy 1: All terms must be present (AND)
        if len(search_terms) <= 3:
            # Use lowercase for case-insensitive search
            conditions = [f'content like "%{word.lower()}%"' for word in search_terms]
            expr = " and ".join(conditions)
            
            print(f"[DEBUG] Keyword search expression (AND): {expr}")
            
            results = collection.query(
                expr=expr,
                output_fields=["content", "source", "page", "hash", "doc_id"],
                limit=20
            )
            all_results.extend(results)
            print(f"[DEBUG] AND search found {len(results)} results")
            
            # Also search in source field for important terms
            if important_terms and len(all_results) < 10:
                source_conditions = [f'source like "%{word.lower()}%"' for word in important_terms]
                source_expr = " or ".join(source_conditions)
                
                print(f"[DEBUG] Source field search expression: {source_expr}")
                
                source_results = collection.query(
                    expr=source_expr,
                    output_fields=["content", "source", "page", "hash", "doc_id"],
                    limit=20
                )
                
                # Add unique results from source search
                existing_hashes = {r.get("hash") for r in all_results}
                for r in source_results:
                    if r.get("hash") not in existing_hashes:
                        all_results.append(r)
                
                print(f"[DEBUG] Source field search added {len(source_results)} results")
        
        # Strategy 2: Any important term (OR) - if AND didn't find enough
        if len(all_results) < 5 and important_terms:
            conditions = [f'content like "%{word}%"' for word in important_terms]
            expr = " or ".join(conditions)
            
            print(f"[DEBUG] Keyword search expression (OR): {expr}")
            
            results = collection.query(
                expr=expr,
                output_fields=["content", "source", "page", "hash", "doc_id"],
                limit=20
            )
            
            # Add unique results
            existing_hashes = {r.get("hash") for r in all_results}
            for r in results:
                if r.get("hash") not in existing_hashes:
                    all_results.append(r)
            
            print(f"[DEBUG] OR search added {len(results)} results")
            
        # Strategy 3: Search in source field for all search terms if still not enough results
        if len(all_results) < 10 and search_terms:
            source_conditions = [f'source like "%{word.lower()}%"' for word in search_terms]
            source_expr = " or ".join(source_conditions)
            
            print(f"[DEBUG] Extended source field search expression: {source_expr}")
            
            source_results = collection.query(
                expr=source_expr,
                output_fields=["content", "source", "page", "hash", "doc_id"],
                limit=20
            )
            
            # Add unique results from source search
            existing_hashes = {r.get("hash") for r in all_results}
            for r in source_results:
                if r.get("hash") not in existing_hashes:
                    all_results.append(r)
            
            print(f"[DEBUG] Extended source search added {len(source_results)} results")
        
        print(f"[DEBUG] Keyword search total found {len(all_results)} results")
        
        # Convert to document format
        from langchain.schema import Document
        docs = []
        for r in all_results:
            doc = Document(
                page_content=r.get("content", ""),
                metadata={
                    "source": r.get("source", ""),
                    "page": r.get("page", 0),
                    "hash": r.get("hash", ""),
                    "doc_id": r.get("doc_id", "")
                }
            )
            docs.append(doc)
        
        return docs
        
    except Exception as e:
        print(f"[ERROR] Keyword search failed: {str(e)}")
        return []
    finally:
        connections.disconnect(alias="keyword_search")

def extract_metadata_hints(question: str) -> dict:
    """Extract potential metadata filters from the query"""
    hints = {}
    
    # Check for date patterns
    import re
    year_pattern = r'\b(20\d{2})\b'
    years = re.findall(year_pattern, question)
    if years:
        hints['years'] = years
    
    # Check for specific document indicators
    if 'pdf' in question.lower() or 'document' in question.lower():
        hints['doc_type'] = 'pdf'
    
    return hints

def suggest_chunking_improvements(query_analysis: dict) -> dict:
    """Suggest chunking improvements based on query patterns"""
    suggestions = {
        'chunk_size': 512,  # Default
        'chunk_overlap': 128,  # Default
        'strategy': 'sentence_based',
        'recommendations': []
    }
    
    if query_analysis['is_short'] or query_analysis['has_acronyms']:
        # Smaller chunks for specific term matching
        suggestions['chunk_size'] = 256
        suggestions['chunk_overlap'] = 64
        suggestions['strategy'] = 'paragraph_based'
        suggestions['recommendations'].append(
            "Consider smaller chunks (256 tokens) for better acronym and specific term matching"
        )
    
    if query_analysis['has_proper_nouns']:
        # Ensure proper context around entities
        suggestions['chunk_overlap'] = 256
        suggestions['recommendations'].append(
            "Increase chunk overlap to preserve entity context across boundaries"
        )
    
    if query_analysis['is_specific']:
        # Precise matching needs smaller chunks
        suggestions['chunk_size'] = 200
        suggestions['strategy'] = 'sentence_based'
        suggestions['recommendations'].append(
            "Use sentence-based chunking for precise term matching"
        )
    
    return suggestions

def analyze_query_type(question: str) -> dict:
    """Analyze query to determine optimal search strategy"""
    analysis = {
        'is_short': len(question.split()) <= 3,
        'has_acronyms': False,
        'has_proper_nouns': False,
        'is_specific': False,
        'keyword_priority': 0.5  # Default 50/50 balance
    }
    
    # Check for acronyms (all caps words)
    words = question.split()
    acronyms = [w for w in words if w.isupper() and len(w) >= 2]
    if acronyms:
        analysis['has_acronyms'] = True
        analysis['keyword_priority'] = 0.7  # Favor keyword search for acronyms
    
    # Check for proper nouns (capitalized words not at start)
    proper_nouns = [w for i, w in enumerate(words) if i > 0 and w[0].isupper()]
    if proper_nouns:
        analysis['has_proper_nouns'] = True
        analysis['keyword_priority'] = max(analysis['keyword_priority'], 0.6)
    
    # Check if query is very specific (contains quotes, specific terms)
    if '"' in question or any(term in question.lower() for term in ['exactly', 'specific', 'precise']):
        analysis['is_specific'] = True
        analysis['keyword_priority'] = 0.8
    
    # Short queries often work better with keyword search
    if analysis['is_short']:
        analysis['keyword_priority'] = max(analysis['keyword_priority'], 0.7)
    
    return analysis

def handle_rag_query(question: str, thinking: bool = False, collections: List[str] = None, collection_strategy: str = "auto", trace=None) -> tuple:
    """Handle RAG queries with hybrid search (vector + keyword) - returns context only
    
    Args:
        question: The user's question
        thinking: Whether to enable extended thinking
        collections: List of collection names to search (None = auto-detect)
        collection_strategy: "auto", "specific", or "all"
    """
    # Handle case where question might be passed as dict (from recent tool calling changes)
    if isinstance(question, dict):
        question = question.get('query', '')
    
    # Ensure question is a string
    if not isinstance(question, str):
        question = str(question) if question else ''
    
    print(f"[DEBUG] handle_rag_query: question = {question}, thinking = {thinking}, collections = {collections}, strategy = {collection_strategy}")
    
    # Create RAG span for tracing
    rag_span = None
    tracer = None
    if trace:
        try:
            from app.core.langfuse_integration import get_tracer
            tracer = get_tracer()
            if tracer.is_enabled():
                rag_span = tracer.create_rag_span(trace, question, collections)
        except Exception as e:
            logger.warning(f"Failed to create RAG span: {e}")
    
    # Check cache first for performance - DISABLED FOR FRESH RESULTS
    # query_hash = hash(question.lower().strip())
    # current_time = time.time()
    
    # if query_hash in _rag_cache:
    #     cached_result, timestamp = _rag_cache[query_hash]
    #     cache_settings = get_rag_cache_settings()
    #     cache_ttl = cache_settings['ttl']
    #     if current_time - timestamp < cache_ttl:
    #         print(f"[DEBUG] handle_rag_query: Using cached result for query")
    #         return cached_result
    #     else:
    #         # Remove expired cache entry
    #         del _rag_cache[query_hash]
    print(f"[DEBUG] handle_rag_query: Cache disabled, performing fresh document retrieval")
    
    try:
        embedding_cfg = get_embedding_settings()
        print(f"[DEBUG] embedding_cfg type: {type(embedding_cfg)}, value: {embedding_cfg}")
        vector_db_cfg = get_vector_db_settings()
        print(f"[DEBUG] vector_db_cfg type: {type(vector_db_cfg)}, value: {vector_db_cfg}")
        llm_cfg = get_llm_settings()
        # print(f"[DEBUG] llm_cfg type: {type(llm_cfg)}, value: {llm_cfg}")
    except Exception as e:
        print(f"[ERROR] Failed to get settings: {str(e)}")
        raise
    
    # Analyze query to determine search strategy
    query_analysis = analyze_query_type(question)
    print(f"[DEBUG] Query analysis: {query_analysis}")
    
    # Get chunking suggestions based on query
    chunking_suggestions = suggest_chunking_improvements(query_analysis)
    if chunking_suggestions['recommendations']:
        print(f"[DEBUG] Chunking recommendations: {', '.join(chunking_suggestions['recommendations'])}")
    
    # Extract metadata hints for filtering
    metadata_hints = extract_metadata_hints(question)
    print(f"[DEBUG] Metadata hints: {metadata_hints}")
    
    # Set up embeddings
    embedding_endpoint = embedding_cfg.get("embedding_endpoint")
    embedding_model = embedding_cfg.get("embedding_model")
    
    print(f"[DEBUG] handle_rag_query: Embedding config - endpoint: {embedding_endpoint}, model: {embedding_model}")
    
    print(f"[DEBUG] Setting up embedding function...")
    if embedding_endpoint and isinstance(embedding_endpoint, str) and embedding_endpoint.strip():
        print(f"[DEBUG] Creating HTTP embedding function with endpoint: {embedding_endpoint}")
        try:
            embeddings = HTTPEmbeddingFunction(embedding_endpoint)
            print(f"[DEBUG] HTTP embedding function created successfully")
        except Exception as e:
            print(f"[ERROR] Failed to create HTTP embedding function: {str(e)}")
            raise
    else:
        # Use default model if embedding_model is not a valid string
        model_name = embedding_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5")
        if not isinstance(model_name, str) or not model_name.strip():
            model_name = "BAAI/bge-base-en-v1.5"
        print(f"[DEBUG] Creating HuggingFace embeddings with model: {model_name}")
        try:
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            print(f"[DEBUG] HuggingFace embedding function created successfully")
        except Exception as e:
            print(f"[ERROR] Failed to create HuggingFace embedding function: {str(e)}")
            raise
    
    # Determine which collections to search
    # Handle both old and new vector database configuration formats
    if "milvus" in vector_db_cfg:
        # Legacy format
        milvus_cfg = vector_db_cfg["milvus"]
        uri = milvus_cfg.get("MILVUS_URI")
        token = milvus_cfg.get("MILVUS_TOKEN")
    else:
        # New format - find milvus database in databases array
        milvus_cfg = None
        for db in vector_db_cfg.get("databases", []):
            if db.get("id") == "milvus" and db.get("enabled", False):
                milvus_cfg = db.get("config", {})
                break
        
        if not milvus_cfg:
            milvus_cfg = {}
        
        uri = milvus_cfg.get("MILVUS_URI")
        token = milvus_cfg.get("MILVUS_TOKEN")
    
    # Get collections to search based on strategy
    collections_to_search = []
    
    if collection_strategy == "specific" and collections:
        # Use only specified collections
        collections_to_search = collections
        print(f"[DEBUG] Using specific collections: {collections_to_search}")
    elif collection_strategy == "all":
        # Get all available collections from registry
        from app.core.collection_registry_cache import get_all_collections
        all_collections = get_all_collections()
        collections_to_search = [col["collection_name"] for col in all_collections]
        print(f"[DEBUG] Searching all collections: {collections_to_search}")
    else:  # "auto" strategy
        pass
    
    # Create collection analysis span for all strategies
    collection_analysis_span = None
    if rag_span and tracer:
        try:
            collection_analysis_span = tracer.create_nested_span(
                rag_span,
                name="collection-analysis",
                metadata={
                    "operation": "collection_selection",
                    "strategy": collection_strategy,
                    "query_length": len(question)
                }
            )
        except Exception as e:
            logger.warning(f"Failed to create collection analysis span: {e}")
    
    if collection_strategy == "auto":
        # Auto collection selection logic
        
        # Auto-detect relevant collections based on query
        from app.core.document_classifier import get_document_classifier
        classifier = get_document_classifier()
        
        # Use query analysis to determine collection type
        collection_type = classifier.classify_document(question, {"query": True})
        target_collection = classifier.get_target_collection(collection_type)
        
        # End collection analysis span
        if collection_analysis_span and tracer:
            try:
                tracer.end_span_with_result(collection_analysis_span, {
                    "collection_type": collection_type,
                    "target_collection": target_collection,
                    "classification_success": True
                }, True)
            except Exception as e:
                logger.warning(f"Failed to end collection analysis span: {e}")
        
        if target_collection:
            collections_to_search = [target_collection]
            # Also add default_knowledge for broader search
            if target_collection != "default_knowledge":
                collections_to_search.append("default_knowledge")
        else:
            collections_to_search = ["default_knowledge"]
            
        print(f"[DEBUG] Auto-detected collections for type '{collection_type}': {collections_to_search}")
    
    # Fallback to default if no collections determined
    if not collections_to_search:
        collections_to_search = ["default_knowledge"]
        print(f"[DEBUG] Using default collection as fallback")
    
    print(f"[DEBUG] handle_rag_query: Will search collections: {collections_to_search}")
    
    # Retrieve relevant documents - optimized for performance
    # IMPORTANT: Milvus with COSINE distance returns lower scores for more similar vectors
    # Score range: 0 (identical) to 2 (opposite)
    # Get RAG settings from new centralized configuration
    from app.core.rag_settings_cache import get_document_retrieval_settings, get_search_strategy_settings
    doc_settings = get_document_retrieval_settings()
    search_settings = get_search_strategy_settings()
    
    SIMILARITY_THRESHOLD = doc_settings.get('similarity_threshold', 1.5)
    NUM_DOCS = doc_settings.get('num_docs_retrieve', 20)
    
    # Use LLM to expand queries for better recall
    logger.info(f"[RAG_SEARCH] Starting query expansion for: {question[:100]}...")
    print(f"[DEBUG] Starting query expansion for: {question}")
    
    try:
        queries_to_try = llm_expand_query(question, llm_cfg)
        logger.info(f"[RAG_SEARCH] Query expansion completed. Got {len(queries_to_try)} variations")
        print(f"[DEBUG] Query expansion completed successfully. Expanded queries: {queries_to_try}")
    except Exception as e:
        logger.error(f"[RAG_SEARCH] Query expansion failed with error: {str(e)}")
        print(f"[ERROR] Query expansion failed: {str(e)}, using original query only")
        queries_to_try = [question.lower()]  # Fallback to original query
    
    print(f"[DEBUG] Starting hybrid search process...")
    
    try:
        # HYBRID SEARCH: Run both vector and keyword search across all collections
        all_docs = []
        seen_ids = set()
        
        print(f"[DEBUG] Collections to search: {collections_to_search}")
        print(f"[DEBUG] Starting search across {len(collections_to_search)} collections...")
        
        # Search each collection
        for i, collection_name in enumerate(collections_to_search):
            print(f"[DEBUG] Processing collection {i+1}/{len(collections_to_search)}: {collection_name}")
            print(f"[DEBUG] Creating Milvus store for collection: {collection_name}...")
            print(f"[DEBUG] Using URI: {uri[:50]}{'...' if len(str(uri)) > 50 else ''}")
            print(f"[DEBUG] Using embeddings type: {type(embeddings).__name__}")
            
            # Create Milvus store for this collection with timeout handling
            try:
                import signal
                import asyncio
                from concurrent.futures import TimeoutError
                
                # Add timeout to Milvus store creation
                def timeout_handler(signum, frame):
                    raise TimeoutError("Milvus store creation timed out after 30 seconds")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # 30 second timeout
                
                print(f"[DEBUG] Starting Milvus store creation...")
                milvus_store = Milvus(
                    embedding_function=embeddings,
                    collection_name=collection_name,
                    connection_args={"uri": uri, "token": token},
                    text_field="content"
                )
                signal.alarm(0)  # Cancel timeout
                print(f"[DEBUG] Milvus store created successfully for {collection_name}")
                
            except TimeoutError as e:
                print(f"[ERROR] Milvus store creation timed out for collection {collection_name}: {e}")
                signal.alarm(0)  # Cancel timeout
                continue  # Skip this collection
            except Exception as e:
                print(f"[ERROR] Failed to create Milvus store for collection {collection_name}: {str(e)}")
                signal.alarm(0)  # Cancel timeout
                continue  # Skip this collection
            
            # 1. Vector search with query expansion
            print(f"[DEBUG] Starting vector search with {len(queries_to_try)} queries for collection {collection_name}")
            for j, query in enumerate(queries_to_try):
                try:
                    print(f"[DEBUG] Processing query {j+1}/{len(queries_to_try)}: '{query[:50]}{'...' if len(query) > 50 else ''}'")
                    # Normalize query to lowercase for consistent vector search
                    normalized_query = query.lower().strip()
                    print(f"[DEBUG] Normalized query: '{normalized_query[:50]}{'...' if len(normalized_query) > 50 else ''}'")
                    
                    # Create vector search span for tracing
                    vector_search_span = None
                    if collection_analysis_span and tracer:
                        try:
                            vector_search_span = tracer.create_nested_span(
                                collection_analysis_span,
                                name="vector-search",
                                metadata={
                                    "operation": "milvus_similarity_search",
                                    "collection": collection_name,
                                    "query": normalized_query[:100],
                                    "k": NUM_DOCS
                                }
                            )
                        except Exception as e:
                            logger.warning(f"Failed to create vector search span: {e}")
                    
                    print(f"[DEBUG] Starting vector search operation...")
                    vector_search_start = time.time()
                    
                    # Add timeout to vector search
                    try:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(60)  # 60 second timeout for vector search
                        
                        if hasattr(milvus_store, 'similarity_search_with_score'):
                            print(f"[DEBUG] Using similarity_search_with_score method...")
                            docs = milvus_store.similarity_search_with_score(normalized_query, k=NUM_DOCS)
                        else:
                            print(f"[DEBUG] Using fallback similarity_search method...")
                            docs = [(doc, 0.0) for doc in milvus_store.similarity_search(normalized_query, k=NUM_DOCS)]
                        
                        signal.alarm(0)  # Cancel timeout
                        vector_search_end = time.time()
                        print(f"[DEBUG] Vector search completed successfully")
                        
                    except TimeoutError as e:
                        print(f"[ERROR] Vector search timed out after 60 seconds for query: {normalized_query[:50]}")
                        signal.alarm(0)  # Cancel timeout
                        continue  # Skip this query
                    except Exception as e:
                        print(f"[ERROR] Vector search failed for query '{normalized_query[:50]}': {str(e)}")
                        signal.alarm(0)  # Cancel timeout
                        continue  # Skip this query
                    
                    # End vector search span with results
                    if vector_search_span:
                        try:
                            tracer.end_span_with_result(
                                vector_search_span,
                                {"docs_found": len(docs), "search_time": vector_search_end - vector_search_start},
                                success=True
                            )
                        except Exception as e:
                            logger.warning(f"Failed to end vector search span: {e}")
                    
                    print(f"[DEBUG] Vector search in {collection_name} for '{normalized_query[:50]}...' took {vector_search_end - vector_search_start:.2f} seconds, found {len(docs)} docs")
                    
                    # Add unique documents with collection metadata
                    for doc, score in docs:
                        # Add collection info to metadata
                        if not hasattr(doc, 'metadata'):
                            doc.metadata = {}
                        doc.metadata['source_collection'] = collection_name
                        
                        # Create a unique ID based on content hash
                        doc_id = hash(doc.page_content)
                        if doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            all_docs.append((doc, score))
                            
                except Exception as e:
                    print(f"[ERROR] handle_rag_query: Failed for query '{query}' in collection {collection_name}: {str(e)}")
            
            # 2. ALWAYS perform keyword search in parallel (not just as fallback)
            print(f"[DEBUG] Running keyword search in {collection_name}")
            
            # Create keyword search span for tracing
            keyword_search_span = None
            if collection_analysis_span and tracer:
                try:
                    keyword_search_span = tracer.create_nested_span(
                        collection_analysis_span,
                        name="keyword-search",
                        metadata={
                            "operation": "milvus_keyword_search",
                            "collection": collection_name,
                            "query": question[:100]
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to create keyword search span: {e}")
            
            keyword_search_start = time.time()
            try:
                print(f"[DEBUG] Starting keyword search operation...")
                # Add timeout to keyword search
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(45)  # 45 second timeout for keyword search
                
                keyword_docs = keyword_search_milvus(
                    question,
                    collection_name,
                    uri=milvus_cfg.get("MILVUS_URI"),
                    token=milvus_cfg.get("MILVUS_TOKEN")
                )
                signal.alarm(0)  # Cancel timeout
                keyword_search_end = time.time()
                print(f"[DEBUG] Keyword search completed successfully")
                
            except TimeoutError as e:
                print(f"[ERROR] Keyword search timed out after 45 seconds for collection {collection_name}: {e}")
                signal.alarm(0)  # Cancel timeout
                keyword_docs = []  # Use empty results
                keyword_search_end = time.time()
            except Exception as e:
                print(f"[ERROR] Keyword search failed for collection {collection_name}: {str(e)}")
                signal.alarm(0)  # Cancel timeout
                keyword_docs = []  # Use empty results
                keyword_search_end = time.time()
            
            # End keyword search span with results
            if keyword_search_span:
                try:
                    tracer.end_span_with_result(
                        keyword_search_span,
                        {"docs_found": len(keyword_docs), "search_time": keyword_search_end - keyword_search_start},
                        success=True
                    )
                except Exception as e:
                    logger.warning(f"Failed to end keyword search span: {e}")
            
            print(f"[DEBUG] Keyword search in {collection_name} found {len(keyword_docs)} docs")
            
            # Add keyword search results with a favorable score (0.8 = good match in cosine distance)
            keyword_boost_count = 0
            for doc in keyword_docs:
                # Add collection info to metadata
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata['source_collection'] = collection_name
                
                doc_id = hash(doc.page_content)
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append((doc, 0.8))  # Good match score for keyword results
                    print(f"[DEBUG] Added keyword-only doc: {doc.page_content[:100]}...")
                else:
                    # If document already found by vector search, boost its importance
                    for i, (existing_doc, existing_score) in enumerate(all_docs):
                        if hash(existing_doc.page_content) == doc_id:
                            # Improve score if found by both methods
                            old_score = existing_score
                            all_docs[i] = (existing_doc, min(existing_score * 0.7, 0.5))
                            keyword_boost_count += 1
                            print(f"[DEBUG] Boosted doc score from {old_score} to {all_docs[i][1]}")
                            break
            
            print(f"[DEBUG] Keyword search in {collection_name} added {len(keyword_docs) - keyword_boost_count} new docs and boosted {keyword_boost_count} existing docs")
        
        print(f"[DEBUG] Total documents from all collections: {len(all_docs)}")
        
        if not all_docs:
            print(f"[ERROR] handle_rag_query: No documents retrieved from vector or keyword search")
            
            # Last resort: Try with the most specific terms from the query
            # Extract any capitalized words or unique terms
            fallback_terms = []
            for word in question.split():
                cleaned = re.sub(r'[^\w]', '', word)
                if len(cleaned) > 2 and (cleaned.isupper() or cleaned[0].isupper()):
                    fallback_terms.append(cleaned.lower())
            
            if fallback_terms:
                print(f"[DEBUG] Trying fallback search with terms: {fallback_terms}")
                try:
                    from pymilvus import Collection, connections
                    connections.connect(uri=uri, token=token, alias="fallback_search")
                    # Use first collection from search list for fallback
                    fallback_collection = collections_to_search[0] if collections_to_search else "default_knowledge"
                    collection_obj = Collection(fallback_collection, using="fallback_search")
                    collection_obj.load()
                    
                    # Try with the first fallback term
                    expr = f'content like "%{fallback_terms[0]}%"'
                    results = collection_obj.query(
                        expr=expr,
                        output_fields=["content", "source", "page", "hash", "doc_id"],
                        limit=10
                    )
                    
                    if results:
                        print(f"[DEBUG] Fallback search found {len(results)} results")
                        from langchain.schema import Document
                        for r in results:
                            doc = Document(
                                page_content=r.get("content", ""),
                                metadata={
                                    "source": r.get("source", ""),
                                    "page": r.get("page", 0),
                                    "hash": r.get("hash", ""),
                                    "doc_id": r.get("doc_id", ""),
                                    "source_collection": fallback_collection
                                }
                            )
                            all_docs.append((doc, 1.0))
                    
                    connections.disconnect(alias="fallback_search")
                except Exception as e:
                    print(f"[ERROR] Fallback search failed: {str(e)}")
            
            if not all_docs:
                # End collection analysis span and RAG span with no results
                if collection_analysis_span and tracer:
                    try:
                        tracer.end_span_with_result(
                            collection_analysis_span,
                            {"documents_found": 0, "fallback_attempted": True},
                            success=False,
                            error="No documents found after vector, keyword, and fallback search"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to end collection analysis span: {e}")
                
                if rag_span and tracer:
                    try:
                        tracer.end_span_with_result(rag_span, {"documents_found": 0}, True)
                    except Exception as e:
                        logger.warning(f"Failed to end RAG span: {e}")
                return "", ""
            
        print(f"[DEBUG] Total unique documents found: {len(all_docs)} (vector + keyword)")
        
        # Sort by score (lower is better for cosine distance)
        docs = sorted(all_docs, key=lambda x: x[1])[:NUM_DOCS]
        
    except Exception as e:
        print(f"[ERROR] handle_rag_query: Failed to search vector store: {str(e)}")
        # End collection analysis span and RAG span with error
        if collection_analysis_span and tracer:
            try:
                tracer.end_span_with_result(
                    collection_analysis_span,
                    None,
                    success=False,
                    error=str(e)
                )
            except Exception as span_error:
                logger.warning(f"Failed to end collection analysis span with error: {span_error}")
        
        if rag_span and tracer:
            try:
                tracer.end_span_with_result(rag_span, {"error": str(e)}, False, str(e))
            except Exception as e:
                logger.warning(f"Failed to end RAG span: {e}")
        return "", ""
    
    print(f"[DEBUG] handle_rag_query: Retrieved {len(docs)} documents")
    print(f"[DEBUG] handle_rag_query: Question = '{question}'")
    
    # Debug: Print all retrieved documents with scores
    if docs:
        print(f"[DEBUG] handle_rag_query: First 3 documents:")
        for i, (doc, score) in enumerate(docs[:3]):
            print(f"  Doc {i}: score={score:.4f}, content_preview='{doc.page_content[:100]}...'")
    else:
        print(f"[DEBUG] handle_rag_query: No documents retrieved from vector store!")
        # End collection analysis span and RAG span with no documents
        if collection_analysis_span and tracer:
            try:
                tracer.end_span_with_result(
                    collection_analysis_span,
                    {"documents_found": 0, "message": "No documents retrieved from vector store"},
                    success=False,
                    error="No documents found in any collection"
                )
            except Exception as e:
                logger.warning(f"Failed to end collection analysis span: {e}")
        
        if rag_span and tracer:
            try:
                tracer.end_span_with_result(rag_span, {"documents_found": 0}, True)
            except Exception as e:
                logger.warning(f"Failed to end RAG span: {e}")
        return "", ""
    
    # Fix: For COSINE distance metric, LOWER scores are BETTER matches
    # Filter and rerank documents
    filtered_and_ranked = []
    query_analysis = analyze_query_type(question)  # Get query analysis for scoring
    for doc, score in docs:
        # Convert distance to similarity for easier understanding
        similarity = 1 - (score / 2)  # Convert [0,2] to [1,0]
        print(f"[DEBUG] handle_rag_query: Doc distance={score:.3f}, similarity={similarity:.3f}, content preview = {doc.page_content[:100]}")
        
        if score <= SIMILARITY_THRESHOLD:  # Filter by distance threshold
            # Enhanced BM25-based relevance calculation with corpus stats
            from app.core.rag_settings_cache import get_bm25_settings
            try:
                bm25_settings = get_bm25_settings()
                bm25_enabled = bm25_settings.get('enable_bm25', True)
                bm25_weight = bm25_settings.get('bm25_weight', 0.3)
            except:
                bm25_enabled = True
                bm25_weight = 0.3
            
            if bm25_enabled:
                # Calculate BM25-enhanced relevance score
                keyword_relevance = calculate_relevance_score(question, doc.page_content)
                
                # Apply BM25 weight from settings
                enhanced_keyword_score = keyword_relevance * (1 + bm25_weight)
                print(f"[DEBUG] BM25 enhanced score: {keyword_relevance:.3f} -> {enhanced_keyword_score:.3f}")
            else:
                # Fallback to legacy scoring
                keyword_relevance = calculate_relevance_score(question, doc.page_content)
                enhanced_keyword_score = keyword_relevance
            
            # Hybrid search: Balance vector similarity and enhanced keyword relevance
            # Use query analysis to determine optimal weighting, with special handling for partnership queries
            base_keyword_weight = query_analysis['keyword_priority']
            
            # Boost keyword weight for partnership/entity-specific queries
            from app.core.rag_settings_cache import get_search_strategy_settings
            try:
                search_settings = get_search_strategy_settings()
                partnership_boost = search_settings.get('partnership_keyword_boost', 0.2)
            except:
                partnership_boost = 0.2
            
            # Detect partnership queries and adjust weighting
            query_lower = question.lower()
            is_partnership_query = any(term in query_lower for term in [
                'partnership', 'collaboration', 'relationship', 'between'
            ])
            
            if is_partnership_query:
                # Increase keyword weight for partnership queries to prioritize specific company content
                keyword_weight = min(0.8, base_keyword_weight + partnership_boost)
                print(f"[DEBUG] Partnership query detected, boosted keyword weight: {base_keyword_weight:.2f} -> {keyword_weight:.2f}")
            else:
                keyword_weight = base_keyword_weight
            
            vector_weight = 1 - keyword_weight
            combined_score = (similarity * vector_weight) + (enhanced_keyword_score * keyword_weight)
            
            
            filtered_and_ranked.append((doc, score, similarity, keyword_relevance, combined_score))
    
    # Sort by combined score (higher is better)
    filtered_and_ranked.sort(key=lambda x: x[4], reverse=True)
    
    # Qwen3-Reranker-4B based re-ranking for top candidates
    if filtered_and_ranked and llm_cfg:
        try:
            # Check if reranker is enabled via settings (not just environment)
            from app.core.reranker_config import RerankerConfig
            from app.core.rag_settings_cache import get_reranking_settings
            
            # Check both config and database settings
            try:
                rerank_settings = get_reranking_settings()
                enable_qwen_reranker = rerank_settings.get('enable_qwen_reranker', True)
            except:
                enable_qwen_reranker = True
            
            if not enable_qwen_reranker:
                raise ImportError("Qwen reranker disabled in RAG settings")
            
            # Try to initialize reranker regardless of Docker environment if enabled in settings
            if not RerankerConfig.is_enabled() and not enable_qwen_reranker:
                raise ImportError("Reranker disabled in environment config")
            
            # Try to use Qwen3-Reranker-4B if available
            from app.rag.qwen_reranker import get_qwen_reranker
            
            # Re-rank top 20 candidates (we can handle more with dedicated model)
            num_to_rerank = min(20, len(filtered_and_ranked))
            print(f"[DEBUG] Starting Qwen reranker re-ranking of top {num_to_rerank} documents")
            
            # Get reranker instance
            reranker = get_qwen_reranker()
            
            # If reranker is not available, fall back to LLM reranking
            if reranker is None:
                raise ImportError("Reranker not available")
            
            # Prepare documents for reranking
            docs_to_rerank = [(item[0], item[4]) for item in filtered_and_ranked[:num_to_rerank]]
            
            # Determine task type based on query
            task_type = "general"
            if any(keyword in question.lower() for keyword in ["code", "function", "class", "method", "api"]):
                task_type = "code"
            elif any(keyword in question.lower() for keyword in ["technical", "documentation", "specification"]):
                task_type = "technical"
            
            instruction = reranker.create_task_specific_instruction(task_type)
            
            # Perform reranking with hybrid scoring (uses settings for rerank_weight)
            rerank_results = reranker.rerank_with_hybrid_score(
                query=question,
                documents=docs_to_rerank,
                rerank_weight=None,  # Use settings default
                instruction=instruction
            )
            
            # Update the filtered_and_ranked list with new scores
            for i, result in enumerate(rerank_results):
                if i < len(filtered_and_ranked):
                    # Find the original item
                    for j, item in enumerate(filtered_and_ranked):
                        if item[0] == result.document:
                            # Update with hybrid score
                            filtered_and_ranked[j] = (
                                item[0], item[1], item[2], item[3], 
                                result.metadata["hybrid_score"]
                            )
                            break
            
            # Re-sort with updated scores
            filtered_and_ranked.sort(key=lambda x: x[4], reverse=True)
            print(f"[DEBUG] Qwen3-Reranker-4B re-ranking complete")
            
        except ImportError:
            print(f"[DEBUG] Qwen3-Reranker-4B not available, falling back to LLM re-ranking")
            # Fallback to original LLM-based re-ranking
            # Re-rank top 15 candidates (balance between quality and cost)
            num_to_rerank = min(15, len(filtered_and_ranked))
            print(f"[DEBUG] Starting LLM re-ranking of top {num_to_rerank} documents")
            top_candidates = filtered_and_ranked[:num_to_rerank]
            
            rerank_prompt = f"""Score how relevant each document is to the question on a scale of 0-10.
Question: {question}

Documents to score:
"""
            for i, (doc, _, _, _, _) in enumerate(top_candidates):
                # Take first 300 chars of each doc for context
                content_preview = doc.page_content[:300].replace('\n', ' ')
                rerank_prompt += f"\nDoc {i+1}: {content_preview}...\n"
            
            rerank_prompt += "\nProvide ONLY the scores in format: Doc1:X, Doc2:Y, Doc3:Z, etc. No explanations."
            
            try:
                # Use non-streaming endpoint for faster, more reliable reranking
                # Get LLM URL from config to handle Docker environment
                import os
                llm_api_url = llm_cfg.get('main_llm', {}).get('model_server', 'http://localhost:11434')
                # Detect Docker environment more reliably
                is_docker = os.path.exists('/root') or os.environ.get('DOCKER_ENVIRONMENT')
                if 'localhost' in llm_api_url and is_docker:
                    llm_api_url = llm_api_url.replace('localhost', 'host.docker.internal')
                    print(f"[DEBUG] Docker environment detected, using Docker URL: {llm_api_url}")
                llm_api_url = f"{llm_api_url}/api/generate"
                
                # Get the main LLM model for reranking
                main_llm_config = llm_cfg.get('main_llm', {})
                model_name = main_llm_config.get('model', 'qwen3:0.6b')  # Use faster model for reranking
                
                payload = {
                    "model": model_name,
                    "prompt": rerank_prompt,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "options": {
                        "num_predict": 80,  # Ollama uses options.num_predict instead of max_tokens
                        "temperature": 0.1,
                        "top_p": 0.9
                    },
                    "stream": False
                }
                
                print(f"[DEBUG] LLM reranking using model: {model_name} at {llm_api_url}")
                
                scores_text = ""
                # Use centralized timeout configuration
                from app.core.timeout_settings_cache import get_timeout_value
                http_timeout = get_timeout_value("api_network", "http_request_timeout", 30)
                
                with httpx.Client(timeout=http_timeout) as client:
                    response = client.post(llm_api_url, json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        # Ollama returns response in different format
                        scores_text = result.get("response", "")
                    else:
                        print(f"[DEBUG] LLM API returned status {response.status_code}")
                        print(f"[DEBUG] Response content: {response.text[:200]}")
                        raise Exception(f"LLM API error: {response.status_code}")
                
                # Parse scores
                llm_scores = {}
                for part in scores_text.split(','):
                    if ':' in part:
                        doc_part, score_part = part.split(':', 1)
                        doc_digits = ''.join(filter(str.isdigit, doc_part))
                        if doc_digits:  # Only process if we have digits
                            try:
                                doc_num = int(doc_digits)
                                score_digits = ''.join(filter(lambda x: x.isdigit() or x == '.', score_part))
                                if score_digits:  # Only process if we have a score
                                    score = float(score_digits)
                                    llm_scores[doc_num] = score / 10.0  # Normalize to 0-1
                            except (ValueError, ZeroDivisionError):
                                continue  # Skip invalid entries
                
                print(f"[DEBUG] LLM scores: {llm_scores}")
                
                # Update scores with LLM re-ranking
                if llm_scores:
                    for i, item in enumerate(top_candidates):
                        if (i + 1) in llm_scores:
                            # Combine original score with LLM score (50/50 weight)
                            original_score = item[4]
                            llm_score = llm_scores[i + 1]
                            new_score = (original_score * 0.4) + (llm_score * 0.6)
                            # Update the tuple with new score
                            filtered_and_ranked[i] = (item[0], item[1], item[2], item[3], new_score)
                    
                    # Re-sort with updated scores
                    filtered_and_ranked.sort(key=lambda x: x[4], reverse=True)
                    print(f"[DEBUG] Re-ranking complete")
                    
            except Exception as e:
                print(f"[DEBUG] LLM re-ranking failed: {str(e)}")
                
        except Exception as e:
            print(f"[DEBUG] Qwen3-Reranker-4B re-ranking failed: {str(e)}")
    
    # Sort by final score
    filtered_and_ranked.sort(key=lambda x: x[4], reverse=True)
    
    # Get settings from proper RAG settings cache
    from app.core.rag_settings_cache import get_document_retrieval_settings
    doc_retrieval_settings = get_document_retrieval_settings()
    # For standard chat, use the max_documents_mcp setting which is designed for agent/chat responses
    # This is different from num_docs_retrieve which is for initial retrieval
    max_final_docs = doc_retrieval_settings.get('max_documents_mcp', 10)
    
    # Take top documents after reranking, but ensure diversity
    filtered_docs = []
    filtered_docs_with_scores = []  # Keep track of scores
    seen_content_hashes = set()
    
    for item in filtered_and_ranked:
        doc = item[0]
        # Use similarity score (item[2]) which is already normalized 0-1
        # or combined score (item[4]) which incorporates keyword relevance
        score = item[2]  # Using similarity score for consistency
        # Create a simple hash of the first 200 chars to avoid duplicate content
        content_hash = hash(doc.page_content[:200])
        
        if content_hash not in seen_content_hashes:
            seen_content_hashes.add(content_hash)
            filtered_docs.append(doc)
            filtered_docs_with_scores.append((doc, score))
            
        if len(filtered_docs) >= max_final_docs:  # Take up to configurable number of diverse documents
            break
    
    print(f"[DEBUG] handle_rag_query: After filtering and reranking: {len(filtered_docs)} documents")
    print(f"[DEBUG] handle_rag_query: Filtered {len(docs) - len(filtered_and_ranked)} documents by similarity threshold {SIMILARITY_THRESHOLD}")
    
    context = "\n\n".join([doc.page_content for doc in filtered_docs])
    
    # Extract sources with collection information and content
    # Use default content preview limit - this setting wasn't in the RAG config
    content_preview_limit = 500
    
    sources = []
    for i, doc in enumerate(filtered_docs):
        # Get the score from filtered_docs_with_scores
        score = filtered_docs_with_scores[i][1] if i < len(filtered_docs_with_scores) else 0.8
        source_info = {
            "content": doc.page_content[:content_preview_limit] + "..." if len(doc.page_content) > content_preview_limit else doc.page_content,
            "file": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", 0),
            "score": score,  # Include the actual similarity score
            "collection": doc.metadata.get("source_collection", "default_knowledge")
        }
        sources.append(source_info)
    
    # Debug: Check if scores are in sources
    print(f"[DEBUG] handle_rag_query: Sources with scores:")
    for i, src in enumerate(sources[:3]):  # Show first 3
        print(f"  - Source {i}: file={src.get('file', 'Unknown')}, score={src.get('score', 'None')}")
    
    # Enhanced relevance detection
    if context.strip():
        print(f"[RAG DEBUG] Query: {question}")
        print(f"[RAG DEBUG] Retrieved {len(filtered_docs)} documents")
        print(f"[RAG DEBUG] Context preview: {context[:200] if context else 'No context'}...")
        
        # Calculate relevance score
        relevance_score = calculate_relevance_score(question, context)
        
        # Get clean keywords for debug (same as used in calculation)
        query_keywords = set(question.lower().split())
        context_words = set(context.lower().split())
        stop_words = {
            'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an',
            'find', 'get', 'show', 'tell', 'about', 'info', 'information', 'news', 'what', 'how', 'when', 'where',
            'why', 'can', 'could', 'would', 'should', 'give', 'provide', 'need', 'want', 'like', 'please', 'help',
            'it', 'this', 'that', 'these', 'those'
        }
        clean_query_keywords = query_keywords - stop_words
        clean_context_words = context_words - stop_words
        
        print(f"[RAG DEBUG] Relevance score: {relevance_score:.2f}")
        print(f"[RAG DEBUG] Query keywords (filtered): {clean_query_keywords}")
        print(f"[RAG DEBUG] Context has {len(clean_context_words)} unique words")
        print(f"[RAG DEBUG] Overlapping words: {clean_query_keywords.intersection(clean_context_words)}")
        
        # If relevance is very low, return no context (will trigger LLM fallback)
        # Get relevance thresholds from proper RAG settings
        from app.core.rag_settings_cache import get_agent_settings
        agent_settings = get_agent_settings()
        
        # Dynamic threshold based on query complexity
        # Using complex_query_threshold from agent_settings as minimum relevance for complex queries
        min_relevance_complex = agent_settings.get('complex_query_threshold', 0.15)
        # Using min_relevance_score for simple queries
        min_relevance_simple = agent_settings.get('min_relevance_score', 0.25)
        # Consider query complex if it has more than 2 significant words
        query_complexity_threshold = 2
        
        min_relevance = min_relevance_complex if len(query_keywords) > query_complexity_threshold else min_relevance_simple
        if relevance_score < min_relevance:
            print(f"[RAG DEBUG] Low relevance detected ({relevance_score:.2f} < {min_relevance}), no context returned")
            result = ("", [])
            # Cache the result for performance
            # End RAG span with success
            if rag_span and tracer:
                try:
                    tracer.end_span_with_result(rag_span, {
                        "documents_found": len(filtered_docs),
                        "context_length": len(context),
                        "search_context_length": len(context)
                    }, True)
                except Exception as e:
                    logger.warning(f"Failed to end RAG span: {e}")
            
            # _cache_rag_result(query_hash, result, current_time)  # Cache disabled
            return result
        
        # Context is relevant, return it for hybrid processing
        print(f"[RAG DEBUG] Good relevance ({relevance_score:.2f}), returning context for hybrid approach")
        result = (context, sources)  # Return sources for frontend transparency
        
        # End RAG span with success
        if rag_span and tracer:
            try:
                tracer.end_span_with_result(rag_span, {
                    "documents_found": len([doc for doc, _, _, _, _ in filtered_and_ranked]) if 'filtered_and_ranked' in locals() else 0,
                    "context_length": len(context),
                    "search_context_length": len(sources)
                }, True)
            except Exception as e:
                logger.warning(f"Failed to end RAG span: {e}")
        
        # Cache the result for performance - DISABLED
        # _cache_rag_result(query_hash, result, current_time)
        return result
    else:
        print(f"[RAG DEBUG] No relevant documents found")
        
        # End RAG span with no relevant documents
        if rag_span and tracer:
            try:
                tracer.end_span_with_result(rag_span, {"documents_found": 0, "relevance_issue": True}, True)
            except Exception as e:
                logger.warning(f"Failed to end RAG span: {e}")
        
        result = ("", [])
        # Cache the result for performance - DISABLED
        # _cache_rag_result(query_hash, result, current_time)
        return result

def apply_reranking_to_sources(question: str, sources: list) -> list:
    """Apply reranking to sources obtained from tools (like MCP rag_knowledge_search)"""
    if not sources or len(sources) <= 1:
        return sources
    
    try:
        # Import here to avoid circular imports
        from app.rag.qwen_reranker import get_qwen_reranker
        from app.core.rag_settings_cache import get_reranking_settings
        
        # Check if reranker is enabled
        try:
            rerank_settings = get_reranking_settings()
            enable_qwen_reranker = rerank_settings.get('enable_qwen_reranker', True)
        except:
            enable_qwen_reranker = True
        
        if not enable_qwen_reranker:
            print(f"[DEBUG] Qwen reranker disabled, skipping tool results reranking")
            return sources
        
        # Convert sources to format expected by reranker
        class MockDoc:
            def __init__(self, content):
                self.page_content = content
        
        documents_for_reranking = []
        for source in sources:
            doc = MockDoc(source.get('content', ''))
            score = source.get('score', 0.8)
            documents_for_reranking.append((doc, score))
        
        print(f"[DEBUG] Applying reranking to {len(documents_for_reranking)} tool result documents")
        
        # Try to get reranker instance
        reranker = get_qwen_reranker()
        if reranker:
            # Use reranker
            rerank_results = reranker.rerank(
                query=question,
                documents=documents_for_reranking,
                top_k=len(documents_for_reranking)
            )
            
            # Convert back to sources format
            reranked_sources = []
            for result in rerank_results:
                # Find original source that matches this content
                original_content = result.document.page_content
                for source in sources:
                    if source.get('content', '') == original_content:
                        # Create new source with reranked score
                        reranked_source = source.copy()
                        reranked_source['score'] = result.score
                        reranked_sources.append(reranked_source)
                        break
            
            print(f"[DEBUG] Reranked {len(reranked_sources)} tool result documents using Qwen3-Reranker-4B")
            return reranked_sources
        else:
            print(f"[DEBUG] Qwen reranker not available for tool results, using original order")
            return sources
            
    except Exception as e:
        print(f"[DEBUG] Tool results reranking failed: {e}")
        return sources

def make_llm_call(prompt: str, thinking: bool, context: str, llm_cfg: dict) -> str:
    """Make a synchronous LLM API call and return the complete response"""
    llm_api_url = "http://localhost:8000/api/v1/generate_stream"
    
    # Determine which LLM config to use based on what's passed
    if 'main_llm' in llm_cfg or 'second_llm' in llm_cfg or 'query_classifier' in llm_cfg:
        # Full settings dict passed, need to determine which config to use
        main_llm_settings = llm_cfg
        # Check if caller specified which LLM to use via thinking parameter or other means
        # For now, default to main_llm config
        mode_config = get_main_llm_full_config(main_llm_settings)
    else:
        # Specific LLM config passed directly (e.g., from second_llm)
        mode_config = llm_cfg
    
    # Get system_prompt if available and prepend to prompt
    system_prompt = mode_config.get("system_prompt", "")
    if system_prompt:
        prompt = f"{system_prompt}\n\n{prompt}"
        print(f"[DEBUG make_llm_call] System prompt found and prepended: {system_prompt[:100]}...")
    else:
        print(f"[DEBUG make_llm_call] No system prompt found in config")
    
    # Get max_tokens from config
    max_tokens_raw = mode_config.get("max_tokens", 16384)
    try:
        max_tokens = int(max_tokens_raw)
        if max_tokens > 32768:
            max_tokens = 16384
    except (ValueError, TypeError):
        max_tokens = 16384
    
    payload = {
        "prompt": prompt,
        "temperature": mode_config.get("temperature", 0.7),
        "top_p": mode_config.get("top_p", 1.0),
        "max_tokens": max_tokens
    }
    
    # Make synchronous request and collect all tokens
    text = ""
    try:
        # Use centralized timeout configuration  
        from app.core.timeout_settings_cache import get_timeout_value
        http_timeout = get_timeout_value("api_network", "http_streaming_timeout", 120)
        
        with httpx.Client(timeout=http_timeout) as client:
            with client.stream("POST", llm_api_url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    if line.startswith("data: "):
                        token = line.replace("data: ", "")
                        if token.strip():
                            text += token
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        return f"Error: {str(e)}"
    
    return text

def streaming_llm_call(prompt: str, thinking: bool, context: str, conversation_id: str, llm_cfg: dict):
    """Generator for streaming LLM responses"""
    llm_api_url = "http://localhost:8000/api/v1/generate_stream"
    # CRITICAL FIX: Ensure we get main LLM config, not query classifier config
    main_llm_settings = get_llm_settings()
    mode_config = get_main_llm_full_config(main_llm_settings)
    llm_cfg = main_llm_settings  # Override llm_cfg with clean main LLM settings
    
    # Get max_tokens from config
    max_tokens_raw = llm_cfg.get("max_tokens", 16384)
    try:
        max_tokens = int(max_tokens_raw)
        if max_tokens > 32768:
            max_tokens = 16384
    except (ValueError, TypeError):
        max_tokens = 16384
    
    payload = {
        "prompt": prompt,
        "temperature": mode_config.get("temperature", 0.7),
        "top_p": mode_config.get("top_p", 1.0),
        "max_tokens": max_tokens
    }
    
    def token_stream():
        text = ""
        try:
            # Use centralized timeout configuration  
            from app.core.timeout_settings_cache import get_timeout_value
            http_timeout = get_timeout_value("api_network", "http_streaming_timeout", 120)
            
            with httpx.Client(timeout=http_timeout) as client:
                with client.stream("POST", llm_api_url, json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        if isinstance(line, bytes):
                            line = line.decode("utf-8")
                        if line.startswith("data: "):
                            token = line.replace("data: ", "")
                            if token.strip():
                                text += token
                                yield json.dumps({"token": token}) + "\n"
                    
                    # Send completion event with full response
                    yield json.dumps({
                        "answer": text,
                        "source": "LLM",
                        "conversation_id": conversation_id
                    }) + "\n"
                    
                    # Store assistant response in conversation history
                    if conversation_id and text:
                        store_conversation_message(conversation_id, "assistant", text)
                        
        except Exception as e:
            print(f"[ERROR] Streaming LLM call failed: {e}")
            yield json.dumps({
                "error": f"LLM streaming failed: {str(e)}",
                "source": "ERROR"
            }) + "\n"
    
    return token_stream()

def get_llm_response_direct(question: str, thinking: bool = False, stream: bool = False, conversation_id: str = None):
    """Get direct LLM response without RAG retrieval
    
    Args:
        question: The user's question
        thinking: Whether to enable extended thinking
        stream: Whether to stream the response
        conversation_id: Conversation ID for context
    """
    print(f"[DEBUG] get_llm_response_direct: question = {question}")
    print(f"[DEBUG] get_llm_response_direct: conversation_id = {conversation_id}")
    
    # Store the user's question in conversation history
    if conversation_id:
        store_conversation_message(conversation_id, "user", question)
    
    # Get LLM settings and validate - check for both old and new schema
    llm_cfg = get_llm_settings()
    
    # Check for new schema first
    if 'main_llm' in llm_cfg and 'thinking_mode_params' in llm_cfg and 'non_thinking_mode_params' in llm_cfg:
        # New schema - validate main_llm structure
        main_llm = llm_cfg.get('main_llm', {})
        required_main_fields = ["model", "max_tokens"]
        missing_main = [f for f in required_main_fields if f not in main_llm or main_llm[f] is None]
        if missing_main:
            raise RuntimeError(f"Missing required main_llm config fields: {', '.join(missing_main)}")
    else:
        # Old schema - check for thinking_mode and non_thinking_mode
        required_fields = ["model", "thinking_mode", "non_thinking_mode", "max_tokens"]
        missing = [f for f in required_fields if f not in llm_cfg or llm_cfg[f] is None]
        if missing:
            raise RuntimeError(f"Missing required LLM config fields: {', '.join(missing)}")
    
    # Get conversation history
    conversation_history = get_conversation_history(conversation_id) if conversation_id else ""
    
    # Build prompt without RAG context
    prompt = build_prompt(question, thinking)
    if conversation_history:
        prompt = f"Previous conversation:\n{conversation_history}\n\nCurrent question: {prompt}"
    
    # Make LLM call
    if stream:
        return streaming_llm_call(prompt, thinking, "", conversation_id, llm_cfg)
    else:
        response = make_llm_call(prompt, thinking, "", llm_cfg)
        
        # Store assistant response
        if conversation_id:
            store_conversation_message(conversation_id, "assistant", response)
        
        return response

async def rag_answer(question: str, thinking: bool = False, stream: bool = False, conversation_id: str = None, use_langgraph: bool = True, collections: List[str] = None, collection_strategy: str = "auto", query_type: str = None, trace=None, hybrid_context=None):
    """Main function with hybrid RAG+LLM approach prioritizing answer quality
    
    Args:
        question: The user's question
        thinking: Whether to enable extended thinking
        stream: Whether to stream the response
        conversation_id: Conversation ID for context
        use_langgraph: Whether to use LangGraph agents
        collections: List of collection names to search (None = auto-detect)
        collection_strategy: "auto", "specific", or "all"
        query_type: Pre-computed query type (RAG/TOOLS/LLM) to skip classification
        hybrid_context: Hybrid RAG context from orchestrator (includes strategy and results)
    """
    print(f"[DEBUG] rag_answer: incoming question = {question}")
    print(f"[DEBUG] rag_answer: conversation_id = {conversation_id}")
    print(f"[DEBUG] rag_answer: collections = {collections}, strategy = {collection_strategy}")
    
    # Log hybrid context if available
    if hybrid_context:
        print(f"[DEBUG] rag_answer: hybrid_context present with strategy = {hybrid_context.get('strategy', 'unknown')}")
        print(f"[DEBUG] rag_answer: hybrid sources = {hybrid_context.get('sources', [])}")
    else:
        print(f"[DEBUG] rag_answer: no hybrid_context provided")
    
    # Store the user's question in conversation history
    if conversation_id:
        store_conversation_message(conversation_id, "user", question)
        
        # Detect if this is a simple query after complex generation task
        # and clear context bleeding if needed
        question_lower = question.lower()
        if any(simple_indicator in question_lower for simple_indicator in [
            'what is', 'who is', 'when is', 'where is', 'how much', 'what time',
            'current', 'today', 'now', 'date', 'time'
        ]) and len(question.split()) <= 10:
            # This looks like a simple factual query, ensure no complex context bleeding
            print(f"[DEBUG] Detected simple query after potential complex task, using limited context")
    
    # Retrieve conversation history early to enhance question for tool execution
    enhanced_question = question
    if conversation_id:
        # Get conversation history limits from settings to avoid hardcoding
        llm_settings = get_llm_settings()
        conversation_config = llm_settings.get('conversation_settings', {})
        max_history_messages = conversation_config.get('max_history_messages', 3)
        
        # Use configurable conversation history to prevent context bleeding
        early_conversation_history = get_limited_conversation_history(
            conversation_id, 
            max_messages=max_history_messages,
            current_query=question
        )
        
        if early_conversation_history:
            # Create enhanced question with conversation context
            enhanced_question = f"Previous conversation:\n{early_conversation_history}\n\nCurrent question: {question}"
            print(f"[DEBUG] rag_answer: Enhanced question with conversation context for tool execution")
    
    # Temporarily disable LangGraph due to Redis initialization issues
    use_langgraph = False
    
    # Use LangGraph implementation if enabled
    if use_langgraph:
        try:
            from app.langchain.langgraph_rag import enhanced_rag_answer
            print(f"[DEBUG] Using LangGraph enhanced RAG implementation")
            return enhanced_rag_answer(
                question=question,
                conversation_id=conversation_id,
                thinking=thinking,
                stream=stream
            )
        except Exception as e:
            print(f"[ERROR] LangGraph implementation failed: {str(e)}")
            print(f"[DEBUG] Falling back to original implementation")
    
    # Original implementation follows...
    # Validate configuration - check for both old and new schema
    llm_cfg = get_llm_settings()
    
    # Check for new schema first
    if 'main_llm' in llm_cfg and 'thinking_mode_params' in llm_cfg and 'non_thinking_mode_params' in llm_cfg:
        # New schema - validate main_llm structure
        main_llm = llm_cfg.get('main_llm', {})
        required_main_fields = ["model", "max_tokens"]
        missing_main = [f for f in required_main_fields if f not in main_llm or main_llm[f] is None]
        if missing_main:
            raise RuntimeError(f"Missing required main_llm config fields: {', '.join(missing_main)}")
    else:
        # Old schema - check for thinking_mode and non_thinking_mode
        required_fields = ["model", "thinking_mode", "non_thinking_mode", "max_tokens"]
        missing = [f for f in required_fields if f not in llm_cfg or llm_cfg[f] is None]
        if missing:
            raise RuntimeError(f"Missing required LLM config fields: {', '.join(missing)}")
    
    # Initialize rag_sources at the top level to ensure it's always available
    rag_sources = []
    
    # If hybrid_context is provided, extract sources from it
    if hybrid_context and 'sources' in hybrid_context:
        print(f"[DEBUG] rag_answer: Extracting sources from hybrid_context")
        for source in hybrid_context.get('sources', []):
            # Mark sources from hybrid context as temporary documents
            source_info = {
                "content": source.get('content', ''),
                "file": source.get('filename', source.get('source', 'Unknown')),
                "page": source.get('page', 0),
                "score": source.get('score', 0.8),  # Preserve the actual score
                "collection": "temp_documents",  # Mark as temp documents
                "is_temporary": True,  # Explicit flag
                "metadata": source.get('metadata', {})
            }
            rag_sources.append(source_info)
        print(f"[DEBUG] rag_answer: Extracted {len(rag_sources)} sources from hybrid_context")
    
    # STEP 1: Use pre-computed query_type if provided, otherwise classify
    print(f"[DEBUG] rag_answer: RECEIVED query_type parameter = '{query_type}'")
    if query_type:
        # Use pre-computed classification from API layer to avoid double work
        query_type = query_type.upper()  # Normalize case
        print(f"[DEBUG] rag_answer: Using pre-computed query_type = {query_type}")
    else:
        # Use efficient classification to avoid unnecessary RAG calls
        query_type = await classify_query_type_efficient(question, llm_cfg)
        print(f"[DEBUG] rag_answer: query_type = {query_type}")
    
    # STEP 2: If large generation, check if chunking is actually needed
    if query_type == "LARGE_GENERATION":
        large_output_analysis = detect_large_output_potential(question)
        target_count = large_output_analysis["estimated_items"]
        
        # CRITICAL FIX: Only use chunking for truly large requests (>100 items)
        if target_count <= 100:
            print(f"[DEBUG] rag_answer: {target_count} items is manageable - using single LLM call instead of chunking")
            # Convert to regular LLM processing for better quality
            query_type = "LLM"  # Override to use single call
            rag_context = ""
        else:
            print(f"[DEBUG] rag_answer: {target_count} items requires chunking - proceeding with chunked processing")
            rag_context = ""  # Skip RAG entirely for large generation to avoid delays
            print(f"[DEBUG] rag_answer: Proceeding without RAG context for chunked processing")
    else:
        # Initialize tool variables for potential fallback use
        tool_calls = []
        tool_context = ""
        # rag_sources already initialized at top level
        
        # STEP 2a: Enhanced hybrid approach - always consider both RAG and TOOLS
        if query_type == "TOOLS":
            print(f"[DEBUG] rag_answer: TOOLS query detected - executing tools, then LLM synthesis")
            rag_context = ""  # Start with no RAG, but will execute tools and synthesize
            
            # ENHANCED: Always execute tools and prepare for synthesis
            try:
                print(f"[DEBUG] rag_answer: Executing tools for TOOLS query")
                
                # DIRECT EXECUTION BYPASS for high-confidence tool queries
                search_results = []
                search_context = ""
                
                try:
                    from app.core.query_classifier_settings_cache import get_query_classifier_settings
                    from app.langchain.enhanced_query_classifier import EnhancedQueryClassifier
                    
                    classifier_settings = get_query_classifier_settings()
                    threshold = classifier_settings.get("direct_execution_threshold", 0.6)
                    
                    classifier = EnhancedQueryClassifier()
                    results = await classifier.classify(question)
                    
                    
                    if results and results[0].confidence >= threshold and results[0].suggested_tools:
                        # Use LLM-suggested tool (first one is most relevant)
                        suggested_tools = results[0].suggested_tools
                        tool_name = suggested_tools[0]  # LLM already ordered by relevance
                        confidence = results[0].confidence
                        
                        print(f"[DEBUG] rag_answer: DIRECT BYPASS triggered - {tool_name} (confidence: {confidence:.3f}) from LLM suggestions: {suggested_tools}")
                        
                        # Execute tool directly without intelligent planning
                        direct_result = call_mcp_tool(tool_name, {}, trace=trace)
                        
                        if direct_result and not direct_result.get("error"):
                            # Create search_results in the expected format
                            search_results = [{
                                "tool": tool_name,
                                "success": True,
                                "result": direct_result,
                                "execution_time": 0.5
                            }]
                            search_context = f"\n{tool_name}: {json.dumps(direct_result, indent=2) if isinstance(direct_result, dict) else direct_result}\n"
                            print(f"[DEBUG] rag_answer: DIRECT EXECUTION SUCCESS - bypassing execute_tools_first")
                        else:
                            print(f"[DEBUG] rag_answer: Direct execution failed, falling back to execute_tools_first")
                except Exception as e:
                    print(f"[DEBUG] rag_answer: Direct bypass failed: {e}, falling back to execute_tools_first")
                
                # Only run intelligent planning if direct execution didn't work
                print(f"[DEBUG] rag_answer: Checking if intelligent planning needed. search_results: {len(search_results) if search_results else 0}")
                if not search_results:
                    print(f"[DEBUG] rag_answer: Running intelligent planning since search_results is empty")
                    search_results, search_context = await execute_intelligent_planning(enhanced_question, thinking, trace)
                else:
                    print(f"[DEBUG] rag_answer: Skipping intelligent planning - direct execution already provided {len(search_results)} results")
                if search_results and search_context:
                    print(f"[DEBUG] rag_answer: TOOLS execution successful, will synthesize with LLM")
                    tool_calls = search_results
                    tool_context = search_context
                    
                    # Extract documents from knowledge_search tool results
                    for result in search_results:
                        if is_rag_tool(result.get('tool', '')) and result.get('success'):
                            tool_result = result.get('result', {})
                            # Handle JSON-RPC response format
                            if isinstance(tool_result, dict) and 'result' in tool_result:
                                rag_result = tool_result['result']
                                if isinstance(rag_result, dict) and 'documents' in rag_result:
                                    # Replace existing rag_sources with tool results (don't append to avoid stacking)
                                    tool_rag_sources = []
                                    for doc in rag_result['documents']:
                                        if isinstance(doc, dict):
                                            source_info = {
                                                "content": doc.get('content', ''),
                                                "file": doc.get('source', 'Unknown'),
                                                "page": doc.get('metadata', {}).get('page', 0),
                                                "score": doc.get('relevance_score', doc.get('score', 0.8)),  # Add score from tool results
                                                "collection": doc.get('collection', 'default_knowledge')
                                            }
                                            tool_rag_sources.append(source_info)
                                    
                                    # Apply reranking to tool results if available
                                    rag_sources = apply_reranking_to_sources(question, tool_rag_sources)
                                    print(f"[DEBUG] rag_answer: Replaced rag_sources with {len(rag_sources)} documents from {get_rag_tool_name()} results (reranked)")
                    
                    # Keep query_type as TOOLS but ensure synthesis happens
                else:
                    print(f"[DEBUG] rag_answer: TOOLS execution returned no results, falling back to LLM")
            except Exception as e:
                print(f"[DEBUG] rag_answer: TOOLS execution failed: {e}, falling back to LLM")
        elif query_type == "LLM":
            print(f"[DEBUG] rag_answer: LLM query detected - checking for tool enhancement opportunities")
            rag_context = ""  # Start with no RAG, but may add tool enhancement
            
            # ENHANCEMENT: Check if query could benefit from current information
            # Get keywords from LLM settings to avoid hardcoding
            llm_settings = get_llm_settings()
            enhancement_config = llm_settings.get('tool_enhancement', {})
            current_info_keywords = enhancement_config.get('current_info_keywords', [
                'latest', 'current', 'recent', 'today', 'now', 'news', 'price', 'weather', 'stock'
            ])
            enable_llm_tool_enhancement = enhancement_config.get('enable_llm_enhancement', True)
            
            if enable_llm_tool_enhancement and any(keyword in question.lower() for keyword in current_info_keywords):
                print(f"[DEBUG] rag_answer: LLM query contains current info keywords, adding tool enhancement")
                try:
                    # Execute search tools to get current information
                    search_results, search_context = await execute_intelligent_planning(enhanced_question, thinking, trace)
                    if search_results and search_context:
                        print(f"[DEBUG] rag_answer: Tool enhancement successful for LLM query")
                        tool_calls = search_results
                        tool_context = search_context
                        query_type = "HYBRID_LLM_TOOLS"  # Mark as hybrid for synthesis
                except Exception as e:
                    print(f"[DEBUG] rag_answer: Tool enhancement failed for LLM query: {e}")
        elif query_type == "RAG":
            # Check if we already have sources from hybrid_context
            if hybrid_context and rag_sources:
                print(f"[DEBUG] rag_answer: RAG query with hybrid_context - skipping duplicate RAG retrieval")
                print(f"[DEBUG] rag_answer: Using {len(rag_sources)} sources from hybrid_context")
                rag_context = ""  # Context is already in the enhanced question
            else:
                # Only RAG queries need internal knowledge retrieval
                print(f"[DEBUG] rag_answer: RAG query detected - performing knowledge retrieval")
                import time
                rag_start_time = time.time()
                rag_context, rag_sources = handle_rag_query(question, thinking, collections, collection_strategy, trace=trace)
                rag_end_time = time.time()
                print(f"[DEBUG] rag_answer: RAG retrieval took {rag_end_time - rag_start_time:.2f} seconds")
                print(f"[DEBUG] rag_answer: RAG context length = {len(rag_context) if rag_context else 0}")
        elif query_type == "SYNTHESIS":
            print(f"[DEBUG] rag_answer: SYNTHESIS query detected - skipping RAG retrieval, using provided context")
            print(f"[EXECUTION TRACKER] SYNTHESIS mode: No additional RAG search needed")
            # Check if pre_formatted_context is provided in hybrid_context
            if hybrid_context and 'pre_formatted_context' in hybrid_context:
                rag_context = hybrid_context['pre_formatted_context']
                print(f"[DEBUG] rag_answer: Using pre_formatted_context from hybrid_context, length: {len(rag_context)}")
            else:
                rag_context = ""  # No additional RAG needed
            # rag_sources already extracted from hybrid_context above
        else:
            # Unknown query type, default to RAG for safety
            print(f"[DEBUG] rag_answer: Unknown query_type '{query_type}' - defaulting to RAG retrieval")
            import time
            rag_start_time = time.time()
            rag_context, rag_sources = handle_rag_query(question, thinking, collections, collection_strategy, trace=trace)
            rag_end_time = time.time()
            print(f"[DEBUG] rag_answer: RAG retrieval took {rag_end_time - rag_start_time:.2f} seconds")
            print(f"[DEBUG] rag_answer: RAG context length = {len(rag_context) if rag_context else 0}")
            
            # INTELLIGENT FALLBACK: If RAG finds no relevant docs, try web search for current info
            if not rag_context or len(rag_context.strip()) < 50:  # No meaningful RAG context
                print(f"[DEBUG] rag_answer: No relevant RAG context found, attempting web search fallback")
                try:
                    # Execute google_search tool to get current information
                    search_results, search_context = await execute_intelligent_planning(enhanced_question, thinking, trace)
                    if search_results and search_context:
                        print(f"[DEBUG] rag_answer: Web search fallback successful, got {len(search_context)} chars")
                        # Keep original empty rag_context, but we'll have search_context for synthesis
                        # This will be handled in unified synthesis as RAG+TOOLS+LLM
                        # Set tool context for later synthesis
                        tool_calls = search_results
                        tool_context = search_context
                        query_type = "TOOLS"  # Change classification to include tools
                        print(f"[DEBUG] rag_answer: Updated query_type to TOOLS due to web search fallback")
                    else:
                        print(f"[DEBUG] rag_answer: Web search fallback failed, proceeding with LLM-only response")
                except Exception as e:
                    print(f"[DEBUG] rag_answer: Web search fallback error: {e}, proceeding with LLM-only response")
    
    # STEP 3: Handle large generation immediately if detected
    if query_type == "LARGE_GENERATION":
        print(f"[DEBUG] rag_answer: Routing to chunked large generation system")
        
        # Check if LLM settings are configured before proceeding
        try:
            llm_settings_test = get_llm_settings()
            if not llm_settings_test:
                raise RuntimeError("LLM settings not configured")
        except Exception as e:
            print(f"[ERROR] Cannot use chunked generation: {e}")
            # Fall back to regular LLM processing with a helpful message
            error_message = "⚠️ Large generation detected but LLM settings not configured. Please configure LLM settings in the Settings page first."
            
            if stream:
                def error_stream():
                    yield json.dumps({"token": error_message}) + "\n"
                    yield json.dumps({
                        "answer": error_message,
                        "source": "ERROR",
                        "context": "",
                        "query_type": "ERROR"
                    }) + "\n"
                return error_stream()
            else:
                return {
                    "answer": error_message,
                    "source": "ERROR",
                    "context": "",
                    "query_type": "ERROR"
                }
        
        # Import here to avoid circular imports
        from app.langchain.multi_agent_system_simple import MultiAgentSystem
        
        # Get large output analysis for parameters
        large_output_analysis = detect_large_output_potential(question)
        target_count = large_output_analysis["estimated_items"]
        
        # Create multi-agent system for chunked processing
        try:
            system = MultiAgentSystem(conversation_id=conversation_id)
        except Exception as e:
            print(f"[ERROR] Failed to create MultiAgentSystem: {e}")
            # Fall back with error message
            error_message = f"⚠️ Chunked generation failed to initialize: {str(e)}"
            
            if stream:
                def error_stream():
                    yield json.dumps({"token": error_message}) + "\n"
                    yield json.dumps({
                        "answer": error_message,
                        "source": "ERROR",
                        "context": "",
                        "query_type": "ERROR"
                    }) + "\n"
                return error_stream()
            else:
                return {
                    "answer": error_message,
                    "source": "ERROR",
                    "context": "",
                    "query_type": "ERROR"
                }
        
        # Transform question into a generation task with RAG context if available
        if rag_context:
            enhanced_query = f"""Based on the following context from our knowledge base, {question}

Context from knowledge base:
{rag_context}

Please generate the requested items incorporating relevant information from the context above."""
        else:
            enhanced_query = question
        
        # Get conversation history limits from settings to avoid hardcoding
        llm_settings = get_llm_settings()
        conversation_config = llm_settings.get('conversation_settings', {})
        max_history_messages = conversation_config.get('max_history_messages', 3)
        
        # Get LIMITED conversation history for context to prevent bleeding
        conversation_history_text = get_limited_conversation_history(
            conversation_id, 
            max_messages=max_history_messages,
            current_query=question
        ) if conversation_id else ""
        conversation_history_list = get_full_conversation_history(conversation_id) if conversation_id else []
        # Also limit the list version to prevent bleeding (2 messages per interaction = user + assistant)
        max_list_messages = max_history_messages * 2
        if len(conversation_history_list) > max_list_messages:
            conversation_history_list = conversation_history_list[-max_list_messages:]
        
        # Stream chunked generation events, converting them to RAG format
        def chunked_generation_stream():
            import asyncio
            async def run_chunked():
                async for event in system.stream_large_generation_events(
                    query=enhanced_query,
                    target_count=target_count,
                    conversation_history=conversation_history_list
                ):
                    # Convert chunked events to RAG streaming format
                    print(f"[DEBUG] Processing chunked event: type={event.get('type')}, keys={list(event.keys())}")
                    
                    if event.get("type") == "chunk_completed":
                        # Stream chunk content as tokens
                        content = event.get("content", "")
                        print(f"[DEBUG] chunk_completed event - content length: {len(content)}")
                        if content:
                            print(f"[DEBUG] Streaming chunk content: {content[:100]}...")
                            # Stream content more naturally as sentences/lines rather than individual words
                            sentences = content.replace('\n', ' ').split('. ')
                            for sentence in sentences:
                                if sentence.strip():
                                    yield json.dumps({"token": sentence.strip() + ". "}) + "\n"
                    elif event.get("type") == "task_completed":
                        # This is the actual final completion from Redis manager
                        final_results = event.get("final_results", [])
                        answer = "\n".join(final_results)
                        
                        print(f"[DEBUG] Task completed with {len(final_results)} items")
                        
                        # Store assistant's response in conversation history
                        if conversation_id and answer:
                            store_conversation_message(conversation_id, "assistant", answer)
                        
                        yield json.dumps({
                            "answer": answer,
                            "source": "RAG+CHUNKED",
                            "context": rag_context,
                            "query_type": "RAG+CHUNKED",
                            "metadata": {
                                "target_count": target_count,
                                "actual_count": len(final_results),
                                "execution_summary": event.get("summary", {}),
                                "quality_report": event.get("quality_report", {})
                            }
                        }) + "\n"
                    elif event.get("type") == "large_generation_completed":
                        # Handle the quality-checked final completion
                        final_results = event.get("final_results", [])
                        answer = "\n".join(final_results)
                        
                        print(f"[DEBUG] Large generation completed with {len(final_results)} items")
                        
                        # Store assistant's response in conversation history
                        if conversation_id and answer:
                            store_conversation_message(conversation_id, "assistant", answer)
                        
                        yield json.dumps({
                            "answer": answer,
                            "source": "RAG+CHUNKED",
                            "context": rag_context,
                            "query_type": "RAG+CHUNKED",
                            "metadata": {
                                "target_count": target_count,
                                "actual_count": len(final_results),
                                "execution_summary": event.get("execution_summary", {}),
                                "quality_report": event.get("quality_report", {})
                            }
                        }) + "\n"
                    elif event.get("type") in ["decomposition_started", "task_decomposed", "execution_started", "chunk_started"]:
                        # Progress events - send as progress metadata, not content tokens
                        chunk_number = event.get('chunk_number', event.get('total_chunks', '...'))
                        yield json.dumps({
                            "progress": f"Processing chunk {chunk_number}",
                            "chunk_number": chunk_number,
                            "total_chunks": event.get('total_chunks', ''),
                            "type": "progress"
                        }) + "\n"
                    elif event.get("type") in ["chunk_failed", "large_generation_error"]:
                        # Error events - send as error metadata, not content tokens
                        yield json.dumps({
                            "error": event.get('error', 'Unknown error'),
                            "type": "error"
                        }) + "\n"
                    else:
                        # Log unhandled events for debugging
                        print(f"[DEBUG] Unhandled chunked event: {event.get('type')} - {event}")
                        
                        # Handle any other completion events we might have missed
                        if 'completed' in event.get('type', ''):
                            final_results = event.get("final_results", [])
                            if final_results:
                                answer = "\n".join(final_results)
                                print(f"[DEBUG] Found completion event with {len(final_results)} items")
                                
                                # Store assistant's response in conversation history
                                if conversation_id and answer:
                                    store_conversation_message(conversation_id, "assistant", answer)
                                
                                yield json.dumps({
                                    "answer": answer,
                                    "source": "RAG+CHUNKED",
                                    "context": rag_context,
                                    "query_type": "RAG+CHUNKED",
                                    "metadata": {
                                        "target_count": target_count,
                                        "actual_count": len(final_results),
                                        "execution_summary": event.get("summary", {}),
                                        "event_type": event.get('type')
                                    }
                                }) + "\n"
            
            # Run async generator in sync context with timeout
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async_gen = run_chunked()
                timeout_counter = 0
                max_timeout = 300  # 5 minutes total timeout
                
                while True:
                    try:
                        # Use a shorter timeout per iteration to avoid hanging
                        event = loop.run_until_complete(
                            asyncio.wait_for(async_gen.__anext__(), timeout=30)
                        )
                        yield event
                        timeout_counter = 0  # Reset timeout counter on successful event
                    except StopAsyncIteration:
                        print(f"[DEBUG] Chunked generation completed normally")
                        break
                    except asyncio.TimeoutError:
                        timeout_counter += 30
                        print(f"[DEBUG] Chunked generation timeout: {timeout_counter}/{max_timeout}s")
                        if timeout_counter >= max_timeout:
                            print(f"[ERROR] Chunked generation timed out after {max_timeout}s")
                            yield json.dumps({
                            "progress": f"Timeout: Generation stopped after {max_timeout}s",
                            "type": "timeout_error"
                        }) + "\n"
                            yield json.dumps({
                                "answer": "Generation timed out. This may be due to missing LLM configuration or system issues.",
                                "source": "ERROR",
                                "context": "",
                                "query_type": "ERROR"
                            }) + "\n"
                            break
                        else:
                            # Continue waiting but show progress - send as progress metadata
                            yield json.dumps({
                                "progress": f"Still processing... {timeout_counter}s",
                                "timeout_counter": timeout_counter,
                                "type": "timeout_progress"
                            }) + "\n"
                    except Exception as e:
                        print(f"[ERROR] Chunked generation exception: {e}")
                        yield json.dumps({
                            "error": str(e),
                            "type": "exception_error"
                        }) + "\n"
                        yield json.dumps({
                            "answer": f"Generation failed: {str(e)}",
                            "source": "ERROR", 
                            "context": "",
                            "query_type": "ERROR"
                        }) + "\n"
                        break
            finally:
                loop.close()
        
        if stream:
            return chunked_generation_stream()
        else:
            # For non-stream mode, collect all results
            all_results = []
            for chunk in chunked_generation_stream():
                if chunk.strip():
                    try:
                        data = json.loads(chunk)
                        if "answer" in data:
                            return {
                                "answer": data["answer"],
                                "context": data.get("context", ""),
                                "source": data["source"],
                                "query_type": data["query_type"],
                                "metadata": data.get("metadata", {})
                            }
                    except json.JSONDecodeError:
                        continue
            
            # Fallback if no final answer found
            return {
                "answer": "Large generation task completed. Please check the streaming output for results.",
                "context": rag_context,
                "source": "RAG+CHUNKED", 
                "query_type": "RAG+CHUNKED"
            }
    
    # STEP 3: Check if tools can add value (for non-large generation queries)
    print(f"[DEBUG] rag_answer: Step 3 - Checking tool applicability")
    print(f"[DEBUG] rag_answer: After Step 3 log, query_type={query_type}")
    
    # Initialize tool variables if not already set by fallback
    print(f"[DEBUG] rag_answer: Initializing tool variables")
    if 'tool_calls' not in locals():
        tool_calls = []
        tool_context = ""
        print(f"[DEBUG] rag_answer: Initialized empty tool_calls and tool_context")
    else:
        print(f"[DEBUG] rag_answer: tool_calls already exists with {len(tool_calls)} items")
    
    print(f"[DEBUG] rag_answer: About to check if query_type == 'TOOLS' (current: {query_type})")
    if query_type == "TOOLS":
        # Only execute tools if not already executed in web search fallback
        if not tool_calls:  # Check if fallback already populated these
            
            # DIRECT EXECUTION BYPASS for high-confidence tool queries
            print(f"[DEBUG] rag_answer: Checking for direct execution bypass")
            try:
                from app.core.query_classifier_settings_cache import get_query_classifier_settings
                from app.langchain.enhanced_query_classifier import EnhancedQueryClassifier
                
                classifier_settings = get_query_classifier_settings()
                threshold = classifier_settings.get("direct_execution_threshold", 0.6)
                
                classifier = EnhancedQueryClassifier()
                results = await classifier.classify(question)
                
                if results and results[0].confidence >= threshold and results[0].suggested_tools:
                    tool_name = results[0].suggested_tools[0]
                    confidence = results[0].confidence
                    print(f"[DEBUG] rag_answer: DIRECT BYPASS triggered - {tool_name} (confidence: {confidence:.3f})")
                    
                    # Execute tool directly without intelligent planning
                    direct_result = call_mcp_tool(tool_name, {}, trace=trace)
                    
                    if direct_result and not direct_result.get("error"):
                        # Populate tool_calls to skip intelligent executor
                        tool_calls = [{
                            "tool": tool_name,
                            "success": True,
                            "result": direct_result,
                            "execution_time": 0.5
                        }]
                        tool_context = f"\n{tool_name}: {json.dumps(direct_result, indent=2) if isinstance(direct_result, dict) else direct_result}\n"
                        
                        # Extract documents from direct tool call results
                        if is_rag_tool(tool_name):
                            # Handle JSON-RPC response format
                            if isinstance(direct_result, dict) and 'result' in direct_result:
                                rag_result = direct_result['result']
                                if isinstance(rag_result, dict) and 'documents' in rag_result:
                                    # Replace existing rag_sources with tool results (don't append to avoid stacking)
                                    tool_rag_sources = []
                                    for doc in rag_result['documents']:
                                        if isinstance(doc, dict):
                                            source_info = {
                                                "content": doc.get('content', ''),
                                                "file": doc.get('source', 'Unknown'),
                                                "page": doc.get('metadata', {}).get('page', 0),
                                                "score": doc.get('relevance_score', doc.get('score', 0.8)),  # Add score from tool results
                                                "collection": doc.get('collection', 'default_knowledge')
                                            }
                                            tool_rag_sources.append(source_info)
                                    
                                    # Replace instead of append to prevent document stacking
                                    rag_sources = tool_rag_sources
                                    print(f"[DEBUG] rag_answer: Replaced rag_sources with {len(rag_sources)} documents from direct tool execution")
                        
                        print(f"[DEBUG] rag_answer: DIRECT EXECUTION SUCCESS - bypassing intelligent executor")
                    else:
                        print(f"[DEBUG] rag_answer: Direct execution failed, falling back to intelligent executor")
            except Exception as e:
                print(f"[DEBUG] rag_answer: Direct bypass check failed: {e}")
            
            # Try intelligent tool executor only if direct execution didn't work
            if not tool_calls:
                try:
                    from app.langchain.intelligent_tool_executor import execute_task_with_intelligent_tools
                    print(f"[DEBUG] rag_answer: Using intelligent tool executor for task: {question}")
                    
                    # Create standard chat span for proper hierarchy
                    chat_span = None
                    if trace:
                        try:
                            from app.core.langfuse_integration import get_tracer
                            tracer = get_tracer()
                            if tracer.is_enabled():
                                chat_span = tracer.create_standard_chat_span(trace, question, thinking)
                        except Exception as e:
                            logger.warning(f"Failed to create standard chat span: {e}")
                    
                    # Execute task with intelligent planning (pass chat_span as trace for proper nesting)
                    execution_events = await execute_task_with_intelligent_tools(
                        task=enhanced_question,
                        context={
                            "conversation_id": conversation_id,
                            "collections": collections,
                            "collection_strategy": collection_strategy
                        },
                        trace=chat_span if chat_span else trace,  # Use chat_span for proper hierarchy
                        mode="standard"  # Explicitly set mode
                    )
                    
                    # Convert execution events to legacy format for compatibility
                    tool_calls = []
                    tool_context = ""
                    
                    for event in execution_events:
                        if event.get("type") == "tool_complete" and event.get("success"):
                            tool_calls.append({
                                "tool": event.get("tool_name"),
                                "success": True,
                                "result": event.get("result"),
                                "execution_time": event.get("execution_time")
                            })
                            
                            # Build context from successful tool results
                            if event.get("result"):
                                tool_context += f"\n{event.get('tool_name')} result: {event.get('result')}\n"
                        
                        elif event.get("type") == "execution_complete":
                            results = event.get("results", {})
                            if results:
                                tool_context += f"\nFinal results from {len(results)} tools:\n"
                                for tool_name, result in results.items():
                                    tool_context += f"- {tool_name}: {result}\n"
                    
                    print(f"[DEBUG] rag_answer: Intelligent executor returned:")
                    print(f"  - tool_calls: {len(tool_calls)} calls")
                    print(f"  - tool_context length: {len(tool_context)}")
                    print(f"  - execution_events: {len(execution_events)} events")
                    
                    # Extract documents from knowledge_search tool results
                    for tc in tool_calls:
                        if is_rag_tool(tc.get('tool', '')) and tc.get('success'):
                            tool_result = tc.get('result', {})
                            # Handle JSON-RPC response format
                            if isinstance(tool_result, dict) and 'result' in tool_result:
                                rag_result = tool_result['result']
                                if isinstance(rag_result, dict) and 'documents' in rag_result:
                                    # Replace existing rag_sources with tool results (don't append to avoid stacking)
                                    tool_rag_sources = []
                                    for doc in rag_result['documents']:
                                        if isinstance(doc, dict):
                                            source_info = {
                                                "content": doc.get('content', ''),
                                                "file": doc.get('source', 'Unknown'),
                                                "page": doc.get('metadata', {}).get('page', 0),
                                                "score": doc.get('relevance_score', doc.get('score', 0.8)),  # Add score from tool results
                                                "collection": doc.get('collection', 'default_knowledge')
                                            }
                                            tool_rag_sources.append(source_info)
                                    
                                    # Apply reranking to tool results if available
                                    rag_sources = apply_reranking_to_sources(question, tool_rag_sources)
                                    print(f"[DEBUG] rag_answer: Replaced rag_sources with {len(rag_sources)} documents from intelligent executor results (reranked)")
                    
                    # End chat span with results
                    if chat_span:
                        try:
                            from app.core.langfuse_integration import get_tracer
                            tracer = get_tracer()
                            tracer.end_span_with_result(
                                chat_span,
                                {
                                    "tool_calls_count": len(tool_calls),
                                    "execution_events_count": len(execution_events),
                                    "tools_executed": [tc.get('tool') for tc in tool_calls if tc.get('success')],
                                    "context_length": len(tool_context)
                                },
                                success=len(tool_calls) > 0
                            )
                        except Exception as e:
                            logger.warning(f"Failed to end standard chat span: {e}")
                    
                except Exception as e:
                    print(f"[DEBUG] rag_answer: Intelligent tool executor failed: {e}, trying simple tool executor")
                    
                    # Fallback to simple tool executor
                    try:
                        from app.langchain.simple_tool_executor import identify_and_execute_tools
                        tool_calls, tool_context = identify_and_execute_tools(question, trace=trace)
                        print(f"[DEBUG] rag_answer: Simple tool executor returned:")
                        print(f"  - tool_calls: {len(tool_calls) if tool_calls else 0} calls")
                        print(f"  - tool_context length: {len(tool_context) if tool_context else 0}")
                        if tool_calls:
                            for tc in tool_calls:
                                print(f"  - Tool: {tc.get('tool', 'unknown')} - Success: {tc.get('success', False)}")
                                
                                # Extract documents from simple tool executor results
                                if is_rag_tool(tc.get('tool', '')) and tc.get('success'):
                                    tool_result = tc.get('result', {})
                                    # Handle JSON-RPC response format
                                    if isinstance(tool_result, dict) and 'result' in tool_result:
                                        rag_result = tool_result['result']
                                        if isinstance(rag_result, dict) and 'documents' in rag_result:
                                            # Replace existing rag_sources with tool results (don't append to avoid stacking)
                                            tool_rag_sources = []
                                            for doc in rag_result['documents']:
                                                if isinstance(doc, dict):
                                                    source_info = {
                                                        "content": doc.get('content', ''),
                                                        "file": doc.get('source', 'Unknown'),
                                                        "page": doc.get('metadata', {}).get('page', 0),
                                                        "collection": doc.get('collection', 'default_knowledge')
                                                    }
                                                    tool_rag_sources.append(source_info)
                                            
                                            # Apply reranking to tool results if available
                                            rag_sources = apply_reranking_to_sources(question, tool_rag_sources)
                                            print(f"[DEBUG] rag_answer: Replaced rag_sources with {len(rag_sources)} documents from simple tool executor results (reranked)")
                    except Exception as e2:
                        print(f"[DEBUG] rag_answer: Simple tool executor also failed: {e2}, trying original method")
                        # Fallback to original method
                        tool_calls, tool_context = await execute_intelligent_planning(enhanced_question, thinking, trace)
                        print(f"[DEBUG] rag_answer: execute_tools_first returned:")
                        print(f"  - tool_calls: {len(tool_calls) if tool_calls else 0} calls")
                        print(f"  - tool_context length: {len(tool_context) if tool_context else 0}")
                        
                        # Extract documents from fallback results
                        if tool_calls:
                            for tc in tool_calls:
                                if is_rag_tool(tc.get('tool', '')) and tc.get('success'):
                                    tool_result = tc.get('result', {})
                                    # Handle JSON-RPC response format
                                    if isinstance(tool_result, dict) and 'result' in tool_result:
                                        rag_result = tool_result['result']
                                        if isinstance(rag_result, dict) and 'documents' in rag_result:
                                            # Replace existing rag_sources with tool results (don't append to avoid stacking)
                                            tool_rag_sources = []
                                            for doc in rag_result['documents']:
                                                if isinstance(doc, dict):
                                                    source_info = {
                                                        "content": doc.get('content', ''),
                                                        "file": doc.get('source', 'Unknown'),
                                                        "page": doc.get('metadata', {}).get('page', 0),
                                                        "collection": doc.get('collection', 'default_knowledge')
                                                    }
                                                    tool_rag_sources.append(source_info)
                                            
                                            # Replace instead of append to prevent document stacking
                                            rag_sources = tool_rag_sources
                                            print(f"[DEBUG] rag_answer: Replaced rag_sources with {len(rag_sources)} documents from fallback results")
            
        if not tool_calls:
            print(f"[DEBUG] rag_answer: No tools actually executed despite TOOLS classification")
            # For TOOLS queries, we should still prioritize tool-based response
            # even if tool execution failed
            tool_context = f"Tool execution was attempted for query: '{question}' but no tools were successfully executed. This may indicate missing tool configuration or execution errors."
        else:
            print(f"[DEBUG] rag_answer: Tools already executed in web search fallback, skipping duplicate execution")
    
    print(f"[DEBUG] rag_answer: Finished TOOLS section, moving to conversation history (query_type={query_type})")
    
    # Get conversation history with smart filtering and freshness scoring for ALL query types
    conversation_history = ""
    conversation_messages = None
    history_prompt = ""
    if conversation_id:
        # Get conversation history limits from settings to avoid hardcoding
        llm_settings = get_llm_settings()
        conversation_config = llm_settings.get('conversation_settings', {})
        max_history_messages = conversation_config.get('max_history_messages', 6)
        max_age_hours = conversation_config.get('max_age_hours', 24.0)
        
        # Try to use the enhanced conversation manager with freshness scoring
        try:
            from app.core.simple_conversation_manager import conversation_manager
            conversation_messages = await conversation_manager.get_conversation_history(
                conversation_id, 
                limit=max_history_messages,
                max_age_hours=max_age_hours
            )
            # Format as string for backward compatibility
            if conversation_messages:
                conversation_history = conversation_manager.format_history_for_prompt(
                    conversation_messages, question
                ).replace(f"Current question: {question}\n\nPlease answer the current question while considering the context of our previous conversation.", "")
                conversation_history = conversation_history.replace("Previous conversation:\n", "")
            
        except Exception as e:
            print(f"[DEBUG] Failed to get enhanced conversation history: {e}, falling back to legacy method")
            # Fallback to legacy method
            conversation_history = get_limited_conversation_history(
                conversation_id, 
                max_messages=max_history_messages,
                current_query=question
            )
        
        if conversation_history:
            history_prompt = f"Previous conversation:\n{conversation_history}\n\n"
    
    # STEP 4: UNIFIED LLM SYNTHESIS - All responses flow through unified synthesis
    print(f"[DEBUG] rag_answer: Using unified LLM synthesis approach")
    print(f"[DEBUG] rag_answer: Before synthesis - query_type={query_type}, tool_context length={len(tool_context) if tool_context else 0}")
    print(f"[DEBUG] rag_answer: Before synthesis - tool_calls count={len(tool_calls) if tool_calls else 0}")
    
    # CRITICAL FIX: For RAG/TOOLS queries with documents, force synthesis mode
    # This prevents LLM from generating tool syntax when we already have results
    synthesis_mode = False
    if query_type in ["RAG", "TOOLS"] and (rag_sources or tool_calls):
        synthesis_mode = True
        print(f"[DEBUG] rag_answer: SYNTHESIS MODE ACTIVATED - Already have {len(rag_sources)} documents and {len(tool_calls)} tool results")
    
    # Use build_messages_for_synthesis directly to get messages array with information hierarchy
    messages, source, context, system_prompt = build_messages_for_synthesis(
        question=question,
        query_type=query_type,
        rag_context=rag_context,
        tool_context=tool_context,
        conversation_history=conversation_history,
        thinking=thinking,
        rag_sources=rag_sources,
        conversation_messages=conversation_messages,  # Pass the enhanced conversation messages with freshness scores
        conversation_id=conversation_id  # Pass conversation_id for cache cleanup
    )
    
    # Keep prompt for backward compatibility and logging
    prompt = "\n\n".join([msg["content"] for msg in messages])
    
    
    # Thinking is now handled in unified_llm_synthesis function
    
    print(f"[DEBUG] rag_answer: final prompt = {prompt[:500]}...")
    print(f"[DEBUG] rag_answer: prompt contains 'tool': {'tool>' in prompt.lower()}")
    print(f"[DEBUG] rag_answer: prompt contains get_rag_tool_name(): {get_rag_tool_name() in prompt.lower()}")
    print(f"[DEBUG] rag_answer: source = {source}")
    
    # Generate response using direct LLM instead of HTTP API for better streaming
    # CRITICAL FIX: Ensure we get main LLM config, not query classifier config
    main_llm_settings = get_llm_settings()
    mode_config = get_main_llm_full_config(main_llm_settings)
    
    # Force use of main LLM config instead of potentially contaminated llm_cfg
    print(f"[DEBUG] FIXED: Using main LLM config instead of llm_cfg")
    print(f"[DEBUG] Main LLM model: {mode_config.get('model')}")
    llm_cfg = main_llm_settings  # Override llm_cfg with clean main LLM settings
    
    # Dynamic max_tokens based on request type
    if "generate" in question.lower() and any(word in question.lower() for word in ["questions", "items", "list"]):
        # For generation tasks, calculate appropriate token limit
        import re
        numbers = re.findall(r'\b(\d+)\b', question)
        target_num = max([int(n) for n in numbers], default=10) if numbers else 10
        # Estimate ~100 tokens per question + 500 buffer for instructions
        estimated_tokens = target_num * 100 + 500
        # Use higher cap for models with larger context windows
        llm_model = mode_config.get("model", "").lower()
        token_cap = 32768 if "deepseek" in llm_model else 8192
        max_tokens = min(estimated_tokens, token_cap)
        print(f"[DEBUG] rag_answer: Using dynamic max_tokens={max_tokens} for {target_num} items (cap: {token_cap})")
    else:
        # Use configured max_tokens from main LLM config (FIXED: was using llm_cfg instead of mode_config)
        max_tokens_raw = mode_config.get("max_tokens", 16384)
        # Handle both string and int values
        try:
            max_tokens = int(max_tokens_raw)
            # Sanity check - only limit if max_tokens is >= context window (obvious user error)
            # Make this configurable based on model capabilities
            context_length = mode_config.get("context_length", 40960)
            # Only apply safety limit when max_tokens equals or exceeds context_length (user error)
            if max_tokens >= context_length:
                max_output_ratio = 0.75  # Use max 75% of context for output (more generous)
                max_safe_tokens = int(context_length * max_output_ratio)
                print(f"[WARNING] max_tokens {max_tokens} equals/exceeds context window {context_length}. Using {max_safe_tokens} for output (75% of context).")
                max_tokens = max_safe_tokens
            # Otherwise, trust the configured max_tokens value
        except (ValueError, TypeError):
            # Use fallback based on context length instead of hardcoded value
            context_length = mode_config.get("context_length", 40960)
            fallback_tokens = int(context_length * 0.3)  # Conservative 30% for fallback
            print(f"[WARNING] Invalid max_tokens value: {max_tokens_raw}, using {fallback_tokens} (30% of {context_length} context)")
            max_tokens = fallback_tokens
    
    print(f"[DEBUG] rag_answer: Final max_tokens being sent: {max_tokens}")
    print(f"[DEBUG] rag_answer: Model: {mode_config.get('model', 'unknown')}")
    print(f"[DEBUG] rag_answer: Temperature: {mode_config.get('temperature', 0.7)}")
    
    payload = {
        "prompt": prompt,
        "temperature": mode_config.get("temperature", 0.7),
        "top_p": mode_config.get("top_p", 1.0),
        "max_tokens": max_tokens
    }
    
    if stream:
        print(f"[DEBUG] STREAM=TRUE - Using direct OllamaLLM like multi-agent")
        
        async def stream_tokens():
            from app.llm.ollama import OllamaLLM
            from app.llm.base import LLMConfig
            
            # Ensure rag_sources is available in this nested function scope
            nonlocal rag_sources
            
            # Create LLM config exactly like multi-agent
            llm_config = LLMConfig(
                model_name=mode_config["model"],
                temperature=float(mode_config.get("temperature", 0.7)),
                top_p=float(mode_config.get("top_p", 1.0)),
                max_tokens=int(max_tokens)
            )
            
            # Get Ollama URL from LLM config
            ollama_url = mode_config.get("model_server", "http://ollama:11434")
            print(f"[DEBUG] Ollama URL from config: {ollama_url}")
            
            # If localhost, change to host.docker.internal for Docker containers only
            import os
            if "localhost" in ollama_url and (os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')):
                ollama_url = ollama_url.replace("localhost", "host.docker.internal")
                print(f"[DEBUG] Converted to Docker URL: {ollama_url}")
            elif "localhost" in ollama_url:
                print(f"[DEBUG] Running locally, keeping localhost URL: {ollama_url}")
            
            # Create LLM instance
            llm = OllamaLLM(llm_config, base_url=ollama_url)
            
            # Create LLM generation span for main response
            llm_generation_span = None
            if trace:
                try:
                    from app.core.langfuse_integration import get_tracer
                    tracer = get_tracer()
                    if tracer.is_enabled():
                        llm_generation_span = tracer.create_llm_generation_span(
                            trace,
                            model=llm_config.model_name,
                            prompt=prompt,
                            operation="main_generation"
                        )
                except Exception as e:
                    logger.warning(f"Failed to create LLM generation span: {e}")
            
            # Stream tokens exactly like multi-agent
            response_text = ""
            stream_chunk_count = 0
            print(f"[DEBUG] Starting streaming generation...")
            
            # Use chat_stream with messages array instead of generate_stream with concatenated prompt
            try:
                async for response_chunk in llm.chat_stream(messages):
                    stream_chunk_count += 1
                    if stream_chunk_count == 1:
                        print(f"[DEBUG] First streaming chunk received: '{response_chunk.text[:50]}...'")
                    elif stream_chunk_count % 20 == 0:
                        print(f"[DEBUG] Streamed {stream_chunk_count} chunks, total length: {len(response_text)} chars")
                    
                    response_text += response_chunk.text
                    
                    # Stream tokens in real-time (but filter out tool syntax)
                    if response_chunk.text.strip():
                        # Don't stream tool syntax tokens to prevent user from seeing them
                        if "<tool>" not in response_chunk.text and "</tool>" not in response_chunk.text:
                            yield json.dumps({
                                "token": response_chunk.text
                            }) + "\n"
                
                print(f"[DEBUG] Streaming completed! Total chunks: {stream_chunk_count}, final length: {len(response_text)} chars")
            except Exception as e:
                print(f"[ERROR] Streaming generation failed: {e}")
                logger.error(f"LLM streaming generation failed: {e}")
                # Yield error information to client
                yield json.dumps({
                    "error": f"Generation failed: {e}"
                }) + "\n"
                return
            
            # End LLM generation span with actual output
            if llm_generation_span:
                try:
                    tracer = get_tracer()
                    usage = tracer.estimate_token_usage(prompt, response_text)
                    # Update the generation with output and usage
                    llm_generation_span.end(
                        output=response_text,
                        usage=usage,
                        metadata={
                            "response_length": len(response_text),
                            "operation": "main_generation",
                            "success": True
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to end LLM generation span: {e}")
            
            # Parse and execute tool calls using enhanced error handling
            # CRITICAL FIX: Skip tool execution if we already have RAG sources or tool results
            # This prevents double execution and tool syntax being shown to users
            from app.langchain.enhanced_query_classifier import EnhancedQueryClassifier
            classifier = EnhancedQueryClassifier()
            has_explicit_search_intent = await classifier.detect_explicit_search_intent(question)
            
            # Check if response contains tool syntax that needs execution
            contains_tool_syntax = "<tool>" in response_text and "</tool>" in response_text
            
            skip_knowledge_search = (
                (query_type == "RAG" and rag_context and not has_explicit_search_intent) or
                (query_type == "SYNTHESIS") or  # Skip tool execution for synthesis mode
                (rag_sources and len(rag_sources) > 0) or  # Skip if we already have RAG sources
                (tool_calls and len(tool_calls) > 0)  # Skip if we already executed tools
            )
            
            if skip_knowledge_search:
                logger.info(f"[EXECUTION TRACKER] SKIPPING tool execution - already have results or synthesis mode")
                print(f"[EXECUTION TRACKER] SKIPPING tool execution - query_type={query_type}, has_rag_sources={bool(rag_sources)}, has_tool_calls={bool(tool_calls)}")
                # If response contains tool syntax but we're skipping execution, we need to synthesize
                if contains_tool_syntax and (rag_sources or tool_calls):
                    print(f"[WARNING] LLM generated tool syntax but we already have results - need synthesis")
            else:
                logger.info(f"[EXECUTION TRACKER] EXECUTION #2: Tool parsing from streaming LLM response")
                print(f"[EXECUTION TRACKER] EXECUTION #2: Tool parsing from streaming LLM response")
            
            tool_results = extract_and_execute_tool_calls(
                response_text, 
                trace=trace, 
                use_enhanced_error_handling=True,
                skip_knowledge_search=skip_knowledge_search,
                original_query=question
            )
            
            # Only process tool results if we actually executed tools
            if tool_results and not skip_knowledge_search:
                print(f"[DEBUG] Executed {len(tool_results)} tools from LLM response")
                
                # Extract documents from tool results
                for tr in tool_results:
                    if is_rag_tool(tr.get('tool', '')) and tr.get('success'):
                        tool_result = tr.get('result', {})
                        # Handle JSON-RPC response format
                        if isinstance(tool_result, dict) and 'result' in tool_result:
                            rag_result = tool_result['result']
                            if isinstance(rag_result, dict) and 'documents' in rag_result:
                                # Replace existing rag_sources with tool results (don't append to avoid stacking)
                                tool_rag_sources = []
                                for doc in rag_result['documents']:
                                    if isinstance(doc, dict):
                                        source_info = {
                                            "content": doc.get('content', ''),
                                            "file": doc.get('source', 'Unknown'),
                                            "page": doc.get('metadata', {}).get('page', 0),
                                            "collection": doc.get('collection', 'default_knowledge')
                                        }
                                        tool_rag_sources.append(source_info)
                                
                                # Replace instead of append to prevent document stacking
                                rag_sources = tool_rag_sources
                                print(f"[DEBUG] rag_answer: Replaced rag_sources with {len(rag_result['documents'])} documents from LLM tool call results")
                
                # Stream tool execution results
                for result in tool_results:
                    yield json.dumps({
                        "tool_execution": {
                            "tool": result.get('tool'),
                            "success": result.get('success', False),
                            "result": result.get('result') if result.get('success') else result.get('error')
                        }
                    }) + "\n"
                
                # If tools were executed successfully, we need to synthesize the response with tool results
                successful_tool_results = [r for r in tool_results if r.get('success') and not r.get('skipped')] if tool_results else []
                if successful_tool_results and not skip_knowledge_search:
                    # Build enhanced response with tool results (excluding skipped tools)
                    tool_context = "\n\nTool Results:\n"
                    for tr in successful_tool_results:
                        # Handle tool result structure properly
                        if 'result' in tr and tr.get('success', False):
                            tool_context += f"\n{tr['tool']}: {json.dumps(tr['result'], indent=2)}\n"
                        elif 'error' in tr:
                            tool_context += f"\n{tr['tool']}: Error - {tr['error']}\n"
                        else:
                            tool_context += f"\n{tr['tool']}: No result available\n"
                    
                    # Create a follow-up prompt to synthesize with tool results using the actual user question
                    synthesis_prompt = f"""You need to provide a final answer based on the tool results below.

{tool_context}

Based on these search results, provide a comprehensive answer to the user's question: "{question}". Format the information clearly and include the most relevant findings."""
                    
                    # Convert to messages format for proper system/user separation
                    synthesis_messages = [
                        {"role": "system", "content": system_prompt if 'system_prompt' in locals() else "You are a helpful AI assistant."},
                        {"role": "user", "content": synthesis_prompt}
                    ]
                    
                    # Create LLM generation span for synthesis
                    synthesis_generation_span = None
                    if trace:
                        try:
                            from app.core.langfuse_integration import get_tracer
                            tracer = get_tracer()
                            if tracer.is_enabled():
                                synthesis_generation_span = tracer.create_llm_generation_span(
                                    trace,
                                    model=llm_config.model_name,
                                    prompt=synthesis_prompt,
                                    operation="tool_synthesis"
                                )
                        except Exception as e:
                            logger.warning(f"Failed to create synthesis generation span: {e}")
                    
                    # Generate final response with tool results
                    final_response = ""
                    # Use chat_stream with messages array instead of generate_stream with concatenated prompt
                    async for response_chunk in llm.chat_stream(synthesis_messages):
                        final_response += response_chunk.text
                        if response_chunk.text.strip():
                            yield json.dumps({
                                "token": response_chunk.text
                            }) + "\n"
                    
                    # End synthesis generation span
                    if synthesis_generation_span:
                        try:
                            tracer = get_tracer()
                            usage = tracer.estimate_token_usage(synthesis_prompt, final_response)
                            synthesis_generation_span.end(
                                output=final_response,
                                usage=usage,
                                metadata={
                                    "response_length": len(final_response),
                                    "tool_results_count": len(tool_results),
                                    "operation": "llm_tool_synthesis",
                                    "result_type": "dict" if tool_results else "text"
                                }
                            )
                        except Exception as e:
                            logger.warning(f"Failed to end synthesis generation span: {e}")
                    
                    response_text = final_response
            
            # Convert rag_sources to frontend format
            documents = []
            if rag_sources:
                print(f"[DEBUG] rag_answer: Converting {len(rag_sources)} rag_sources to documents")
                for i, source_info in enumerate(rag_sources[:3]):  # Debug first 3
                    print(f"  - Source {i}: has score={source_info.get('score', 'None')}")
                for i, source_info in enumerate(rag_sources):
                    # Create document entry matching frontend interface
                    doc_entry = {
                        "content": source_info.get("content", f"Document {i+1} content"),
                        "source": source_info.get("file", "Unknown"),
                        "relevance_score": source_info.get("score", source_info.get("relevance_score", 0.8)),  # Use actual score if available
                        "metadata": {
                            "page": source_info.get("page"),
                            "doc_id": f"doc_{i+1}",
                            "collection": source_info.get("collection", "default_knowledge")
                        }
                    }
                    documents.append(doc_entry)
                    
            print(f"[DEBUG] rag_answer: Including {len(documents)} documents in response")
            
            # Send completion event
            yield json.dumps({
                "answer": response_text,
                "source": source + ("+TOOLS" if tool_results else ""),
                "conversation_id": conversation_id
            }) + "\n"
            
            # Store conversation - ensure we store the synthesized text, not tool syntax
            if conversation_id and response_text:
                # Check if response contains tool syntax and skip storing if so
                if "<tool>" not in response_text and "</tool>" not in response_text:
                    store_conversation_message(conversation_id, "assistant", response_text)
                    print(f"[DEBUG] Stored synthesized response to conversation (length: {len(response_text)})")
                else:
                    print(f"[WARNING] Skipping storage of response with tool syntax to prevent contamination")
        
        return stream_tokens()
    
    # Non-streaming version - collect all tokens
    else:
        from app.llm.ollama import OllamaLLM
        from app.llm.base import LLMConfig
        import os
        
        # Ensure rag_sources is available for non-streaming path
        if 'rag_sources' not in locals():
            rag_sources = []
        
        llm_config = LLMConfig(
            model_name=mode_config["model"],
            temperature=float(mode_config.get("temperature", 0.7)),
            top_p=float(mode_config.get("top_p", 1.0)),
            max_tokens=int(max_tokens)
        )
        
        ollama_url = mode_config.get("model_server", "http://ollama:11434")
        # If localhost, change to host.docker.internal for Docker containers
        if "localhost" in ollama_url:
            ollama_url = ollama_url.replace("localhost", "host.docker.internal")
        llm = OllamaLLM(llm_config, base_url=ollama_url)
        
        # Create LLM generation span for non-streaming response
        llm_generation_span = None
        if trace:
            try:
                from app.core.langfuse_integration import get_tracer
                tracer = get_tracer()
                if tracer.is_enabled():
                    llm_generation_span = tracer.create_llm_generation_span(
                        trace,
                        model=llm_config.model_name,
                        prompt=prompt,
                        operation="non_streaming_generation"
                    )
            except Exception as e:
                logger.warning(f"Failed to create LLM generation span: {e}")
        
        response_text = ""
        chunk_count = 0
        print(f"[DEBUG] Starting non-streaming generation...")
        # Use chat_stream with messages array instead of generate_stream with concatenated prompt
        try:
            async for response_chunk in llm.chat_stream(messages):
                chunk_count += 1
                if chunk_count == 1:
                    print(f"[DEBUG] First response chunk received: '{response_chunk.text[:50]}...'")
                elif chunk_count % 20 == 0:
                    print(f"[DEBUG] Received {chunk_count} chunks, total length: {len(response_text)} chars")
                response_text += response_chunk.text
            
            print(f"[DEBUG] Generation completed! Total chunks: {chunk_count}, final length: {len(response_text)} chars")
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            logger.error(f"LLM generation failed: {e}")
            raise
        
        # End LLM generation span
        if llm_generation_span:
            try:
                tracer = get_tracer()
                usage = tracer.estimate_token_usage(prompt, response_text)
                tracer.end_span_with_result(llm_generation_span, {
                    "response_length": len(response_text),
                    "usage": usage
                }, True)
            except Exception as e:
                logger.warning(f"Failed to end LLM generation span: {e}")
        
        # Parse and execute tool calls using enhanced error handling
        # Skip redundant knowledge_search if RAG was already performed, unless explicit search intent detected
        from app.langchain.enhanced_query_classifier import EnhancedQueryClassifier
        classifier = EnhancedQueryClassifier()
        has_explicit_search_intent = await classifier.detect_explicit_search_intent(question)
        skip_knowledge_search = (
            (query_type == "RAG" and rag_context and not has_explicit_search_intent) or
            (query_type == "SYNTHESIS")  # Skip tool execution for synthesis mode to prevent triple execution
        )
        
        if skip_knowledge_search:
            logger.info(f"[EXECUTION TRACKER] SKIPPING tool execution - query_type={query_type}, rag_context={'present' if rag_context else 'absent'}")
            print(f"[EXECUTION TRACKER] SKIPPING tool execution - query_type={query_type}")
        else:
            logger.info(f"[EXECUTION TRACKER] EXECUTION #3: Tool parsing from non-streaming LLM response")
            print(f"[EXECUTION TRACKER] EXECUTION #3: Tool parsing from non-streaming LLM response")
        
        tool_results = extract_and_execute_tool_calls(
            response_text, 
            trace=trace, 
            use_enhanced_error_handling=True,
            skip_knowledge_search=skip_knowledge_search,
            original_query=question
        )
        
        # Filter out skipped tool results from synthesis
        successful_tool_results = [r for r in tool_results if r.get('success') and not r.get('skipped')]
        if successful_tool_results:
            # Build enhanced response with tool results (excluding skipped tools)
            tool_context = "\n\nTool Results:\n"
            for tr in successful_tool_results:
                tool_context += f"\n{tr['tool']}: {json.dumps(tr['result'], indent=2)}\n"
            
            # Create a follow-up prompt to synthesize with tool results
            synthesis_prompt = f"""{response_text}

{tool_context}

Please provide a complete answer using the tool results above."""
            
            # Convert to messages format for proper system/user separation
            synthesis_messages = [
                {"role": "system", "content": system_prompt if 'system_prompt' in locals() else "You are a helpful AI assistant."},
                {"role": "user", "content": synthesis_prompt}
            ]
            
            # Create LLM generation span for final synthesis
            final_synthesis_span = None
            if trace:
                try:
                    from app.core.langfuse_integration import get_tracer
                    tracer = get_tracer()
                    if tracer.is_enabled():
                        final_synthesis_span = tracer.create_llm_generation_span(
                            trace,
                            model=llm_config.model_name,
                            prompt=synthesis_prompt,
                            operation="final_synthesis"
                        )
                except Exception as e:
                    logger.warning(f"Failed to create final synthesis span: {e}")
            
            # Generate final response with tool results
            final_response = ""
            # Use chat_stream with messages array instead of generate_stream with concatenated prompt
            async for response_chunk in llm.chat_stream(synthesis_messages):
                final_response += response_chunk.text
            
            # End final synthesis span
            if final_synthesis_span:
                try:
                    tracer = get_tracer()
                    usage = tracer.estimate_token_usage(synthesis_prompt, final_response)
                    tracer.end_span_with_result(final_synthesis_span, {
                        "response_length": len(final_response),
                        "usage": usage,
                        "tool_results_count": len(tool_results)
                    }, True)
                except Exception as e:
                    logger.warning(f"Failed to end final synthesis span: {e}")
            
            response_text = final_response
            source = source + "+TOOLS"
        
        if conversation_id and response_text:
            store_conversation_message(conversation_id, "assistant", response_text)
        
        # Convert rag_sources to frontend format for non-streaming response
        documents = []
        if rag_sources:
            for i, source_info in enumerate(rag_sources):
                doc_entry = {
                    "content": source_info.get("content", f"Document {i+1} content"),
                    "source": source_info.get("file", "Unknown"),
                    "relevance_score": source_info.get("score", source_info.get("relevance_score", 0.8)),  # Use actual score if available
                    "metadata": {
                        "page": source_info.get("page"),
                        "doc_id": f"doc_{i+1}",
                        "collection": source_info.get("collection", "default_knowledge")
                    }
                }
                documents.append(doc_entry)
        
        return {
            "answer": response_text,
            "source": source,
            "context": context,
            "documents": documents
        }

# Keep existing helper functions
def get_mcp_tools_context(force_reload=False, include_examples=True):
    """
    Prepare comprehensive MCP tools context for LLM prompt with full parameter details
    
    Args:
        force_reload: Force reload cache from database
        include_examples: Include usage examples for better LLM understanding
    
    Returns:
        Formatted string with all available tools, their parameters, and usage examples
    """
    from app.core.mcp_tools_cache import reload_enabled_mcp_tools
    
    # Get tools data - reload if forced or cache is empty
    enabled_tools = get_enabled_mcp_tools()
    
    if force_reload or not enabled_tools:
        print("[DEBUG] MCP Tools: Reloading cache from database...")
        enabled_tools = reload_enabled_mcp_tools()
    
    # Validate critical tools have parameters
    critical_tools = ['google_search', 'get_datetime']
    for tool_name in critical_tools:
        if tool_name in enabled_tools:
            tool_info = enabled_tools[tool_name]
            if not tool_info.get('parameters') or tool_info.get('parameters') == {}:
                print(f"[DEBUG] MCP Tools: {tool_name} missing parameters, forcing reload...")
                enabled_tools = reload_enabled_mcp_tools()
                break
    
    if not enabled_tools:
        return "\n⚠️ No MCP tools available. Tool execution disabled."
    
    # Build comprehensive tools context
    tools_context = []
    tools_context.append("=" * 60)
    tools_context.append("🛠️  AVAILABLE MCP TOOLS")
    tools_context.append("=" * 60)
    
    # Group tools by category for better organization
    tool_categories = {
        "search": [],
        "email": [],
        "jira": [],
        "datetime": [],
        "other": []
    }
    
    for tool_name, tool_info in enabled_tools.items():
        # Categorize tools
        if "search" in tool_name.lower() or "google" in tool_name.lower():
            category = "search"
        elif "email" in tool_name.lower() or "gmail" in tool_name.lower():
            category = "email"
        elif "jira" in tool_name.lower():
            category = "jira"
        elif "datetime" in tool_name.lower() or "time" in tool_name.lower():
            category = "datetime"
        else:
            category = "other"
        
        tool_categories[category].append((tool_name, tool_info))
    
    # Format each category
    for category, tools in tool_categories.items():
        if not tools:
            continue
            
        tools_context.append(f"\n📂 {category.upper()} TOOLS:")
        tools_context.append("-" * 40)
        
        for tool_name, tool_info in tools:
            # Parse parameters if they're a string
            parameters = tool_info.get("parameters", {})
            if isinstance(parameters, str):
                try:
                    import json
                    parameters = json.loads(parameters)
                except json.JSONDecodeError:
                    print(f"[WARNING] Failed to parse parameters for {tool_name}")
                    parameters = {}
            
            # Build tool description
            tool_desc = []
            tool_desc.append(f"🔧 Tool: {tool_name}")
            tool_desc.append(f"   Description: {tool_info.get('description', 'No description')}")
            
            # Add parameter details
            if parameters and isinstance(parameters, dict):
                if "required" in parameters:
                    required_params = parameters.get("required", [])
                    if required_params:
                        tool_desc.append(f"   Required Parameters: {', '.join(required_params)}")
                
                if "properties" in parameters:
                    props = parameters["properties"]
                    tool_desc.append("   Parameters:")
                    for param_name, param_info in props.items():
                        param_type = param_info.get("type", "unknown")
                        param_desc = param_info.get("description", "No description")
                        required_mark = "* " if param_name in parameters.get("required", []) else "  "
                        tool_desc.append(f"   {required_mark}- {param_name} ({param_type}): {param_desc}")
                        
                        # Add constraints
                        constraints = []
                        if "maxLength" in param_info:
                            constraints.append(f"max length: {param_info['maxLength']}")
                        if "minimum" in param_info:
                            constraints.append(f"min: {param_info['minimum']}")
                        if "maximum" in param_info:
                            constraints.append(f"max: {param_info['maximum']}")
                        if "default" in param_info:
                            constraints.append(f"default: {param_info['default']}")
                        if constraints:
                            tool_desc.append(f"       ({', '.join(constraints)})")
            
            # Add usage examples for critical tools
            if include_examples:
                if tool_name == "google_search":
                    tool_desc.append("   💡 Example: <tool>google_search({\"query\": \"latest AI news\"})</tool>")
                elif tool_name == "get_datetime":
                    tool_desc.append("   💡 Example: <tool>get_datetime({})</tool>")
                elif "gmail" in tool_name and "send" in tool_name:
                    tool_desc.append("   💡 Example: <tool>gmail_send({\"to\": [\"user@example.com\"], \"subject\": \"Test\", \"body\": \"Hello\"})</tool>")
            
            tools_context.append("\n".join(tool_desc))
            tools_context.append("")  # Empty line between tools
    
    # Add usage instructions
    if include_examples:
        tools_context.append("\n" + "=" * 60)
        tools_context.append("📋 TOOL USAGE INSTRUCTIONS")
        tools_context.append("=" * 60)
        tools_context.append("To use a tool, format your response as:")
        tools_context.append("<tool>tool_name(parameters)</tool>")
        tools_context.append("")
        tools_context.append("Where:")
        tools_context.append("• tool_name: exact name from the list above")
        tools_context.append("• parameters: JSON object with required parameters")
        tools_context.append("• Use {} for tools that need no parameters")
        tools_context.append("")
        tools_context.append("IMPORTANT: Only use tools when they can help answer the user's question!")
    
    context = "\n".join(tools_context)
    
    # Enhanced debug logging
    tools_by_category = {cat: len(tools) for cat, tools in tool_categories.items() if tools}
    print(f"[DEBUG] MCP Tools Context: {len(enabled_tools)} total tools, categories: {tools_by_category}")
    print(f"[DEBUG] MCP Tools Context length: {len(context)} characters")
    
    # Log critical tools status
    for tool_name in critical_tools:
        if tool_name in enabled_tools:
            tool_info = enabled_tools[tool_name]
            has_params = bool(tool_info.get('parameters'))
            print(f"[DEBUG] MCP Tools: {tool_name} - parameters: {has_params}")
    
    return context

def refresh_gmail_token(server_id):
    """Refresh Gmail OAuth token"""
    # Ensure logger is available in this function context
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        from app.core.oauth_token_manager import oauth_token_manager
        
        # Invalidate the cached token to force refresh
        oauth_token_manager.invalidate_token(server_id, "gmail")
        
        # Get fresh token (this will trigger refresh if needed)
        oauth_creds = oauth_token_manager.get_valid_token(
            server_id=server_id,
            service_name="gmail"
        )
        
        if oauth_creds and oauth_creds.get("access_token"):
            return {
                "message": "Token refreshed successfully",
                "access_token": oauth_creds.get("access_token"),
                "expires_at": oauth_creds.get("expires_at")
            }
        else:
            return {"error": "Failed to refresh token: No valid credentials available"}
    except Exception as e:
        return {"error": f"Exception refreshing token: {str(e)}"}

def _map_tool_parameters_service(tool_name: str, params: dict) -> tuple[str, dict]:
    """Map common parameter mismatches to correct parameter names for service layer"""
    # Ensure logger is available in this function context
    import logging
    logger = logging.getLogger(__name__)
    
    mapped_params = params.copy()
    original_tool_name = tool_name
    
    # Tool name correction mapping - fix common LLM mistakes (using configurable RAG tool name)
    rag_tool_name = get_rag_tool_name()
    
    tool_name_mapping = {
        # Common search tool mistakes
        'search_knowledge': rag_tool_name,
        'search_internal_knowledge': rag_tool_name,
        'internal_knowledge_search': rag_tool_name,
        'search_knowledge_base': rag_tool_name,
        'knowledge_base_search': rag_tool_name,
        'search_documents': rag_tool_name,
        'document_search': rag_tool_name,
        'rag_search': rag_tool_name,
        'internal_search': rag_tool_name,
        get_rag_tool_name(): rag_tool_name,
        
        # Gmail/email tool variations
        'search_email': 'find_email',
        'email_search': 'find_email',
        'gmail_search': 'find_email',
        
        # Common datetime tool mistakes
        'get_time': 'get_datetime',
        'current_time': 'get_datetime',
        'datetime': 'get_datetime',
        
        # Jira tool variations
        'jira_search': 'jira_list_issues',
        'search_jira': 'jira_list_issues',
    }
    
    # Apply tool name correction if needed
    if tool_name in tool_name_mapping:
        corrected_name = tool_name_mapping[tool_name]
        logger.info(f"[TOOL CORRECTION] Mapped '{original_tool_name}' -> '{corrected_name}'")
        tool_name = corrected_name
    
    # Apply search query optimization for search tools if enabled
    if _is_search_tool(tool_name) and 'query' in mapped_params:
        try:
            from app.langchain.search_query_optimizer import optimize_search_query
            import asyncio
            import concurrent.futures
            import threading
            
            original_query = mapped_params['query']
            
            def run_optimization():
                """Run optimization in a new thread with its own event loop"""
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(optimize_search_query(original_query))
                finally:
                    loop.close()
            
            # Run optimization in thread pool to avoid event loop conflicts
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_optimization)
                optimized_query = future.result(timeout=15)  # 15 second timeout
            
            if optimized_query and optimized_query != original_query:
                mapped_params['query'] = optimized_query
                logger.info(f"[SEARCH OPTIMIZATION] '{tool_name}' query optimized: '{original_query}' → '{optimized_query}'")
            
        except Exception as e:
            logger.warning(f"[SEARCH OPTIMIZATION] Failed for '{tool_name}': {e}")
    
    return tool_name, mapped_params


def _is_search_tool(tool_name: str) -> bool:
    """Check if a tool is a search tool that would benefit from query optimization"""
    search_keywords = ['search', 'google', 'web', 'tavily', 'find', 'lookup']
    tool_lower = tool_name.lower()
    return any(keyword in tool_lower for keyword in search_keywords)

def call_internal_service(tool_name: str, parameters: dict, tool_info: dict) -> dict:
    """Handle calls to internal services"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        if tool_name == get_rag_tool_name():
            # Import and call RAG service
            import asyncio
            from app.mcp_services.rag_mcp_service import execute_rag_search, execute_rag_search_sync
            
            # Extract parameters with defaults from RAG config
            query = parameters.get('query', '')
            collections = parameters.get('collections')
            
            # Get max_documents from RAG settings instead of hardcoding
            from app.core.rag_settings_cache import get_document_retrieval_settings
            doc_settings = get_document_retrieval_settings()
            max_documents = parameters.get('max_documents') or doc_settings.get('max_documents_mcp', 8)
            
            include_content = parameters.get('include_content', True)
            
            # Execute RAG search
            try:
                # Try to get the running loop
                asyncio.get_running_loop()
                logger.info("RAG search: Running in async context, using sync version")
                # If we're in an async context, use the sync version directly
                # This avoids the thread/event loop issues
                return execute_rag_search_sync(query, collections, max_documents, include_content)
                
            except RuntimeError:
                # No event loop running, safe to call async version
                logger.info("RAG search: No event loop, using asyncio.run")
                return asyncio.run(execute_rag_search(query, collections, max_documents, include_content))
                
        else:
            logger.error(f"Unknown internal service: {tool_name}")
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32601,
                    "message": f"Unknown internal service: {tool_name}"
                }
            }
            
    except Exception as e:
        logger.error(f"Internal service error for {tool_name}: {e}")
        return {
            "jsonrpc": "2.0", 
            "error": {
                "code": -32000,
                "message": f"Internal service failed: {str(e)}"
            }
        }

def call_mcp_tool(tool_name, parameters, trace=None, _skip_span_creation=False):
    """
    Call an MCP tool using the unified MCP service with enhanced OAuth handling
    
    This function now uses the unified architecture for both HTTP and stdio MCP servers
    with automatic OAuth token refresh and comprehensive error handling.
    """
    
    # Ensure logger is available in this function context
    import logging
    import json
    logger = logging.getLogger(__name__)
    
    
    # Apply parameter mapping for common mismatches FIRST (including search optimization)
    tool_name, parameters = _map_tool_parameters_service(tool_name, parameters)
    
    # Create tool span AFTER optimization so Langfuse shows optimized parameters
    tool_span = None
    tracer = None
    if trace and not _skip_span_creation:
        try:
            from app.core.langfuse_integration import get_tracer
            tracer = get_tracer()
            if tracer.is_enabled():
                # Sanitize parameters for Langfuse (now using optimized parameters)
                safe_parameters = {}
                if isinstance(parameters, dict):
                    for key, value in parameters.items():
                        # Ensure key and value are strings and limit length
                        safe_key = str(key)[:100]
                        safe_value = str(value)[:500] if value is not None else ""
                        safe_parameters[safe_key] = safe_value
                tool_span = tracer.create_tool_span(trace, str(tool_name), safe_parameters)
            else:
                pass
        except Exception as e:
            logger.warning(f"Failed to create tool span for {tool_name}: {e}")
            tool_span = None
            tracer = None
    else:
        pass
    
    try:
        from app.core.mcp_tools_cache import get_enabled_mcp_tools
        enabled_tools = get_enabled_mcp_tools()
        
        if tool_name not in enabled_tools:
            logger.warning(f"Tool {tool_name} not found in enabled tools")
            # End tool span with error
            if tool_span and tracer:
                try:
                    tracer.end_span_with_result(tool_span, {"error": f"Tool {tool_name} is disabled or not available"}, False, f"Tool {tool_name} not found in enabled tools")
                except Exception as e:
                    logger.warning(f"Failed to end tool span for {tool_name}: {e}")
            return {"error": f"Tool {tool_name} is disabled or not available"}
        
        tool_info = enabled_tools[tool_name]
        endpoint = tool_info["endpoint"]
        
        # Replace localhost with the actual hostname from manifest if available
        server_hostname = tool_info.get("server_hostname")
        
        # Check if we're running inside Docker
        import os
        in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
        
        if in_docker and server_hostname and "localhost" in endpoint:
            endpoint = endpoint.replace("localhost", server_hostname)
        elif not in_docker and server_hostname and server_hostname in endpoint:
            endpoint = endpoint.replace(server_hostname, "localhost")
        
        method = tool_info.get("method", "POST")
        headers = tool_info.get("headers") or {}
        
        # Handle MCP server pattern: use endpoint_prefix if tool-specific endpoint doesn't work
        endpoint_prefix = tool_info.get("endpoint_prefix")
        if endpoint_prefix and endpoint.endswith(f"/invoke/{tool_name}"):
            endpoint = endpoint_prefix
        
        # Add API key authentication if available
        api_key = tool_info.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Inject OAuth credentials for Gmail tools (skipping for brevity)
        
        # Use unified MCP service for both HTTP and stdio tools
        try:
            # Clean up parameters (remove agent parameter that shouldn't be sent to tool)
            clean_parameters = {k: v for k, v in parameters.items() if k != "agent"}
            
            # Check for internal services first
            if tool_info.get('endpoint', '').startswith('internal://'):
                logger.info(f"[INTERNAL] Calling internal service: {tool_name}")
                return call_internal_service(tool_name, clean_parameters, tool_info)
            
            # Check for APINode workflow tools
            if tool_info.get('workflow_context') and tool_name.startswith('workflow_api_'):
                logger.info(f"[APINODE] Calling APINode workflow tool: {tool_name}")
                from app.automation.integrations.apinode_mcp_bridge import apinode_mcp_bridge
                import asyncio
                
                # Get workflow context
                workflow_context = tool_info.get('workflow_context', {})
                workflow_id = workflow_context.get('workflow_id', 'unknown')
                
                # Create a dummy execution_id for this tool call
                import time
                execution_id = f"tool_call_{int(time.time() * 1000)}"
                
                try:
                    # Execute APINode tool
                    result = asyncio.run(apinode_mcp_bridge.execute_apinode_tool(
                        tool_name=tool_name,
                        parameters=clean_parameters,
                        workflow_id=workflow_id,
                        execution_id=execution_id
                    ))
                    
                    # End tool span with success
                    if tool_span and tracer:
                        try:
                            tracer.end_span_with_result(tool_span, result, True)
                        except Exception as e:
                            logger.warning(f"Failed to end tool span for {tool_name}: {e}")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"[APINODE] APINode tool execution failed: {str(e)}")
                    error_result = {"error": f"APINode tool execution failed: {str(e)}"}
                    
                    # End tool span with error
                    if tool_span and tracer:
                        try:
                            tracer.end_span_with_result(tool_span, error_result, False, str(e))
                        except Exception as span_e:
                            logger.warning(f"Failed to end tool span for {tool_name}: {span_e}")
                    
                    return error_result
            
            
            # Import unified service
            from app.core.unified_mcp_service import call_mcp_tool_unified
            import asyncio
            
            # Call the unified service (handle async in sync context)
            try:
                # Check if we're in an event loop
                asyncio.get_running_loop()
                # We're in an async context, create a new event loop in a thread
                import threading
                result_container = {'result': None, 'exception': None}
                
                def run_in_thread():
                    try:
                        # Create new event loop for this thread
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        result_container['result'] = new_loop.run_until_complete(
                            call_mcp_tool_unified(tool_info, tool_name, clean_parameters)
                        )
                        new_loop.close()
                    except Exception as e:
                        logger.error(f"Exception in thread for {tool_name}: {e}")
                        result_container['exception'] = e
                
                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join(timeout=5.0)  # 5 second timeout for MCP tool calls
                
                if thread.is_alive():
                    logger.error(f"MCP tool call {tool_name} timed out after 5 seconds")
                    result_container['exception'] = Exception(f"MCP tool '{tool_name}' operation timed out after 5 seconds")
                
                if result_container['exception']:
                    raise result_container['exception']
                result = result_container['result']
                
            except RuntimeError:
                # No running loop, we can use asyncio.run()
                result = asyncio.run(call_mcp_tool_unified(tool_info, tool_name, clean_parameters))
            
            
            # End tool span with result
            if tool_span and tracer:
                try:
                    success = "error" not in result
                    tracer.end_span_with_result(tool_span, result, success, result.get("error"))
                except Exception as e:
                    logger.warning(f"Failed to end tool span for {tool_name}: {e}")
            else:
                pass
            
            return result
            
        except Exception as e:
            error_msg = f"Unified MCP service failed for {tool_name}: {str(e)}"
            logger.error(error_msg)
            if tool_span and tracer:
                try:
                    tracer.end_span_with_result(tool_span, None, False, error_msg)
                except:
                    pass
            return {"error": error_msg}
    
    except Exception as e:
        error_msg = f"Unexpected error calling tool {tool_name}: {str(e)}"
        logger.error(error_msg)
        if tool_span and tracer:
            try:
                tracer.end_span_with_result(tool_span, None, False, error_msg)
            except:
                pass
        return {"error": error_msg}

def call_mcp_tool_with_generic_endpoint(tool_name, parameters, tool_info):
    """Fallback function to retry with generic MCP endpoint"""
    # Ensure logger is available in this function context
    import logging
    logger = logging.getLogger(__name__)
    
    endpoint_prefix = tool_info.get("endpoint_prefix")
    if not endpoint_prefix:
        return {"error": f"No fallback endpoint available for {tool_name}"}
    
    print(f"[DEBUG] Retrying {tool_name} with generic endpoint: {endpoint_prefix}")
    
    headers = tool_info.get("headers") or {}
    api_key = tool_info.get("api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    if "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"
    
    payload = {
        "name": tool_name,
        "arguments": parameters if parameters else {}
    }
    
    try:
        response = requests.post(endpoint_prefix, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        print(f"[DEBUG] Retry successful for {tool_name}: {result}")
        return result
    except Exception as e:
        error_msg = f"Retry failed for {tool_name}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return {"error": error_msg}

async def extract_and_execute_tool_calls_async(text, stream_callback=None, trace=None, use_enhanced_error_handling=True, skip_knowledge_search=False, original_query=None):
    """
    Async version of extract_and_execute_tool_calls with enhanced error handling
    """
    import re  # Import at function level to avoid scope issues
    
    if "NO_TOOLS_NEEDED" in text:
        print("[DEBUG] Tool extraction: LLM indicated no tools needed")
        return []
        
    # Enhanced pattern to catch more tool call variations + handle malformed calls
    tool_calls_patterns = [
        r'<tool>(.*?)\((.*?)\)</tool>',  # Standard format: <tool>name(params)</tool>
        r'<tool>(.*?):\s*(.*?)</tool>',  # Alternative format: <tool>name: params</tool>
        r'<tool>(\w+)(?:\s+.*?)?</tool>', # Malformed: <tool>name with extra text</tool> - extract just the tool name
        r'Tool:\s*(.*?)\((.*?)\)',       # Plain format
    ]
    
    tool_calls = []
    for pattern in tool_calls_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            print(f"[DEBUG] Tool extraction: Found {len(matches)} tool calls with pattern: {pattern}")
            # Handle both tuple (tool_name, params) and single tool_name matches
            normalized_matches = []
            for match in matches:
                if isinstance(match, tuple) and len(match) == 2:
                    # Standard format: (tool_name, params)
                    normalized_matches.append(match)
                elif isinstance(match, str):
                    # Malformed format: just tool_name, add empty params
                    print(f"[DEBUG] Tool extraction: Fixed malformed tool call: {match}")
                    normalized_matches.append((match, "{}"))
                else:
                    # Fallback for unexpected formats
                    print(f"[WARNING] Tool extraction: Unexpected match format: {match}")
                    continue
            tool_calls.extend(normalized_matches)
            break  # Use first matching pattern only
    
    # If no tool calls found, return empty list
    if not tool_calls:
        print("[DEBUG] Tool extraction: No tool calls found in LLM response")
        print(f"[DEBUG] Tool extraction: Scanned text: {text[:200]}...")
        return []
    
    # Deduplicate tool calls (same tool with same parameters)
    unique_tools = []
    seen_calls = set()
    for tool_name, params_str in tool_calls:
        call_signature = (tool_name.strip(), params_str.strip())
        if call_signature not in seen_calls:
            unique_tools.append((tool_name, params_str))
            seen_calls.add(call_signature)
        else:
            print(f"[DEBUG] Tool extraction: Skipping duplicate call to {tool_name}")
    
    tool_calls = unique_tools
    
    # Limit tool calls to prevent infinite loops (configurable via MCP settings)
    from app.core.mcp_tools_cache import get_max_tool_calls
    MAX_TOOL_CALLS = get_max_tool_calls()
    if len(tool_calls) > MAX_TOOL_CALLS:
        print(f"[WARNING] Tool extraction: Limited tool calls from {len(tool_calls)} to {MAX_TOOL_CALLS} to prevent infinite loops")
        tool_calls = tool_calls[:MAX_TOOL_CALLS]
    
    print(f"[DEBUG] Tool extraction: {len(tool_calls)} unique tools to execute")
    
    results = []
    for i, (tool_name, params_str) in enumerate(tool_calls):
        try:
            tool_name = tool_name.strip()
            params_str = params_str.strip()
            
            if stream_callback:
                stream_callback(f"🔧 Executing tool: {tool_name}")
            
            print(f"[DEBUG] Tool execution [{i+1}/{len(tool_calls)}]: {tool_name}")
            
            # Skip knowledge_search tools if skip_knowledge_search is enabled (prevents redundant RAG operations)
            # Handle both exact matches and malformed tool names containing get_rag_tool_name()
            if skip_knowledge_search and is_rag_tool(tool_name):
                print(f"[DEBUG] Skipping knowledge_search tool call to prevent redundant RAG operation (tool_name: {tool_name})")
                if stream_callback:
                    stream_callback(f"⏭️ Skipping knowledge_search (RAG already performed)")
                results.append({
                    "tool": tool_name,
                    "parameters": params_str,
                    "success": True,
                    "result": {"message": "Skipped to prevent redundant RAG operation"},
                    "skipped": True
                })
                continue
            
            # Parse parameters with enhanced error handling
            if params_str == "{}" or params_str == "" or params_str.lower() == "none":
                params = {}
            else:
                try:
                    # Handle common parameter format issues
                    if not params_str.startswith('{'):
                        # Try to wrap as object if it looks like key-value
                        if ':' in params_str:
                            params_str = '{' + params_str + '}'
                        else:
                            # Assume it's a single string parameter for query-like tools
                            if 'search' in tool_name.lower():
                                params = {"query": params_str.strip('"\'`')}
                            else:
                                params = {}
                    else:
                        params = json.loads(params_str)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Failed to parse parameters for {tool_name}: {params_str}")
                    print(f"[ERROR] JSON decode error: {e}")
                    # Try fallback parsing for common cases
                    if 'search' in tool_name.lower():
                        params = {"query": params_str.strip('"\'`{}()')}
                    else:
                        params = {}
            
            print(f"[DEBUG] Tool execution: {tool_name} with params: {params}")
            
            # Create tool span if trace is provided
            tool_span = None
            if trace:
                from app.core.langfuse_integration import get_tracer
                tracer = get_tracer()
                if tracer.is_enabled():
                    print(f"[DEBUG] RAG chat creating tool span for {tool_name}")
                    tool_span = tracer.create_tool_span(trace, tool_name, params)
                    print(f"[DEBUG] RAG chat tool span created: {tool_span is not None}")
            
            # Execute the tool with enhanced error handling
            if use_enhanced_error_handling:
                from app.core.tool_error_handler import call_mcp_tool_with_retry
                
                # Execute with enhanced error handling (uses server-specific configuration automatically)
                result = await call_mcp_tool_with_retry(tool_name, params, trace=trace)
            else:
                # Execute the tool (skip span creation since we already created it above)
                result = call_mcp_tool(tool_name, params, trace=trace, _skip_span_creation=True)
            
            # Enhanced result processing with error type support
            if "error" in result:
                error_msg = result['error']
                error_type = result.get('error_type', 'unknown')
                attempts = result.get('attempts', 1)
                
                if attempts > 1:
                    print(f"[ERROR] Tool {tool_name} failed after {attempts} attempts ({error_type}): {error_msg}")
                    if stream_callback:
                        stream_callback(f"❌ Tool {tool_name} failed after {attempts} retries ({error_type}): {error_msg}")
                else:
                    print(f"[ERROR] Tool {tool_name} returned error ({error_type}): {error_msg}")
                    if stream_callback:
                        stream_callback(f"❌ Tool {tool_name} failed ({error_type}): {error_msg}")
                
                # End tool span with error
                if tool_span:
                    print(f"[DEBUG] RAG chat ending tool span for {tool_name} with error: {error_msg}")
                    tracer.end_span_with_result(tool_span, None, False, error_msg)
                
                results.append({
                    "tool": tool_name,
                    "parameters": params,
                    "success": False,
                    "error": error_msg,
                    "error_type": error_type,
                    "attempts": attempts
                })
            else:
                print(f"[SUCCESS] Tool {tool_name} executed successfully")
                
                # End tool span with success
                if tool_span:
                    print(f"[DEBUG] RAG chat ending tool span for {tool_name} with success")
                    tracer.end_span_with_result(tool_span, result, True)
                
                # Special formatting for different tool types
                if tool_name == "get_datetime" and "result" in result:
                    try:
                        from datetime import datetime
                        iso_date = result["result"]
                        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
                        formatted_date = dt.strftime("%B %d, %Y at %I:%M %p")
                        result["formatted_date"] = formatted_date
                    except Exception as e:
                        print(f"[ERROR] Failed to format date: {str(e)}")
                
                results.append({
                    "tool": tool_name,
                    "parameters": params,
                    "success": True,
                    "result": result
                })
                
                if stream_callback:
                    stream_callback(f"✅ Tool {tool_name} completed successfully")
        
        except Exception as e:
            error_msg = f"Unexpected error executing {tool_name}: {str(e)}"
            print(f"[ERROR] {error_msg}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            
            results.append({
                "tool": tool_name,
                "parameters": params if 'params' in locals() else {},
                "success": False,
                "error": error_msg,
                "error_type": "exception"
            })
            
            if stream_callback:
                stream_callback(f"❌ Tool {tool_name} failed: {error_msg}")
    
    print(f"[DEBUG] Tool execution complete: {len(results)} results")
    return results

def extract_and_execute_tool_calls(text, stream_callback=None, trace=None, use_enhanced_error_handling=False, skip_knowledge_search=False, original_query=None):
    """
    Synchronous wrapper for extract_and_execute_tool_calls_async
    
    Args:
        text: LLM response text to scan for tool calls
        stream_callback: Optional callback to stream tool execution updates
        trace: Optional Langfuse trace for span creation
        use_enhanced_error_handling: Whether to use enhanced error handling (default: False for compatibility)
        skip_knowledge_search: Whether to skip redundant knowledge_search calls (default: False)
        original_query: The original query for redundancy checking (default: None)
    
    Returns:
        List of tool execution results
    """
    import asyncio
    
    # If enhanced error handling is requested, use async version
    if use_enhanced_error_handling:
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, need to run in thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(extract_and_execute_tool_calls_async(text, stream_callback, trace, use_enhanced_error_handling, skip_knowledge_search, original_query))
                )
                return future.result()
        except RuntimeError:
            # No running loop, we can use asyncio.run
            try:
                return asyncio.run(extract_and_execute_tool_calls_async(text, stream_callback, trace, use_enhanced_error_handling, skip_knowledge_search, original_query))
            except Exception as e:
                print(f"[ERROR] Async execution failed: {e}")
                # Fall back to sync implementation
                use_enhanced_error_handling = False
    
    # Fallback to original synchronous implementation for compatibility
    import re  # Import at function level to avoid scope issues
    
    if "NO_TOOLS_NEEDED" in text:
        print("[DEBUG] Tool extraction: LLM indicated no tools needed")
        return []
        
    # Enhanced pattern to catch more tool call variations + handle malformed calls
    tool_calls_patterns = [
        r'<tool>(.*?)\((.*?)\)</tool>',  # Standard format: <tool>name(params)</tool>
        r'<tool>(.*?):\s*(.*?)</tool>',  # Alternative format: <tool>name: params</tool>
        r'<tool>(\w+)(?:\s+.*?)?</tool>', # Malformed: <tool>name with extra text</tool> - extract just the tool name
        r'Tool:\s*(.*?)\((.*?)\)',       # Plain format
    ]
    
    tool_calls = []
    for pattern in tool_calls_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            print(f"[DEBUG] Tool extraction: Found {len(matches)} tool calls with pattern: {pattern}")
            # Handle both tuple (tool_name, params) and single tool_name matches
            normalized_matches = []
            for match in matches:
                if isinstance(match, tuple) and len(match) == 2:
                    # Standard format: (tool_name, params)
                    normalized_matches.append(match)
                elif isinstance(match, str):
                    # Malformed format: just tool_name, add empty params
                    print(f"[DEBUG] Tool extraction: Fixed malformed tool call: {match}")
                    normalized_matches.append((match, "{}"))
                else:
                    # Fallback for unexpected formats
                    print(f"[WARNING] Tool extraction: Unexpected match format: {match}")
                    continue
            tool_calls.extend(normalized_matches)
            break  # Use first matching pattern only
    
    # If no tool calls found, return empty list
    if not tool_calls:
        print("[DEBUG] Tool extraction: No tool calls found in LLM response")
        print(f"[DEBUG] Tool extraction: Scanned text: {text[:200]}...")
        return []
    
    # Deduplicate tool calls (same tool with same parameters)
    unique_tools = []
    seen_calls = set()
    for tool_name, params_str in tool_calls:
        call_signature = (tool_name.strip(), params_str.strip())
        if call_signature not in seen_calls:
            unique_tools.append((tool_name, params_str))
            seen_calls.add(call_signature)
        else:
            print(f"[DEBUG] Tool extraction: Skipping duplicate call to {tool_name}")
    
    tool_calls = unique_tools
    
    # Limit tool calls to prevent infinite loops (configurable via MCP settings)
    from app.core.mcp_tools_cache import get_max_tool_calls
    MAX_TOOL_CALLS = get_max_tool_calls()
    if len(tool_calls) > MAX_TOOL_CALLS:
        print(f"[WARNING] Tool extraction: Limited tool calls from {len(tool_calls)} to {MAX_TOOL_CALLS} to prevent infinite loops")
        tool_calls = tool_calls[:MAX_TOOL_CALLS]
    
    print(f"[DEBUG] Tool extraction: {len(tool_calls)} unique tools to execute")
    
    results = []
    for i, (tool_name, params_str) in enumerate(tool_calls):
        try:
            tool_name = tool_name.strip()
            params_str = params_str.strip()
            
            if stream_callback:
                stream_callback(f"🔧 Executing tool: {tool_name}")
            
            print(f"[DEBUG] Tool execution [{i+1}/{len(tool_calls)}]: {tool_name}")
            
            # Parse parameters with enhanced error handling
            if params_str == "{}" or params_str == "" or params_str.lower() == "none":
                params = {}
            else:
                try:
                    # Handle common parameter format issues
                    if not params_str.startswith('{'):
                        # Try to wrap as object if it looks like key-value
                        if ':' in params_str:
                            params_str = '{' + params_str + '}'
                        else:
                            # Assume it's a single string parameter for query-like tools
                            if 'search' in tool_name.lower():
                                params = {"query": params_str.strip('"\'`')}
                            else:
                                params = {}
                    else:
                        params = json.loads(params_str)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Failed to parse parameters for {tool_name}: {params_str}")
                    print(f"[ERROR] JSON decode error: {e}")
                    # Try fallback parsing for common cases
                    if 'search' in tool_name.lower():
                        params = {"query": params_str.strip('"\'`{}()')}
                    else:
                        params = {}
            
            print(f"[DEBUG] Tool execution: {tool_name} with params: {params}")
            
            # Create tool span if trace is provided
            tool_span = None
            if trace:
                from app.core.langfuse_integration import get_tracer
                tracer = get_tracer()
                if tracer.is_enabled():
                    print(f"[DEBUG] RAG chat creating tool span for {tool_name}")
                    tool_span = tracer.create_tool_span(trace, tool_name, params)
                    print(f"[DEBUG] RAG chat tool span created: {tool_span is not None}")
            
            # Execute the tool with enhanced error handling
            if use_enhanced_error_handling:
                from app.core.tool_error_handler import call_mcp_tool_with_retry
                
                # Execute with enhanced error handling in a new event loop (uses server-specific configuration automatically)
                try:
                    result = asyncio.run(call_mcp_tool_with_retry(tool_name, params, trace=trace))
                except Exception as e:
                    print(f"[ERROR] Enhanced error handling failed: {e}, falling back to direct call")
                    result = call_mcp_tool(tool_name, params, trace=trace, _skip_span_creation=True)
            else:
                # Execute the tool (skip span creation since we already created it above)
                result = call_mcp_tool(tool_name, params, trace=trace, _skip_span_creation=True)
            
            # Enhanced result processing with error type support
            if "error" in result:
                error_msg = result['error']
                error_type = result.get('error_type', 'unknown')
                attempts = result.get('attempts', 1)
                
                if attempts > 1:
                    print(f"[ERROR] Tool {tool_name} failed after {attempts} attempts ({error_type}): {error_msg}")
                    if stream_callback:
                        stream_callback(f"❌ Tool {tool_name} failed after {attempts} retries ({error_type}): {error_msg}")
                else:
                    print(f"[ERROR] Tool {tool_name} returned error ({error_type}): {error_msg}")
                    if stream_callback:
                        stream_callback(f"❌ Tool {tool_name} failed ({error_type}): {error_msg}")
                
                # End tool span with error
                if tool_span:
                    print(f"[DEBUG] RAG chat ending tool span for {tool_name} with error: {error_msg}")
                    tracer.end_span_with_result(tool_span, None, False, error_msg)
                
                results.append({
                    "tool": tool_name,
                    "parameters": params,
                    "success": False,
                    "error": error_msg,
                    "error_type": error_type,
                    "attempts": attempts
                })
            else:
                print(f"[SUCCESS] Tool {tool_name} executed successfully")
                
                # End tool span with success
                if tool_span:
                    print(f"[DEBUG] RAG chat ending tool span for {tool_name} with success")
                    tracer.end_span_with_result(tool_span, result, True)
                
                # Special formatting for different tool types
                if tool_name == "get_datetime" and "result" in result:
                    try:
                        from datetime import datetime
                        iso_date = result["result"]
                        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
                        formatted_date = dt.strftime("%B %d, %Y at %I:%M %p")
                        result["formatted_date"] = formatted_date
                    except Exception as e:
                        print(f"[ERROR] Failed to format date: {str(e)}")
                
                results.append({
                    "tool": tool_name,
                    "parameters": params,
                    "success": True,
                    "result": result
                })
                
                if stream_callback:
                    stream_callback(f"✅ Tool {tool_name} completed successfully")
                    
        except Exception as e:
            print(f"[ERROR] Failed to execute tool call {tool_name}: {str(e)}")
            results.append({
                "tool": tool_name,
                "parameters": params_str,
                "success": False,
                "error": str(e)
            })
            if stream_callback:
                stream_callback(f"❌ Tool {tool_name} failed: {str(e)}")
    
    print(f"[DEBUG] Tool extraction complete: {len(results)} tools executed")


def validate_response_against_tools(response_text: str, tool_context: str) -> Dict[str, Any]:
    """
    Validate that LLM response doesn't contradict tool results.
    
    Args:
        response_text: The generated response from LLM
        tool_context: The context from tool execution results
    
    Returns:
        dict: Validation results with conflict detection and confidence
    """
    import re
    
    if not response_text or not tool_context:
        return {
            'has_conflict': False,
            'confidence': 0.0,
            'conflicts': [],
            'validation_passed': True
        }
    
    response_lower = response_text.lower()
    tool_lower = tool_context.lower()
    
    detected_conflicts = []
    
    # Critical existence vs non-existence patterns
    existence_patterns = [
        {
            'entity_variants': [
                'chatgpt-5', 'chatgpt 5', 'gpt-5', 'gpt 5', 'chatgpt5', 'gpt5'
            ],
            'positive_tool_indicators': [
                'is out', 'released', 'available', 'launched', 'exists', 'smarter than all of us',
                'now.*available', 'has.*released', 'jul.*2025', 'july.*2025'
            ],
            'negative_response_indicators': [
                'does not exist', 'doesn\'t exist', 'not exist', 'no.*official.*release',
                'persistent myth', 'is.*myth', 'not.*real', 'no.*credible.*evidence',
                'as of.*does not exist'
            ],
            'confidence_boost_phrases': [
                # Tool context phrases that indicate high confidence
                'is out', 'smarter than all of us', 'jul.*2025', 'july.*2025'
            ]
        }
    ]
    
    for pattern in existence_patterns:
        # Check if any entity variant is mentioned in either context
        entity_in_tools = any(
            re.search(entity, tool_lower) for entity in pattern['entity_variants']
        )
        entity_in_response = any(
            re.search(entity, response_lower) for entity in pattern['entity_variants'] 
        )
        
        if not (entity_in_tools or entity_in_response):
            continue
            
        # Check for positive indicators in tool context
        positive_in_tools = any(
            re.search(indicator, tool_lower) for indicator in pattern['positive_tool_indicators']
        )
        
        # Check for negative indicators in response
        negative_in_response = any(
            re.search(indicator, response_lower) for indicator in pattern['negative_response_indicators']
        )
        
        if positive_in_tools and negative_in_response:
            # Calculate confidence based on specific phrases
            confidence = 0.8  # Base confidence for existence conflicts
            
            # Boost confidence for high-certainty tool phrases
            if any(re.search(phrase, tool_lower) for phrase in pattern['confidence_boost_phrases']):
                confidence = 0.95
                
            # Boost confidence for definitive negative statements in response
            if any(phrase in response_lower for phrase in ['does not exist', 'persistent myth', 'no official release']):
                confidence = min(confidence + 0.1, 1.0)
            
            detected_conflicts.append({
                'type': 'existence_contradiction',
                'confidence': confidence,
                'tool_evidence': [
                    indicator for indicator in pattern['positive_tool_indicators'] 
                    if re.search(indicator, tool_lower)
                ],
                'response_evidence': [
                    indicator for indicator in pattern['negative_response_indicators']
                    if re.search(indicator, response_lower)
                ],
                'entity_detected': next(
                    (entity for entity in pattern['entity_variants'] if re.search(entity, tool_lower)),
                    pattern['entity_variants'][0]
                )
            })
    
    # Determine overall validation result
    has_conflict = len(detected_conflicts) > 0
    max_confidence = max([c['confidence'] for c in detected_conflicts]) if detected_conflicts else 0.0
    
    validation_result = {
        'has_conflict': has_conflict,
        'confidence': max_confidence,
        'conflicts': detected_conflicts,
        'validation_passed': not has_conflict or max_confidence < 0.7,
        'severity': 'critical' if max_confidence >= 0.9 else 'high' if max_confidence >= 0.7 else 'medium'
    }
    
    if has_conflict:
        logger.warning(f"[RESPONSE VALIDATION] Detected {len(detected_conflicts)} conflicts with confidence {max_confidence:.2f}")
        for conflict in detected_conflicts:
            logger.warning(f"[CONFLICT] {conflict['type']}: {conflict['entity_detected']} - confidence {conflict['confidence']:.2f}")
    
    return validation_result