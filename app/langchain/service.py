import requests
import re
import httpx
import json
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from app.core.llm_settings_cache import get_llm_settings
from app.core.embedding_settings_cache import get_embedding_settings
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.core.mcp_tools_cache import get_enabled_mcp_tools
from app.api.v1.endpoints.document import HTTPEmbeddingFunction
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from app.rag.bm25_processor import BM25Processor, BM25CorpusManager

# Enhanced conversation management with Redis fallback
_conversation_cache = {}  # In-memory fallback

# Simple query cache for RAG results (in-memory, expires after 5 minutes)
import time
_rag_cache = {}  # {query_hash: (result, timestamp)}
RAG_CACHE_TTL = 300  # 5 minutes
RAG_CACHE_MAX_SIZE = 100  # Maximum cache entries to prevent memory bloat

def _cache_rag_result(query_hash: int, result: tuple, current_time: float):
    """Helper function to cache RAG results with cleanup"""
    _rag_cache[query_hash] = (result, current_time)
    # Clean up cache if it gets too large
    if len(_rag_cache) > RAG_CACHE_MAX_SIZE:
        # Remove oldest entries
        sorted_cache = sorted(_rag_cache.items(), key=lambda x: x[1][1])
        for old_key, _ in sorted_cache[:20]:  # Remove 20 oldest entries
            del _rag_cache[old_key]

def get_redis_conversation_client():
    """Get Redis client specifically for conversation storage"""
    try:
        from app.core.redis_client import get_redis_client
        return get_redis_client()
    except Exception as e:
        print(f"[DEBUG] Redis client not available for conversations: {e}")
        return None

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

def get_limited_conversation_history(conversation_id: str, max_messages: int = 2) -> str:
    """Get limited conversation history for simple factual queries"""
    if not conversation_id:
        return ""
    
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
    """Store a message in conversation history with Redis support"""
    if not conversation_id:
        return
    
    from app.core.large_generation_utils import get_config_accessor
    config = get_config_accessor()
    
    message = {
        "role": role,
        "content": content,
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

def build_prompt(prompt: str, thinking: bool = False, is_internal: bool = False) -> str:
    if thinking:
        return (
            "Please show your reasoning step by step before giving the final answer.\n"
            + prompt
        )
    return prompt

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
    
    # Extract numbers that might indicate quantity (exclude years and contextual numbers)
    import re
    all_numbers = re.findall(r'\b(\d+)\b', question)
    
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
- "What time is it now?" → TOOLS (get_datetime tool available)
- "Explain machine learning" → LLM (general knowledge)
- "How's Apple's AI progress?" → RAG (check for market research/competitor analysis docs first)
- "How's DBS progress in AI?" → RAG (check for industry reports/analysis first)
- "What is photosynthesis?" → LLM (pure general knowledge)
- "Send email to John" → TOOLS (outlook_send_message available)
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
    with httpx.Client(timeout=30.0) as client:  # Add timeout to prevent hanging
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
            # Simple heuristics for tool matching
            if any(keyword in question_lower for keyword in ["time", "date", "now"]) and "get_datetime" in available_tools:
                tool_can_help = True
            elif any(keyword in question_lower for keyword in ["email", "send", "message"]) and "outlook_send_message" in available_tools:
                tool_can_help = True
            elif any(keyword in question_lower for keyword in ["jira", "issue", "ticket"]) and any("jira" in tool for tool in available_tools):
                tool_can_help = True
        
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

def execute_tools_first(question: str, thinking: bool = False) -> tuple:
    """
    Execute tools first, then generate answer with tool results
    Returns: (tool_results, updated_question_with_context)
    """
    print(f"[DEBUG] execute_tools_first: question = {question}, thinking = {thinking}")
    
    # Get tools context
    mcp_tools_context = get_mcp_tools_context()
    
    # First, ask LLM to identify which tools to use
    tool_selection_prompt = f"""IMPORTANT: Output ONLY the tool call format shown below, no other text or explanation.

Question: {question}

{mcp_tools_context}

Based on the question and available tools, output the exact tool call in this format:
<tool>tool_name(parameters)</tool>

Examples:
- For "What time is it?": <tool>get_datetime({{}})</tool>
- For "What is today date & time?": <tool>get_datetime({{}})</tool>
- For "find emails from amazon": <tool>search_emails({{"query": "from:amazon"}})</tool>
- For "search gmail for amazon": <tool>search_emails({{"query": "amazon"}})</tool>
- For "find latest gmail": <tool>search_emails({{"query": ""}})</tool>
- For "search gmail for unread": <tool>search_emails({{"query": "is:unread"}})</tool>
- For "show today's emails": <tool>search_emails({{"query": "newer_than:1d"}})</tool>
- For "send email": <tool>send_email({{"to": ["email@example.com"], "subject": "...", "body": "..."}})</tool>

If the question is asking about emails or Gmail, you MUST use search_emails.
If no tools are needed to answer the question: NO_TOOLS_NEEDED

CRITICAL: You MUST output ONLY the tool call or NO_TOOLS_NEEDED, nothing else."""

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
    with httpx.Client(timeout=30.0) as client:  # Add timeout to prevent hanging
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
    
    return tool_results, updated_question, results_context

def llm_expand_query(question: str, llm_cfg: dict) -> list:
    """Use LLM to generate alternative queries for better retrieval"""
    
    # Check if llm_cfg is valid
    if not isinstance(llm_cfg, dict):
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
        
        try:
            text = ""
            with httpx.Client(timeout=10) as client:
                with client.stream("POST", llm_api_url, json=payload) as response:
                    for line in response.iter_lines():
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
            
            # Parse alternatives
            alternatives = [q.strip() for q in text.strip().split('\n') if q.strip() and len(q.strip()) > 5]
            expanded_queries.extend(alternatives[:2])  # Take up to 2 alternatives
            
        except Exception as e:
            print(f"[DEBUG] LLM query expansion failed: {str(e)}")
    
    # Normalize all queries to lowercase for consistent searching
    normalized_queries = [query.lower().strip() for query in expanded_queries]
    
    print(f"[DEBUG] Expanded queries: {normalized_queries}")
    return normalized_queries

def keyword_search_milvus(question: str, collection_name: str, uri: str, token: str) -> list:
    """Enhanced keyword search with smart term extraction"""
    from pymilvus import Collection, connections
    import re
    
    try:
        connections.connect(uri=uri, token=token, alias="keyword_search")
        collection = Collection(collection_name, using="keyword_search")
        collection.load()
        
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

def handle_rag_query(question: str, thinking: bool = False, collections: List[str] = None, collection_strategy: str = "auto") -> tuple:
    """Handle RAG queries with hybrid search (vector + keyword) - returns context only
    
    Args:
        question: The user's question
        thinking: Whether to enable extended thinking
        collections: List of collection names to search (None = auto-detect)
        collection_strategy: "auto", "specific", or "all"
    """
    print(f"[DEBUG] handle_rag_query: question = {question}, thinking = {thinking}, collections = {collections}, strategy = {collection_strategy}")
    
    # Check cache first for performance
    query_hash = hash(question.lower().strip())
    current_time = time.time()
    
    if query_hash in _rag_cache:
        cached_result, timestamp = _rag_cache[query_hash]
        if current_time - timestamp < RAG_CACHE_TTL:
            print(f"[DEBUG] handle_rag_query: Using cached result for query")
            return cached_result
        else:
            # Remove expired cache entry
            del _rag_cache[query_hash]
    
    try:
        embedding_cfg = get_embedding_settings()
        print(f"[DEBUG] embedding_cfg type: {type(embedding_cfg)}, value: {embedding_cfg}")
        vector_db_cfg = get_vector_db_settings()
        print(f"[DEBUG] vector_db_cfg type: {type(vector_db_cfg)}, value: {vector_db_cfg}")
        llm_cfg = get_llm_settings()
        print(f"[DEBUG] llm_cfg type: {type(llm_cfg)}, value: {llm_cfg}")
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
    
    if embedding_endpoint and isinstance(embedding_endpoint, str) and embedding_endpoint.strip():
        embeddings = HTTPEmbeddingFunction(embedding_endpoint)
        print(f"[DEBUG] handle_rag_query: Using HTTP embedding endpoint")
    else:
        # Use default model if embedding_model is not a valid string
        model_name = embedding_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5")
        if not isinstance(model_name, str) or not model_name.strip():
            model_name = "BAAI/bge-base-en-v1.5"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        print(f"[DEBUG] handle_rag_query: Using HuggingFace embeddings with model: {model_name}")
    
    # Determine which collections to search
    milvus_cfg = vector_db_cfg["milvus"]
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
        # Auto-detect relevant collections based on query
        from app.core.document_classifier import get_document_classifier
        classifier = get_document_classifier()
        
        # Use query analysis to determine collection type
        collection_type = classifier.classify_document(question, {"query": True})
        target_collection = classifier.get_target_collection(collection_type)
        
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
    SIMILARITY_THRESHOLD = 1.5  # Very inclusive for initial retrieval (let re-ranking filter)
    NUM_DOCS = 20  # Reduced from 50 to 20 for better performance while maintaining quality
    
    # Use LLM to expand queries for better recall
    queries_to_try = llm_expand_query(question, llm_cfg)
    
    try:
        # HYBRID SEARCH: Run both vector and keyword search across all collections
        all_docs = []
        seen_ids = set()
        
        # Search each collection
        for collection_name in collections_to_search:
            print(f"[DEBUG] Searching collection: {collection_name}")
            
            # Create Milvus store for this collection
            milvus_store = Milvus(
                embedding_function=embeddings,
                collection_name=collection_name,
                connection_args={"uri": uri, "token": token},
                text_field="content"
            )
            
            # 1. Vector search with query expansion
            for query in queries_to_try:
                try:
                    # Normalize query to lowercase for consistent vector search
                    normalized_query = query.lower().strip()
                    vector_search_start = time.time()
                    docs = milvus_store.similarity_search_with_score(normalized_query, k=NUM_DOCS) if hasattr(milvus_store, 'similarity_search_with_score') else [(doc, 0.0) for doc in milvus_store.similarity_search(normalized_query, k=NUM_DOCS)]
                    vector_search_end = time.time()
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
            keyword_docs = keyword_search_milvus(
                question,
                collection_name,
                uri=milvus_cfg.get("MILVUS_URI"),
                token=milvus_cfg.get("MILVUS_TOKEN")
            )
            
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
                return "", ""
            
        print(f"[DEBUG] Total unique documents found: {len(all_docs)} (vector + keyword)")
        
        # Sort by score (lower is better for cosine distance)
        docs = sorted(all_docs, key=lambda x: x[1])[:NUM_DOCS]
        
    except Exception as e:
        print(f"[ERROR] handle_rag_query: Failed to search vector store: {str(e)}")
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
            # Calculate keyword-based relevance for reranking (hybrid search)
            keyword_relevance = calculate_relevance_score(question, doc.page_content)
            
            # Hybrid search: Balance vector similarity and keyword relevance
            # Use query analysis to determine optimal weighting
            keyword_weight = query_analysis['keyword_priority']
            vector_weight = 1 - keyword_weight
            combined_score = (similarity * vector_weight) + (keyword_relevance * keyword_weight)
            
            filtered_and_ranked.append((doc, score, similarity, keyword_relevance, combined_score))
    
    # Sort by combined score (higher is better)
    filtered_and_ranked.sort(key=lambda x: x[4], reverse=True)
    
    # Qwen3-Reranker-4B based re-ranking for top candidates
    if filtered_and_ranked and llm_cfg:
        try:
            # Check if reranker is enabled
            from app.core.reranker_config import RerankerConfig
            
            if not RerankerConfig.is_enabled():
                raise ImportError("Reranker is disabled in this environment")
            
            # Try to use Qwen3-Reranker-4B if available
            from app.rag.qwen_reranker import get_qwen_reranker
            
            # Re-rank top 20 candidates (we can handle more with dedicated model)
            num_to_rerank = min(20, len(filtered_and_ranked))
            print(f"[DEBUG] Starting Qwen3-Reranker-4B re-ranking of top {num_to_rerank} documents")
            
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
            
            # Perform reranking with hybrid scoring
            rerank_results = reranker.rerank_with_hybrid_score(
                query=question,
                documents=docs_to_rerank,
                rerank_weight=0.7,  # Give 70% weight to Qwen reranker
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
                llm_api_url = "http://localhost:8000/api/v1/generate_stream"
                payload = {
                    "prompt": rerank_prompt,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 100
                }
                
                scores_text = ""
                with httpx.Client(timeout=15) as client:
                    with client.stream("POST", llm_api_url, json=payload) as response:
                        for line in response.iter_lines():
                            if not line:
                                continue
                            if isinstance(line, bytes):
                                line = line.decode("utf-8")
                            if line.startswith("data: "):
                                token = line.replace("data: ", "")
                                scores_text += token
                
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
    
    # Take top documents after reranking, but ensure diversity
    filtered_docs = []
    seen_content_hashes = set()
    
    for item in filtered_and_ranked:
        doc = item[0]
        # Create a simple hash of the first 200 chars to avoid duplicate content
        content_hash = hash(doc.page_content[:200])
        
        if content_hash not in seen_content_hashes:
            seen_content_hashes.add(content_hash)
            filtered_docs.append(doc)
            
        if len(filtered_docs) >= 8:  # Take up to 8 diverse documents
            break
    
    print(f"[DEBUG] handle_rag_query: After filtering and reranking: {len(filtered_docs)} documents")
    print(f"[DEBUG] handle_rag_query: Filtered {len(docs) - len(filtered_and_ranked)} documents by similarity threshold {SIMILARITY_THRESHOLD}")
    
    context = "\n\n".join([doc.page_content for doc in filtered_docs])
    
    # Extract sources with collection information
    sources = []
    for doc in filtered_docs:
        source_info = {
            "file": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", 0),
            "collection": doc.metadata.get("source_collection", "default_knowledge")
        }
        sources.append(source_info)
    
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
        # Dynamic threshold based on query complexity
        min_relevance = 0.15 if len(query_keywords) > 2 else 0.25
        if relevance_score < min_relevance:
            print(f"[RAG DEBUG] Low relevance detected ({relevance_score:.2f} < {min_relevance}), no context returned")
            result = ("", "")
            # Cache the result for performance
            _cache_rag_result(query_hash, result, current_time)
            return result
        
        # Context is relevant, return it for hybrid processing
        print(f"[RAG DEBUG] Good relevance ({relevance_score:.2f}), returning context for hybrid approach")
        result = (context, "")  # Return empty string for prompt since we handle prompting in main function
        # Cache the result for performance
        _cache_rag_result(query_hash, result, current_time)
        return result
    else:
        print(f"[RAG DEBUG] No relevant documents found")
        result = ("", "")
        # Cache the result for performance
        _cache_rag_result(query_hash, result, current_time)
        return result

def make_llm_call(prompt: str, thinking: bool, context: str, llm_cfg: dict) -> str:
    """Make a synchronous LLM API call and return the complete response"""
    llm_api_url = "http://localhost:8000/api/v1/generate_stream"
    mode_config = llm_cfg["thinking_mode"] if thinking else llm_cfg["non_thinking_mode"]
    
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
    
    # Make synchronous request and collect all tokens
    text = ""
    try:
        with httpx.Client(timeout=60.0) as client:
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
    mode_config = llm_cfg["thinking_mode"] if thinking else llm_cfg["non_thinking_mode"]
    
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
            with httpx.Client(timeout=60.0) as client:
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
    
    # Get LLM settings
    llm_cfg = get_llm_settings()
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

def rag_answer(question: str, thinking: bool = False, stream: bool = False, conversation_id: str = None, use_langgraph: bool = True, collections: List[str] = None, collection_strategy: str = "auto"):
    """Main function with hybrid RAG+LLM approach prioritizing answer quality
    
    Args:
        question: The user's question
        thinking: Whether to enable extended thinking
        stream: Whether to stream the response
        conversation_id: Conversation ID for context
        use_langgraph: Whether to use LangGraph agents
        collections: List of collection names to search (None = auto-detect)
        collection_strategy: "auto", "specific", or "all"
    """
    print(f"[DEBUG] rag_answer: incoming question = {question}")
    print(f"[DEBUG] rag_answer: conversation_id = {conversation_id}")
    print(f"[DEBUG] rag_answer: collections = {collections}, strategy = {collection_strategy}")
    
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
    # Validate configuration
    llm_cfg = get_llm_settings()
    required_fields = ["model", "thinking_mode", "non_thinking_mode", "max_tokens"]
    missing = [f for f in required_fields if f not in llm_cfg or llm_cfg[f] is None]
    if missing:
        raise RuntimeError(f"Missing required LLM config fields: {', '.join(missing)}")
    
    # STEP 1: Check if this requires large generation (chunked processing) FIRST
    query_type = classify_query_type(question, llm_cfg)
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
        # STEP 2a: For TOOLS queries, skip RAG and execute tools immediately
        if query_type == "TOOLS":
            print(f"[DEBUG] rag_answer: TOOLS query detected - skipping RAG retrieval")
            rag_context = ""  # No RAG needed for tool queries
        else:
            # For non-large generation and non-tools queries, do full RAG retrieval
            print(f"[DEBUG] rag_answer: Step 2a - Attempting full RAG retrieval")
            import time
            rag_start_time = time.time()
            rag_context, _ = handle_rag_query(question, thinking, collections, collection_strategy)
            rag_end_time = time.time()
            print(f"[DEBUG] rag_answer: RAG retrieval took {rag_end_time - rag_start_time:.2f} seconds")
            print(f"[DEBUG] rag_answer: RAG context length = {len(rag_context) if rag_context else 0}")
    
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
        
        # Get LIMITED conversation history for context to prevent bleeding
        conversation_history_text = get_limited_conversation_history(conversation_id, max_messages=3) if conversation_id else ""
        conversation_history_list = get_full_conversation_history(conversation_id) if conversation_id else []
        # Also limit the list version to prevent bleeding
        if len(conversation_history_list) > 6:  # 3 user + 3 assistant messages max
            conversation_history_list = conversation_history_list[-6:]
        
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
    
    tool_calls = []
    tool_context = ""
    if query_type == "TOOLS":
        tool_calls, _, tool_context = execute_tools_first(question, thinking)
        if not tool_calls:
            print(f"[DEBUG] rag_answer: No tools actually executed despite TOOLS classification")
            # For TOOLS queries, we should still prioritize tool-based response
            # even if tool execution failed
    
    # Get conversation history with smart filtering for ALL query types
    conversation_history = ""
    history_prompt = ""
    if conversation_id:
        # ALWAYS use limited conversation history to prevent context bleeding
        conversation_history = get_limited_conversation_history(conversation_id, max_messages=3)
        
        if conversation_history:
            history_prompt = f"Previous conversation:\n{conversation_history}\n\n"
    
    # STEP 4: Hybrid approach - combine available sources for optimal answer
    context = ""
    source = ""
    
    if rag_context and tool_calls:
        # BEST CASE: RAG + TOOLS + LLM synthesis
        source = "RAG+TOOLS+LLM"
        context = f"RAG Context:\n{rag_context}\n\nTool Results:\n{tool_context}"
        prompt = f"""{history_prompt}You have both internal knowledge base information and tool results to answer this question comprehensively.

Internal Knowledge Base Context:
{rag_context}

Tool Results:
{tool_context}

Question: {question}

Please provide a comprehensive answer that synthesizes information from both the internal knowledge base and the tool results, along with your general knowledge to give the most complete and insightful response.

Answer:"""
        print(f"[DEBUG] rag_answer: Using RAG+TOOLS+LLM hybrid approach")
        
    elif rag_context:
        # IDEAL CASE: RAG + LLM synthesis 
        source = "RAG+LLM"
        context = rag_context
        prompt = f"""{history_prompt}You have relevant information from our internal knowledge base. Use this along with your general knowledge to provide a comprehensive, insightful answer.

Internal Knowledge Base Context:
{rag_context}

Question: {question}

Please synthesize the specific information from our knowledge base with broader context and analysis to give the most complete and valuable response.

Answer:"""
        print(f"[DEBUG] rag_answer: Using RAG+LLM hybrid approach")
        
    elif tool_calls:
        # TOOLS + LLM
        source = "TOOLS+LLM"
        context = tool_context
        prompt = f"""{history_prompt}I have executed tools to gather current information. Now I need to provide a comprehensive answer based on the tool results and relevant general knowledge.

Tool Results:
{tool_context}

Question: {question}

Please provide a direct, helpful answer using the tool results above."""
        print(f"[DEBUG] rag_answer: Using TOOLS+LLM approach")
        
    elif query_type == "TOOLS":
        # TOOLS query but no tools executed - still treat as tool query
        source = "TOOLS"
        context = ""
        
        # Get available tools for the prompt
        mcp_tools_context = get_mcp_tools_context()
        
        prompt = f"""{history_prompt}The user is asking for information that requires using tools.

Question: {question}

{mcp_tools_context}

Based on the available tools above, please answer the question. For date/time queries, provide the current date and time in Singapore timezone (Asia/Singapore)."""
        print(f"[DEBUG] rag_answer: Using TOOLS approach (direct answer)")
        
    elif query_type == "RAG":
        # RAG was attempted but no documents found - still use RAG+LLM to indicate RAG was checked
        source = "RAG+LLM"
        context = ""
        prompt = f"""{history_prompt}No specific documents were found in our internal knowledge base for this query, but I'll provide a comprehensive answer based on general knowledge.

Question: {question}

Answer:"""
        print(f"[DEBUG] rag_answer: Using RAG+LLM approach (no documents found)")
        
    else:
        # FALLBACK: Pure LLM (including large single generations)
        source = "LLM"
        
        # Enhanced prompt for large generation requests
        if "generate" in question.lower() and any(word in question.lower() for word in ["questions", "items", "list"]):
            # Extract number for better prompt
            import re
            numbers = re.findall(r'\b(\d+)\b', question)
            target_num = max([int(n) for n in numbers], default=10) if numbers else 10
            
            prompt = f"""{history_prompt}Generate exactly {target_num} high-quality, unique interview questions for the specified role and context.

Requirements:
- Each question must be completely unique and cover different aspects
- Avoid any semantic duplicates or variations of the same concept  
- Number each question clearly (1. 2. 3. etc.)
- Ensure professional interview quality
- Cover diverse competency areas

{question}

Generate exactly {target_num} questions:"""
        else:
            prompt = f"{history_prompt}Question: {question}\n\nAnswer:"
        
        print(f"[DEBUG] rag_answer: Using pure LLM approach (no relevant context or tools)")
    
    # Apply thinking wrapper if needed
    prompt = build_prompt(prompt, thinking=thinking)
    
    print(f"[DEBUG] rag_answer: final prompt = {prompt[:200]}...")
    print(f"[DEBUG] rag_answer: source = {source}")
    
    # Generate response using internal API
    llm_api_url = "http://localhost:8000/api/v1/generate_stream"
    mode_config = llm_cfg["thinking_mode"] if thinking else llm_cfg["non_thinking_mode"]
    
    # Dynamic max_tokens based on request type
    if "generate" in question.lower() and any(word in question.lower() for word in ["questions", "items", "list"]):
        # For generation tasks, calculate appropriate token limit
        import re
        numbers = re.findall(r'\b(\d+)\b', question)
        target_num = max([int(n) for n in numbers], default=10) if numbers else 10
        # Estimate ~100 tokens per question + 500 buffer for instructions
        estimated_tokens = target_num * 100 + 500
        # Use higher cap for models with larger context windows
        llm_model = llm_cfg.get("model", "").lower()
        token_cap = 32768 if "deepseek" in llm_model else 8192
        max_tokens = min(estimated_tokens, token_cap)
        print(f"[DEBUG] rag_answer: Using dynamic max_tokens={max_tokens} for {target_num} items (cap: {token_cap})")
    else:
        # Use configured max_tokens
        max_tokens_raw = llm_cfg.get("max_tokens", 16384)
        # Handle both string and int values
        try:
            max_tokens = int(max_tokens_raw)
            # Sanity check - if someone set max_tokens to context window size, fix it
            if max_tokens > 32768:
                print(f"[WARNING] max_tokens {max_tokens} is too high (likely set to context window). Using 16384 for output.")
                max_tokens = 16384
        except (ValueError, TypeError):
            print(f"[WARNING] Invalid max_tokens value: {max_tokens_raw}, using 16384")
            max_tokens = 16384
    
    print(f"[DEBUG] rag_answer: Final max_tokens being sent: {max_tokens}")
    print(f"[DEBUG] rag_answer: Model: {llm_cfg.get('model', 'unknown')}")
    print(f"[DEBUG] rag_answer: Temperature: {mode_config.get('temperature', 0.7)}")
    
    payload = {
        "prompt": prompt,
        "temperature": mode_config.get("temperature", 0.7),
        "top_p": mode_config.get("top_p", 1.0),
        "max_tokens": max_tokens
    }
    
    if stream:
        def token_stream():
            # Import time locally to avoid scope issues
            import time
            # Capture max_tokens in local scope
            tokens_limit = max_tokens
            text = ""
            streaming_start_time = time.time()
            print(f"[DEBUG] Starting LLM streaming at {streaming_start_time}")
            try:
                with httpx.Client(timeout=60.0) as client:  # 1 minute timeout, more reasonable
                    with client.stream("POST", llm_api_url, json=payload) as response:
                        response.raise_for_status()  # Raise exception for HTTP errors
                        for line in response.iter_lines():
                            if not line:
                                continue
                            if isinstance(line, bytes):
                                line = line.decode("utf-8")
                            if line.startswith("data: "):
                                token = line.replace("data: ", "")
                                if token.strip():  # Only process non-empty tokens
                                    text += token
                                    yield json.dumps({"token": token}) + "\n"
                        
                        # Ensure response is fully consumed
                        streaming_end_time = time.time()
                        print(f"[DEBUG] HTTP streaming completed in {streaming_end_time - streaming_start_time:.2f} seconds, total text length: {len(text)}")
                        print(f"[DEBUG] Starting post-processing of response")
            except httpx.TimeoutException:
                print(f"[ERROR] HTTP timeout after 60 seconds")
                # Send completion event to stop cursor
                yield json.dumps({
                    "answer": "Request timed out. Please try again.",
                    "source": "ERROR",
                    "error": "Request timed out"
                }) + "\n"
                return
            except httpx.HTTPStatusError as e:
                print(f"[ERROR] HTTP error: {e.response.status_code} - {e.response.text}")
                # Send completion event to stop cursor
                yield json.dumps({
                    "answer": f"HTTP error: {e.response.status_code}. Please try again.",
                    "source": "ERROR", 
                    "error": f"HTTP error: {e.response.status_code}"
                }) + "\n"
                return
            except Exception as e:
                print(f"[ERROR] Streaming error: {e}")
                # Send completion event to stop cursor  
                yield json.dumps({
                    "answer": f"Streaming failed: {str(e)}. Please try again.",
                    "source": "ERROR",
                    "error": f"Streaming failed: {str(e)}"
                }) + "\n"
                return
            
            print(f"[DEBUG] Post-processing response: text_length={len(text)}")
            try:
                import re  # Import re inside the function to fix scoping issue
                # Extract reasoning and preserve formatting in answer
                reasoning = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
                print(f"[DEBUG] Found {len(reasoning)} reasoning sections")
                # Remove thinking tags but preserve all formatting
                answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
                
                # Clean up the answer while preserving structure
                if answer:
                    # Remove excessive blank lines (more than 2)
                    answer = re.sub(r'\n{3,}', '\n\n', answer)
                    # Remove leading/trailing whitespace but preserve internal structure
                    answer = answer.strip()
                    # Ensure proper spacing after periods for readability
                    answer = re.sub(r'\.([A-Z])', r'. \1', answer)
                    # Fix any missing spaces after colons
                    answer = re.sub(r':([A-Z])', r': \1', answer)
                else:
                    answer = ""
                
                # Send completion event with the exact format the frontend expects first for immediate UI response
                completion_event = {
                    "answer": answer,  # This is the key field the frontend looks for
                    "source": source,
                    "reasoning": reasoning[0] if reasoning else None,
                    "tool_calls": tool_calls,
                    "query_type": source,
                    "metadata": {
                        "streaming_complete": True,
                        "total_tokens": len(text),
                        "response_length": len(answer),
                        "max_tokens_used": tokens_limit,
                        "completion_timestamp": datetime.now().isoformat()
                    }
                }
                
                print(f"[DEBUG] Sending completion event: answer_length={len(answer)}, source={source}")
                try:
                    completion_json = json.dumps(completion_event, ensure_ascii=False)
                    print(f"[DEBUG] Completion JSON length: {len(completion_json)}")
                    yield completion_json + "\n"
                    
                    # Store conversation message after sending response for better performance
                    if conversation_id and answer:
                        try:
                            store_conversation_message(conversation_id, "assistant", answer)
                        except Exception as storage_error:
                            print(f"[ERROR] Failed to store conversation message: {storage_error}")
                    
                except Exception as json_error:
                    print(f"[ERROR] Failed to serialize completion event: {json_error}")
                    # Send a minimal completion event as fallback
                    fallback_event = {"answer": answer, "source": source}
                    yield json.dumps(fallback_event, ensure_ascii=False) + "\n"
            except Exception as e:
                print(f"[ERROR] Completion processing error: {e}")
                import traceback
                traceback.print_exc()
                # Always send a completion event even on error to stop the cursor
                yield json.dumps({
                    "answer": f"Error: {str(e)}",
                    "source": "ERROR",
                    "error": f"Completion failed: {str(e)}"
                }) + "\n"
        
        return token_stream()
    
    else:
        text = ""
        with httpx.Client(timeout=30.0) as client:  # Add timeout to prevent hanging
            with client.stream("POST", llm_api_url, json=payload) as response:
                for line in response.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    if line.startswith("data: "):
                        token = line.replace("data: ", "")
                        text += token
        
        reasoning = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
        answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        
        print(f"[RAG ROUTER] Final answer preview: {answer[:100]}...")
        print(f"[RAG ROUTER] Source used: {source}")
        
        # Store assistant's response in conversation history
        if conversation_id and answer:
            store_conversation_message(conversation_id, "assistant", answer)
        
        return {
            "answer": answer,
            "context": context,
            "reasoning": reasoning[0] if reasoning else None,
            "raw": text,
            "route": source,  # Updated to use source
            "source": source,
            "tool_calls": tool_calls,
            "query_type": source,  # Use source as query_type for backward compatibility
            "metadata": {
                "total_tokens": len(text),
                "response_length": len(answer),
                "max_tokens_used": max_tokens,
                "completion_timestamp": datetime.now().isoformat()
            }
        }

# Keep existing helper functions
def get_mcp_tools_context():
    """Prepare MCP tools context for LLM prompt"""
    enabled_tools = get_enabled_mcp_tools()
    if not enabled_tools:
        return ""
    
    tools_context = []
    for _, tool_info in enabled_tools.items():
        tool_desc = {
            "name": tool_info["name"],
            "description": tool_info["description"],
            "parameters": tool_info["parameters"]
        }
        tools_context.append(json.dumps(tool_desc, indent=2))
    
    return "\nAvailable tools:\n" + "\n".join(tools_context)

def refresh_gmail_token(server_id):
    """Refresh Gmail OAuth token"""
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

def call_mcp_tool(tool_name, parameters):
    """Call an MCP tool and return the result using info from Redis cache"""
    print(f"[DEBUG] call_mcp_tool called with tool_name='{tool_name}', parameters={parameters}")
    enabled_tools = get_enabled_mcp_tools()
    if tool_name not in enabled_tools:
        return {"error": f"Tool {tool_name} not found in enabled tools"}
    
    tool_info = enabled_tools[tool_name]
    endpoint = tool_info["endpoint"]
    print(f"[DEBUG] Tool info keys: {list(tool_info.keys())}")
    print(f"[DEBUG] Endpoint: {endpoint}")
    
    # Replace localhost with the actual hostname from manifest if available
    # This handles the case where endpoints are stored with localhost but need to use Docker hostname
    server_hostname = tool_info.get("server_hostname")
    
    # Check if we're running inside Docker
    import os
    in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
    
    if in_docker and server_hostname and "localhost" in endpoint:
        # When calling from inside Docker, use the Docker hostname
        endpoint = endpoint.replace("localhost", server_hostname)
        print(f"[DEBUG] Replaced localhost with server hostname '{server_hostname}' in endpoint (Docker environment)")
    elif not in_docker and server_hostname and server_hostname in endpoint:
        # When calling from host (e.g., user's browser), replace Docker hostname with localhost
        endpoint = endpoint.replace(server_hostname, "localhost")
        print(f"[DEBUG] Replaced server hostname '{server_hostname}' with localhost in endpoint (Host environment)")
    
    method = tool_info.get("method", "POST")
    headers = tool_info.get("headers") or {}
    
    # Handle MCP server pattern: use endpoint_prefix if tool-specific endpoint doesn't work
    # Most MCP servers use a generic /invoke endpoint with tool name in payload
    endpoint_prefix = tool_info.get("endpoint_prefix")
    if endpoint_prefix and endpoint.endswith(f"/invoke/{tool_name}"):
        # Use the generic invoke endpoint instead
        endpoint = endpoint_prefix
        print(f"[DEBUG] Using generic MCP endpoint: {endpoint}")
    
    print(f"[DEBUG] Calling MCP tool {tool_name} at {endpoint} with params: {parameters}")
    
    # Add API key authentication if available
    api_key = tool_info.get("api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        print(f"[DEBUG] Added API key authentication for {tool_name}")
    
    # Inject OAuth credentials for Gmail tools
    # Check if this is a Gmail tool by looking at the endpoint or tool name
    is_gmail_tool = ("gmail" in tool_name.lower() or 
                     "email" in tool_name.lower() or 
                     "Gmail MCP" in endpoint)
    print(f"[DEBUG] Checking OAuth injection: is_gmail_tool={is_gmail_tool}, server_id={tool_info.get('server_id')}")
    if is_gmail_tool and tool_info.get("server_id"):
        try:
            from app.core.oauth_token_manager import oauth_token_manager
            
            # Get valid token (will refresh automatically if needed)
            oauth_creds = oauth_token_manager.get_valid_token(
                server_id=tool_info["server_id"],
                service_name="gmail"
            )
            
            if oauth_creds and "google_access_token" not in parameters:
                # Gmail MCP server expects these exact parameter names
                parameters.update({
                    "google_access_token": oauth_creds.get("access_token", ""),
                    "google_refresh_token": oauth_creds.get("refresh_token", ""),
                    "google_client_id": oauth_creds.get("client_id", ""),
                    "google_client_secret": oauth_creds.get("client_secret", "")
                    # Note: token_uri is NOT passed as the Gmail MCP server has it hardcoded
                })
                print(f"[DEBUG] Injected OAuth credentials for {tool_name} from cache/refresh")
                print(f"[DEBUG] Token expires at: {oauth_creds.get('expires_at', 'Unknown')}")
            else:
                print(f"[WARNING] No valid OAuth credentials available for {tool_name}")
        except Exception as e:
            print(f"[ERROR] Failed to inject OAuth credentials: {e}")
    
    # Check if this is a stdio-based tool
    if endpoint.startswith("stdio://"):
        print(f"[DEBUG] Using stdio bridge for {tool_name}")
        # For stdio tools, we need to get the server configuration
        if tool_info.get("server_id"):
            from app.core.db import SessionLocal, MCPServer
            db = SessionLocal()
            try:
                server = db.query(MCPServer).filter(MCPServer.id == tool_info["server_id"]).first()
                if server and server.config_type == "command":
                    # Use stdio bridge for command-based servers
                    import asyncio
                    from app.core.mcp_stdio_bridge import call_mcp_tool_via_stdio
                    
                    server_config = {
                        "command": server.command,
                        "args": server.args or []
                    }
                    
                    # Run the async function
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            call_mcp_tool_via_stdio(server_config, tool_name, parameters)
                        )
                        
                        # Check if this is a Gmail token error in stdio response
                        error_str = str(result.get("error", "")) if isinstance(result, dict) else ""
                        
                        # Extract the actual error message from the response content
                        if isinstance(result, dict) and "content" in result:
                            content = result.get("content", [])
                            if content and isinstance(content, list) and len(content) > 0:
                                text_content = content[0].get("text", "")
                                if text_content:
                                    error_str = text_content
                        
                        if ("gmail" in tool_name.lower() and
                            ("credentials do not contain" in error_str.lower() or
                             "token" in error_str.lower() or
                             "expired" in error_str.lower() or
                             "invalid_grant" in error_str.lower())):
                            print(f"[DEBUG] Detected Gmail token error in stdio response: {error_str[:200]}")
                            
                            # Check for invalid_grant (token revoked)
                            if "invalid_grant" in error_str.lower():
                                print(f"[ERROR] Gmail refresh token has been revoked")
                                result = {
                                    "error": "Gmail authorization has been revoked. Please re-authorize Gmail access in the MCP Servers settings.",
                                    "error_type": "auth_revoked",
                                    "server_id": tool_info["server_id"]
                                }
                            else:
                                print(f"[DEBUG] Attempting to refresh token...")
                                
                                refresh_result = refresh_gmail_token(tool_info["server_id"])
                                if refresh_result and not refresh_result.get("error"):
                                    print(f"[DEBUG] Token refreshed successfully, retrying stdio tool call...")
                                    # Invalidate the tool cache to ensure fresh OAuth credentials are loaded
                                    from app.core.mcp_tools_cache import reload_enabled_mcp_tools
                                    reload_enabled_mcp_tools()
                                    
                                    # Re-inject OAuth credentials with fresh token
                                    oauth_creds = oauth_token_manager.get_valid_token(
                                        server_id=tool_info["server_id"],
                                        service_name="gmail"
                                    )
                                    if oauth_creds:
                                        parameters.update({
                                            "google_access_token": oauth_creds.get("access_token", ""),
                                            "google_refresh_token": oauth_creds.get("refresh_token", ""),
                                            "google_client_id": oauth_creds.get("client_id", ""),
                                            "google_client_secret": oauth_creds.get("client_secret", "")
                                            # Note: token_uri is NOT passed as the Gmail MCP server has it hardcoded
                                        })
                                    
                                    # Retry the tool call with refreshed token
                                    result = loop.run_until_complete(
                                        call_mcp_tool_via_stdio(server_config, tool_name, parameters)
                                    )
                                else:
                                    print(f"[ERROR] Failed to refresh token: {refresh_result}")
                                    # Check if refresh also failed due to invalid_grant
                                    if "invalid_grant" in str(refresh_result.get("error", "")):
                                        result = {
                                            "error": "Gmail authorization has been revoked. Please re-authorize Gmail access in the MCP Servers settings.",
                                            "error_type": "auth_revoked",
                                            "server_id": tool_info["server_id"]
                                        }
                        
                        return result
                    finally:
                        loop.close()
                else:
                    return {"error": "Server configuration not found for stdio tool"}
            finally:
                db.close()
        else:
            return {"error": "No server_id for stdio tool"}
    
    # Original HTTP-based logic
    # MCP server pattern: always use generic invoke format
    if "/invoke" in endpoint and (endpoint.endswith("/invoke") or endpoint_prefix):
        # Standard MCP format: {"name": "tool_name", "arguments": {parameters}}
        payload = {
            "name": tool_name,
            "arguments": parameters if parameters else {}
        }
        print(f"[DEBUG] Using MCP invoke format")
    else:
        # Fallback to direct parameters for other API patterns
        payload = parameters if parameters else {}
        print(f"[DEBUG] Using direct parameters format")
    
    print(f"[DEBUG] Final payload: {payload}")
    
    try:
        # Set default headers
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        
        if method.upper() == "GET":
            response = requests.get(endpoint, params=payload, headers=headers, timeout=30)
        else:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
        
        response.raise_for_status()
        
        # Handle different response types
        try:
            result = response.json()
            print(f"[DEBUG] MCP tool {tool_name} response: {result}")
            return result
        except json.JSONDecodeError:
            # If response is not JSON, return as text
            return {"result": response.text}
            
    except requests.exceptions.Timeout:
        error_msg = f"Timeout calling tool {tool_name} (30s timeout exceeded)"
        print(f"[ERROR] {error_msg}")
        return {"error": error_msg}
    except requests.exceptions.ConnectionError:
        error_msg = f"Connection error calling tool {tool_name} at {endpoint}. Check if MCP server is running."
        print(f"[ERROR] {error_msg}")
        return {"error": error_msg}
    except requests.exceptions.HTTPError as e:
        try:
            error_detail = e.response.text
        except:
            error_detail = str(e)
        error_msg = f"HTTP {e.response.status_code} error calling tool {tool_name}: {error_detail}"
        print(f"[ERROR] {error_msg}")
        
        # Check if this is a Gmail token expiration error
        if ("gmail" in tool_name.lower() and 
            ("credentials do not contain" in error_detail.lower() or 
             "token" in error_detail.lower() or
             "401" in error_detail or
             e.response.status_code == 401)):
            print(f"[DEBUG] Detected Gmail token error, attempting to refresh token...")
            
            # Try to refresh the token
            if tool_info.get("server_id"):
                refresh_result = refresh_gmail_token(tool_info["server_id"])
                if refresh_result and not refresh_result.get("error"):
                    print(f"[DEBUG] Token refreshed successfully, retrying tool call...")
                    # Invalidate the tool cache to ensure fresh OAuth credentials are loaded
                    from app.core.mcp_tools_cache import reload_enabled_mcp_tools
                    reload_enabled_mcp_tools()
                    # Retry the tool call with refreshed token
                    return call_mcp_tool(tool_name, parameters)
                else:
                    print(f"[ERROR] Failed to refresh token: {refresh_result}")
                    # Check if it's an invalid_grant error (token revoked)
                    if "invalid_grant" in str(refresh_result.get("error", "")):
                        return {
                            "error": "Gmail authorization has been revoked. Please re-authorize Gmail access in the MCP Servers settings.",
                            "error_type": "auth_revoked",
                            "server_id": tool_info.get("server_id")
                        }
        
        # If we get 404 and were using tool-specific endpoint, try generic endpoint
        if e.response.status_code == 404 and endpoint_prefix and not endpoint.endswith("/invoke"):
            print(f"[DEBUG] Retrying with generic endpoint due to 404...")
            return call_mcp_tool_with_generic_endpoint(tool_name, parameters, tool_info)
        
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error calling tool {tool_name}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return {"error": error_msg}

def call_mcp_tool_with_generic_endpoint(tool_name, parameters, tool_info):
    """Fallback function to retry with generic MCP endpoint"""
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

def extract_and_execute_tool_calls(text):
    """Extract tool calls from LLM output and execute them"""
    import re  # Import at function level to avoid scope issues
    
    if "NO_TOOLS_NEEDED" in text:
        return []
        
    tool_calls_pattern = r'<tool>(.*?)\((.*?)\)</tool>'
    tool_calls = re.findall(tool_calls_pattern, text, re.DOTALL)
    
    # Fallback: If no tool calls found in proper format, try to extract from thinking
    if not tool_calls:
        print("[DEBUG] No tool calls in proper format, attempting fallback extraction")
        text_lower = text.lower()
        
        # Look for tool mentions in thinking or anywhere in the text
        if ("get_datetime" in text or "date" in text_lower or "time" in text_lower) and not ("gmail" in text_lower or "email" in text_lower):
            print("[DEBUG] Detected date/time query, executing get_datetime")
            tool_calls = [("get_datetime", "{}")]
        elif any(indicator in text_lower for indicator in ["gmail", "email", "mail", "amazon", "from:", "subject:", "unread"]):
            print("[DEBUG] Detected email/Gmail query, executing search_emails")
            
            # Try to extract query parameters from the original question
            query_params = {}
            
            # Check for specific patterns
            if "amazon" in text_lower:
                query_params["query"] = "from:amazon"
            elif "unread" in text_lower:
                query_params["query"] = "is:unread"
            elif "today" in text_lower:
                query_params["query"] = "newer_than:1d"
            elif "from:" in text_lower:
                # Extract from: pattern
                import re
                from_match = re.search(r'from:\s*(\S+)', text_lower)
                if from_match:
                    query_params["query"] = f"from:{from_match.group(1)}"
                else:
                    query_params["query"] = ""
            else:
                # Default to empty query (gets recent emails)
                query_params["query"] = ""
            
            tool_calls = [("search_emails", json.dumps(query_params))]
        elif "send_email" in text and "send" in text_lower:
            # Would need more sophisticated parsing for email parameters
            pass
        elif "outlook_send_message" in text and "email" in text_lower:
            # Would need more sophisticated parsing for email parameters
            pass
        elif "jira" in text and ("ticket" in text_lower or "issue" in text_lower):
            # Would need more sophisticated parsing for JIRA parameters
            pass
    
    results = []
    for tool_name, params_str in tool_calls:
        try:
            tool_name = tool_name.strip()
            params_str = params_str.strip()
            
            if params_str == "{}" or params_str == "":
                params = {}
            else:
                try:
                    params = json.loads(params_str)
                except json.JSONDecodeError:
                    print(f"[ERROR] Failed to parse parameters for {tool_name}: {params_str}")
                    params = {}
            
            print(f"[DEBUG] Executing tool call: {tool_name} with params: {params}")
            result = call_mcp_tool(tool_name, params)
            
            # Special handling for date/time responses
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
                "result": result
            })
        except Exception as e:
            print(f"[ERROR] Failed to execute tool call {tool_name}: {str(e)}")
            results.append({
                "tool": tool_name,
                "error": str(e)
            })
    
    return results