"""
Conversation Context Manager
Intelligently manages conversation history to prevent context confusion
"""
import re
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class ConversationContextManager:
    """Manages conversation context intelligently based on query type and relevance"""
    
    def __init__(self):
        # Query type patterns
        self.simple_query_patterns = [
            r"what\s+(?:is|are)\s+(?:the\s+)?(?:current\s+)?(?:date|time|day|year|month)",
            r"what\s+time\s+is\s+it",
            r"today'?s\s+date",
            r"current\s+(?:date|time)",
            r"hello|hi|hey",
            r"how\s+are\s+you",
            r"thank\s+you|thanks",
            r"goodbye|bye",
            r"yes|no|okay|ok",
        ]
        
        self.context_reset_patterns = [
            r"(?:forget|ignore|disregard)\s+(?:the\s+)?(?:previous|last|above)",
            r"new\s+(?:topic|question|subject)",
            r"start\s+(?:over|fresh|new)",
            r"change\s+(?:topic|subject)",
            r"different\s+(?:topic|question)",
        ]
        
        self.context_sensitive_patterns = [
            r"(?:based\s+on|according\s+to|from)\s+(?:the\s+)?(?:previous|last|above)",
            r"as\s+(?:mentioned|discussed|stated)",
            r"(?:follow|following)\s+up",
            r"(?:related|relating)\s+to",
            r"(?:more|additional)\s+(?:info|information|details?)\s+(?:on|about)",
            r"(?:explain|elaborate)\s+(?:more|further)",
            r"what\s+(?:did|do)\s+(?:I|you)\s+(?:just|previously)",
        ]
        
        # Topic clustering keywords
        self.topic_keywords = {
            'technical': ['code', 'programming', 'api', 'function', 'error', 'debug', 'implement'],
            'data': ['data', 'database', 'query', 'sql', 'table', 'record', 'analysis'],
            'general': ['date', 'time', 'weather', 'hello', 'thanks', 'help', 'explain'],
            'document': ['document', 'file', 'pdf', 'upload', 'extract', 'content'],
            'ai': ['ai', 'model', 'llm', 'neural', 'machine learning', 'training'],
            'temporal': ['time', 'date', 'schedule', 'calendar', 'deadline', 'appointment', 'meeting', 'business hours']
        }
        
        # Integration with temporal context manager
        try:
            from app.core.temporal_context_manager import get_temporal_context_manager
            self.temporal_manager = get_temporal_context_manager()
            logger.debug("Integrated with temporal context manager")
        except Exception as e:
            logger.warning(f"Could not integrate with temporal context manager: {e}")
            self.temporal_manager = None
    
    def should_include_history(self, current_query: str, conversation_history: List[Dict]) -> bool:
        """Determine if conversation history should be included for current query with temporal awareness"""
        query_lower = current_query.lower().strip()
        
        # Check for explicit context reset
        for pattern in self.context_reset_patterns:
            if re.search(pattern, query_lower):
                return False
        
        # Use temporal context manager for time-related queries
        if self.temporal_manager:
            try:
                is_time_related, _, confidence = self.temporal_manager.detect_time_related_query(query_lower)
                if is_time_related and confidence > 0.6:
                    # For time-related queries, be very selective about including history
                    # Only include if explicitly referencing previous context
                    for pattern in self.context_sensitive_patterns:
                        if re.search(pattern, query_lower):
                            logger.debug("Including history for time-related query with context reference")
                            return True
                    logger.debug("Excluding history for standalone time-related query")
                    return False
            except Exception as e:
                logger.debug(f"Temporal history decision failed: {e}")
        
        # Check if it's a simple standalone query
        for pattern in self.simple_query_patterns:
            if re.search(pattern, query_lower):
                return False
        
        # Check if query explicitly references previous context
        for pattern in self.context_sensitive_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # If history exists, check topic relevance
        if conversation_history:
            return self._is_topic_relevant(current_query, conversation_history)
        
        return False
    
    def _is_topic_relevant(self, current_query: str, history: List[Dict]) -> bool:
        """Check if current query is topically related to recent history"""
        current_topic = self._extract_topic(current_query)
        
        # Check last 2-3 messages for topic relevance
        recent_messages = history[-3:] if len(history) >= 3 else history
        for msg in recent_messages:
            if msg.get('role') == 'user':
                hist_topic = self._extract_topic(msg.get('content', ''))
                if current_topic == hist_topic and current_topic != 'general':
                    return True
        
        return False
    
    def _extract_topic(self, text: str) -> str:
        """Extract primary topic from text with enhanced temporal detection"""
        text_lower = text.lower()
        topic_scores = {}
        
        # Use temporal context manager for better temporal detection
        if self.temporal_manager:
            try:
                is_time_related, _, confidence = self.temporal_manager.detect_time_related_query(text_lower)
                if is_time_related and confidence > 0.5:
                    return 'temporal'
            except Exception as e:
                logger.debug(f"Temporal topic detection failed: {e}")
        
        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            return max(topic_scores, key=topic_scores.get)
        return 'general'
    
    def filter_conversation_history(
        self, 
        conversation_history: List[Dict], 
        current_query: str,
        max_messages: int = 4
    ) -> List[Dict]:
        """Filter and limit conversation history based on relevance"""
        if not conversation_history:
            return []
        
        query_lower = current_query.lower()
        
        # For queries explicitly referencing context, include more history
        is_contextual = any(
            re.search(pattern, query_lower) 
            for pattern in self.context_sensitive_patterns
        )
        
        if is_contextual:
            max_messages = min(6, max_messages * 2)
        
        # Filter out system messages and overly long responses
        filtered = []
        for msg in conversation_history[-max_messages:]:
            content = msg.get('content', '')
            
            # Skip system-like messages
            if any(marker in content.lower() for marker in [
                '[debug]', '[error]', '[info]', 'conversation_id:', 'timestamp:'
            ]):
                continue
            
            # Skip overly long responses that might confuse context
            if len(content) > 500 and msg.get('role') == 'assistant':
                # Only include if directly relevant
                if not self._is_content_relevant(content, current_query):
                    continue
            
            filtered.append(msg)
        
        return filtered[-max_messages:]
    
    def _is_content_relevant(self, content: str, query: str) -> bool:
        """Check if content is relevant to current query"""
        # Extract key terms from query
        query_terms = set(
            word.lower() for word in re.findall(r'\b\w+\b', query)
            if len(word) > 3 and word.lower() not in ['what', 'when', 'where', 'how', 'why']
        )
        
        # Check for term overlap
        content_terms = set(
            word.lower() for word in re.findall(r'\b\w+\b', content)
            if len(word) > 3
        )
        
        overlap = query_terms.intersection(content_terms)
        return len(overlap) >= min(2, len(query_terms) // 2)
    
    def format_conversation_history(
        self, 
        history: List[Dict], 
        include_timestamps: bool = False
    ) -> str:
        """Format conversation history for inclusion in prompt"""
        if not history:
            return ""
        
        formatted_lines = []
        
        for msg in history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get('content', '').strip()
            
            # Truncate overly long content
            if len(content) > 200:
                content = content[:197] + "..."
            
            line = f"{role}: {content}"
            
            if include_timestamps and 'timestamp' in msg:
                try:
                    ts = datetime.fromisoformat(msg['timestamp'])
                    time_ago = self._format_time_ago(ts)
                    line = f"{line} ({time_ago})"
                except:
                    pass
            
            formatted_lines.append(line)
        
        return "\n".join(formatted_lines)
    
    def _format_time_ago(self, timestamp: datetime) -> str:
        """Format timestamp as human-readable time ago"""
        now = datetime.now()
        delta = now - timestamp
        
        if delta < timedelta(minutes=1):
            return "just now"
        elif delta < timedelta(hours=1):
            minutes = int(delta.total_seconds() / 60)
            return f"{minutes}m ago"
        elif delta < timedelta(days=1):
            hours = int(delta.total_seconds() / 3600)
            return f"{hours}h ago"
        else:
            days = delta.days
            return f"{days}d ago"
    
    def get_context_window_size(self, query_type: str) -> int:
        """Get appropriate context window size based on query type"""
        context_sizes = {
            'simple': 0,      # No context for simple queries
            'contextual': 6,  # More context for follow-up questions
            'technical': 4,   # Moderate context for technical discussions
            'general': 2,     # Minimal context for general queries
            'temporal': 1,    # Very minimal context for time-related queries (they need fresh data)
        }
        
        return context_sizes.get(query_type, 2)
    
    def classify_query_type(self, query: str) -> str:
        """Classify query type for context management with temporal awareness"""
        query_lower = query.lower()
        
        # Check for temporal queries first (highest priority for context management)
        if self.temporal_manager:
            try:
                is_time_related, _, confidence = self.temporal_manager.detect_time_related_query(query_lower)
                if is_time_related and confidence > 0.6:
                    return 'temporal'
            except Exception as e:
                logger.debug(f"Temporal query classification failed: {e}")
        
        # Check patterns in order of precedence
        if any(re.search(p, query_lower) for p in self.simple_query_patterns):
            return 'simple'
        
        if any(re.search(p, query_lower) for p in self.context_sensitive_patterns):
            return 'contextual'
        
        topic = self._extract_topic(query)
        if topic in ['technical', 'data', 'ai']:
            return 'technical'
        elif topic == 'temporal':
            return 'temporal'
        
        return 'general'


# Enhanced version of get_limited_conversation_history
def get_smart_conversation_history(
    conversation_id: str, 
    current_query: str,
    conversation_history: Optional[List[Dict]] = None,
    max_messages: int = 4
) -> str:
    """
    Get intelligently filtered conversation history based on query context
    
    Args:
        conversation_id: Conversation ID
        current_query: The current user query
        conversation_history: Optional pre-loaded history
        max_messages: Maximum messages to include
        
    Returns:
        Formatted conversation history string
    """
    manager = ConversationContextManager()
    
    # Load history if not provided
    if conversation_history is None:
        # This would load from Redis/memory - simplified here
        from app.langchain.service import get_redis_conversation_client, _conversation_cache
        conversation_history = []
        
        try:
            redis_client = get_redis_conversation_client()
            if redis_client:
                redis_key = f"conversation:{conversation_id}"
                history_json = redis_client.lrange(redis_key, 0, -1)
                if history_json:
                    conversation_history = [json.loads(msg) for msg in history_json]
        except:
            # Fallback to memory cache
            conversation_history = _conversation_cache.get(conversation_id, [])
    
    # Check if we should include history at all
    if not manager.should_include_history(current_query, conversation_history):
        return ""
    
    # Classify query and get appropriate window size
    query_type = manager.classify_query_type(current_query)
    window_size = manager.get_context_window_size(query_type)
    
    if window_size == 0:
        return ""
    
    # Filter and format history
    filtered_history = manager.filter_conversation_history(
        conversation_history, 
        current_query,
        max_messages=window_size
    )
    
    return manager.format_conversation_history(filtered_history)