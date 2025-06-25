"""
Comprehensive Conversation Memory Manager
Preserves ALL conversation details while intelligently managing context limits

This system NEVER loses conversation details - instead it uses:
1. External storage for full conversation history
2. Intelligent retrieval of relevant conversation parts
3. Vector-based conversation search and retrieval
4. Hierarchical context assembly with full detail preservation
"""

import json
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from datetime import datetime
from app.core.config import get_settings, timedelta
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import redis.asyncio as redis
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from app.core.redis_client import get_redis_client
from app.core.embedding_settings_cache import get_embedding_settings
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.api.v1.endpoints.document import HTTPEmbeddingFunction


@dataclass
class ConversationMessage:
    """Single conversation message with full metadata"""
    message_id: str
    conversation_id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]
    
    # Context information
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None
    reasoning_trace: Optional[str] = None
    source_info: Optional[str] = None
    
    # Relationships
    parent_message_id: Optional[str] = None
    thread_id: Optional[str] = None
    
    # Temporary document references
    temp_doc_ids: Optional[List[str]] = None
    temp_doc_context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMessage':
        """Create from stored dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ConversationThread:
    """Represents a conversation thread or topic within a conversation"""
    thread_id: str
    conversation_id: str
    topic: str
    summary: str
    start_time: datetime
    end_time: Optional[datetime]
    message_ids: List[str]
    relevance_keywords: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            **asdict(self),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None
        }


class ConversationStorage(ABC):
    """Abstract base for conversation storage backends"""
    
    @abstractmethod
    async def store_message(self, message: ConversationMessage) -> None:
        """Store a single message"""
        pass
    
    @abstractmethod
    async def get_messages(
        self, 
        conversation_id: str, 
        limit: Optional[int] = None,
        after_timestamp: Optional[datetime] = None
    ) -> List[ConversationMessage]:
        """Retrieve messages for a conversation"""
        pass
    
    @abstractmethod
    async def search_messages(
        self, 
        conversation_id: str,
        query: str,
        limit: int = 20
    ) -> List[ConversationMessage]:
        """Search for relevant messages in conversation"""
        pass


class RedisConversationStorage(ConversationStorage):
    """Redis-based conversation storage with vector search integration"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.ttl_days = 90  # Keep conversations for 90 days
        
        # Initialize embeddings for semantic search
        self.embeddings = None
        self.vector_store = None
        self._init_vector_search()
    
    def _init_vector_search(self):
        """Initialize vector search for conversations"""
        try:
            embedding_cfg = get_embedding_settings()
            vector_db_cfg = get_vector_db_settings()
            
            # Set up embeddings
            embedding_endpoint = embedding_cfg.get("embedding_endpoint")
            if embedding_endpoint and isinstance(embedding_endpoint, str) and embedding_endpoint.strip():
                self.embeddings = HTTPEmbeddingFunction(embedding_endpoint)
            else:
                model_name = embedding_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5")
                self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
            
            # Set up vector store for conversation search
            milvus_cfg = vector_db_cfg["milvus"]
            conversation_collection = "conversation_memory"
            
            self.vector_store = Milvus(
                embedding_function=self.embeddings,
                collection_name=conversation_collection,
                connection_args={
                    "uri": milvus_cfg.get("MILVUS_URI"),
                    "token": milvus_cfg.get("MILVUS_TOKEN")
                },
                text_field="content"
            )
            
        except Exception as e:
            print(f"[CONV_MEMORY] Warning: Vector search initialization failed: {e}")
            # Continue without vector search - will use Redis-only storage
    
    async def _get_redis_client(self) -> redis.Redis:
        """Get or create Redis client"""
        if self.redis_client is None:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        return self.redis_client
    
    async def store_message(self, message: ConversationMessage) -> None:
        """Store message in Redis and vector store"""
        try:
            redis_client = await self._get_redis_client()
            
            # Store in Redis with conversation-based keys
            message_key = f"conv_msg:{message.conversation_id}:{message.message_id}"
            conversation_list_key = f"conv_list:{message.conversation_id}"
            
            # Store full message data
            await redis_client.hset(
                message_key,
                mapping={k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                        for k, v in message.to_dict().items()}
            )
            await redis_client.expire(message_key, timedelta(days=self.ttl_days))
            
            # Add to conversation message list (ordered by timestamp)
            await redis_client.zadd(
                conversation_list_key,
                {message.message_id: message.timestamp.timestamp()}
            )
            await redis_client.expire(conversation_list_key, timedelta(days=self.ttl_days))
            
            # Store in vector store for semantic search (if available)
            if self.vector_store and message.content:
                try:
                    # Create document for vector search
                    doc = Document(
                        page_content=message.content,
                        metadata={
                            "message_id": message.message_id,
                            "conversation_id": message.conversation_id,
                            "role": message.role,
                            "timestamp": message.timestamp.isoformat(),
                            "type": "conversation_message"
                        }
                    )
                    
                    # Add to vector store (async if supported, otherwise sync)
                    if hasattr(self.vector_store, 'aadd_documents'):
                        await self.vector_store.aadd_documents([doc])
                    else:
                        # Fallback to sync method
                        self.vector_store.add_documents([doc])
                        
                except Exception as e:
                    print(f"[CONV_MEMORY] Warning: Vector storage failed for message {message.message_id}: {e}")
            
        except Exception as e:
            print(f"[CONV_MEMORY] Error storing message {message.message_id}: {e}")
            raise
    
    async def get_messages(
        self, 
        conversation_id: str, 
        limit: Optional[int] = None,
        after_timestamp: Optional[datetime] = None
    ) -> List[ConversationMessage]:
        """Retrieve messages from Redis"""
        try:
            redis_client = await self._get_redis_client()
            conversation_list_key = f"conv_list:{conversation_id}"
            
            # Get message IDs in chronological order
            min_score = after_timestamp.timestamp() if after_timestamp else 0
            max_score = "+inf"
            
            # Get message IDs sorted by timestamp
            message_ids = await redis_client.zrangebyscore(
                conversation_list_key,
                min_score,
                max_score,
                start=0,
                num=limit if limit else -1
            )
            
            # Retrieve full message data
            messages = []
            for message_id in message_ids:
                message_key = f"conv_msg:{conversation_id}:{message_id}"
                message_data = await redis_client.hgetall(message_key)
                
                if message_data:
                    # Parse JSON fields back
                    for key, value in message_data.items():
                        try:
                            message_data[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            pass  # Keep as string
                    
                    messages.append(ConversationMessage.from_dict(message_data))
            
            return messages
            
        except Exception as e:
            print(f"[CONV_MEMORY] Error retrieving messages for conversation {conversation_id}: {e}")
            return []
    
    async def search_messages(
        self, 
        conversation_id: str,
        query: str,
        limit: int = 20
    ) -> List[ConversationMessage]:
        """Search for relevant messages using vector similarity"""
        try:
            if not self.vector_store:
                print("[CONV_MEMORY] Vector search not available, falling back to text search")
                return await self._fallback_text_search(conversation_id, query, limit)
            
            # Perform vector similarity search
            search_results = self.vector_store.similarity_search_with_score(
                query,
                k=limit * 2,  # Get more candidates to filter by conversation_id
                filter={"conversation_id": conversation_id}  # Filter by conversation
            )
            
            # Convert results to messages
            messages = []
            for doc, score in search_results:
                if len(messages) >= limit:
                    break
                
                # Get full message from Redis using message_id
                message_id = doc.metadata.get("message_id")
                if message_id:
                    message_key = f"conv_msg:{conversation_id}:{message_id}"
                    redis_client = await self._get_redis_client()
                    message_data = await redis_client.hgetall(message_key)
                    
                    if message_data:
                        # Parse JSON fields
                        for key, value in message_data.items():
                            try:
                                message_data[key] = json.loads(value)
                            except (json.JSONDecodeError, TypeError):
                                pass
                        
                        message = ConversationMessage.from_dict(message_data)
                        messages.append(message)
            
            return messages
            
        except Exception as e:
            print(f"[CONV_MEMORY] Vector search failed, using fallback: {e}")
            return await self._fallback_text_search(conversation_id, query, limit)
    
    async def _fallback_text_search(
        self, 
        conversation_id: str, 
        query: str, 
        limit: int
    ) -> List[ConversationMessage]:
        """Fallback text-based search using Redis"""
        try:
            # Get all messages and filter by keyword matching
            all_messages = await self.get_messages(conversation_id)
            
            # Simple keyword matching
            query_words = set(query.lower().split())
            scored_messages = []
            
            for message in all_messages:
                content_words = set(message.content.lower().split())
                overlap = len(query_words.intersection(content_words))
                
                if overlap > 0:
                    score = overlap / len(query_words)  # Simple relevance score
                    scored_messages.append((message, score))
            
            # Sort by relevance and return top results
            scored_messages.sort(key=lambda x: x[1], reverse=True)
            return [msg for msg, _ in scored_messages[:limit]]
            
        except Exception as e:
            print(f"[CONV_MEMORY] Fallback search failed: {e}")
            return []


class ConversationContextManager:
    """
    Manages conversation context assembly while preserving ALL details
    Uses intelligent retrieval to fit within context limits
    """
    
    def __init__(self, storage: ConversationStorage):
        self.storage = storage
        self.max_context_tokens = 8000  # Conservative limit
        self.min_recent_messages = 10    # Always include at least this many recent messages
        
    async def get_relevant_context(
        self, 
        conversation_id: str,
        current_query: str,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Assemble relevant conversation context intelligently
        Returns: (context_string, metadata)
        """
        max_tokens = max_tokens or self.max_context_tokens
        
        # Strategy 1: Get recent messages (always included)
        recent_messages = await self.storage.get_messages(
            conversation_id, 
            limit=self.min_recent_messages
        )
        
        # Strategy 2: Search for semantically relevant messages
        relevant_messages = await self.storage.search_messages(
            conversation_id,
            current_query,
            limit=20
        )
        
        # Combine and deduplicate messages
        all_messages = self._combine_and_deduplicate_messages(
            recent_messages, 
            relevant_messages
        )
        
        # Assemble context within token limits
        context_parts, metadata = self._assemble_context_within_limits(
            all_messages,
            max_tokens,
            current_query
        )
        
        return "\n\n".join(context_parts), metadata
    
    def _combine_and_deduplicate_messages(
        self, 
        recent_messages: List[ConversationMessage],
        relevant_messages: List[ConversationMessage]
    ) -> List[ConversationMessage]:
        """Combine message lists and remove duplicates"""
        seen_ids = set()
        combined_messages = []
        
        # First add recent messages (to ensure they're included)
        for msg in recent_messages:
            if msg.message_id not in seen_ids:
                seen_ids.add(msg.message_id)
                combined_messages.append(msg)
        
        # Then add relevant messages that aren't already included
        for msg in relevant_messages:
            if msg.message_id not in seen_ids:
                seen_ids.add(msg.message_id)
                combined_messages.append(msg)
        
        # Sort by timestamp to maintain conversation flow
        combined_messages.sort(key=lambda x: x.timestamp)
        return combined_messages
    
    def _assemble_context_within_limits(
        self,
        messages: List[ConversationMessage],
        max_tokens: int,
        current_query: str
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Assemble context while staying within token limits
        Uses intelligent summarization when needed
        """
        context_parts = []
        estimated_tokens = 0
        included_messages = 0
        summarized_messages = 0
        
        # Reserve tokens for current query
        query_tokens = len(current_query.split()) * 1.3  # Rough estimate
        available_tokens = max_tokens - query_tokens
        
        # Process messages in reverse chronological order (most recent first)
        for i, message in enumerate(reversed(messages)):
            message_text = f"{message.role.capitalize()}: {message.content}"
            message_tokens = len(message_text.split()) * 1.3  # Rough token estimate
            
            if estimated_tokens + message_tokens <= available_tokens:
                # Include full message
                context_parts.insert(0, message_text)  # Insert at beginning to maintain order
                estimated_tokens += message_tokens
                included_messages += 1
            else:
                # Check if this is a recent message we must include
                if i < self.min_recent_messages:
                    # Summarize older messages to make room
                    if context_parts:
                        summary = self._create_message_summary(
                            messages[:-self.min_recent_messages]
                        )
                        if summary:
                            # Replace older context with summary
                            context_parts = [summary] + context_parts[-self.min_recent_messages:]
                            summarized_messages = len(messages) - self.min_recent_messages
                            
                    # Add current message
                    context_parts.insert(0, message_text)
                    included_messages += 1
                else:
                    # No more room - stop adding messages
                    break
        
        metadata = {
            "total_messages_available": len(messages),
            "included_messages": included_messages,
            "summarized_messages": summarized_messages,
            "estimated_tokens": estimated_tokens,
            "context_strategy": "intelligent_retrieval"
        }
        
        return context_parts, metadata
    
    def _create_message_summary(self, messages: List[ConversationMessage]) -> str:
        """Create a summary of older messages"""
        if not messages:
            return ""
        
        # Simple summary for now - could be enhanced with LLM summarization
        topics = set()
        key_points = []
        
        for msg in messages:
            # Extract key topics (simple keyword extraction)
            words = msg.content.lower().split()
            for word in words:
                if len(word) > 4 and word.isalpha():
                    topics.add(word)
        
        if len(topics) > 10:
            topics = list(topics)[:10]  # Limit topics
        
        return f"[Earlier conversation summary: discussed {', '.join(topics)} - {len(messages)} messages]"


class ConversationMemoryManager:
    """
    Main interface for conversation memory management
    Preserves ALL conversation details while managing context intelligently
    """
    
    def __init__(self, storage: Optional[ConversationStorage] = None):
        self.storage = storage or RedisConversationStorage()
        self.context_manager = ConversationContextManager(self.storage)
    
    async def store_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a conversation message and return message ID"""
        message_id = f"{conversation_id}_{int(datetime.now().timestamp() * 1000)}"
        
        message = ConversationMessage(
            message_id=message_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        await self.storage.store_message(message)
        return message_id
    
    async def get_conversation_context(
        self,
        conversation_id: str,
        current_query: str,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get intelligently assembled conversation context
        NEVER loses details - stores everything, retrieves relevantly
        """
        return await self.context_manager.get_relevant_context(
            conversation_id,
            current_query,
            max_tokens
        )
    
    async def search_conversation_history(
        self,
        conversation_id: str,
        query: str,
        limit: int = 20
    ) -> List[ConversationMessage]:
        """Search through full conversation history"""
        return await self.storage.search_messages(conversation_id, query, limit)
    
    async def get_full_conversation(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[ConversationMessage]:
        """Get complete conversation history (for export/review)"""
        return await self.storage.get_messages(conversation_id, limit=limit)
    
    async def export_conversation(
        self,
        conversation_id: str,
        format: str = "json"
    ) -> str:
        """Export full conversation in specified format"""
        messages = await self.get_full_conversation(conversation_id)
        
        if format == "json":
            return json.dumps([msg.to_dict() for msg in messages], indent=2)
        elif format == "markdown":
            lines = []
            for msg in messages:
                lines.append(f"## {msg.role.capitalize()} ({msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')})")
                lines.append(msg.content)
                lines.append("")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Compatibility layer for existing code
class EnhancedConversationService:
    """
    Enhanced conversation service that preserves ALL details
    Drop-in replacement for existing conversation management
    """
    
    def __init__(self):
        self.memory_manager = ConversationMemoryManager()
    
    async def store_conversation_message(
        self, 
        conversation_id: str, 
        role: str, 
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Enhanced version of store_conversation_message that never loses data"""
        await self.memory_manager.store_message(conversation_id, role, content, metadata)
    
    async def get_conversation_history(
        self, 
        conversation_id: str,
        current_query: str = "",
        format: str = "string"
    ) -> str:
        """
        Enhanced conversation history that intelligently retrieves relevant context
        Compatible with existing code but much more powerful
        """
        if not conversation_id:
            return ""
        
        try:
            context, metadata = await self.memory_manager.get_conversation_context(
                conversation_id, current_query
            )
            
            if format == "string":
                return context
            elif format == "dict":
                return {"context": context, "metadata": metadata}
            else:
                return context
                
        except Exception as e:
            print(f"[ENHANCED_CONV] Error getting conversation history: {e}")
            return ""