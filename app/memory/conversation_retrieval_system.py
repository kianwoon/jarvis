"""
Conversation Retrieval System
Integrates conversation memory with existing RAG infrastructure
Provides vector-based conversation search and contextual retrieval
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from datetime import datetime, timedelta
import json
import uuid

from langchain.schema import Document
from langchain_community.vectorstores import Milvus
from app.memory.conversation_memory_manager import ConversationMemoryManager, ConversationMessage
from app.memory.context_assembly_engine import ContextAssemblyEngine, ContextStrategy
from app.core.embedding_settings_cache import get_embedding_settings
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.api.v1.endpoints.document import HTTPEmbeddingFunction
from langchain_community.embeddings import HuggingFaceEmbeddings


class ConversationVectorStore:
    """
    Vector store specifically optimized for conversation search
    Integrates with existing RAG infrastructure
    """
    
    def __init__(self):
        self.collection_name = "conversation_memory_v2"
        self.vector_store = None
        self.embeddings = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize vector store for conversations"""
        try:
            embedding_cfg = get_embedding_settings()
            vector_db_cfg = get_vector_db_settings()
            
            # Set up embeddings (same as main RAG system)
            embedding_endpoint = embedding_cfg.get("embedding_endpoint")
            if embedding_endpoint and isinstance(embedding_endpoint, str) and embedding_endpoint.strip():
                self.embeddings = HTTPEmbeddingFunction(embedding_endpoint)
            else:
                model_name = embedding_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5")
                self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
            
            # Set up vector store
            milvus_cfg = vector_db_cfg["milvus"]
            
            self.vector_store = Milvus(
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                connection_args={
                    "uri": milvus_cfg.get("MILVUS_URI"),
                    "token": milvus_cfg.get("MILVUS_TOKEN")
                },
                text_field="content"
            )
            
            print(f"[CONV_VECTOR] Initialized conversation vector store: {self.collection_name}")
            
        except Exception as e:
            print(f"[CONV_VECTOR] Failed to initialize vector store: {e}")
            self.vector_store = None
    
    async def index_conversation_message(self, message: ConversationMessage) -> bool:
        """Index a single conversation message for vector search"""
        if not self.vector_store:
            return False
        
        try:
            # Create document for vector storage
            doc = Document(
                page_content=message.content,
                metadata={
                    "message_id": message.message_id,
                    "conversation_id": message.conversation_id,
                    "role": message.role,
                    "timestamp": message.timestamp.isoformat(),
                    "content_type": "conversation_message",
                    "thread_id": message.thread_id or "",
                    "parent_message_id": message.parent_message_id or "",
                    # Add searchable metadata
                    "day_of_week": message.timestamp.strftime("%A"),
                    "hour_of_day": str(message.timestamp.hour),
                    "date": message.timestamp.date().isoformat()
                }
            )
            
            # Add to vector store
            if hasattr(self.vector_store, 'aadd_documents'):
                await self.vector_store.aadd_documents([doc])
            else:
                # Fallback to sync method
                self.vector_store.add_documents([doc])
            
            return True
            
        except Exception as e:
            print(f"[CONV_VECTOR] Failed to index message {message.message_id}: {e}")
            return False
    
    async def search_conversation_messages(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        limit: int = 20,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        role_filter: Optional[str] = None
    ) -> List[Tuple[ConversationMessage, float]]:
        """
        Search conversation messages using vector similarity
        Returns list of (message, relevance_score) tuples
        """
        if not self.vector_store:
            return []
        
        try:
            # Build metadata filter
            metadata_filter = {"content_type": "conversation_message"}
            
            if conversation_id:
                metadata_filter["conversation_id"] = conversation_id
            
            if role_filter:
                metadata_filter["role"] = role_filter
            
            # Perform vector search
            search_results = self.vector_store.similarity_search_with_score(
                query,
                k=limit * 2,  # Get more results to filter
                filter=metadata_filter
            )
            
            # Convert results and apply additional filters
            filtered_results = []
            
            for doc, score in search_results:
                try:
                    # Reconstruct message from metadata
                    metadata = doc.metadata
                    message = ConversationMessage(
                        message_id=metadata["message_id"],
                        conversation_id=metadata["conversation_id"],
                        role=metadata["role"],
                        content=doc.page_content,
                        timestamp=datetime.fromisoformat(metadata["timestamp"]),
                        metadata={},
                        thread_id=metadata.get("thread_id"),
                        parent_message_id=metadata.get("parent_message_id")
                    )
                    
                    # Apply time range filter if specified
                    if time_range:
                        start_time, end_time = time_range
                        if not (start_time <= message.timestamp <= end_time):
                            continue
                    
                    # Convert distance to similarity score (assuming cosine distance)
                    similarity_score = max(0.0, 1.0 - score / 2.0)
                    
                    filtered_results.append((message, similarity_score))
                    
                except Exception as e:
                    print(f"[CONV_VECTOR] Error processing search result: {e}")
                    continue
            
            # Sort by relevance and limit results
            filtered_results.sort(key=lambda x: x[1], reverse=True)
            return filtered_results[:limit]
            
        except Exception as e:
            print(f"[CONV_VECTOR] Search failed: {e}")
            return []
    
    async def get_conversation_timeline(
        self,
        conversation_id: str,
        query: str,
        context_window: timedelta = timedelta(hours=2)
    ) -> List[ConversationMessage]:
        """
        Get conversation timeline around relevant messages
        Finds relevant messages then expands context around them
        """
        # First find relevant messages
        relevant_results = await self.search_conversation_messages(
            query, conversation_id, limit=10
        )
        
        if not relevant_results:
            return []
        
        # Get time ranges around relevant messages
        timeline_messages = set()
        
        for message, score in relevant_results:
            # Define time window around this message
            start_time = message.timestamp - context_window
            end_time = message.timestamp + context_window
            
            # Get messages in this time window
            window_messages = await self.search_conversation_messages(
                "",  # Empty query to get all messages
                conversation_id,
                limit=50,
                time_range=(start_time, end_time)
            )
            
            # Add to timeline
            for msg, _ in window_messages:
                timeline_messages.add(msg.message_id)
        
        # Convert back to message objects and sort by timestamp
        all_messages = [msg for msg, _ in relevant_results]
        all_messages.sort(key=lambda x: x.timestamp)
        
        return all_messages


class EnhancedConversationRetriever:
    """
    Enhanced conversation retrieval that integrates with RAG system
    Provides intelligent context assembly and retrieval
    """
    
    def __init__(self):
        self.memory_manager = ConversationMemoryManager()
        self.context_engine = ContextAssemblyEngine(self.memory_manager)
        self.vector_store = ConversationVectorStore()
        
        # Integration settings
        self.max_context_tokens = {
            'standard_chat': 8000,
            'rag_enhanced': 6000,  # Leave room for RAG content
            'multi_agent': 4000,   # Conservative for agent systems
            'reasoning_mode': 5000  # Leave room for reasoning
        }
    
    async def get_contextual_conversation_history(
        self,
        conversation_id: str,
        current_query: str,
        context_type: str = 'standard_chat',
        strategy: Optional[ContextStrategy] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get contextually relevant conversation history
        Optimized for different use cases (RAG, multi-agent, etc.)
        """
        
        max_tokens = self.max_context_tokens.get(context_type, 8000)
        strategy = strategy or ContextStrategy.ADAPTIVE
        
        try:
            # Use context assembly engine for intelligent retrieval
            context, metadata = await self.context_engine.assemble_context(
                conversation_id=conversation_id,
                current_query=current_query,
                strategy=strategy,
                context_type=context_type,
                max_tokens=max_tokens
            )
            
            return context, metadata
            
        except Exception as e:
            print(f"[ENHANCED_RETRIEVER] Error getting contextual history: {e}")
            return "", {"error": str(e)}
    
    async def hybrid_conversation_search(
        self,
        conversation_id: str,
        query: str,
        include_timeline: bool = True,
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and keyword matching
        Similar to the RAG hybrid search approach
        """
        
        results = []
        
        try:
            # Vector similarity search
            vector_results = await self.vector_store.search_conversation_messages(
                query, conversation_id, limit=max_results
            )
            
            # Add vector results
            for message, score in vector_results:
                results.append({
                    "message": message,
                    "relevance_score": score,
                    "search_type": "vector_similarity",
                    "timestamp": message.timestamp,
                    "role": message.role
                })
            
            # If timeline is requested, get contextual messages around relevant ones
            if include_timeline and vector_results:
                timeline_messages = await self.vector_store.get_conversation_timeline(
                    conversation_id, query
                )
                
                # Add timeline context (with lower priority)
                existing_message_ids = {result["message"].message_id for result in results}
                
                for message in timeline_messages:
                    if message.message_id not in existing_message_ids:
                        results.append({
                            "message": message,
                            "relevance_score": 0.3,  # Lower score for timeline context
                            "search_type": "timeline_context",
                            "timestamp": message.timestamp,
                            "role": message.role
                        })
            
            # Sort by relevance and timestamp
            results.sort(key=lambda x: (x["relevance_score"], x["timestamp"]), reverse=True)
            
            return results[:max_results]
            
        except Exception as e:
            print(f"[ENHANCED_RETRIEVER] Hybrid search failed: {e}")
            return []
    
    async def get_conversation_summary(
        self,
        conversation_id: str,
        summary_type: str = "key_points",
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Generate conversation summary for context preservation
        Different summary types for different needs
        """
        
        try:
            # Get all conversation messages in range
            if time_range:
                messages = await self.memory_manager.storage.get_messages(
                    conversation_id, after_timestamp=time_range[0]
                )
                messages = [
                    msg for msg in messages 
                    if time_range[0] <= msg.timestamp <= time_range[1]
                ]
            else:
                messages = await self.memory_manager.storage.get_messages(
                    conversation_id, limit=100
                )
            
            if not messages:
                return {"summary": "", "message_count": 0}
            
            if summary_type == "key_points":
                summary = self._generate_key_points_summary(messages)
            elif summary_type == "timeline":
                summary = self._generate_timeline_summary(messages)
            elif summary_type == "topics":
                summary = self._generate_topics_summary(messages)
            else:
                summary = self._generate_basic_summary(messages)
            
            return {
                "summary": summary,
                "message_count": len(messages),
                "time_range": {
                    "start": messages[0].timestamp.isoformat(),
                    "end": messages[-1].timestamp.isoformat()
                },
                "summary_type": summary_type
            }
            
        except Exception as e:
            print(f"[ENHANCED_RETRIEVER] Summary generation failed: {e}")
            return {"summary": "", "error": str(e)}
    
    def _generate_key_points_summary(self, messages: List[ConversationMessage]) -> str:
        """Generate a key points summary"""
        # Extract important statements (simple heuristic)
        key_points = []
        
        for msg in messages:
            # Look for messages with decision/conclusion indicators
            content_lower = msg.content.lower()
            if any(indicator in content_lower for indicator in [
                'decided', 'conclusion', 'important', 'key', 'solution',
                'agreed', 'final', 'result', 'outcome'
            ]):
                # Take first sentence or up to 100 chars
                first_sentence = msg.content.split('.')[0][:100]
                if len(first_sentence) > 20:
                    key_points.append(f"- {first_sentence}")
        
        if not key_points:
            # Fallback: extract longer messages as potentially important
            longer_messages = [msg for msg in messages if len(msg.content) > 100]
            for msg in longer_messages[-5:]:  # Last 5 longer messages
                summary_text = msg.content[:80] + "..."
                key_points.append(f"- {summary_text}")
        
        return "\n".join(key_points[:10])  # Limit to 10 key points
    
    def _generate_timeline_summary(self, messages: List[ConversationMessage]) -> str:
        """Generate a chronological timeline summary"""
        timeline_parts = []
        
        # Group messages by time periods
        current_date = None
        daily_summaries = []
        
        for msg in messages:
            msg_date = msg.timestamp.date()
            
            if current_date != msg_date:
                if daily_summaries:
                    # Summarize previous day
                    summary = f"{current_date}: {len(daily_summaries)} messages exchanged"
                    timeline_parts.append(summary)
                
                current_date = msg_date
                daily_summaries = []
            
            daily_summaries.append(msg)
        
        # Add final day
        if daily_summaries and current_date:
            summary = f"{current_date}: {len(daily_summaries)} messages exchanged"
            timeline_parts.append(summary)
        
        return "\n".join(timeline_parts)
    
    def _generate_topics_summary(self, messages: List[ConversationMessage]) -> str:
        """Generate a topics-based summary"""
        # Simple topic extraction
        word_frequency = {}
        
        for msg in messages:
            words = msg.content.lower().split()
            for word in words:
                if len(word) > 4 and word.isalpha():
                    word_frequency[word] = word_frequency.get(word, 0) + 1
        
        # Get top topics
        top_topics = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)[:15]
        
        topic_summary = "Main topics discussed: " + ", ".join([topic for topic, count in top_topics])
        return topic_summary
    
    def _generate_basic_summary(self, messages: List[ConversationMessage]) -> str:
        """Generate a basic statistical summary"""
        user_messages = [msg for msg in messages if msg.role == "user"]
        assistant_messages = [msg for msg in messages if msg.role == "assistant"]
        
        avg_user_length = sum(len(msg.content) for msg in user_messages) / len(user_messages) if user_messages else 0
        avg_assistant_length = sum(len(msg.content) for msg in assistant_messages) / len(assistant_messages) if assistant_messages else 0
        
        return f"Conversation: {len(user_messages)} user messages, {len(assistant_messages)} assistant responses. Average lengths: user {avg_user_length:.0f} chars, assistant {avg_assistant_length:.0f} chars."


# Enhanced integration wrapper for existing code
class ConversationContextProvider:
    """
    Drop-in replacement for existing conversation history functions
    Provides enhanced capabilities while maintaining backward compatibility
    """
    
    def __init__(self):
        self.retriever = EnhancedConversationRetriever()
        
        # Backward compatibility settings
        self.legacy_format = True
        self.auto_migrate = True
    
    async def get_conversation_history(
        self,
        conversation_id: str,
        current_query: str = "",
        max_messages: Optional[int] = None,
        format: str = "string"
    ) -> str:
        """
        Enhanced replacement for get_conversation_history()
        Maintains backward compatibility while providing better results
        """
        
        if not conversation_id:
            return ""
        
        try:
            # Determine context type based on usage patterns
            context_type = "standard_chat"
            if max_messages and max_messages < 5:
                context_type = "multi_agent"
            elif any(keyword in current_query.lower() for keyword in ['document', 'search', 'find']):
                context_type = "rag_enhanced"
            
            # Get enhanced context
            context, metadata = await self.retriever.get_contextual_conversation_history(
                conversation_id, current_query, context_type
            )
            
            if format == "string":
                return context
            elif format == "dict":
                return {
                    "context": context,
                    "metadata": metadata
                }
            else:
                return context
                
        except Exception as e:
            print(f"[CONV_CONTEXT_PROVIDER] Error: {e}")
            return ""
    
    async def store_conversation_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Enhanced message storage with vector indexing
        """
        try:
            # Store in enhanced memory system
            message_id = await self.retriever.memory_manager.store_message(
                conversation_id, role, content, metadata
            )
            
            # Also index for vector search
            message = ConversationMessage(
                message_id=message_id,
                conversation_id=conversation_id,
                role=role,
                content=content,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            
            await self.retriever.vector_store.index_conversation_message(message)
            
        except Exception as e:
            print(f"[CONV_CONTEXT_PROVIDER] Storage error: {e}")
    
    async def search_conversation(
        self,
        conversation_id: str,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        New capability: search within conversation history
        """
        return await self.retriever.hybrid_conversation_search(
            conversation_id, query, limit=limit
        )